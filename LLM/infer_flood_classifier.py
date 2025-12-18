#!/usr/bin/env python3
import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel

# optional plotting libs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    _PLOTTING_AVAILABLE = True
except Exception:
    _PLOTTING_AVAILABLE = False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--adapter_subdir", default="adapter")
    ap.add_argument("--text", help="Single text to classify (use with no --eval_file)")
    ap.add_argument("--eval_file", help="CSV/JSONL file with texts and labels to run batch inference for confusion matrix")
    ap.add_argument("--text_column", default="summary")
    ap.add_argument("--label_column", default="loss_category")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--save_confusion", action="store_true", help="Save confusion matrix PNG/CSV when eval_file is provided")
    return ap.parse_args()

def load_map(p):
    with open(p) as f:
        m = json.load(f)
    return {int(k): v for k, v in m["id2label"].items()}

def save_confusion_matrix(true_labels, pred_labels, label_names, out_png, out_csv=None):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(label_names))))
    if _PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(4, len(label_names))))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    if out_csv:
        np.savetxt(out_csv, cm, fmt="%d", delimiter=",")

def load_model_and_tokenizer(base_model, output_dir, adapter_subdir):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"

    ad = os.path.join(output_dir, adapter_subdir)
    if os.path.isdir(ad):
        cfg = AutoConfig.from_pretrained(base_model, num_labels=None)  # will set below
        # we will load config with correct num_labels later if label map is present
        mdl = AutoModelForSequenceClassification.from_pretrained(base_model, config=cfg, device_map="auto")
        mdl = PeftModel.from_pretrained(mdl, ad)
    else:
        mdl = AutoModelForSequenceClassification.from_pretrained(output_dir, device_map="auto")
    return tok, mdl

def predict_batch(model, tokenizer, texts, device, batch_size=16, max_length=512):
    model.to(device)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
            all_logits.append(logits)
    return np.vstack(all_logits)

def main():
    a = parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    # load label mapping
    label_map_path = os.path.join(a.output_dir, "label_mapping.json")
    if not os.path.isfile(label_map_path):
        raise SystemExit(f"label_mapping.json not found in {a.output_dir}")
    id2label = load_map(label_map_path)
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    label2id = {v: k for k, v in id2label.items()}

    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"

    ad = os.path.join(a.output_dir, a.adapter_subdir)
    if os.path.isdir(ad):
        cfg = AutoConfig.from_pretrained(a.base_model, num_labels=len(label_names), id2label=id2label, label2id={v: k for k, v in id2label.items()})
        mdl = AutoModelForSequenceClassification.from_pretrained(a.base_model, config=cfg, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None), device_map="auto")
        mdl = PeftModel.from_pretrained(mdl, ad)
    else:
        cfg = AutoConfig.from_pretrained(a.output_dir, num_labels=len(label_names), id2label=id2label, label2id={v: k for k, v in id2label.items()})
        mdl = AutoModelForSequenceClassification.from_pretrained(a.output_dir, config=cfg, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None), device_map="auto")

    device = next(mdl.parameters()).device if any(True for _ in mdl.parameters()) else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Single-text inference
    if a.text and not a.eval_file:
        enc = tok(a.text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred = int(torch.argmax(logits, dim=-1).item())
        print("Prediction:", id2label[pred])
        print("Probabilities:", {id2label[i]: float(p) for i, p in enumerate(probs)})
        return

    # Batch inference for confusion matrix / evaluation
    if a.eval_file:
        # load file (support CSV and JSONL)
        if a.eval_file.endswith(".csv"):
            df = pd.read_csv(a.eval_file)
        else:
            df = pd.read_json(a.eval_file, lines=True)

        if a.text_column not in df.columns:
            raise SystemExit(f"text column '{a.text_column}' not found in {a.eval_file}")
        texts = df[a.text_column].astype(str).tolist()
        has_labels = a.label_column in df.columns
        if has_labels:
            raw_labels = df[a.label_column].tolist()
            # map string labels to ids if necessary
            if isinstance(raw_labels[0], str):
                try:
                    labels = [label2id[l] for l in raw_labels]
                except KeyError as e:
                    raise SystemExit(f"Label value not found in label mapping: {e}")
            else:
                labels = [int(x) for x in raw_labels]
            labels = np.asarray(labels, dtype=int)
        else:
            labels = None

        logits = predict_batch(mdl, tok, texts, device, batch_size=a.batch_size)
        preds = np.argmax(logits, axis=-1).astype(int)

        # output predictions appended to file
        out_df = df.copy()
        out_df["pred_id"] = preds
        out_df["pred_label"] = [id2label[int(p)] for p in preds]
        out_path_preds = os.path.join(a.output_dir, "predictions.csv")
        out_df.to_csv(out_path_preds, index=False)
        print(f"Wrote predictions to {out_path_preds}")

        if has_labels:
            # confusion matrix + report
            if not _PLOTTING_AVAILABLE and a.save_confusion:
                print("Plotting libraries not available (matplotlib/seaborn/scikit-learn). Confusion PNG will not be created, raw CSV will be saved.")
            cm_png = os.path.join(a.output_dir, "confusion_matrix.png")
            cm_csv = os.path.join(a.output_dir, "confusion_matrix.csv")
            if a.save_confusion:
                if not _PLOTTING_AVAILABLE:
                    # still save raw matrix
                    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
                    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
                    print(f"Saved confusion matrix CSV to {cm_csv} (PNG not available)")
                else:
                    save_confusion_matrix(labels, preds, label_names, cm_png, cm_csv)
                    print(f"Saved confusion matrix PNG to {cm_png} and CSV to {cm_csv}")

            # print textual classification report
            if _PLOTTING_AVAILABLE:
                print(classification_report(labels, preds, target_names=label_names, digits=4))
            else:
                # fallback minimal metrics
                from collections import Counter
                correct = int((labels == preds).sum())
                print(f"Accuracy: {correct}/{len(labels)} = {correct/len(labels):.4f}")
        else:
            print("No labels present in eval file; saved predictions only.")

if __name__ == "__main__":
    main()
