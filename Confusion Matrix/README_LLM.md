# Using Your Trained LLM Adapters for Confusion Matrix Analysis

## Overview
This guide shows how to use your trained LLM adapters (LoRA/PEFT) for generating confusion matrices instead of traditional ML models.

## Setup Instructions

### 1. Install Required Packages
```powershell
# Install PyTorch (choose appropriate version for your system)
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have a compatible GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements_llm.txt
```

### 2. Prepare Your Model Paths
Update the `model_configs` section in `confusion_matrix_llm.py` with your actual model paths:

```python
model_configs = [
    {
        'name': 'Your Model Name 1',
        'base_model': 'path/to/your/base/model',  # e.g., 'bert-base-uncased'
        'adapter_path': r'C:\path\to\your\adapter1'
    },
    {
        'name': 'Your Model Name 2', 
        'base_model': 'path/to/your/base/model2',
        'adapter_path': r'C:\path\to\your\adapter2'
    }
]
```

## Model Structure Expected

Your adapter folders should contain:
```
your_adapter_folder/
├── adapter_config.json
├── adapter_model.bin (or adapter_model.safetensors)
└── README.md (optional)
```

## Key Differences from Traditional ML

### 1. **No Feature Engineering Required**
- Traditional: TF-IDF vectorization needed
- LLM: Raw text input directly

### 2. **Model Loading**
```python
# Traditional
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# LLM Adapter
model = LLMAdapter(base_model_path, adapter_path, model_name)
predictions = model.predict(texts)
```

### 3. **Prediction Format**
- Traditional: Returns numeric class indices
- LLM: Returns string labels that need mapping back to indices

## Usage Examples

### For LoRA Adapters:
```python
model_configs = [
    {
        'name': 'LoRA-BERT',
        'base_model': 'bert-base-uncased',
        'adapter_path': r'C:\models\lora_bert_adapter'
    }
]
```

### For Different Base Models:
```python
model_configs = [
    {
        'name': 'Fine-tuned RoBERTa',
        'base_model': 'roberta-base',
        'adapter_path': r'C:\models\roberta_adapter'
    },
    {
        'name': 'Fine-tuned DistilBERT',
        'base_model': 'distilbert-base-uncased',
        'adapter_path': r'C:\models\distilbert_adapter'
    }
]
```

## Performance Considerations

### Memory Usage:
- Each model loads the full base model + adapter
- Consider using smaller base models or model offloading for multiple models
- Batch size affects GPU memory usage

### Speed Optimization:
- Use GPU if available (`torch.cuda.is_available()`)
- Adjust batch size in the `predict` method
- Consider using `torch.compile()` for faster inference (PyTorch 2.0+)

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   - Reduce batch size in the `predict` method
   - Use CPU instead: set `device=-1` in pipeline

2. **Model Loading Errors**
   - Verify adapter paths exist
   - Check adapter compatibility with base model
   - Ensure adapter was saved correctly

3. **Label Mapping Issues**
   - Ensure your model outputs match the expected class names
   - Check the `label_to_encoded` mapping

### Debug Mode:
Add these lines to see what your model is predicting:
```python
# After getting predictions
print("Sample predictions:", y_pred_strings_model1[:5])
print("Expected format:", list(class_names))
```

## Advanced Features

### Multiple GPU Support:
```python
# Modify the LLMAdapter class
self.classifier = pipeline(
    "text-classification",
    model=self.model,
    tokenizer=self.tokenizer,
    device_map="auto",  # Automatic multi-GPU
    torch_dtype=torch.float16
)
```

### Custom Classification Threshold:
```python
def predict_with_confidence(self, texts, threshold=0.8):
    """Return predictions only above confidence threshold"""
    # Implementation details...
```

## Output Files Generated:
- `llm_confusion_matrix_comparison.png/tiff`: Visualization
- `llm_predictions_results.csv`: Detailed predictions for analysis
