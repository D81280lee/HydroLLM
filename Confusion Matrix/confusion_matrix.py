import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib as mpl

# Set publication-quality style parameters
plt.style.use('default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 1.2

def create_publication_confusion_matrix(y_true, y_pred, model_name, class_names, ax):
    """
    Creates a publication-ready confusion matrix heatmap
    """
    # Compute confusion matrix and normalize
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    
    # Customize the plot
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(class_names, fontsize=14)
    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=10, fontsize=16)
    ax.set_ylabel('True Label', fontweight='bold', labelpad=10, fontsize=16)
    ax.set_title(f'{model_name}\nNormalized Confusion Matrix', 
                 fontweight='bold', pad=15, fontsize=16)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(class_names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names)), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return im

# 1. Load and prepare your data
print("Loading and preprocessing data...")
df = pd.read_csv('data_for_confusion.csv')  # Replace with your actual file path

# Convert text to features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['summary']).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['loss_category'])
class_names = label_encoder.classes_

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train models
print("Training models...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# 4. Generate predictions
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# 5. Create the publication-ready figure
print("Creating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Create confusion matrices
im1 = create_publication_confusion_matrix(y_test, y_pred_rf, 'Random Forest', class_names, ax1)
im2 = create_publication_confusion_matrix(y_test, y_pred_xgb, 'XGBoost', class_names, ax2)

# Add a shared colorbar
cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Proportion of True Class', rotation=270, labelpad=15, fontweight='bold', fontsize=16)

# Adjust layout and save
plt.tight_layout(rect=(0, 0, 0.9, 0.95))  # Make room for colorbar

# Save as high-resolution TIFF/PNG for publication
plt.savefig('confusion_matrix_comparison.tiff', dpi=300, bbox_inches='tight', 
            format='tiff', facecolor='white')
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white')

plt.show()

# 6. Print performance metrics in a table format
print("\n" + "="*70)
print("COMPREHENSIVE PERFORMANCE METRICS")
print("="*70)

from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

metrics_data = []
for model_name, y_pred in [('Random Forest', y_pred_rf), ('XGBoost', y_pred_xgb)]:
    metrics_data.append({
        'Model': model_name,
        'Macro F1': f"{f1_score(y_test, y_pred, average='macro'):.3f}",
        'Weighted Accuracy': f"{accuracy_score(y_test, y_pred):.3f}",
        "Cohen's Kappa": f"{cohen_kappa_score(y_test, y_pred):.3f}"
    })

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))