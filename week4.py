# =============================================================================
# Title: Assignment 4.2 - Building Classification Models
# Author: Pankaj Yadav
# Date: [Current Date]
# Modified By: Pankaj Yadav
# Description: Building and evaluating classification models on Movie Reviews Dataset
# =============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import unicodedata
import sys

# Load the dataset
df = pd.read_csv('labeledTrainData.tsv', sep='\t')

# Preprocessing (same as Assignment 3.2)
# Remove HTML tags
df['review'] = df['review'].str.replace(r'<.*?>', ' ', regex=True)

# Convert to lowercase
df['review'] = df['review'].str.lower()

# Remove punctuation
punctuation = dict.fromkeys(
    (i for i in range(sys.maxunicode)
     if unicodedata.category(chr(i)).startswith('P')),
    None
)
df['review'] = df['review'].apply(lambda x: x.translate(punctuation))

# Remove extra whitespaces and numbers
df['review'] = df['review'].str.replace(r'\s+', ' ', regex=True)
df['review'] = df['review'].str.replace(r'\d+', '', regex=True)

# Tokenization
df['tokenized_review'] = df['review'].apply(word_tokenize)

# Stop words removal
stop_words = set(stopwords.words('english'))
df['tokenized_review'] = df['tokenized_review'].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

# Stemming
ps = PorterStemmer()
df['stemmed_review'] = df['tokenized_review'].apply(
    lambda tokens: [ps.stem(word) for word in tokens]
)

# Join tokens back to string
df['stemmed_review_str'] = df['stemmed_review'].apply(lambda tokens: ' '.join(tokens))

print("Preprocessing completed!")
print(f"Dataset shape: {df.shape}")
print(df[['sentiment', 'stemmed_review_str']].head(3))

# Step 2: Split into training and test set (80-20 split)
X = df['stemmed_review_str']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set positive sentiment ratio: {y_train.mean():.2%}")
print(f"Test set positive sentiment ratio: {y_test.mean():.2%}")

# Step 3: Fit and apply TF-IDF vectorization to training set
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print(f"\nTF-IDF training matrix shape: {X_train_tfidf.shape}")

# Step 4: Apply (DO NOT FIT) TF-IDF vectorization to test set
# We use transform() only, not fit_transform(), because we want to use
# the same vocabulary and IDF weights learned from the training data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF test matrix shape: {X_test_tfidf.shape}")
print("\nWhy not fit on test set? Because:")
print("- Fitting on test data would use information from test set")
print("- This would cause data leakage and overly optimistic performance")
print("- The model must only learn from training data")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

# Step 5: Train logistic regression
print("\n" + "="*70)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*70)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Step 6: Find model accuracy on test set
y_pred_lr = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

print(f"\nLogistic Regression Test Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# Step 7: Create confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm_lr)

# Step 8: Get precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, 
                          target_names=['Negative', 'Positive']))

# Step 9: Create ROC curve
y_pred_proba_lr = lr_model.predict_proba(X_test_tfidf)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_lr:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nROC AUC Score: {roc_auc_lr:.4f}")

# ============================================================================
# MODEL 2: RANDOM FOREST CLASSIFIER
# ============================================================================

# Step 10: Pick another classification model and repeat steps 5-9
print("\n" + "="*70)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("="*70)

# Step 5: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

# Step 6: Find model accuracy on test set
y_pred_rf = rf_model.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Test Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# Step 7: Create confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm_rf)

# Step 8: Get precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf,
                          target_names=['Negative', 'Positive']))

# Step 9: Create ROC curve
y_pred_proba_rf = rf_model.predict_proba(X_test_tfidf)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2,
         label=f'ROC curve (AUC = {roc_auc_rf:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nROC AUC Score: {roc_auc_rf:.4f}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Compare accuracies
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, rf_accuracy],
    'ROC AUC': [roc_auc_lr, roc_auc_rf]
})

print("\n", comparison_df)

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(comparison_df['Model'], comparison_df['Accuracy'], 
            color=['darkorange', 'green'])
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.7, 1.0])
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

# ROC curves comparison
axes[1].plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
            label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
axes[1].plot(fpr_rf, tpr_rf, color='green', lw=2,
            label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance (for Random Forest)
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
print("="*70)

feature_names = tfidf_vectorizer.get_feature_names_out()
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

print("\nTop 10 features:")
for i, idx in enumerate(indices, 1):
    print(f"{i}. {feature_names[idx]}: {importances[idx]:.6f}")

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices][::-1], 
         importances[indices][::-1], color='green')
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 10 Most Important Features - Random Forest', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()