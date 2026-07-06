import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from imblearn.over_sampling import SMOTE
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier

# ==========================================
# 1. LOAD DATASET & INITIAL PREPROCESSING
# ==========================================
print("=== 1. LOAD DATASET ===")

# Load dataset Parkinson
df = pd.read_csv("parkinsons.data")

print(f"Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print(df.head())

# Hapus kolom nama pasien
df.drop(columns=["name"], inplace=True)

# Target
target_col = "status"

# Semua fitur selain target adalah numerik
num_cols = [col for col in df.columns if col != target_col]
cat_cols = []


# ==========================================
# 2. PREPROCESSING: MISSING VALUES & DUPLICATES
# ==========================================
print("\n=== 2. PREPROCESSING ===")

print(f"Missing values sebelum: {df.isnull().sum().sum()}")

# Imputasi median jika ada missing value
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print(f"Missing values sesudah: {df.isnull().sum().sum()}")

# Hapus data duplikat
print(f"Data duplikat sebelum: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Data duplikat sesudah: {df.duplicated().sum()}")

# Setting tema global Seaborn agar terlihat modern dan bersih
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

# ==========================================
# 3. CLASS DISTRIBUTION
# ==========================================
print("\n=== 3. CLASS DISTRIBUTION ===")

dist_counts = df[target_col].value_counts()
dist_percent = df[target_col].value_counts(normalize=True) * 100

print(dist_counts)

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    x=dist_counts.index.map({0: 'Healthy (0)', 1: 'Parkinson (1)'}),
    y=dist_counts.values,
    hue=dist_counts.index,
    palette=["#2ecc71", "#e74c3c"],
    legend=False
)

plt.title("Distribusi Kelas Parkinson\n(Terdeteksi Imbalanced Data)", fontsize=14, pad=15, fontweight='bold')
plt.xlabel("Status Kesehatan", labelpad=10)
plt.ylabel("Jumlah Sampel", labelpad=10)
plt.ylim(0, max(dist_counts.values) * 1.15)

for i, count in enumerate(dist_counts.values):
    pct = dist_percent.values[i]
    ax.text(
        i, count + 3, f"{count} data\n({pct:.1f}%)",
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333'
    )

plt.tight_layout()
plt.show()

# ==========================================
# 4. VISUALIZATION (HISTOGRAM & BOXPLOT SEBELUM)
# ==========================================
df[num_cols].hist(bins=20, figsize=(15, 12), color='skyblue', edgecolor='black', grid=False)
plt.suptitle("Histogram Distribusi Fitur Numerik", fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

num_features = len(num_cols)
rows = (num_features + 3) // 4

plt.figure(figsize=(16, rows * 3))
for i, col in enumerate(num_cols):
    plt.subplot(rows, 4, i + 1)
    sns.boxplot(y=df[col], color='#f39c12', width=0.5)
    plt.title(col, fontsize=11, fontweight='semibold')
    plt.ylabel('')
    plt.xlabel('')

plt.suptitle("Boxplot Fitur Numerik (SEBELUM Penanganan Outlier)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ==========================================
# 5. OUTLIER HANDLING WITH IQR (CAPPING / WINSORIZATION)
# ==========================================
print("\n=== 5. OUTLIER HANDLING (IQR CAPPING) ===")
print(f"Ukuran data sebelum penanganan outlier: {df.shape}")

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print(f"Ukuran data sesudah penanganan outlier: {df.shape} (Baris tetap utuh!)")

plt.figure(figsize=(16, rows * 3))
for i, col in enumerate(num_cols):
    plt.subplot(rows, 4, i + 1)
    sns.boxplot(y=df[col], color='#3498db', width=0.5)
    plt.title(col, fontsize=11, fontweight='semibold')
    plt.ylabel('')
    plt.xlabel('')

plt.suptitle("Boxplot Fitur Numerik (SESUDAH Penanganan Outlier Capping)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ==========================================
# 6. MIN-MAX SCALING
# ==========================================
print("\n=== 6. MIN-MAX SCALING ===")
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head(7))

# ==========================================
# 7. STRATIFIED SPLIT (80:10:10)
# ==========================================
print("\n=== 7. STRATIFIED SPLIT ===")
X = df.drop(columns=[target_col]).values
y = df[target_col].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ==========================================
# 8. SMOTE (HANYA DI DATA TRAINING)
# ==========================================
print("\n=== 8. SMOTE ===")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Sebelum SMOTE -> Label 1: {sum(y_train==1)}, Label 0: {sum(y_train==0)}")
print(f"Sesudah SMOTE -> Label 1: {sum(y_train_res==1)}, Label 0: {sum(y_train_res==0)}")

# ==========================================
# 9. TABNET TRAINING
# ==========================================
print("\n=== 9. TRAINING TABNET ===")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device_name = "cuda" if torch.cuda.is_available() else "cpu"

# Semua fitur bersifat kontinu (tidak ada kolom kategorikal asli),
# jadi cat_idxs / cat_dims dibiarkan kosong -> TabNet menganggap semua fitur numerik.
model = TabNetClassifier(
    n_d=4,
    n_a=4,
    n_steps=3,
    gamma=1.3,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-3,
    momentum=0.02,
    mask_type="sparsemax",
    seed=SEED,
    verbose=1,
    device_name=device_name,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params=dict(step_size=30, gamma=0.9),
)

# ---- EARLY STOPPING CONFIG ----
# Dataset sangat kecil -> beri patience lebih longgar & epoch lebih banyak
# supaya model tidak berhenti terlalu dini pada minimum lokal yang belum optimal.
patience = 20
max_epochs = 300

model.fit(
    X_train=X_train_res,
    y_train=y_train_res,
    eval_set=[(X_train_res, y_train_res), (X_val, y_val)],
    eval_name=["train", "val"],
    eval_metric=["logloss", "accuracy"],
    max_epochs=max_epochs,
    patience=patience,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
    weights=0,          # data sudah diseimbangkan lewat SMOTE
    drop_last=False,
)

best_epoch = model.best_epoch if hasattr(model, "best_epoch") else len(model.history["loss"]) - patience
print(f"\n>> Model terbaik (early stopping) dipakai dari sekitar epoch: {best_epoch}")

history = model.history.history

# ================= TEST =================
test_proba_all = model.predict_proba(X_test)
test_proba = test_proba_all[:, 1]
test_preds = np.argmax(test_proba_all, axis=1)
test_targets = y_test

# ==========================================
# 9b. REKAP METRIK EVALUASI
# ==========================================
print("\n=== 9b. REKAP METRIK EVALUASI (TEST SET) ===")

test_acc = accuracy_score(test_targets, test_preds)
test_prec = precision_score(test_targets, test_preds, zero_division=0)
test_rec = recall_score(test_targets, test_preds, zero_division=0)
test_f1 = f1_score(test_targets, test_preds, zero_division=0)
test_auc = roc_auc_score(test_targets, test_proba)

cm_test = confusion_matrix(test_targets, test_preds)
tn, fp, fn, tp = cm_test.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

recap_df = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "AUC-ROC"],
    "Nilai": [test_acc, test_prec, test_rec, specificity, test_f1, test_auc]
})
recap_df["Nilai"] = recap_df["Nilai"].map(lambda x: f"{x:.4f}")

print(recap_df.to_string(index=False))
print(f"\nConfusion Matrix -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# 10. CONFUSION MATRIX
print("\n=== 10. CONFUSION MATRIX ===")
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], yticklabels=['Healthy', 'Parkinson'])
plt.title("Confusion Matrix - TabNet")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 11. GRAPH VALIDATION (loss & accuracy history dari pytorch-tabnet)
print("\n=== 11. GRAPH VALIDATION ===")
epochs_ran = len(history['loss'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(range(1, epochs_ran + 1), history['train_logloss'], label='Train Loss', marker='o', markevery=max(1, epochs_ran // 20))
ax1.plot(range(1, epochs_ran + 1), history['val_logloss'], label='Val Loss', marker='o', markevery=max(1, epochs_ran // 20))
ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch})')
ax1.set_title("Loss History")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(range(1, epochs_ran + 1), history['train_accuracy'], label='Train Accuracy', marker='s', markevery=max(1, epochs_ran // 20))
ax2.plot(range(1, epochs_ran + 1), history['val_accuracy'], label='Val Accuracy', marker='s', markevery=max(1, epochs_ran // 20))
ax2.axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch})')
ax2.set_title("Performance History")
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.show()

# 12. ROC CURVE
print("\n=== 12. ROC CURVE ===")
fpr, tpr, _ = roc_curve(test_targets, test_proba)
auc_score = roc_auc_score(test_targets, test_proba)
plt.figure(figsize=(6, 4.5))
plt.plot(fpr, tpr, label=f'TabNet (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

# ==========================================
# 13. FEATURE IMPORTANCE (bawaan TabNet) + SHAP
# ==========================================
print("\n=== 13a. TABNET NATIVE FEATURE IMPORTANCE ===")

feature_names = num_cols  # semua kolom dipakai (tidak ada kolom yang dibuang seperti di SAINT)

importance_df = pd.DataFrame({
    "Fitur": feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 8))
sns.barplot(data=importance_df, x="Importance", y="Fitur", color="#3498db")
plt.title("TabNet Feature Importance (Native Attention Mask)")
plt.tight_layout()
plt.show()

print("\n=== 13b. SHAP EXPLAINABILITY ===")

def custom_predict(x_numpy):
    return model.predict_proba(x_numpy)

X_train_res_shap = X_train_res
X_test_shap = X_test[:10, :]

background = shap.kmeans(X_train_res_shap, 5)
explainer = shap.KernelExplainer(custom_predict, background)
shap_values = explainer.shap_values(X_test_shap)

if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values[:, :, 1]

shap.summary_plot(shap_values_class1, X_test_shap, feature_names=feature_names)