import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'saint'))
from models import SAINT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# 1. LOAD DATASET 
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


# 2. PREPROCESSING: MISSING VALUES & DUPLICATES
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

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

# 3. CLASS DISTRIBUTION
print("\n=== 3. CLASS DISTRIBUTION ===")

dist_counts = df[target_col].value_counts()
dist_percent = df[target_col].value_counts(normalize=True) * 100

print(dist_counts)

plt.figure(figsize=(7, 5))
# Menambahkan palette warna yang kontras tapi soft
ax = sns.barplot(
    x=dist_counts.index.map({0: 'Healthy (0)', 1: 'Parkinson (1)'}),
    y=dist_counts.values,
    hue=dist_counts.index,
    palette=["#2ecc71", "#e74c3c"], 
    legend=False
)

plt.title("Distribusi Kelas Parkinson)", fontsize=14, pad=15, fontweight='bold')
plt.xlabel("Status Kesehatan", labelpad=10)
plt.ylabel("Jumlah Sampel", labelpad=10)
plt.ylim(0, max(dist_counts.values) * 1.15) 


for i, count in enumerate(dist_counts.values):
    pct = dist_percent.values[i]
    ax.text(
        i,
        count + 3,
        f"{count} data\n({pct:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='#333333'
    )

plt.tight_layout()
plt.show()


# 4. VISUALIZATION (HISTOGRAM & BOXPLOT SEBELUM)
# Histogram Fitur Numerik
df[num_cols].hist(bins=20, figsize=(15, 12), color='skyblue', edgecolor='black', grid=False)
plt.suptitle("Histogram Distribusi Fitur Numerik", fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

# Boxplot SEBELUM Outlier Handling 
num_features = len(num_cols)
rows = (num_features + 3) // 4 

plt.figure(figsize=(16, rows * 3))
for i, col in enumerate(num_cols):
    plt.subplot(rows, 4, i + 1)
    sns.boxplot(y=df[col], color='#f39c12', width=0.5)
    plt.title(col, fontsize=11, fontweight='semibold')
    plt.ylabel('') 
    plt.xlabel('')

plt.suptitle("Boxplot SEBELUM Penanganan Outlier)", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# 5. OUTLIER HANDLING WITH IQR (CAPPING / WINSORIZATION)
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

# Boxplot SESUDAH Outlier Handling 
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

# 6. MIN-MAX SCALING
print("\n=== 6. MIN-MAX SCALING ===")
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head(7))

# 7. STRATIFIED SPLIT (80:10:10)
print("\n=== 7. STRATIFIED SPLIT ===")
X = df.drop(columns=[target_col]).values
y = df[target_col].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# 8. SMOTE
print("\n=== 8. SMOTE ===")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Sesudah SMOTE -> Label 1: {sum(y_train_res==1)}, Label 0: {sum(y_train_res==0)}")

train_ds = TensorDataset(torch.tensor(X_train_res, dtype=torch.float32), torch.tensor(y_train_res, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 9. SAINT TRAINING LOOP WITH METRICS
print("\n=== 9. TRAINING SAINT ===")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAINT(
    categories=[2],
    num_continuous=X_train_res.shape[1] - 1,
    dim=32,
    dim_out=2,
    depth=6,
    heads=8,
    attn_dropout=0.3,
    ff_dropout=0.3,
    mlp_act=nn.GELU()
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
epochs = 30

patience = 5         
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
best_epoch = 0

history = {
    'loss': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': []
}

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        x_categ = torch.zeros(
            (inputs.size(0), 1),
            dtype=torch.long,
            device=device
        )

        x_cont = inputs[:, 1:]

        _, outputs = model(x_categ, x_cont)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()

    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            x_categ = torch.zeros(
                (inputs.size(0), 1),
                dtype=torch.long,
                device=device
            )

            x_cont = inputs[:, 1:]

            _, outputs = model(x_categ, x_cont)

            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss /= len(val_loader.dataset)

    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, zero_division=0)
    val_rec = recall_score(all_targets, all_preds, zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Acc: {val_acc:.4f} | "
        f"Prec: {val_prec:.4f} | "
        f"Rec: {val_rec:.4f} | "
        f"F1: {val_f1:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n>> Early stopping di epoch {epoch} (val_loss tidak membaik selama {patience} epoch)")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\n>> Model terbaik dipakai dari epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")

model.eval()

test_preds = []
test_proba = []
test_targets = []

with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(device)

        x_categ = torch.zeros(
            (inputs.size(0), 1),
            dtype=torch.long,
            device=device
        )

        x_cont = inputs[:, 1:]

        _, outputs = model(x_categ, x_cont)

        proba = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_proba.extend(proba.cpu().numpy())
        test_targets.extend(targets.numpy())

# 9b. REKAP METRIK EVALUASI
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
print(f"Model terbaik dipakai dari epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
print(f"Total epoch dijalankan: {len(history['loss'])} dari maksimal {epochs} (early stopping patience={patience})")

# 10. CONFUSION MATRIX
print("\n=== 10. CONFUSION MATRIX ===")
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], yticklabels=['Healthy', 'Parkinson'])
plt.title("Confusion Matrix - SAINT")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 11. GRAPH VALIDATION
print("\n=== 11. GRAPH VALIDATION ===")
epochs_ran = len(history['loss'])  # bisa lebih kecil dari `epochs` kalau early stopping aktif

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(range(1, epochs_ran + 1), history['loss'], label='Train Loss', marker='o')
ax1.plot(range(1, epochs_ran + 1), history['val_loss'], label='Val Loss', marker='o')
ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch})')
ax1.set_title("Loss History")
ax1.legend()

ax2.plot(range(1, epochs_ran + 1), history['val_acc'], label='Val Accuracy', marker='s')
ax2.plot(range(1, epochs_ran + 1), history['val_f1'], label='Val F1-Score', marker='^')
ax2.axvline(best_epoch, color='green', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch})')
ax2.set_title("Performance History")
ax2.legend()
plt.tight_layout()
plt.show()

# 12. ROC CURVE
print("\n=== 12. ROC CURVE ===")
fpr, tpr, _ = roc_curve(test_targets, test_proba)
auc_score = roc_auc_score(test_targets, test_proba)
plt.figure(figsize=(6, 4.5))
plt.plot(fpr, tpr, label=f'SAINT (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

# 13. SHAP
print("\n=== 13. SHAP EXPLAINABILITY ===")
shap_feature_names = num_cols[1:]

def custom_predict(x_numpy):
    tensor_x = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    x_categ_dummy = torch.zeros((tensor_x.size(0), 1), dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        _, outputs = model(x_categ_dummy, tensor_x)
        proba = torch.softmax(outputs, dim=1)
    return proba.cpu().numpy()

X_train_res_shap = X_train_res[:, 1:]
X_test_shap = X_test[:10, 1:]

background = shap.kmeans(X_train_res_shap, 5)
explainer = shap.KernelExplainer(custom_predict, background)
shap_values = explainer.shap_values(X_test_shap)

if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values[:, :, 1]

shap.summary_plot(shap_values_class1, X_test_shap, feature_names=shap_feature_names)


# 14. SAVE MODEL & SCALER UNTUK DEPLOYMENT
print("\n=== 14. SAVE MODEL & SCALER ===")
joblib.dump(scaler, "scaler.pkl")
torch.save(best_model_state, "saint_model.pt")
model_config = {
    "categories": [2],
    "num_continuous": X_train_res.shape[1] - 1,
    "dim": 32,
    "dim_out": 2,
    "depth": 6,
    "heads": 8,
    "attn_dropout": 0.3,
    "ff_dropout": 0.3,
}
joblib.dump(model_config, "model_config.pkl")
joblib.dump(num_cols, "feature_columns.pkl")
print("Tersimpan: scaler.pkl, saint_model.pt, model_config.pkl, feature_columns.pkl")
print(f"Total fitur: {len(num_cols)} kolom -> {num_cols}")