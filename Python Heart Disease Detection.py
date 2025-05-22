# Heart Disease Detection
# Deteksi Penyakit Jantung berdasarkan Data Medis Pasien

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 1. Loading dan Eksplorasi Data
print("Loading dan Eksplorasi Data")
print("--------------------------")

# Mencoba beberapa lokasi file yang mungkin
file_paths = [
    '/kaggle/input/heart-disease-from-cleveland/heart_disease_cleveland.csv',
    '/kaggle/input/heart-disease-from-cleveland/processed.cleveland.data',
    'heart_disease_cleveland.csv',
    'processed.cleveland.data'
]

df = None
for path in file_paths:
    try:
        print(f"Mencoba membaca file dari: {path}")
        if path.endswith('.data'):
            # Untuk file .data dari UCI, kita perlu mendefinisikan nama kolom
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
            df = pd.read_csv(path, names=column_names, na_values='?')
        else:
            df = pd.read_csv(path)
        print(f"Berhasil membaca file dari: {path}")
        break
    except FileNotFoundError:
        print(f"File tidak ditemukan di: {path}")
        continue

# Jika tidak ada file yang ditemukan, buat dataset sample untuk demonstrasi
if df is None:
    print("\nTidak dapat menemukan file dataset. Membuat dataset sample untuk demonstrasi...")
    
    # Buat dataset sample Cleveland Heart Disease
    np.random.seed(42)
    n_samples = 303
    
    # Generate data sample yang mirip dengan dataset asli
    data = {
        'age': np.random.randint(29, 78, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(130, 15, n_samples).astype(int),
        'chol': np.random.normal(240, 50, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n_samples),
        'thalach': np.random.normal(150, 25, n_samples).astype(int),
        'exang': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.exponential(1, n_samples).round(1),
        'slope': np.random.choice([0, 1, 2], n_samples),
        'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.1, 0.7, 0.15]),
        'num': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.54, 0.18, 0.12, 0.1, 0.06])
    }
    
    df = pd.DataFrame(data)
    print("Dataset sample berhasil dibuat!")
    print("Catatan: Ini adalah data sample untuk demonstrasi. Untuk hasil yang akurat, gunakan dataset asli.")

# Menampilkan informasi awal dataset
print("Ukuran Dataset:", df.shape)
print("\nLima data pertama:")
print(df.head())

print("\nInformasi Dataset:")
print(df.info())

print("\nStatistik Deskriptif:")
print(df.describe())

print("\nJumlah nilai yang hilang per kolom:")
print(df.isnull().sum())

# 2. Data Preprocessing
print("\n\nData Preprocessing")
print("-----------------")

# Mengganti nilai '?' dengan NaN
df = df.replace('?', np.nan)

# Mengkonversi kolom dengan tipe objek ke numerik
for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Jumlah nilai yang hilang setelah konversi:")
print(df.isnull().sum())

# Mengisi nilai yang hilang dengan median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

print("\nJumlah nilai yang hilang setelah diisi:")
print(df.isnull().sum())

# Memeriksa distribusi target (num)
# Mengubah num menjadi kategorikal (0: tidak ada penyakit, 1: ada penyakit)
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

print("\nDistribusi Target:")
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True) * 100)

# 3. Visualisasi Data
print("\n\nVisualisasi Data")
print("---------------")

# Plot distribusi target
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribusi Target: Penyakit Jantung')
plt.xlabel('Status Penyakit Jantung (0: Tidak Ada, 1: Ada)')
plt.ylabel('Jumlah')
plt.savefig('target_distribution.png')
plt.close()

# Korelasi antara fitur
plt.figure(figsize=(12, 10))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Fitur')
plt.savefig('correlation_matrix.png')
plt.close()

# Plot beberapa fitur penting terhadap target
plt.figure(figsize=(20, 15))

# Age vs Target
plt.subplot(2, 3, 1)
sns.boxplot(x='target', y='age', data=df)
plt.title('Usia vs Penyakit Jantung')
plt.xlabel('Penyakit Jantung')
plt.ylabel('Usia')

# Trestbps (Tekanan Darah) vs Target
plt.subplot(2, 3, 2)
sns.boxplot(x='target', y='trestbps', data=df)
plt.title('Tekanan Darah vs Penyakit Jantung')
plt.xlabel('Penyakit Jantung')
plt.ylabel('Tekanan Darah (mm Hg)')

# Chol (Kolesterol) vs Target
plt.subplot(2, 3, 3)
sns.boxplot(x='target', y='chol', data=df)
plt.title('Kolesterol vs Penyakit Jantung')
plt.xlabel('Penyakit Jantung')
plt.ylabel('Kolesterol (mg/dl)')

# Thalach (Detak Jantung Maksimum) vs Target
plt.subplot(2, 3, 4)
sns.boxplot(x='target', y='thalach', data=df)
plt.title('Detak Jantung Maksimum vs Penyakit Jantung')
plt.xlabel('Penyakit Jantung')
plt.ylabel('Detak Jantung Maksimum')

# Oldpeak vs Target
plt.subplot(2, 3, 5)
sns.boxplot(x='target', y='oldpeak', data=df)
plt.title('ST Depression vs Penyakit Jantung')
plt.xlabel('Penyakit Jantung')
plt.ylabel('ST Depression')

plt.tight_layout()
plt.savefig('feature_vs_target.png')
plt.close()

# 4. Pemilihan Fitur dan Pemrosesan
print("\n\nPemilihan Fitur dan Pemrosesan")
print("-----------------------------")

# Menghapus kolom yang tidak digunakan
df_model = df.drop(['num'], axis=1)

# Memilih fitur berdasarkan korelasi dengan target
correlation_with_target = abs(df_model.corr()['target'])
print("Korelasi fitur dengan target:")
print(correlation_with_target.sort_values(ascending=False))

# Split data menjadi fitur dan target
X = df_model.drop('target', axis=1)
y = df_model['target']

# Membagi data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nUkuran data training: {X_train.shape}")
print(f"Ukuran data testing: {X_test.shape}")

# 5. Pemodelan dengan Random Forest
print("\n\nPemodelan dengan Random Forest")
print("-----------------------------")

# Membangun model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluasi model dengan cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

# Prediksi pada data testing
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluasi model pada data testing
print("\nEvaluasi Random Forest pada data testing:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_rf.png')
plt.close()

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.savefig('roc_curve_rf.png')
plt.close()

# 6. Pemodelan dengan Logistic Regression
print("\n\nPemodelan dengan Logistic Regression")
print("---------------------------------")

# Membangun model Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Evaluasi model dengan cross-validation
cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores_lr}")
print(f"Mean cross-validation score: {cv_scores_lr.mean():.4f}")

# Prediksi pada data testing
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluasi model pada data testing
print("\nEvaluasi Logistic Regression pada data testing:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_lr.png')
plt.close()

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Comparison')
plt.legend(loc='lower right')
plt.savefig('roc_curve_comparison.png')
plt.close()

# 7. Feature Importance
print("\n\nFeature Importance")
print("----------------")

# Feature importance dari Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance - Random Forest')
plt.savefig('feature_importance.png')
plt.close()

# 8. Kesimpulan
print("\n\nKesimpulan")
print("---------")
print("1. Model Random Forest menghasilkan accuracy sebesar {:.2f}%".format(accuracy_score(y_test, y_pred_rf) * 100))
print("2. Model Logistic Regression menghasilkan accuracy sebesar {:.2f}%".format(accuracy_score(y_test, y_pred_lr) * 100))

if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_lr):
    print("3. Model Random Forest lebih baik dalam mendeteksi penyakit jantung pada dataset ini.")
else:
    print("3. Model Logistic Regression lebih baik dalam mendeteksi penyakit jantung pada dataset ini.")

print("4. Fitur terpenting dalam prediksi penyakit jantung adalah:")
for i in range(5):
    print(f"   - {feature_importance.iloc[i]['Feature']}")