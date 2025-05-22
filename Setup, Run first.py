# Setup dan Download Dataset Heart Disease
# Jalankan cell ini terlebih dahulu untuk mempersiapkan dataset

import pandas as pd
import numpy as np
import urllib.request
import os

print("Setup Dataset Cleveland Heart Disease")
print("====================================")

# Method 1: Download dari UCI Repository
def download_cleveland_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    filename = "processed.cleveland.data"
    
    try:
        print("Mengunduh dataset dari UCI Repository...")
        urllib.request.urlretrieve(url, filename)
        print(f"Dataset berhasil diunduh: {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

# Method 2: Buat dataset sample jika download gagal
def create_sample_dataset():
    print("Membuat dataset sample Cleveland Heart Disease...")
    
    np.random.seed(42)
    n_samples = 303
    
    # Generate data yang mirip dengan dataset asli
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
    df.to_csv('heart_disease_cleveland.csv', index=False)
    print("Dataset sample berhasil dibuat: heart_disease_cleveland.csv")
    return 'heart_disease_cleveland.csv'

# Coba download, jika gagal buat sample
filename = download_cleveland_data()
if filename is None:
    filename = create_sample_dataset()

# Load dataset dengan nama kolom yang benar
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

if filename.endswith('.data'):
    df = pd.read_csv(filename, names=column_names, na_values='?')
else:
    df = pd.read_csv(filename)

print(f"\nDataset berhasil dimuat!")
print(f"Ukuran dataset: {df.shape}")
print(f"Kolom: {list(df.columns)}")

# Simpan sebagai CSV untuk kemudahan akses
if not os.path.exists('heart_disease_cleveland.csv'):
    df.to_csv('heart_disease_cleveland.csv', index=False)
    print("Dataset disimpan sebagai 'heart_disease_cleveland.csv'")

print("\nPenjelasan Kolom Dataset:")
print("========================")
column_explanations = {
    'age': 'Usia pasien (tahun)',
    'sex': 'Jenis kelamin (1 = laki-laki, 0 = perempuan)',
    'cp': 'Jenis nyeri dada (0-3)',
    'trestbps': 'Tekanan darah istirahat (mm Hg)',
    'chol': 'Kolesterol serum (mg/dl)',
    'fbs': 'Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak)',
    'restecg': 'Hasil elektrokardiografi istirahat (0-2)',
    'thalach': 'Detak jantung maksimum yang dicapai',
    'exang': 'Angina yang diinduksi oleh olahraga (1 = ya, 0 = tidak)',
    'oldpeak': 'Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat',
    'slope': 'Kemiringan segmen ST latihan puncak',
    'ca': 'Jumlah pembuluh darah utama (0-3)',
    'thal': 'Thalassemia (0-3)', 
    'num': 'Target: diagnosis penyakit jantung (0 = tidak ada, 1-4 = ada)'
}

for col, explanation in column_explanations.items():
    if col in df.columns:
        print(f"- {col}: {explanation}")

print("\n✅ Setup selesai! Anda dapat menjalankan kode analisis selanjutnya.")