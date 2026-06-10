# Model Machien Learning Capstone Project Prediksi Produksi Padi

Proyek ini menggunakan model Linear Regression, Random Forest, dan XGBoost untuk memprediksi hasil panen padi berdasarkan data cuaca.

## Struktur Folder
```
├── dataset/
│   ├── eda_outputs/               # Hasil pembersihan dan feature engineering (file .csv dan .png) 
│   ├── models/                    # Hasil hasil training baseline model (.joblib)
│   └── output_pipeline/           # Hasil log dan metrik baseline model
│   dataset_produksi_padi.csv      # Dataset bersih
│   
├── exploratory_data_analysis/     # File analisis eksplorasi data (.ipynb) 
│
├── k-fold_cross_validation.ipynb  # Uji coba metode validasi k-fold 
├── linear_regression.ipynb        # File koding linear regression
├── model_comparison.ipynb         # File koding komparasi seluruh model
├── random_forest.ipynb            # File koding random forest
├── xgboost.ipynb                  # File koding xgboost
├── requirements.txt       # Daftar library yang diperlukan
└── README.md              # Dokumentasi proyek             
```
    
## Cara Menjalankan
Instruksi menjalankan kode:
1. Clone repositori ini.
2. Install library: `pip install -r requirements.txt`

## Dataset
- Total: 2529 baris
   - Jumlah Data Train: 2023 baris (80.0%)
   - Jumlah Data Validasi: 506 baris (20.0%)
   - Jumlah Data Test: 506 baris

## Hasil Evaluasi
- Pemenang RMSE & $R^2$: XGBoost terpilih sebagai model utama karena memberikan RMSE terendah (25.68) dan skor $R^2$ tertinggi (0,9857) pada data validasi, menunjukkan kemampuan generalisasi yang paling stabil.
- Indikasi Overfitting: Random Forest menunjukkan adanya gap yang cukup besar antara skor Train dan Validation, sedangkan XGBoost memberikan performa yang lebih konsisten (lebih robust terhadap data baru).

### Tabel Perbandingan Performa Model
| Model | Train MAE | Val MAE | Val RMSE | Val $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 18.419,16 | 18.484,60 | 32.594,62 | 0,9770 |
| **Random Forest** | 4.937,11 | 13.242,58 | 27.241,72 | 0,9840 |
| **XGBoost** | 8.187,66 | 13.704,89 | 25.687,45 | 0,9857 |