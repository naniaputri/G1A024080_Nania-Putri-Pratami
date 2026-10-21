# Studi Kasus 3 - Workflow Machine Learning End-to-End

**Materi:** Pengenalan Machine Learning

## Ringkasan materi yang dipakai
- Pemrograman tradisional (rule-based) vs ML (algoritma belajar aturan dari data).
- Workflow 7 langkah: pengumpulan data -> pra-pemrosesan -> split (training 70-80%, testing 20-30%) -> pilih algoritma -> latih -> evaluasi -> deploy.
- Supervised (klasifikasi, regresi) dan Unsupervised (clustering K-Means).
- Metrik: accuracy, precision, recall, F1, confusion matrix; MAE dan R-squared untuk regresi.
- Overfitting vs underfitting; kualitas data menentukan kualitas prediksi.

## Tiga masalah dalam satu skrip (`main.py`)
| Bagian | Jenis | Isi |
|---|---|---|
| A | Klasifikasi | Persetujuan kredit: rule-based manual vs Logistic Regression vs Decision Tree vs Random Forest, plus demo overfitting (kedalaman pohon, 10% label noise) |
| B | Regresi | Prediksi harga rumah (luas, kamar, usia, kota) dengan Linear Regression vs Random Forest, model disimpan (`joblib`) dan dipakai memprediksi rumah baru (*deploy for inference*) |
| C | Clustering | Segmentasi pelanggan (RFM) dengan K-Means, k dipilih dengan silhouette score |

## Hasil (seed 42)
**Klasifikasi kredit (data uji):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Rule-based manual | 0,690 | 0,637 | 0,389 | 0,483 |
| Logistic Regression | 0,740 | 0,710 | 0,510 | **0,594** |
| Decision Tree (depth 4) | 0,740 | 0,771 | 0,430 | 0,552 |
| Random Forest | 0,728 | 0,669 | 0,530 | 0,592 |

- Overfitting: pohon depth 20 -> akurasi train **1,00** tetapi test **0,63**; kedalaman terbaik = 6.
- **Regresi:** R-squared 0,974 (Linear) dan 0,971 (RF); prediksi rumah 120 m2, 3 KT, 5 tahun di Bengkulu ~ Rp 806 juta.
- **Clustering:** k terbaik = 3, menghasilkan segmen hemat / setia / premium yang sesuai dengan pola data.

Grafik: `outputs/ringkasan_ml.png`, model: `outputs/model_harga_rumah.joblib`, metrik: `outputs/ringkasan.json`.

## Catatan
Pada data sintetis ini Logistic Regression setara/lebih baik dari Random Forest karena hubungan data dibangkitkan secara logistik -
contoh bahwa algoritma yang paling kompleks belum tentu terbaik; pilih sesuai karakter data.
