# Cross-Domain WiFi Gesture Recognition (Widar3.0)

Repositori ini berisi implementasi proyek pengenalan gestur berbasis WiFi Sensing (Body-velocity Profile / BVP) dari dataset Widar3.0. Proyek ini membandingkan performa model baseline **CNN-LSTM** dengan **Domain Adversarial Neural Network (DANN)** untuk mengatasi tantangan *cross-domain generalization*.

---

##  Struktur Repositori

- `data/` : Folder tempat sampel dataset/file BVP (`.mat`).
- `notebooks/` :
  - `01_data_preprocessing.ipynb` : Sanitasi & pemrosesan awal data BVP.
  - `02_baseline_model.ipynb` : Pelatihan model baseline CNN-LSTM.
  - `03_dann_model.ipynb` : Pelatihan model DANN (Domain Adaptation).
  - `04_model_comparison_evaluation.ipynb` : Evaluasi, Confusion Matrix, & visualisasi t-SNE.
- `src/` : Kode modular pendukung (seperti `dataset.py`).
- `requirements.txt` : Daftar pustaka Python yang dibutuhkan.

---

##  Cara Menjalankan (Google Colab)

1. Clone repositori ini atau *download* file notebook di folder `notebooks/`.
2. Buka notebook menggunakan **Google Colab**.
3. Pastikan memilih **GPU** pada jenis *Runtime* Colab (`Runtime` > `Change runtime type` > `T4 GPU`).
4. Jalankan notebook secara berurutan mulai dari `01` hingga `04`.

---

##  Hasil Evaluasi
Perbandingan performa antara model CNN-LSTM dan DANN disajikan pada notebook `04` dalam bentuk **Confusion Matrix** dan **t-SNE feature distribution**.
