# KLASIFIKASI KUALITAS KARET REMAH (CRUMB RUBBER) MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK BERBASIS CITRA DIGITAL

## Deskripsi Proyek
Proyek skripsi ini bertujuan untuk mengklasifikasikan kualitas karet remah (crumb rubber) menggunakan teknologi Convolutional Neural Network (CNN) berbasis analisis citra digital. Sistem ini dirancang untuk membantu petani dan industri karet dalam menentukan kualitas karet remah secara otomatis dan akurat.

## Fitur Utama
- ✨ Klasifikasi otomatis kualitas karet remah
- 🖼️ Pemrosesan citra digital menggunakan CNN
- 📊 Analisis akurasi dan performa model
- 🎯 Interface user-friendly untuk upload gambar
- 📈 Visualisasi hasil prediksi

## Teknologi yang Digunakan
- **Deep Learning Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Data Analysis**: NumPy, Pandas, Matplotlib
- **Web Framework**: Flask/Streamlit (untuk interface)
- **Programming Language**: Python 3.8+

## Struktur Dataset
```
dataset/
├── train/
│   ├── grade_a/
│   ├── grade_b/
│   └── grade_c/
├── validation/
│   ├── grade_a/
│   ├── grade_b/
│   └── grade_c/
└── test/
    ├── grade_a/
    ├── grade_b/
    └── grade_c/
```

## Instalasi dan Setup

### Prerequisites
- Python 3.8 atau lebih baru
- pip package manager
- Virtual environment (disarankan)

### Langkah Instalasi
1. Clone repository ini:
```bash
git clone https://github.com/dresar/klasifikasi-kualitas-karet-cnn.git
cd klasifikasi-kualitas-karet-cnn
```

2. Buat virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Penggunaan

### 1. Preprocessing Data
```bash
python src/data_preprocessing.py
```

### 2. Training Model
```bash
python src/train_model.py
```

### 3. Evaluasi Model
```bash
python src/evaluate_model.py
```

### 4. Prediksi Gambar Baru
```bash
python src/predict.py --image path/to/image.jpg
```

### 5. Menjalankan Web Interface
```bash
python app.py
```

## Struktur Proyek
```
├── README.md
├── requirements.txt
├── app.py                 # Web interface
├── config.py             # Konfigurasi project
├── dataset/              # Dataset karet remah
├── models/               # Model yang telah dilatih
├── notebooks/            # Jupyter notebooks untuk eksperimen
│   ├── data_exploration.ipynb
│   ├── model_development.ipynb
│   └── results_analysis.ipynb
├── src/                  # Source code utama
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
├── static/               # File static untuk web
├── templates/            # Template HTML
└── tests/               # Unit tests
```

## Metodologi

### 1. Pengumpulan Data
- Pengambilan foto karet remah dengan berbagai grade kualitas
- Standarisasi kondisi pencahayaan dan sudut pengambilan
- Labeling manual oleh ahli karet

### 2. Preprocessing
- Resize gambar ke ukuran standar (224x224 px)
- Normalisasi pixel values
- Data augmentation untuk meningkatkan variasi dataset

### 3. Arsitektur Model CNN
- Convolutional layers dengan ReLU activation
- MaxPooling layers untuk dimensionality reduction
- Dropout layers untuk mencegah overfitting
- Dense layers untuk klasifikasi final

### 4. Training dan Validasi
- Split data: 70% training, 15% validation, 15% testing
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall, F1-Score

## Hasil yang Diharapkan
- Akurasi klasifikasi > 85%
- Sistem yang dapat digunakan secara real-time
- Interface yang user-friendly untuk petani
- Dokumentasi lengkap untuk pengembangan lanjutan

## Kontribusi
Proyek ini merupakan bagian dari skripsi dan terbuka untuk saran serta masukan. Silakan buat issue atau pull request untuk kontribusi.

## Lisensi
MIT License - see [LICENSE](LICENSE) file for details.

## Kontak
- **Mahasiswa**: Eka Syarif Maulana
- **Email**: eka.ckp16799@gmail.com
- **Universitas**: Universitas Sumatera Utara
- **Program Studi**: Teknologi Informasi

## Acknowledgments
- Dosen pembimbing skripsi
- Petani karet yang telah membantu dalam pengumpulan data
- Referensi penelitian terkait CNN dan image classification

---
*Project ini dikembangkan sebagai bagian dari tugas akhir/skripsi di Universitas Sumatera Utara*