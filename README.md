# Prediksi Harga Kos Madiun

Aplikasi web untuk melihat data kos Madiun dan memperoleh estimasi harga bulanan berdasarkan jarak, rating, dan fasilitas AC. Tujuannya adalah menyediakan demonstrasi interaktif model regresi sekaligus membantu pengguna membaca visualisasi data. UI menggunakan tema **Cute Soft Pastel – Modern Minimalist**.

Linear Regression digunakan untuk memprediksi harga. Groq API hanya digunakan untuk menghasilkan penjelasan berbahasa Indonesia atas diagram; Groq tidak mengubah prediksi, model, atau evaluasi. **Aplikasi tidak menggunakan database dan tidak menyimpan histori prediksi.**

## Fitur

- Dashboard berbahasa Indonesia: statistik dataset, status backend/model, distribusi harga, rata-rata berdasarkan AC, serta grafik jarak/harga dan rating/harga. Setiap diagram menampilkan penjelasan AI jika Groq dikonfigurasi.
- Prediksi lewat `POST /api/predict/` pada Django; hasil menampilkan harga ribu rupiah dan rupiah, input, algoritma, waktu, dan keterangan keterbatasan.
- Dataset dengan pencarian nama, filter AC, harga, jarak, rating, statistik, distribusi harga, dan perbandingan rata-rata berdasarkan AC.
- Evaluasi holdout, baseline DummyRegressor, Leave-One-Out Cross-Validation, serta diagram perbandingan MAE/RMSE dengan penjelasan AI.
- Halaman Tentang Aplikasi. Navigasi tidak menyediakan menu Riwayat Prediksi.

## Dataset dan model

`data/kos_madiun.csv` berisi 30 baris dengan kolom `nama`, `jarak_km`, `harga`, `ac`, dan `rating`. Target `harga` dan prediksi menggunakan satuan **ribu rupiah per bulan**. Fitur model berurutan `jarak_km`, `rating`, `ac_num`; mapping AC adalah `ya=1`, `tidak=0`. Kolom nama bukan fitur.

Training memakai `train_test_split(test_size=0.2, random_state=42)`, menghitung MAE, MSE, RMSE, R² training dan test, lalu melatih model produksi kembali dengan seluruh dataset. Artefak berada di `ml/artifacts/model_kos.joblib`; request prediksi tidak melatih ulang model.

### Evaluasi yang tercatat

Nilai berikut bersumber dari metadata Joblib dan snapshot diagnostik [evaluation_metrics.json](./ml/artifacts/evaluation_metrics.json), yang terikat ke SHA-256 dataset. Angka harga dalam ribu rupiah per bulan.

| Evaluasi | Model | MAE | MSE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Holdout 24:6 | Linear Regression | 246.056 | 78,638.402 | 280.425 | -6.033 |
| Holdout 24:6 | DummyRegressor mean | 168.750 | 39,657.118 | 199.141 | -2.547 |
| Leave-One-Out | Linear Regression | 267.538 | 98,524.047 | 313.885 | -0.309 |
| Leave-One-Out | DummyRegressor mean | 240.000 | 80,570.749 | 283.850 | -0.070 |

R² training model evaluator adalah **0.077**. Hasil menunjukkan Linear Regression belum mengungguli DummyRegressor. Holdout hanya memiliki enam data test dan sangat sensitif terhadap pembagian. LOO merupakan evaluasi tambahan; R² LOO dihitung dari prediksi out-of-fold yang digabungkan, bukan rata-rata R² per fold. Metrik regresi bukan persentase akurasi.

Model evaluator untuk metrik holdout dilatih hanya pada 24 data training. Model produksi di Joblib merupakan fit terpisah pada seluruh 30 baris. Snapshot evaluasi akan ditolak oleh API jika hash dataset atau metrik holdout Joblib tidak cocok; lakukan evaluasi baru sebelum menampilkan angka untuk dataset/model yang berubah.

Dataset kecil dan hasil prediksi merupakan estimasi, bukan jaminan harga pasar. Sumber serta lisensi asli dataset perlu dikonfirmasi kepada dosen.

## Teknologi dan arsitektur

- **Frontend:** Streamlit, Pandas, Altair.
- **Backend:** Django dan Django REST Framework.
- **Machine Learning:** Scikit-learn Linear Regression; model disimpan/dimuat menggunakan Joblib.
- **Penjelasan diagram:** Groq Cloud API melalui HTTP `requests`, model default `openai/gpt-oss-120b`.

Streamlit menampilkan dashboard, formulir prediksi, tabel, grafik, dan penjelasan diagram. Untuk prediksi serta ringkasan evaluasi, Streamlit berkomunikasi dengan Django REST API. Django memuat artefak Joblib dan memproses masukan; Groq dipanggil langsung oleh Streamlit hanya untuk penjelasan diagram. Tidak ada komunikasi Groq dari browser dan tidak ada histori AI yang disimpan.

## Instalasi dan menjalankan di Windows

Gunakan environment Python proyek dan dependency pada `requirements.txt`:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

Jika Python 3.15 tidak tersedia, pilih versi Python yang kompatibel dengan paket di requirements.

Salin `.env.example` menjadi `.env`, lalu isi konfigurasi lokal yang diperlukan. File `.env` diabaikan Git dan tidak boleh dibagikan.

### Konfigurasi Groq API

Atur `GROQ_API_KEY` melalui `.env` lokal, environment variable Windows, atau Streamlit Secrets saat deployment. Jangan pernah menulis API key ke source code, README, atau chat. Model dapat dikonfigurasi dengan `GROQ_MODEL`; nilai default-nya `openai/gpt-oss-120b`.

Contoh `.env` lokal (isi key secara lokal, jangan commit file ini):

```dotenv
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
PREDICTION_MODE=django
```

Untuk deployment Streamlit, tambahkan konfigurasi berikut ke `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = ""
GROQ_MODEL = "openai/gpt-oss-120b"
```

`secrets.toml` juga diabaikan Git. Tanpa API key, diagram dan fitur lain tetap berfungsi, sedangkan aplikasi memberi tahu bahwa penjelasan AI belum diaktifkan. Kegagalan layanan Groq tidak menghentikan aplikasi. Penjelasan yang berhasil dihasilkan di-cache selama 24 jam berdasarkan model dan data diagram.

## Deployment Streamlit Community Cloud

Aplikasi dapat dijalankan tanpa hosting backend terpisah.

1. Di Streamlit Community Cloud, pilih **Create app** dan repository GitHub `afidsetiawan-dotcom/prediksi-harga-kos-madiun`.
2. Pilih branch `main` dan entry point `frontend/app.py`.
3. Pada **Advanced settings**, gunakan Python 3.12 (versi default Community Cloud dan kompatibel dengan dependency yang dipin).
4. Isi **Settings → Secrets** dengan konfigurasi berikut; masukkan API key langsung pada dashboard Streamlit, bukan ke Git.

```toml
GROQ_API_KEY = ""
GROQ_MODEL = "openai/gpt-oss-120b"
PREDICTION_MODE = "direct"
```

5. Simpan pengaturan dan deploy secara manual.

File `frontend/requirements.txt` berisi dependency khusus frontend dan berada di samping entry point sehingga Community Cloud memilihnya sebelum `requirements.txt` root.

Mode `direct` tidak memanggil health, summary, atau endpoint prediksi Django. Aplikasi memuat dataset dari root repository dan menjalankan prosedur training produksi (evaluasi holdout lalu fit Linear Regression pada seluruh data) sekali per proses Streamlit melalui `st.cache_resource`. Evaluasi tetap membaca snapshot yang divalidasi terhadap hash dataset. Mode lokal tetap menggunakan Django API (`PREDICTION_MODE=django`, default untuk `.env.example`).

Repository tidak memuat artefak Joblib (`ml/artifacts/*.joblib` diabaikan Git); mode Community Cloud membuat model langsung dari dataset tanpa menulis atau membutuhkan artefak tersebut. Deployment dilakukan manual melalui dashboard Streamlit Community Cloud.

Untuk training ulang secara sengaja:

```powershell
python ml/train_model.py
```

Artefak Joblib tidak disimpan di Git; buat artefak model sebelum menjalankan backend jika belum tersedia. Training memperbarui artefak Joblib; setelah mengganti dataset atau model, evaluasi diagnostik di `ml/artifacts/evaluation_metrics.json` harus dijalankan dan diverifikasi ulang. Jangan menyajikan snapshot lama sebagai metrik model baru.

Jalankan backend di terminal pertama:

```powershell
python backend\manage.py runserver 127.0.0.1:8000
```

Jalankan frontend di terminal kedua:

```powershell
streamlit run frontend\app.py
```

Frontend memakai `http://127.0.0.1:8000/api/` secara default. Jika alamat berbeda:

```powershell
$env:DJANGO_API_URL = "http://127.0.0.1:8000/api/"
```

Navigasi frontend terdiri dari Dashboard, Prediksi Harga, Dataset, Evaluasi Model, dan Tentang Aplikasi. Hasil prediksi hanya ditampilkan pada sesi aktif dan tidak dicatat sebagai histori.

Buka alamat Streamlit yang ditampilkan, biasanya `http://localhost:8501`.

## Struktur proyek

```text
backend/                 Django REST API
data/kos_madiun.csv      Dataset aplikasi
frontend/                Aplikasi Streamlit dan integrasi Groq
frontend/requirements.txt Dependency frontend untuk Community Cloud
ml/train_model.py        Pipeline training model
ml/artifacts/            Artefak lokal Joblib (diabaikan Git)
tests/                   Tes integrasi Groq
.streamlit/config.toml   Tema Streamlit
```

## Endpoint API

- `GET /api/health/`: status backend dan ketersediaan model.
- `POST /api/predict/`: menerima JSON `{"jarak_km": 1.5, "rating": 4.2, "ac": "ya"}` dan mengembalikan hasil model.
- `GET /api/summary/`: statistik dataset, metadata model Joblib, dan evaluasi yang tervalidasi terhadap dataset.

Harga prediksi disediakan dalam ribu rupiah dan ekuivalen rupiah. Histori prediksi tidak dikirim atau disimpan oleh backend maupun frontend.

## Pengujian

```powershell
.\.venv\Scripts\python.exe backend\manage.py check
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s backend -p "tests.py"
.\.venv\Scripts\python.exe -m unittest discover -s frontend -p "test_*.py"
```

Tes frontend AppTest menjalankan Django sementara di localhost dengan port acak; tidak memakai atau membuat database.
