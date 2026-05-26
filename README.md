<h1 align="center">Optimasi Telemarketing PortuBank</h1>
<h3 align="center">Model Prediktif Minat Nasabah Terhadap Deposito Berjangka<br>yang Berfokus pada Minimalisasi Biaya Loss</h3>

<p align="center">
  <img src="portubank.png" width="500" alt="PortuBank Banner"/>
</p>

<p align="center">
  <a href="https://termdepositpredictor-alpha.streamlit.app/"><b>Live Demo (Streamlit)</b></a> •
  <a href="https://public.tableau.com/app/profile/muhammad.hafizh.hariyanto/viz/BankTelemarketingProfileofCostumersLikelytoSubscribetoTermDeposit/Dashboard2"><b>Tableau Dashboard</b></a> •
  <a href="https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset/data"><b>Dataset (Kaggle)</b></a> •
  <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing"><b>Dataset (UCI)</b></a>
</p>

---

## Tim Penulis

| Nama | LinkedIn |
|---|---|
| **Muhammad Hafizh Hariyanto** | [LinkedIn Profile](https://www.linkedin.com/in/muhammad-hafizh-hariyanto/) |
| **Baira Rahayu** | [LinkedIn Profile](https://www.linkedin.com/in/baira-rahayu/) |

---

## Ringkasan Proyek

**PortuBank** adalah institusi perbankan ritel Portugal yang mengandalkan strategi **telemarketing langsung** (*outbound calls*) untuk mengakuisisi dana pihak ketiga melalui produk **Term Deposit** (Deposito Berjangka). Dataset mencatat kampanye yang berjalan antara **2008–2013**, di mana hanya **11,3%** dari total nasabah yang dihubungi akhirnya *subscribe* — artinya **hampir 9 dari setiap 10 panggilan berakhir tanpa konversi**.

Proyek ini membangun model prediktif **propensity-to-subscribe** yang menjawab satu problem bisnis secara langsung: **meminimalkan biaya kehilangan nasabah berminat (False Negative)**, di mana setiap FN bernilai **€37,50** (revenue NIM tahunan yang lepas) — sekitar **34× lebih besar** dari biaya satu panggilan tidak produktif (FP = **€1,09**). Asimetri biaya inilah yang mendasari pemilihan **F6-Score** sebagai metrik utama (β=6 → bobot Recall ≈ 36× Precision) dan **threshold tuning** untuk meminimalkan total biaya loss.

---

## Problem Statement

> *"Bagaimana membangun model klasifikasi propensity-to-subscribe yang **meminimalkan biaya kehilangan nasabah berminat (False Negative)** — di mana setiap FN bernilai €37,50, sekitar **34× lebih besar** dari biaya satu panggilan tidak produktif (False Positive €1,09) — sehingga total biaya loss kampanye telemarketing serendah mungkin?"*

### Struktur Biaya Loss

| Komponen | Nilai | Asumsi |
|---|---|---|
| **False Negative (FN)** | **€37,50** | Revenue NIM tahunan yang lepas (€2.500 deposito × 1,5% NIM) |
| **False Positive (FP)** | **€1,09** | Biaya 1 panggilan tidak produktif (€0,484 telepon + €0,606 porsi gaji agen) |
| **Rasio FN : FP** | **34,4 : 1** | Justifikasi β=6 pada F-beta Score (β²=36 ≈ 34,4 ✅) |

*Sumber asumsi biaya: ANACOM (tarif telepon Portugal), Eurofound (gaji minimum Portugal), Plecto (kapasitas panggilan agen), Statista (gaji data scientist).*

---

## Dataset

| Atribut | Detail |
|---|---|
| **Sumber** | Bank Marketing Dataset — UCI ML Repository *(Moro et al., 2014)* |
| **Periode** | Kampanye telemarketing Portugal, 2008–2013 |
| **Ukuran awal** | 41.188 baris × 21 kolom |
| **Ukuran setelah cleaning** | 41.172 baris × 25 kolom (setelah hapus duplikat, drop `duration=0`, dan feature engineering) |
| **Target** | `y` — nasabah subscribe deposito berjangka (`yes`/`no`) |
| **Class imbalance** | 88,7% tidak subscribe : 11,3% subscribe |
| **Referensi** | Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, Elsevier, 62, 22–31. |

---

## Metodologi

### Feature Engineering (5 Fitur Baru)

Berdasarkan temuan EDA, dibuat 5 fitur baru yang menangkap pola non-linear dan konteks bisnis:

| Fitur Baru | Asal | Justifikasi |
|---|---|---|
| `contacted_before` | `pdays` (sentinel 999) | Subscribe rate 63,8% (pernah dihubungi) vs 9,3% (belum) — **7× lipat** |
| `age_group` | `age` | Pola U-shape: usia 60+ rate 45,5% (>4× baseline); 31–50 di bawah rata-rata |
| `euribor_level` | `euribor3m` | Rate 45,7% (very_low ≤1%) vs 4,8% (very_high >4%) — **~10× gradient** |
| `month_season` | `month` | Bulan peak (Mar/Sep/Oct/Dec) rate >40% vs bulan reguler ~9% |
| `campaign_intensity` | `campaign` | Diminishing returns: `once` 13,0% → `excessive` 4,6% |

### Preprocessing Pipeline

- **Missing values**: nilai `unknown` dipertahankan sebagai kategori tersendiri; SimpleImputer `most_frequent` di dalam pipeline
- **Outlier handling**: `duration=0` di-drop; `campaign` di-cap di nilai 20
- **Scaling**: RobustScaler (seluruh fitur numerik tidak terdistribusi normal, dikonfirmasi Shapiro-Wilk)
- **Encoding**: OneHotEncoder untuk fitur kategorikal (kardinalitas rendah ≤12 nilai unik)
- **Eksklusi `duration`**: fitur ini hanya diketahui *setelah* panggilan selesai → data leakage, dieksklusi dari modeling
- **Split data**: Train / Validation / Test dengan stratify

### Benchmark Model

18 kombinasi dievaluasi (6 algoritma × 3 strategi resampling: tanpa resampling, SMOTE, Under-sampling). Model terbaik dari benchmark: **LightGBM + Under-sampling** (F6 Validation = 0,7236), kemudian ditingkatkan melalui:

1. **Hyperparameter tuning** (RandomizedSearchCV) — `n_estimators` 500–1000, `num_leaves` 63–127, `learning_rate` 0,02–0,05
2. **Class-weighting** (`scale_pos_weight=7.87`)
3. **Threshold tuning** di validation set untuk memaksimalkan F6 secara langsung (tidak memakai default 0,5)

### Interpretabilitas Model

SHAP Values digunakan untuk menjelaskan prediksi secara global maupun per nasabah (*local explanation*). Fitur paling berpengaruh:
- **Fitur makro ekonomi** (`nr.employed`, `emp.var.rate`, `cons.price.idx`, `euribor3m`) — dominan secara global
- **`poutcome_success`** — fitur kategorikal terkuat, konsisten dengan EDA (Cramér's V kuat)
- **`contacted_before`** — binary namun sangat informatif

---

## Hasil Model Final

**LightGBM + Under-sampling (ratio 0,5) + Class-Weighting (`scale_pos_weight=7.87`) + Threshold Tuning**

Dievaluasi pada *held-out test set* (8.235 nasabah, 928 subscriber aktual).

| Metrik | Train | Validation | Test | Interpretasi |
|---|---|---|---|---|
| **F6 Score ★** | 0,8266 | **0,8251** | **0,8245** | Metrik utama — minimalisasi biaya loss |
| **Gap (Train − Val)** | — | **+0,0015** ✅ | — | Good Fit — siap di-deploy |
| **ROC-AUC** | — | — | **0,8000** | Diskriminasi model yang baik |

### Penghematan vs Tanpa Model

| Pendekatan | Total Biaya Loss (Test Set) |
|---|---|
| **Tanpa Model** (hubungi semua) | €7.964,63 |
| **Model ML + F6 Threshold** | **€7.948,07** ★ |
| **Penghematan** | **€16,56 (0,2%)** + kemampuan ranking probabilitas |

> Nilai utama model bukan hanya angka penghematan langsung, tetapi pada **kemampuan ranking probabilitas** untuk Priority Level Segmentation dan **skalabilitas** ke populasi produksi puluhan ribu nasabah — sesuatu yang tidak bisa dilakukan pendekatan rule-based.

### Keunggulan Model ML vs Rule-Based

| Dimensi | Rule-Based (EDA Segmentation) | Model ML (LightGBM) |
|---|---|---|
| Variabel yang dipertimbangkan | 3–4 fitur | 20 fitur sekaligus |
| Output | Biner (masuk/tidak masuk segmen) | Skor probabilitas per nasabah |
| Ranking dalam segmen | ❌ Tidak bisa | ✅ Priority Level 1/2/3 |
| Adaptasi kondisi makro | ❌ Statis | ✅ Otomatis |

---

## Priority Level Segmentation

Ketika kapasitas kampanye tidak mencukupi untuk menghubungi semua nasabah yang diprediksi berminat, model menghasilkan **3 tingkat prioritas** berdasarkan rentang probabilitas:

| Tingkat | Cara Pembagian | Strategi |
|---|---|---|
| **Tingkat 1 — Tinggi** | Top 33% probabilitas | Hubungi pertama — alokasikan agen senior |
| **Tingkat 2 — Sedang** | Mid 33% probabilitas | Hubungi setelah Tingkat 1 |
| **Tingkat 3 — Rendah** | Bottom 33% probabilitas (masih di atas threshold) | Hubungi terakhir atau saat kapasitas tersedia |
| **Tidak Dihubungi** | Di bawah threshold model | Tidak masuk daftar kontak |

**Panduan alokasi budget:**
- Budget penuh → Tingkat 1 → 2 → 3 secara berurutan
- Budget sedang (~67%) → Tingkat 1 + 2 → tangkap ~85–90% subscriber, hemat ~33% biaya
- Budget terbatas (~33%) → **Tingkat 1 saja** → tangkap mayoritas subscriber dengan biaya minimal

---

## Insight Bisnis Utama

**Dari EDA & Inferential Analysis (Chi-Square + Mann-Whitney, α=0,05):**

**Segmen prioritas tinggi:**
- `poutcome = success` → subscribe rate ~65% (~6× baseline) — prediktor kategorikal terkuat
- `contacted_before = 1` (warm lead) → rate ~63,8% vs 9,3% (7× lipat)
- Usia 60+ saat `euribor3m` rendah → rate ~45%+
- Bulan Mar/Sep/Oct/Dec → rate >40%

**Segmen tidak efisien:**
- Cold lead tanpa riwayat + `euribor3m` > 3 → rate ~3–5%
- Nasabah dihubungi >3× tanpa konversi → marginal rate <5% → terapkan **"3-Strike Rule"**

---

## Rekomendasi Utama

1. **Gunakan model sebagai mesin ranking** — bukan sekadar filter biner
2. **Quick win interim** — terapkan segmentasi EDA sebelum model diintegrasikan ke CRM
3. **Optimasi timing** — intensifkan kampanye di Mar/Sep/Oct/Dec; kurangi saat `euribor3m > 3`
4. **3-Strike Rule** — hentikan kontak setelah 3 panggilan tidak responsif; alihkan ke warm lead baru
5. **Kembangkan database warm lead** — hanya 3,7% dari nasabah punya riwayat kontak, tapi rate-nya 63,8%
6. **Retraining berkala** — setiap 6–12 bulan, pantau pergeseran kondisi makro, F6 Score pada data terbaru, dan perubahan struktur biaya FN:FP

---

## Struktur Notebook

| Section | Topik | Output Utama |
|---|---|---|
| **1** | Business Understanding | Problem statement, struktur biaya, stakeholder |
| **2** | Data Understanding | Deskripsi 21 fitur, statistik deskriptif |
| **3** | Data Quality Check | Cek duplikat, missing, outlier (IQR), normalitas (Shapiro-Wilk + QQ-Plot), korelasi |
| **4** | Exploratory Data Analysis | Subscription rate per fitur, segmentasi, analisis threshold durasi |
| **5** | Data Preprocessing | 5 fitur baru, handling outlier, pipeline (RobustScaler + OHE), split stratified |
| **6** | Methodology — Data Analytics | Descriptive + Inferential (Chi-Square, Mann-Whitney) |
| **7** | Methodology — Machine Learning | Benchmark 18 kombinasi, tuning, threshold optimization, SHAP, confusion matrix, Priority Level Segmentation, simpan model |
| **8** | Conclusion & Recommendation | Jawaban problem statement, improvement vs iterasi sebelumnya, rekomendasi bisnis |

---

<p align="center">
  <b>Made by Alpha Group</b>
</p>
