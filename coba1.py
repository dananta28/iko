import re
import pickle
import requests
import numpy as np
import string 
import pandas as pd
import streamlit as st
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

st.set_page_config(
    page_title="Analisis Sentimen Berbasis Aspek",
    page_icon="🍽️",
    layout="centered",
)

st.markdown("""
<style>
.badge {
    display: inline-block;
    border-radius: 8px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 4px 6px 4px 0;
}
.hasil-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    background-color: #fafafa;
}
.metric-box {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    background-color: #ffffff;
}
.metric-box h2 {margin-bottom:0; font-size: 1.8rem;}
.metric-box p {margin-top:4px; color:#6b7280; font-size:0.85rem;}
</style>
""", unsafe_allow_html=True)

ASPEK_WARNA = {
    "makanan": "#fecdd3",
    "pelayanan": "#fde68a",
    "tempat": "#a7f3d0",
    "harga": "#e9d5ff",
    "lainnya": "#e5e7eb",
}
SENTIMEN_WARNA = {
    "positif": "#93c5fd",
    "negatif": "#fca5a5",
}

# ===============================
# LOAD FILE DARI GITHUB
# ===============================
def load_pickle_from_github(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(BytesIO(response.content))

@st.cache_resource
def load_artifacts():
    model_url = "https://raw.githubusercontent.com/dananta28/iko/main/model_xgboost1.pkl"
    vectorizer_url = "https://raw.githubusercontent.com/dananta28/iko/main/vectorizer_baru.pkl"
    le_url = "https://raw.githubusercontent.com/dananta28/iko/main/le_lp.pkl"
    mlb_url = "https://raw.githubusercontent.com/dananta28/iko/main/mlb%20(2).pkl"

    model = load_pickle_from_github(model_url)
    vectorizer = load_pickle_from_github(vectorizer_url)
    le_lp = load_pickle_from_github(le_url)
    mlb = load_pickle_from_github(mlb_url)

    return model, vectorizer, le_lp, mlb

@st.cache_data
def load_kamus_normalisasi():
    url = "https://raw.githubusercontent.com/dananta28/iko/main/colloquial-indonesian-lexicon%20(3).csv"
    kamus = pd.read_csv(url)
    kamus = kamus.drop(
        columns=["In-dictionary", "context", "category1", "category2", "category3"],
        errors="ignore"
    )
    return kamus

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

# ===============================
# PREPROCESSING
# ===============================
def case_folding(text):
    return str(text).lower()

def remove_punctuation(text):
    data = re.sub("@[^\\s]+", " ", text)
    data = re.sub(r"http\S*", " ", data)
    data = data.translate(str.maketrans(" ", " ", string.punctuation))
    data = re.sub("[^a-zA-Z]", " ", data)
    data = re.sub("\n", " ", data)
    data = re.sub(r"\b[a-zA-Z]\b", " ", data)
    data = re.sub(r"\s+", " ", data).strip()
    return data

def tokenize(text):
    return word_tokenize(text)

def normalization(tokens, kamus_normalisasi):
    hasil = []
    for kata in tokens:
        if kata in kamus_normalisasi["slang"].values:
            formal = kamus_normalisasi.loc[
                kamus_normalisasi["slang"] == kata, "formal"
            ].values[0]
            hasil.append(formal)
        else:
            hasil.append(kata)
    return hasil

def remove_stopword(tokens):
    stopword_indonesia = set(stopwords.words("indonesian"))
    return [w for w in tokens if w not in stopword_indonesia]

def stemming(tokens, stemmer):
    return [stemmer.stem(w) for w in tokens]

def preprocessing_full(text, kamus_normalisasi, stemmer):
    text = case_folding(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = normalization(tokens, kamus_normalisasi)
    tokens = remove_stopword(tokens)
    tokens = stemming(tokens, stemmer)
    return " ".join(tokens)

# ===============================
# DECODE & RENDER
# ===============================
def decode_prediksi(pred_enc, le_lp, mlb):
    kode_str = le_lp.inverse_transform([pred_enc])[0]
    bits = [int(b) for b in kode_str.split("_")]
    nama = [mlb.classes_[j] for j, b in enumerate(bits) if b == 1]
    hasil = []
    for n in nama:
        parts = n.strip().split(" ")
        sentimen = parts[-1]
        aspek = " ".join(parts[:-1])
        hasil.append((aspek, sentimen))
    return hasil

def render_badge_aspek_sentimen(pasangan):
    html = ""
    for aspek, sentimen in pasangan:
        warna_aspek = ASPEK_WARNA.get(aspek.lower(), "#e5e7eb")
        warna_sentimen = SENTIMEN_WARNA.get(sentimen.lower(), "#d1d5db")
        html += (
            f"<span class='badge' style='background-color:{warna_aspek};'>"
            f"Aspek: {aspek.title()}</span>"
            f"<span class='badge' style='background-color:{warna_sentimen};'>"
            f"Sentimen: {sentimen.title()}</span><br>"
        )
    return html

# ===============================
# LOAD SEMUA ARTIFACT
# ===============================
model, vectorizer, le_lp, mlb = load_artifacts()
kamus_normalisasi = load_kamus_normalisasi()
stemmer = load_stemmer()

# ===============================
# UI
# ===============================
st.title("🍽️ Analisis Sentimen Berbasis Aspek")
st.caption("Studi Kasus: Ulasan Rumah Makan Bebek Sinjay")

tab_prediksi, tab_tentang = st.tabs(["🔍 Coba Analisis", "📊 Tentang Model"])

with tab_prediksi:
    contoh_ulasan = [
        "Pilih contoh ulasan (opsional)",
        "Makanannya enak banget tapi pelayanannya lama",
        "Tempatnya bersih dan nyaman untuk makan keluarga",
        "Pelayanan ramah, makanan juga lezat",
        "Antri lama dan makanan kurang enak",
    ]
    pilihan = st.selectbox("Atau pilih contoh ulasan:", contoh_ulasan)
    default_text = "" if pilihan == contoh_ulasan[0] else pilihan

    input_user = st.text_area(
        "**Masukkan Ulasan**",
        value=default_text,
        placeholder="Contoh: Makanannya enak, tapi pelayanannya agak lama...",
        height=100,
    )
    submit = st.button("🚀 Proses Analisis", type="primary", use_container_width=True)

    if submit:
        if not input_user.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Menganalisis ulasan..."):
                teks_bersih = preprocessing_full(input_user, kamus_normalisasi, stemmer)
                X_input = vectorizer.transform([teks_bersih])
                pred_enc = model.predict(X_input)[0]
                pasangan = decode_prediksi(pred_enc, le_lp, mlb)

            st.markdown("### 📝 Hasil Analisis")
            st.markdown("<div class='hasil-card'>", unsafe_allow_html=True)
            if pasangan:
                st.markdown(render_badge_aspek_sentimen(pasangan), unsafe_allow_html=True)
            else:
                st.write("⚠️ Kombinasi aspek & sentimen tidak dikenali oleh model.")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🔎 Lihat detail preprocessing"):
                st.write("**Teks asli:**", input_user)
                st.write("**Setelah preprocessing:**", teks_bersih)

with tab_tentang:
    st.markdown("#### ⚙️ Tahapan Pemrosesan Teks")
    st.markdown("""
1. **Case folding** — mengubah semua huruf menjadi huruf kecil.
2. **Pembersihan teks** — menghapus URL, mention, angka, dan tanda baca.
3. **Tokenisasi** — memecah teks menjadi kata-kata.
4. **Normalisasi** — mengubah kata slang ke bentuk formal.
5. **Stopword Removal** — menghapus kata-kata umum yang tidak bermakna.
6. **Stemming** — mengubah kata ke bentuk dasar.
7. **TF-IDF Vectorization** — mengubah teks menjadi representasi numerik.
8. **Label Powerset + XGBoost** — memprediksi kombinasi aspek+sentimen.
    """)

st.markdown("---")
st.caption("Dibuat untuk keperluan Skripsi — Analisis Sentimen Berbasis Aspek pada Ulasan Rumah Makan Bebek Sinjay.")
