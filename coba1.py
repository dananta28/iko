import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Berbasis Aspek",
    page_icon="🍽️",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────
# CSS SEDERHANA UNTUK TAMPILAN YANG LEBIH RAPI
# ─────────────────────────────────────────────────────────────────────
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

# Warna untuk tiap aspek & sentimen (dipakai untuk badge hasil prediksi)
ASPEK_WARNA = {
    "makanan": "#fecdd3",     # pink
    "pelayanan": "#fde68a",   # kuning
    "tempat": "#a7f3d0",      # hijau
    "harga": "#e9d5ff",       # ungu
    "lainnya": "#e5e7eb",     # abu-abu
}
SENTIMEN_WARNA = {
    "positif": "#93c5fd",  # biru
    "negatif": "#fca5a5",  # merah
}


# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING TEKS (sama dengan yang dipakai saat melatih model)
# ─────────────────────────────────────────────────────────────────────
def preprocessing(teks: str) -> str:
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', ' ', teks)
    teks = re.sub(r'@\w+|#\w+', ' ', teks)
    teks = re.sub(r'\d+', ' ', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


# ─────────────────────────────────────────────────────────────────────
# MEMUAT MODEL & ARTEFAK (di-cache agar tidak diulang setiap interaksi)
# ─────────────────────────────────────────────────────────────────────
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


vectorizer, mlb, le_lp, model, metrics, distribusi = load_artifacts()


def decode_prediksi(pred_enc: int):
    """Ubah hasil prediksi (kode Label Powerset) menjadi daftar (aspek, sentimen)."""
    if pred_enc < 0 or pred_enc >= len(le_lp.classes_):
        return []
    kode_str = le_lp.classes_[pred_enc]
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


# ─────────────────────────────────────────────────────────────────────
# NAVIGASI / TAB
# ─────────────────────────────────────────────────────────────────────
st.title("🍽️ Analisis Sentimen Berbasis Aspek")
st.caption("Studi Kasus: Ulasan Rumah Makan Bebek Sinjay")

tab_prediksi, tab_tentang = st.tabs(["🔍 Coba Analisis", "📊 Tentang Model & Data"])

# ───────────────────────── TAB 1: PREDIKSI ─────────────────────────
with tab_prediksi:
    st.markdown(
        "Masukkan satu ulasan pelanggan di bawah ini. Sistem akan secara otomatis "
        "mengenali **aspek** apa saja yang dibahas (Makanan, Pelayanan, Tempat) "
        "beserta **sentimennya** (Positif/Negatif)."
    )

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
                teks_bersih = preprocessing(input_user)
                X_input = vectorizer.transform([teks_bersih])
                pred_enc = model.predict(X_input)[0]
                pasangan = decode_prediksi(pred_enc)

                proba = model.predict_proba(X_input)[0]
                top5_idx = np.argsort(proba)[::-1][:5]

            st.markdown("### 📝 Hasil Analisis")
            st.markdown("<div class='hasil-card'>", unsafe_allow_html=True)
            if pasangan:
                st.markdown(render_badge_aspek_sentimen(pasangan), unsafe_allow_html=True)
            else:
                st.write("⚠️ Kombinasi aspek & sentimen pada ulasan ini tidak dikenali oleh model.")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🔎 Lihat detail proses (untuk keperluan presentasi)"):
                st.write("**Teks asli:**", input_user)
                st.write("**Setelah preprocessing:**", teks_bersih)
                st.write("**Top 5 kombinasi paling mungkin menurut model:**")
                tabel_top5 = []
                for idx in top5_idx:
                    kode_str = le_lp.classes_[idx]
                    bits = [int(b) for b in kode_str.split("_")]
                    nama = [mlb.classes_[j] for j, b in enumerate(bits) if b == 1]
                    nama_str = ", ".join(nama) if nama else "tidak ada label"
                    tabel_top5.append({
                        "Kombinasi Aspek + Sentimen": nama_str,
                        "Probabilitas": f"{proba[idx]*100:.1f}%",
                    })
                st.table(pd.DataFrame(tabel_top5))

# ───────────────────────── TAB 2: TENTANG MODEL ─────────────────────────
with tab_tentang:
    st.markdown(
        "Halaman ini menjelaskan secara ringkas **bagaimana model dibuat** dan "
        "**seberapa baik performanya**, sehingga mudah dijelaskan kepada audiens."
    )

    st.markdown("#### 📌 Ringkasan Dataset & Model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-box'><h2>{metrics['jumlah_data']}</h2>"
            f"<p>Total Data Ulasan</p></div>", unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-box'><h2>{metrics['jumlah_aspek_sentimen']}</h2>"
            f"<p>Kategori Aspek+Sentimen</p></div>", unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-box'><h2>{metrics['jumlah_kombinasi_label']}</h2>"
            f"<p>Kombinasi Label (Powerset)</p></div>", unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("#### 🎯 Performa Model Terbaik (data uji)")
    bp = metrics["best_params"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score (Macro)", f"{bp['F1 Macro']*100:.1f}%")
    c2.metric("Precision (Macro)", f"{bp['Precision Macro']*100:.1f}%")
    c3.metric("Recall (Macro)", f"{bp['Recall Macro']*100:.1f}%")
    c4.metric("Hamming Loss", f"{bp['Hamming Loss']*100:.1f}%")

    st.caption(
        "Model dipilih melalui *grid search* (pencarian kombinasi parameter terbaik) "
        "pada algoritma **XGBoost**, dievaluasi dengan skema **Label Powerset** "
        "(setiap kombinasi aspek+sentimen yang unik diperlakukan sebagai satu kelas)."
    )

    st.markdown("#### 📊 Distribusi Label pada Data Latih")
    st.bar_chart(distribusi.set_index("Label"))
    st.caption(
        "Grafik di atas menunjukkan sebaran jumlah ulasan untuk masing-masing "
        "kombinasi aspek+sentimen yang digunakan dalam pelatihan model. "
        "Data yang tidak seimbang (*imbalanced*) seperti ini adalah hal yang umum "
        "terjadi pada data ulasan pelanggan."
    )

    st.markdown("#### ⚙️ Tahapan Pemrosesan Teks")
    st.markdown("""
1. **Case folding** — mengubah semua huruf menjadi huruf kecil.
2. **Pembersihan teks** — menghapus URL, mention/hashtag, angka, dan tanda baca.
3. **TF-IDF Vectorization** — mengubah teks menjadi representasi numerik berdasarkan
   tingkat kepentingan kata dalam ulasan.
4. **Label Powerset Encoding** — menggabungkan kombinasi aspek+sentimen menjadi satu
   label unik agar dapat dipelajari sebagai klasifikasi multi-kelas.
5. **Klasifikasi XGBoost** — model memprediksi kombinasi aspek+sentimen yang paling sesuai
   dengan ulasan baru.
    """)

    with st.expander("📋 Lihat 5 kombinasi parameter terbaik hasil grid search"):
        df_top5 = pd.DataFrame(metrics["top5_skenario"])
        st.dataframe(df_top5, use_container_width=True)

st.markdown("---")
st.caption("Dibuat untuk keperluan Skripsi — Analisis Sentimen Berbasis Aspek pada Ulasan Rumah Makan Bebek Sinjay.")
