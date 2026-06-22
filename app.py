import re
import string
import pickle
import requests
from io import BytesIO

import numpy as np
import pandas as pd
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ===============================
# DOWNLOAD NLTK
# ===============================
nltk.download("punkt")
nltk.download("stopwords")


# ===============================
# TAMPILAN APLIKASI
# ===============================
st.header("Analisis Sentimen Berbasis Aspek Ulasan Bebek Sinjay")
st.write("Menggunakan Label Powerset dan XGBoost")

text_input = st.text_area("Masukkan Teks Ulasan")
submit = st.button("Submit", type="primary")


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
    vectorizer_url = "https://raw.githubusercontent.com/dananta28/iko/main/vectorizer%20(3).pkl"
    le_url = "https://raw.githubusercontent.com/dananta28/iko/main/label_encoder_lp%20(2).pkl"
    mlb_url = "https://raw.githubusercontent.com/dananta28/iko/main/mlb%20(1).pkl"

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
# FUNGSI PREPROCESSING
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
                kamus_normalisasi["slang"] == kata,
                "formal"
            ].values[0]
            hasil.append(formal)
        else:
            hasil.append(kata)

    return hasil


def remove_stopword(tokens):
    stopword_indonesia = set(stopwords.words("indonesian"))
    hasil = []

    for word in tokens:
        if word not in stopword_indonesia:
            hasil.append(word)

    return hasil


def stemming(tokens, stemmer):
    hasil = []

    for word in tokens:
        hasil.append(stemmer.stem(word))

    return hasil


# ===============================
# DECODE LABEL POWERSET
# ===============================
def decode_label_powerset(kode_str, mlb):
    label_bin = np.array([[int(x) for x in kode_str.split("_")]])
    label_aktif = mlb.inverse_transform(label_bin)[0]

    if len(label_aktif) == 0:
        return "Tidak terdeteksi aspek/sentimen"

    return ", ".join(label_aktif)


# ===============================
# PROSES UTAMA
# ===============================
if submit:
    if text_input.strip() == "":
        st.warning("Kamu belum memasukkan teks ulasan 😊")

    else:
        with st.spinner("Memproses ulasan..."):

            kamus_normalisasi = load_kamus_normalisasi()
            stemmer = load_stemmer()
            model, vectorizer, le_lp, mlb = load_artifacts()

            df_mentah = pd.DataFrame({"ulasan": [text_input]})

            df_mentah["Case Folding"] = df_mentah["ulasan"].apply(case_folding)

            df_mentah["Remove Punctuation"] = df_mentah["Case Folding"].apply(
                remove_punctuation
            )

            df_mentah["Tokenized"] = df_mentah["Remove Punctuation"].apply(tokenize)

            df_mentah["Normalisasi"] = df_mentah["Tokenized"].apply(
                lambda x: normalization(x, kamus_normalisasi)
            )

            df_mentah["Stopword Removal"] = df_mentah["Normalisasi"].apply(
                remove_stopword
            )

            df_mentah["Stemming"] = df_mentah["Stopword Removal"].apply(
                lambda x: stemming(x, stemmer)
            )

            df_mentah["HasilProcessing"] = df_mentah["Stemming"].apply(
                lambda x: " ".join(x)
            )

            X_input = vectorizer.transform(df_mentah["HasilProcessing"])

            y_pred_enc = model.predict(X_input)

            y_pred_str = le_lp.inverse_transform(y_pred_enc)[0]

            hasil_label = decode_label_powerset(y_pred_str, mlb)

        st.success("Prediksi selesai!")

        st.subheader("Hasil Prediksi")
        st.write(f'Ulasan: **"{text_input}"**')
        st.write(f"Hasil Prediksi Aspek dan Sentimen: **{hasil_label}**")
        st.write(f"Label Powerset: `{y_pred_str}`")

        with st.expander("Lihat Detail Preprocessing"):
            st.write("Case Folding:", df_mentah["Case Folding"].iloc[0])
            st.write("Remove Punctuation:", df_mentah["Remove Punctuation"].iloc[0])
            st.write("Tokenized:", df_mentah["Tokenized"].iloc[0])
            st.write("Normalisasi:", df_mentah["Normalisasi"].iloc[0])
            st.write("Stopword Removal:", df_mentah["Stopword Removal"].iloc[0])
            st.write("Stemming:", df_mentah["Stemming"].iloc[0])
            st.write("Hasil Akhir:", df_mentah["HasilProcessing"].iloc[0])
