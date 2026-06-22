import streamlit as st

st.header("Tes Deploy Streamlit")
st.write("Jika halaman ini muncul, berarti deploy dasar Streamlit sudah berhasil.")

st.divider()
st.subheader("Cek versi library")

try:
    import numpy
    st.success(f"numpy: {numpy.__version__}")
except Exception as e:
    st.error(f"numpy gagal di-import: {e}")

try:
    import pandas
    st.success(f"pandas: {pandas.__version__}")
except Exception as e:
    st.error(f"pandas gagal di-import: {e}")

try:
    import sklearn
    st.success(f"scikit-learn: {sklearn.__version__}")
except Exception as e:
    st.error(f"scikit-learn gagal di-import: {e}")

try:
    import xgboost
    st.success(f"xgboost: {xgboost.__version__}")
except Exception as e:
    st.error(f"xgboost gagal di-import: {e}")

try:
    import nltk
    st.success(f"nltk: {nltk.__version__}")
except Exception as e:
    st.error(f"nltk gagal di-import: {e}")

try:
    import Sastrawi
    st.success("Sastrawi: berhasil di-import")
except Exception as e:
    st.error(f"Sastrawi gagal di-import: {e}")

try:
    import requests
    st.success(f"requests: {requests.__version__}")
except Exception as e:
    st.error(f"requests gagal di-import: {e}")

try:
    import streamlit as _st
    st.success(f"streamlit: {_st.__version__}")
except Exception as e:
    st.error(f"streamlit gagal di-import: {e}")

st.divider()
st.subheader("Tes input sederhana")
nama = st.text_input("Coba ketik sesuatu")
if nama:
    st.write(f"Kamu mengetik: **{nama}**")
