import streamlit as st
import requests
st.title("Spam / Ham Detector")
text=st.text_area("Enter an email")
if st.button("Predict"):
    res=requests.post("http://api:8000/predict",json={"email":text})
    st.write(res.json())