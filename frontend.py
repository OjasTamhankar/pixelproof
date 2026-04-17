import streamlit as st
import requests

st.title("Fake vs Real Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}

        try:
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            result = response.json()

            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']:.4f}")

        except:
            st.error("API not running")