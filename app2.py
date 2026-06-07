
# app.py
import streamlit as st
from PIL import Image
import easyocr
import pyttsx3
import numpy as np
from colorspacious import cspace_convert
from transformers import pipeline
import os

# ---------------- Page Setup ----------------
st.set_page_config(page_title="AI Accessibility Toolkit", layout="wide")
st.title("AI Accessibility Toolkit")
st.write("An AI-powered tool to support dyslexic, visually impaired, and colorblind users.")

# ---------------- Dyslexia-Friendly CSS ----------------
st.markdown("""
<style>
textarea, input {
    font-family: 'OpenDyslexic', Arial, sans-serif !important;
    font-size: 18px;
    line-height: 1.5em;
    letter-spacing: 0.03em;
}
.stButton>button {
    font-family: 'OpenDyslexic', Arial, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Custom Model ----------------
@st.cache_resource
def load_custom_model():
    model_path = "trained_model"
    if os.path.exists(model_path):
        try:
            return pipeline("summarization", model=model_path, tokenizer=model_path)
        except:
            return None
    return None

custom_model = load_custom_model()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Text-to-Speech",
    "Custom AI Text Simplifier",
    "OCR + AI Cleanup",
    "Dyslexia-Friendly Typing",
    "Colorblind Filter"
])

# ---------------- 1. Text-to-Speech ----------------
with tab1:
    st.header("Text-to-Speech")
    text_input = st.text_area("Type text to read aloud:", height=150)

    if st.button("Speak"):
        if text_input.strip() == "":
            st.warning("Please type some text first!")
        else:
            try:
                engine = pyttsx3.init()
                engine.say(text_input)
                engine.runAndWait()
                st.success("Text spoken successfully!")
            except Exception as e:
                st.error(f"TTS Error: {e}")

# ---------------- 2. Custom AI Text Simplifier ----------------
with tab2:
    st.header("Custom AI Text Simplifier")
    st.info("This feature uses my trained model from the ASSET dataset.")

    user_text = st.text_area("Enter text to simplify:", height=150)

    if st.button("Simplify Text"):
        if user_text.strip() == "":
            st.warning("Please enter text first.")
        else:
            if custom_model is None:
                st.error("⚠ No trained model found.\nTrain your model and save it in 'trained_model/'")
            else:
                output = custom_model(
                    user_text,
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )[0]['summary_text']

                st.text_area("Simplified Text:", output, height=150)

# ---------------- 3. OCR + AI Cleanup ----------------
with tab3:
    st.header("AI Reading from Images + Cleanup")
    uploaded_file = st.file_uploader("Upload an image with text", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        try:
            reader = easyocr.Reader(['en'])
            result = reader.readtext(np.array(image))
            raw_text = "\n".join([res[1] for res in result])

            st.subheader("Extracted OCR Text")
            st.text_area("OCR Output:", raw_text, height=150)

            # AI cleanup using your model
            st.subheader("AI-Cleaned Text")
            if custom_model is None:
                st.error("⚠ Custom model not found. Train your model first.")
            else:
                cleaned = custom_model(
                    raw_text,
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )[0]['summary_text']

                st.text_area("Cleaned Text:", cleaned, height=150)

        except Exception as e:
            st.error(f"OCR Error: {e}")

# ---------------- 4. Dyslexia-Friendly Typing ----------------
with tab4:
    st.header("Dyslexia-Friendly Typing Area")

    # Embed OpenDyslexic font
    st.markdown("""
    <style>
    @font-face {
        font-family: 'OpenDyslexic';
        src: url('OpenDyslexic3-Regular.ttf') format('truetype');
    }
    textarea, input, .stButton>button {
        font-family: 'OpenDyslexic', Arial, sans-serif !important;
        font-size: 18px;
        line-height: 1.5em;
        letter-spacing: 0.03em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dyslexia-friendly typing area
    dyslexia_text = st.text_area("Type here:", height=150)
    st.write("You typed:", dyslexia_text)


# ---------------- 5. Colorblind Filter ----------------
with tab5:
    st.header("Colorblind Image Simulation")
    cb_type = st.selectbox(
        "Choose colorblindness type:",
        ["Normal", "Protanopia (red-blind)", "Deuteranopia (green-blind)", "Tritanopia (blue-blind)"]
    )

    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="cb")

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(image)

        if cb_type == "Normal":
            cb_img = img_np
        else:
            lms = cspace_convert(img_np, "sRGB1", "CAM02-UCS")
            if cb_type == "Protanopia":
                lms[..., 0] = 0
            elif cb_type == "Deuteranopia":
                lms[..., 1] = 0
            elif cb_type == "Tritanopia":
                lms[..., 2] = 0

            cb_img = np.clip(
                cspace_convert(lms, "CAM02-UCS", "sRGB1") * 255,
                0, 255
            ).astype(np.uint8)

        st.image(cb_img, caption=f"Simulated: {cb_type}", use_column_width=True)
