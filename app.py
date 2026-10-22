import tempfile
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

st.title("🦺 PPE Detection (YOLO11)")
st.write("Image ya video upload karein, model helmet, vest, gloves, goggles, boots (aur unke 'no-' classes) detect karega.")

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.write("Classes:", ", ".join(model.names.values()))


def show_counts(result):
    names = [model.names[int(c)] for c in result.boxes.cls]
    if names:
        st.write("**Detections:**", dict(Counter(names)))
    else:
        st.write("Kuch detect nahi hua.")


tab_img, tab_vid = st.tabs(["📷 Image", "🎞️ Video"])

with tab_img:
    img_file = st.file_uploader("Image upload karein", type=["jpg", "jpeg", "png"], key="img")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        result = model.predict(np.array(image), conf=conf, verbose=False)[0]
        annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(annotated, caption="Detection", use_container_width=True)
        show_counts(result)

with tab_vid:
    vid_file = st.file_uploader("Video upload karein", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if vid_file and st.button("Process video"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_box = st.empty()
        bar = st.progress(0.0)
        i = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % 2:  # har doosra frame skip: cloud par speed ke liye
                continue
            result = model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
            frame_box.image(result.plot()[:, :, ::-1], use_container_width=True)
            bar.progress(min(i / total, 1.0))
        cap.release()
        bar.progress(1.0)
        st.success("Video process ho gayi.")
