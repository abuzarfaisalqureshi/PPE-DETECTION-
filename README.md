# PPE-DETECTION-
# 🦺 PPE Detection App (YOLO11 + Streamlit)

Yeh ek Streamlit web app hai jo YOLO11n model se images aur videos mein PPE (Personal Protective Equipment) detect karti hai.

## Features

- Image upload karein aur original ke saath detection result dekhein
- Video upload karein aur frame-by-frame detection dekhein
- Confidence threshold slider (sidebar mein)
- Har detection ki class-wise count

## Detected Classes

| Class | Matlab |
|-------|--------|
| `helmet`, `vest`, `gloves`, `goggles`, `boots` | PPE pehna hua hai |
| `no-helmet`, `no-vest`, `no-gloves`, `no-goggles`, `no-boots` | PPE nahi pehna |

## Model Details

- Model: YOLO11n (`yolo11n.pt` se fine-tune kiya gaya)
- Epochs: 20
- Image size: 640
- Framework: Ultralytics

## Project Structure

```
.
├── app.py             # Streamlit app
├── best.pt            # Trained YOLO11n weights
├── requirements.txt   # Python dependencies
└── README.md
```

## Local Setup

```bash
# 1. Dependencies install karein
pip install -r requirements.txt

# 2. App chalayein
streamlit run app.py
```

App browser mein `http://localhost:8501` par khul jaye gi.

## Streamlit Cloud par Deploy

1. Teeno files (`app.py`, `best.pt`, `requirements.txt`) aur yeh README GitHub repo ke root mein push karein.
2. [share.streamlit.io](https://share.streamlit.io) par jayein aur GitHub se login karein.
3. **New app** dabayein, apna repo aur branch select karein.
4. Main file path `app.py` rakhein.
5. Agar build fail ho to **Advanced settings** mein Python 3.11 select karein.
6. **Deploy** dabayein.

## Notes

- Video processing free CPU par slow ho sakti hai, is liye pehle chhoti video se test karein.
- App har doosra frame process karti hai taake speed behtar rahe.
- Confidence slider se detection ki sensitivity adjust karein: kam value par zyada detections aati hain, lekin false positives bhi barh sakte hain.

## Credits

- [Ultralytics YOLO11](https://docs.ultralytics.com)
- [Streamlit](https://streamlit.io)
