import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import tempfile
import numpy as np
import os
from PIL import Image
from datetime import datetime
import smtplib
from email.message import EmailMessage
import time

# -------------------------
# Model definition
# -------------------------
class AccidentCNN(nn.Module):
    def __init__(self):
        super(AccidentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 13 * 13, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = AccidentCNN()
    model.load_state_dict(torch.load("saved_models/accident_model.pth", map_location=device))
    model.eval()
    model.to(device)
    return model

# -------------------------
# Preprocessing
# -------------------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img_tensor

# -------------------------
# Email Alert
# -------------------------
def send_email_alert(image_path):
    msg = EmailMessage()
    msg['Subject'] = '🚨 Accident Alert from CCTV Detection System'
    msg['From'] = 'shiv79029@gmail.com'  # replace with your Gmail
    msg['To'] = 'shiv79029@gmail.com'  # replace with recipient
    msg.set_content('An accident has been detected. Please check the attached frame.')

    with open(image_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login('shiv79029@gmail.com', 'aiuy iknq ggwx yxmo')  # Use Gmail App Password
            smtp.send_message(msg)
    except Exception as e:
        st.warning(f"Email alert failed: {e}")

# -------------------------
# Real-Time Detection
# -------------------------
def real_time_detection(video_source):
    model = load_model()
    cap = cv2.VideoCapture(video_source)
    stframe = st.empty()
    st.markdown("### Detecting... Alerts will be sent on detection.")

    threshold = 0.7
    last_alert_time = 0
    cooldown_seconds = 8
    os.makedirs("accident_frames", exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = preprocess_frame(frame)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            acc_prob = probs[0][1].item()

        label = "Accident" if acc_prob > threshold else "No Accident"
        color = (0, 0, 255) if label == "Accident" else (0, 255, 0)

        cv2.putText(frame, f"{label} ({acc_prob:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        stframe.image(frame, channels='BGR')

        current_time = time.time()
        if acc_prob > threshold and (current_time - last_alert_time) > cooldown_seconds:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = f"accident_frames/accident_{timestamp}.jpg"
            cv2.imwrite(frame_path, frame.copy())
            send_email_alert(frame_path)
            st.image(frame_path, caption="📸 Accident Frame Sent", use_column_width=True)
            last_alert_time = current_time
            st.error(f"🚨 Accident Detected! Email sent at {timestamp}.")

    cap.release()
    st.success("✅ Stream ended or camera disconnected.")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚗 Real-Time Accident Detection System")

mode = st.radio("Select Input Mode:", ["Upload Video", "Live CCTV"])

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        real_time_detection(tfile.name)

elif mode == "Live CCTV":
    ip_address = st.text_input("Enter CCTV Camera IP (e.g., rtsp://ip:port/stream)")
    if st.button("Start Stream"):
        if ip_address:
            real_time_detection(ip_address)
        else:
            st.warning("Please enter a valid CCTV stream address.")
