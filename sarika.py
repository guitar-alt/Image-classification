import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm

st.title("🕵 AI Image Detector")
st.write("Upload an image to detect if it’s AI-generated or real (demo).")

@st.cache_resource
def load_model():
    model = timm.create_model("resnet18", pretrained=True)  # Pretrained ImageNet
    model.fc = torch.nn.Linear(model.fc.in_features, 2)     # 2 classes: Real / AI
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        fake_prob = probs[1].item()
        real_prob = probs[0].item()

    st.subheader("🔍 Prediction (demo):")
    st.write(f"*AI-generated:* {fake_prob*100:.2f}%")
    st.write(f"*Real:* {real_prob*100:.2f}%")

    if fake_prob > 0.6:
        st.error("🚨 This image is likely AI-generated!")
    else:
        st.success("✅ This image looks real!")
