import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import urllib.request
import os

st.title("🕵 AI Image Detector")
st.write("Upload an image to detect if it’s AI-generated or real.")

# -------------------------------
# Step 1: Load or download the model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "resnet18_ai.pth"
    
    # Public GitHub URL (replace this with your own model if needed)
    url = "https://github.com/your-username/your-repo/raw/main/resnet18_ai.pth"

    # Download the model if it does not exist
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        try:
            urllib.request.urlretrieve(url, model_path)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error("Failed to download model. Using placeholder model.")
            print("Download error:", e)
            model_path = None

    # Load the model
    if model_path and os.path.exists(model_path):
        model = timm.create_model("resnet18", pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        # Placeholder model if download failed
        model = timm.create_model("resnet18", pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.eval()
    return model

model = load_model()

# -------------------------------
# Step 2: Upload image
# -------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Step 3: Preprocess the image
    # -------------------------------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    # -------------------------------
    # Step 4: Predict
    # -------------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        fake_prob = probs[1].item()
        real_prob = probs[0].item()

    # -------------------------------
    # Step 5: Show results
    # -------------------------------
    st.subheader("🔍 Prediction:")
    st.write(f"*AI-generated:* {fake_prob*100:.2f}%")
    st.write(f"*Real:* {real_prob*100:.2f}%")

    if fake_prob > 0.6:
        st.error("🚨 This image is likely AI-generated!")
    else:
        st.success("✅ This image looks real!")
