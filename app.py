
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

st.set_page_config(page_title="Self-Supervised Learning", page_icon="🧠")

st.title("Self-Supervised Learning for Visual Representation")
st.write("Upload an image to classify it using the trained CIFAR-10 model.")

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    model.load_state_dict(
        torch.load("supervised_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        try:
            model = load_model()
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_index = probabilities.argmax().item()

            st.success(f"Predicted class: {classes[predicted_index]}")
            st.write(
                f"Confidence: {probabilities[predicted_index].item() * 100:.2f}%"
            )

            st.subheader("Class probabilities")
            for i, name in enumerate(classes):
                st.write(f"{name}: {probabilities[i].item() * 100:.2f}%")

        except Exception as error:
            st.error(f"Error loading model: {error}")
