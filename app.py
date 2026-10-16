
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Self-Supervised Learning for Visual Representation")
st.write(
    "Upload an image to classify it into one of the 10 CIFAR-10 classes."
)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    checkpoint = torch.load(
        "supervised_model_full.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Classify Image", type="primary"):
        try:
            model = load_model()
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_index = probabilities.argmax().item()

            st.success(
                f"Predicted class: {classes[predicted_index]}"
            )
            st.metric(
                "Model confidence",
                f"{probabilities[predicted_index].item() * 100:.2f}%"
            )

            st.subheader("Class probabilities")

            probability_data = {
                "Class": classes,
                "Probability (%)": [
                    round(probabilities[i].item() * 100, 2)
                    for i in range(len(classes))
                ]
            }

            st.bar_chart(
                probability_data,
                x="Class",
                y="Probability (%)",
                horizontal=True
            )

            with st.expander("View exact probabilities"):
                st.dataframe(
                    probability_data,
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as error:
            st.error("Could not classify this image.")
            st.exception(error)
