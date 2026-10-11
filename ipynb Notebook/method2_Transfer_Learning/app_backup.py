import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_PATH = r"C:\Users\Vinay\Downloads\archive\bird_species_project"

DATASET_PATH = r"C:\Users\Vinay\Downloads\archive\CUB_200_2011"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "efficientnet_b0_cub_finetuned.pth"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bird Species Identifier",
    page_icon="🐦",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🐦 Bird Species Identifier")

st.write(
    "Upload an image of a bird and the model will predict "
    "its species using a fine-tuned EfficientNet-B0 model."
)

st.divider()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    class_names = {}

    classes_file = os.path.join(
        DATASET_PATH,
        "classes.txt"
    )

    with open(classes_file, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(" ", 1)
            class_names[int(class_id)] = class_name

    return class_names


class_names = load_class_names()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        200
    )

    # Load our trained model
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model, weights


model, weights = load_model()

transform = weights.transforms()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a bird image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Bird Image",
        width=500
    )

    st.divider()

    if st.button("🔍 Identify Bird", use_container_width=True):

        with st.spinner("Analyzing image..."):

            # Preprocess image
            input_image = transform(image)
            input_image = input_image.unsqueeze(0)

            # Prediction
            with torch.no_grad():

                output = model(input_image)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                confidence = probabilities[
                    0,
                    predicted_class
                ].item()

        # CUB classes are 1-200
        predicted_id = predicted_class + 1

        predicted_name = class_names[
            predicted_id
        ]

        # Remove "001." from the displayed name
        display_name = predicted_name.split(
            ".",
            1
        )[1].replace("_", " ")

        # ====================================================
        # RESULT
        # ====================================================

        st.success("Prediction completed!")

        st.subheader("🐦 Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Bird Species",
                display_name
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.progress(
            confidence
        )

        st.caption(
            f"CUB-200-2011 Class ID: {predicted_id}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About this model"):

    st.write(
        """
        **Model:** EfficientNet-B0

        **Learning approach:** Transfer Learning + Fine-tuning

        **Dataset:** CUB-200-2011

        **Number of bird species:** 200

        **Training images:** 4,000

        **Test images:** 400

        **Test accuracy:** 55.00%

        The model was first trained with the pretrained
        EfficientNet feature extractor frozen, followed by
        fine-tuning of the deeper feature layers.
        """
    )