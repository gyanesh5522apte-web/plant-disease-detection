import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide",
)

MODEL_PATH = "models/mobilenet_finetuned.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Bacterial spot",
    "Early blight",
    "Healthy",
    "Late blight",
    "Leaf mold",
    "Septoria leaf spot",
    "Spider mites",
    "Target spot",
    "Tomato mosaic virus",
    "Yellow leaf curl virus",
]

CONFIDENCE_LEVEL = 0.70

DISEASE_INFO = {
    "Bacterial spot": (
        "A disease caused by bacteria. It shows as small dark spots on the leaves and spreads faster in warm, wet weather.",
        "Remove the affected leaves and avoid watering from above. Ask an agricultural advisor about treatment.",
    ),
    "Early blight": (
        "A fungal disease. It shows as brown spots with rings, like a target, usually on the older, lower leaves first.",
        "Remove the infected lower leaves and keep the leaves dry. Ask an agricultural advisor about treatment.",
    ),
    "Healthy": (
        "No signs of the 9 diseases this app knows were found on this leaf.",
        "Keep checking the plant regularly. The model can miss early or partial damage, so ask an expert if the leaf looks unusual.",
    ),
    "Late blight": (
        "A serious disease that spreads fast, especially in cool, wet weather. It shows as large dark, greasy-looking patches.",
        "Act quickly: contact an agricultural advisor as soon as possible, and remove badly affected plants.",
    ),
    "Leaf mold": (
        "A fungal disease, common in humid greenhouses. It shows as yellow patches on top of the leaf and olive-green mould underneath.",
        "Improve air flow around the plants and lower the humidity. Ask an agricultural advisor about treatment.",
    ),
    "Septoria leaf spot": (
        "A fungal disease. It shows as many small round spots with dark edges and pale centres, starting on the lower leaves.",
        "Remove the infected leaves and avoid wetting the leaves when watering. Ask an agricultural advisor about treatment.",
    ),
    "Spider mites": (
        "Tiny pests, not a disease. They cause fine yellow speckles on the leaves, and sometimes thin webbing underneath, often in hot, dry weather.",
        "Check the undersides of the leaves for mites. Ask an agricultural advisor about how to control them.",
    ),
    "Target spot": (
        "A fungal disease. It shows as brown spots with rings, which can look very similar to early blight.",
        "Remove the affected leaves and improve air flow. Ask an agricultural advisor to confirm, since it is easy to confuse with early blight.",
    ),
    "Tomato mosaic virus": (
        "A virus. It shows as a patchy light and dark green pattern, and the leaves can curl or look misshapen. It spreads through hands and tools.",
        "There is no cure. Remove infected plants, and clean hands and tools. Ask an agricultural advisor for next steps.",
    ),
    "Yellow leaf curl virus": (
        "A virus spread by small insects called whiteflies. The leaves turn yellow and curl upwards, and the plant stops growing well.",
        "There is no cure. Control the whiteflies and remove infected plants. Ask an agricultural advisor for next steps.",
    ),
}

SAMPLES = {
    "Bacterial spot": "app/samples/correct_bacterial_spot.png",
    "Early blight": "app/samples/correct_early_blight.png",
    "Healthy": "app/samples/correct_healthy.png",
    "Late blight": "app/samples/correct_late_blight.png",
    "Leaf mold": "app/samples/correct_leaf_mold.png",
    "Septoria leaf spot": "app/samples/correct_septoria_leaf_spot.png",
    "Spider mites": "app/samples/correct_spider_mites.png",
    "Target spot": "app/samples/correct_target_spot.png",
    "Tomato mosaic virus": "app/samples/correct_tomato_mosaic_virus.png",
    "Yellow leaf curl virus": "app/samples/correct_yellow_leaf_curl_virus.png",
    "Late blight (known failure case)": "app/samples/failure_late_blight_called_healthy.png",
}

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(image):
    resized = image.resize(IMAGE_SIZE, Image.BILINEAR)
    image_array = np.array(resized, dtype="float32")
    image_array = np.expand_dims(image_array, axis=0)
    return model.predict(image_array, verbose=0)[0]


model = load_model()

st.title("🍅 Tomato Leaf Disease Detector")
st.write("A quick first check of a tomato leaf photo.")
st.warning("Advisory only, not a diagnosis. Please confirm the result with an agricultural expert.")

left, right = st.columns(2)

image = None

with left:
    st.subheader("1. Choose a leaf")
    source = st.radio("Image source", ["Upload a photo", "Try a sample"], horizontal=True)

    if source == "Upload a photo":
        uploaded_file = st.file_uploader("Upload a photo of one tomato leaf", type=["jpg", "jpeg", "png"])
        st.caption("For best results: one leaf, laid flat on a plain surface, in good light. Photos of whole plants or leaves in the field can give wrong answers.")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your leaf", width=350)
    else:
        sample_name = st.selectbox("Choose a sample leaf from the test set", list(SAMPLES.keys()))
        image = Image.open(SAMPLES[sample_name]).convert("RGB")
        st.image(image, caption="True label: " + sample_name, width=350)
        if "failure" in sample_name:
            st.caption("This leaf really has Late blight, but the model gets it wrong. It shows why the result should be checked by an expert.")

with right:
    st.subheader("2. Result")
    if image is None:
        st.info("Upload a leaf photo to see the result.")
    else:
        probabilities = predict(image)
        top_indices = np.argsort(probabilities)[::-1][:3]
        best = int(top_indices[0])
        confidence = float(probabilities[best])
        best_name = CLASS_NAMES[best]

        st.markdown("### " + best_name)
        if confidence > 0.999:
            st.write("Confidence: over 99.9%")
        else:
            st.write("Confidence: " + str(round(confidence * 100, 1)) + "%")
        st.progress(confidence)

        if confidence < CONFIDENCE_LEVEL:
            level_text = str(int(CONFIDENCE_LEVEL * 100)) + "%"
            st.warning("The model is not sure about this leaf (confidence below " + level_text + "). Treat this result with caution and ask an expert.")

        meaning, next_step = DISEASE_INFO[best_name]
        st.markdown("**What this means**")
        st.write(meaning)
        st.markdown("**Suggested next step**")
        st.write(next_step)

        st.markdown("**Other possibilities**")
        for i in top_indices[1:]:
            i = int(i)
            chance = float(probabilities[i])
            st.write(CLASS_NAMES[i] + ": " + str(round(chance * 100, 1)) + "%")
            st.progress(chance)

st.divider()

with st.expander("How it works"):
    st.markdown(
        "1. The leaf photo is resized to 224 x 224 pixels.\n"
        "2. It goes into a MobileNetV2 model, first trained on general photos (ImageNet) and then fine-tuned on tomato leaves.\n"
        "3. The model gives a probability for each of the 10 classes.\n"
        "4. The most likely class is shown, with the next two below it.\n"
        "5. If the top probability is below 70%, the app shows a warning."
    )

with st.expander("Limitations"):
    st.markdown(
        "- Works only for tomato leaves: 9 diseases and healthy. Any other image is still forced into one of these 10 classes.\n"
        "- Trained on lab photos: one leaf on a plain background. On field photos (several leaves, soil, shadows) it can be wrong, even with high confidence and no warning.\n"
        "- Weakest on Early blight (78% found in testing), which it often confuses with Target spot.\n"
        "- It can call a diseased leaf healthy: this happened 7 times out of 1,665 diseased test leaves.\n"
        "- It gives advice, not a diagnosis. Always confirm with an agricultural expert."
    )

with st.expander("About the model"):
    st.markdown(
        "- Model: MobileNetV2, fine-tuned (layers 100 to 153 retrained)\n"
        "- Data: PlantVillage from TensorFlow Datasets, 18,159 tomato leaf images, 10 classes\n"
        "- Test set: 1,825 images never used in training\n"
        "- Test accuracy 96.8%, macro-F1 0.952, top-3 accuracy 99.7%\n"
        "- Size: 2.3M parameters, about 37 ms per image on a laptop"
    )