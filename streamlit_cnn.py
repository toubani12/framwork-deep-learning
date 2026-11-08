import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

IMG_SIZE = 224  # same as the notebook
# sorted folder names of flower_photos, same order as CLASS_NAMES in the notebook
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
LOW_CONFIDENCE = 0.60


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modele_flowers.keras")


def predict(model, image):
    # The model already contains its own Rescaling layer (and augmentation, inactive at inference),
    # so it must receive raw 0-255 pixels, resized exactly like during training.
    x = tf.image.resize(tf.keras.utils.img_to_array(image), [IMG_SIZE, IMG_SIZE])
    batch = tf.stack([x, tf.image.flip_left_right(x)])  # test-time augmentation: image + mirror
    return model.predict(batch, verbose=0).mean(axis=0)


model = load_model()

st.title("Flower classification Application")
st.write("Upload an image of a flower to identify it.")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        probs = predict(model, image)

    best = int(np.argmax(probs))
    st.subheader("Prediction Results")
    st.success(f"**Predicted flower:** {CLASSES[best]}  ({probs[best]:.1%})")
    if probs[best] < LOW_CONFIDENCE:
        st.warning("Low confidence: this may not be one of the 5 supported flowers, "
                   "or the photo is unclear.")

    st.write("### Class probabilities")
    for i in np.argsort(probs)[::-1]:
        st.write(f"{CLASSES[i]}: {probs[i]:.2%}")
        st.progress(float(probs[i]))
else:
    st.info("Please upload an image to classify.")
