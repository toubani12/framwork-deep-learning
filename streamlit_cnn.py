from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "modele_flowers.keras"
REPO_URL = "https://github.com/toubani12/framwork-deep-learning"
IMAGE_BASE_URL = "https://raw.githubusercontent.com/toubani12/framwork-deep-learning/main/types_images"

IMG_SIZE = 224  # same as the notebook
# sorted folder names of flower_photos, same order as CLASS_NAMES in the notebook
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
LOW_CONFIDENCE = 0.60


@dataclass(frozen=True)
class Flower:
    name: str
    latin: str
    image: str
    blurb: str

    @property
    def image_url(self):
        return f"{IMAGE_BASE_URL}/{self.image}"


FLOWERS = {
    "daisy": Flower("Daisy", "Bellis perennis", "daisy.jpeg", "White rays around a sunny yellow eye."),
    "dandelion": Flower("Dandelion", "Taraxacum", "dandelion.jpeg", "Golden meadow heads that turn to seed clocks."),
    "roses": Flower("Roses", "Rosa", "rose.jpeg", "Layered, velvety petals in every shade."),
    "sunflowers": Flower("Sunflowers", "Helianthus annuus", "sunflower.jpeg", "Broad golden faces on towering stems."),
    "tulips": Flower("Tulips", "Tulipa", "tulipe.jpeg", "Smooth, cup-shaped blooms of spring."),
}

# st.html strips inline SVG, so these icons are drawn from assets/style.css
FLOWER_ICON = '<span class="icon-flower" aria-hidden="true"></span>'
FLOURISH = '<span class="flourish" aria-hidden="true"></span>'


@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf  # imported here so the page renders before TensorFlow loads

    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, image):
    import tensorflow as tf

    # The model already contains its own Rescaling layer (and augmentation, inactive at inference),
    # so it must receive raw 0-255 pixels, resized exactly like during training.
    x = tf.image.resize(tf.keras.utils.img_to_array(image), [IMG_SIZE, IMG_SIZE])
    batch = tf.stack([x, tf.image.flip_left_right(x)])  # test-time augmentation: image + mirror
    return model.predict(batch, verbose=0).mean(axis=0)


def section_head(anchor, title, subtitle):
    st.html(f'<div class="section-head" id="{anchor}">{FLOURISH}<h2>{title}</h2><p>{subtitle}</p></div>')


def nav_html():
    return f"""
    <nav class="nav" id="top">
      <a class="nav-brand" href="#top">{FLOWER_ICON}Bloom ID</a>
      <div class="nav-links">
        <a href="#identify">Identify</a>
        <a href="#collection">Collection</a>
        <a href="#how">How it works</a>
      </div>
      <a class="nav-tag" href="{REPO_URL}" target="_blank" rel="noopener">View on GitHub</a>
    </nav>
    """


def hero_html():
    petals = "".join(
        f'<div class="petal"><img src="{flower.image_url}" alt="{flower.name}"></div>'
        for flower in FLOWERS.values()
    )
    drifting = "".join(f'<span class="drift d{i}"></span>' for i in range(1, 7))
    return f"""
    <div class="hero-wrap">
      <section class="hero">
        <div aria-hidden="true">{drifting}</div>
        <div class="hero-disc"></div>
        <h1 class="hero-word">Flowers</h1>
        <div class="bloom">{petals}<div class="bloom-heart"></div></div>
        <div class="hero-copy">
          <h2>Name every bloom</h2>
          <p>Share a photo and a neural network trained on thousands of flower pictures
          tells you which of five species it sees.</p>
          <div class="hero-actions">
            <a class="btn" href="#identify">Identify a flower</a>
            <a class="btn-link" href="#collection">Browse species &rarr;</a>
          </div>
        </div>
        <div class="hero-count"><b>05</b><span>species recognized</span></div>
      </section>
    </div>
    """


def cards_html(probs=None):
    has_result = probs is not None
    order = np.argsort(probs)[::-1] if has_result else range(len(CLASSES))
    cards = []
    for rank, index in enumerate(order):
        flower = FLOWERS[CLASSES[index]]
        is_top = has_result and rank == 0
        badge = meter = ""
        if is_top:
            label = "Closest match" if probs[index] < LOW_CONFIDENCE else "Best match"
            badge = f'<span class="card-badge">{label}</span>'
        if has_result:
            meter = (
                f'<div class="meter"><i style="width:{probs[index]:.1%}"></i></div>'
                f'<div class="card-pct">{probs[index]:.1%}</div>'
            )
        cards.append(
            f"""
            <article class="card{' top' if is_top else ''}">
              <div class="card-photo"><img src="{flower.image_url}" alt="{flower.name}" loading="lazy"></div>
              {badge}
              <div class="card-latin">{flower.latin}</div>
              <h3>{flower.name}</h3>
              <p>{flower.blurb}</p>
              {meter}
            </article>
            """
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def how_html():
    steps = [
        ("Share a photo", "Upload a clear, close-up picture. The app reads it in memory and never saves it."),
        ("The network looks twice", f"Your photo and its mirror image are resized to {IMG_SIZE} × {IMG_SIZE} px and "
                                    "read by a MobileNetV2 model."),
        ("Scores become a verdict", "Both readings are averaged into one probability per species, and the highest "
                                    "one is your match."),
    ]
    items = "".join(
        f'<div class="step"><b>{number:02d}</b><h3>{title}</h3><p>{text}</p></div>'
        for number, (title, text) in enumerate(steps, start=1)
    )
    return f'<div class="steps">{items}</div>'


st.set_page_config(page_title="Bloom ID", page_icon=":material/local_florist:", layout="wide")
st.html(ROOT / "assets" / "style.css")

with st.container(key="page"):
    st.html(nav_html())
    st.html(hero_html())

    features = [
        (":material/local_florist:", "Five species", "Daisy to tulip"),
        (":material/photo_library:", "2,568 photos", "Used for training"),
        (":material/neurology:", "MobileNetV2", "Transfer learning"),
        (":material/flip:", "Checked twice", "Photo and its mirror"),
        (":material/lock:", "Never saved", "Stays in memory"),
    ]
    with st.container(horizontal=True, gap="medium", key="features"):
        for icon, title, caption in features:
            with st.container(width="stretch"):
                st.markdown(f"{icon} **{title}**")
                st.caption(caption)

    section_head("identify", "Identify your flower", "Upload a photo of a daisy, dandelion, rose, sunflower or tulip.")

    probs = None
    with st.container(key="studio"):
        upload_column, result_column = st.columns(2, gap="large")

        with upload_column:
            uploaded_file = st.file_uploader(
                "Upload a flower photo", type=["png", "jpg", "jpeg", "webp"], key="uploader"
            )
            st.caption("Works best with one flower, in focus, filling most of the frame.")

        image = None
        if uploaded_file is not None:
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            except (UnidentifiedImageError, OSError):
                upload_column.error("This file could not be read as an image. Please try another photo.")

        with result_column:
            if image is None:
                st.html(
                    f'<div class="empty-arch">{FLOWER_ICON}<b>Your bloom will appear here</b>'
                    "Upload a photo to see what the model thinks it is.</div>"
                )
            else:
                with st.spinner("Waking the model and studying your photo..."):
                    probs = predict(load_model(), image)
                best = int(np.argmax(probs))
                flower = FLOWERS[CLASSES[best]]

                st.caption("This looks like")
                st.header(flower.name, anchor=False)
                st.markdown(f"*{flower.latin}*. {flower.blurb}")
                st.metric("Confidence", f"{probs[best]:.1%}")
                st.progress(float(probs[best]))
                if probs[best] < LOW_CONFIDENCE:
                    st.warning(
                        "This may not be one of the 5 supported flowers, or the photo is unclear.",
                        title="Low confidence",
                        icon=":material/warning:",
                    )

        if image is not None:
            with upload_column:
                with st.container(key="preview"):
                    st.image(image, alt="Your uploaded flower photo", width="stretch")

    section_head(
        "collection",
        "The collection",
        "The five species this model knows. After a prediction, each card shows its probability.",
    )
    st.html(cards_html(probs))

    section_head("how", "How it works", "Three steps from your photo to a name.")
    st.html(how_html())
    st.html(
        '<p class="foot">It only knows these five species, so any other flower gets matched to the closest one. '
        f'Built with TensorFlow and Streamlit. <a href="{REPO_URL}" target="_blank" rel="noopener">Source code</a></p>'
    )
