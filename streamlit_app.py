from utils import load_images_from_bucket, predict_category, label2desc, index2label, image_size, crop_image

from google.oauth2 import service_account
from google.cloud import storage

import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np

import os

#################################################################################
# GCP Credentials
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./petroglyphs-nlp.json"

#################################################################################
# Loading methods (usually cached)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(
        './models/EfficientNetB7_57k.h5py',
        compile=True
    )
    return model


#################################################################################
# Streamlit App

st.title("Captioned Figures Dataset Explorer")

st.markdown(
    """
    Welcome to the captioned figures dataset explorer.
    For each figure in the dataset, you can:
    - validate its caption is consistent with the original document page,
    - propose an alternative caption,
    - modify the figure's category if needed.
    """
)

st.warning("Please wait for the app to load the model used for classification. This may take a few seconds...")

# loading classifier model
model = load_model()

st.subheader("Dataset Content")

st.markdown(
    """The dataset is contained in Google Cloud Storage bucket and metadata is stored in a BigQuery table."""
)

# bucket selector
st.sidebar.subheader("Dataset Selector")
bucket_name = st.sidebar.selectbox(
    label="Select a dataset bucket",
    options=["exploration_development", "petrophysics", "structural", "well_logs"]
)
bucket_figures_name = f"{bucket_name}_figures"
bucket_pages_name = f"{bucket_name}_pages"
file_path = "figs_captions.csv"

# load original dataset
df = pd.read_csv(f'gs://{bucket_figures_name}/{file_path}', sep='|', storage_options={"token": "petroglyphs-nlp.json"})
df.set_index('index', inplace=True)
df.reset_index(inplace=True)

# captioned Figures Validator
st.subheader("Captioned Figure Explorer and Validator")

st.markdown("""
    For each figure in the dataset, you can display the figure and its original context (page).
    You can zoom out to see the context of the figure by using the zoom-out selector.
""")

col1, col2 = st.columns([8, 2])
with col1:
    # images navigation slider
    index = st.slider(label="Image index", min_value=0, max_value=len(df))
with col2:
    expansion = st.number_input(label="Zoom-out", value=1.1, min_value=1.0, max_value=1.5, step=0.1)

# display individual caption image in its page context
storage_client = storage.Client()

# extract the figure and page images from blobs
image_blob_path = df.at[index, 'path']
page_blob_path = f"gs://{bucket_pages_name}/{str(df.at[index, 'document']).zfill(3)}/{df.at[index, 'page_index']}.jpg"
image, page = load_images_from_bucket(image_blob_path, page_blob_path, storage_client)
caption=df.at[index, 'raw_text']

# crop the page image to focus on the figure area
coords = df.at[index, 'coords'].split('|')
coords = [float(coord) for coord in coords]
page_cropped = crop_image(page, coords, page.width, page.height, expansion)

# two columns to display the figure and page area side-by-side
col3, col4 = st.columns(2)
with col3:
    st.write("Page context")
    st.image(page_cropped)

with col4:
    st.write("Captioned Image")
    st.image(image)

    # predict the category of the figure
    category = predict_category(
       image, model, image_size['B7']['IMG_SIZE'], index2label, label2desc
    )

# find the index of the category in the list of categories
categories = sorted(label2desc.values())
category_index = list(categories).index(category)

with st.form(key="figure_validator"):
    st.text_area(label="Caption", value=caption, height=100)
    st.selectbox(label="Category", index=category_index, options=sorted(categories))
    st.selectbox(label="Status", options=["Validated", "Not Validated", "To be reviewed"])
    st.form_submit_button(label="Save changes")