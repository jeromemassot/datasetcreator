from utils import load_images_from_bucket, label2desc

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

# GCS Paths
bucket_figures_name = "daks_figures"
bucket_pages_name = "daks_reports_pages"
file_path = "figs_captions.csv"

#################################################################################
# Loading methods (usually cached)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(
        './models/EfficientNetB0_57k.h5py',
        compile=True
    )
    return model

#################################################################################
# Streamlit App

st.title("Captioned Figures Dataset Explorer")

st.subheader("Welcome")
st.markdown(
    """
    Welcome to the captioned figures dataset explorer. The dataset is contained in Google
    Cloud Storage bucket and metadata is stored in a BigQuery table.\n
    Here, you can browser the dataset content and enrich it by:
    - validating the captioned figures by looking at the caption in the original document page.
    - proposing alternative caption for a particular figure
    - classifying figures per categories
    """
)

st.warning("Please wait for the app to load the model used for classification. This may take a few seconds...")

# loading classifier model
model = load_model()

st.subheader("Dataset Content")

# load original dataset
df = pd.read_csv(f'gs://{bucket_figures_name}/{file_path}', sep='|', storage_options={"token": "petroglyphs-nlp.json"})
df.set_index('index', inplace=True)
df['alternative caption'] = 'None'
df['category'] = 'None'
df['status'] = 'To be reviewed'

# dataset filters
col_a, col_b, col_c = st.columns(3)

with col_a:
    show_only_non_validated = st.checkbox(label="non validated figures only", value=False)
with col_b:
    show_only_non_reviewed = st.checkbox(label="to be reviewed figures only", value=False)
with col_c:
    show_only_validated = st.checkbox(label="validated figures only", value=False)

if show_only_non_validated:
    df = df[~df["status"]=="Validated"]

# display dataset as a table
st.dataframe(df[['caption', 'alternative caption', 'category', 'status']])

# captioned Figures Validator
st.subheader("Captioned Figure Explorer and Validator")

# images navigation slider
index = st.slider(label="Image index", min_value=0, max_value=len(df))

# display individual caption image in its page context
storage_client = storage.Client()

# extract the figure and page images from blobs
image_blob_path = df.at[index, 'path']
page_blob_path = f"gs://{bucket_pages_name}/{str(df.at[index, 'document']).zfill(3)}/{df.at[index, 'page']}.jpg"
image, page = load_images_from_bucket(image_blob_path, page_blob_path, storage_client)

# two columns to display the figure and page image side-by-side
col1, col2 = st.columns(2)
with col1:
    st.write("Captioned Image")
    st.image(image, caption=df.at[index, 'caption'])

with col2:
    st.write("Original page")
    st.image(page)

with st.form(key="figure_validator"):
    st.text_input(label="Alternative caption")
    st.selectbox(label="category", options=sorted(label2desc.values()))
    st.selectbox(label="Status", options=["Validated", "Not Validated", "To be reviewed"])
    st.form_submit_button(label="Save changes")