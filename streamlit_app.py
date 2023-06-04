from utils import (
    load_images_from_bucket, predict_category, label2desc, 
    index2label, image_size, crop_image, overwrite_figure,
    update_bigquery_table
)

from google.oauth2 import service_account
from google.cloud import storage

from streamlit_cropper import st_cropper
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

@st.cache_resource()
def load_model(model_name:str='EfficientNetB0_57k'):
    """
    Load the classifier model from the models folder
    :param model_name: name of the model to load
    :return: the loaded model, image size
    """
    model = tf.keras.models.load_model(
        f'./models/{model_name}.h5py',
        compile=True
    )

    if model_name == "EfficientNetB0_57k":
        selected_image_size = image_size['B0']['IMG_SIZE']
    else:
        selected_image_size = image_size['B7']['IMG_SIZE']

    return model, selected_image_size


@st.cache_resource()
def load_dataset_from_bq(origin:str, status:str='Not Validated') -> pd.DataFrame:
    """
    Load the list of figures and associated metadata as Pandas DataFrame from BigQuery
    :param origin: origin of the figures (also the name of the GCS bucket)
    :param status: status of the figures to load
    :return: figures data as Pandas Dataframe and the GCS client
    """
    # init the GCS client
    storage_client = storage.Client()

    # load the dataset from BigQuery
    query = f"""
        SELECT
            id,
            url,
            category,
            coords,
            caption,
            tags,
            origin,
            document,
            page_index,
            status, 
            backup_url
        FROM
            `petroglyphs-nlp.geosciences_ai_datasets.geosciences-captioned-figures`
        WHERE
            status = '{status}' AND origin = '{origin}'
    """
    df = pd.read_gbq(query=query, dialect='standard')

    return df, storage_client

#################################################################################
# Streamlit App

## Sidebar

# bucket selector
st.sidebar.subheader("Dataset Selector")

bucket_name = st.sidebar.selectbox(
    label="Select a dataset bucket",
    options=["daks", "exploration_development", "petrophysics", "structural", "well_logs"]
)
bucket_figures_name = f"{bucket_name}_figures"
bucket_pages_name = f"{bucket_name}_pages"

# model selector
st.sidebar.subheader("Model Selector")

model_name = st.sidebar.selectbox(
    label="Select a model",
    options=["EfficientNetB0_57k", "EfficientNetB7_57k"]
)

## Main page

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
model, selected_image_size = load_model(model_name)

# load original dataset
df, storage_client = load_dataset_from_bq(bucket_figures_name)

# captioned Figures Validator
st.subheader("Captioned Figures Explorer")

st.markdown("""
    For each figure in the dataset, you can display the figure and its original context (page).
    You can zoom out to see the context of the figure by using the zoom-out selector.
""")

## figure navigation slider
col1, col2 = st.columns([8, 2])
with col1:
    # images navigation slider
    index = st.slider(label="Image index", min_value=0, max_value=len(df)-1)
with col2:
    expansion = st.number_input(label="Zoom-out", value=1.1, min_value=1.0, max_value=1.5, step=0.1)

# display individual caption image in its page context

# extract the figure and page images information
id_image = df.at[index, 'id']
page_index = df.at[index, 'page_index']
document = df.at[index, 'document']
caption = df.at[index, 'caption']
status = df.at[index, 'status']

# extract the figure and page images from blobs
image_blob_path = df.at[index, 'url']
page_blob_path = f"gs://{bucket_pages_name}/{str(df.at[index, 'document']).zfill(3)}/{df.at[index, 'page_index']}.jpg"
image, page = load_images_from_bucket(image_blob_path, page_blob_path, storage_client)

# crop the page image to focus on the figure area
raw_coords = df.at[index, 'coords']
coords = raw_coords.split('|')
coords = [float(coord) for coord in coords]
page_cropped = crop_image(page, coords, page.width, page.height, expansion)

# two columns to display the figure and page area side-by-side
with st.expander(label="Figure and Page Context"):
    col3, col4, col5 = st.columns([9, 1, 9])
    with col3:
        st.image(page_cropped)
        st.write("Page context")
    with col5:
        st.image(image)
        st.write("Captioned Image")

st.markdown("""
If needed, you can overwrite the image by cropping the page to better fit the figure.
Just drag the corners of the box to select the area of interest. The updated figure will be displayed below.
""")
st.warning("""The updated figure will replace the original figure in the database. Do it only if needed.""")

# two columns to page the new figure side-by-side
with st.expander(label="Figure Extraction"):
   
    cropped_img = st_cropper(page, box_color='black', realtime_update=False)
    if cropped_img:
        st.image(cropped_img)
        overwrite_button = st.button(label="Overwrite Figure")
        if overwrite_button:
            filters = {'id': id_image, 'page_index': page_index, 'document': document, 'coords': raw_coords}
            backup_url = overwrite_figure(cropped_img, image, image_blob_path, storage_client)
            msg = update_bigquery_table(filters, 'backup_url', backup_url)
            st.write(msg)
        st.write("Updated Figure")

# predict the category of the figure
category = predict_category(
    image, model, selected_image_size, index2label, label2desc
)

# find the index of the category in the list of categories
categories = sorted(label2desc.values())
category_index = list(categories).index(category)

st.markdown("""You can now validate or modify the metadata attached to the figure. Click the Save button when done.""")
with st.form(key="figure_validator"):
    caption = st.text_area(label="Caption", value=caption, height=100)
    tags = st.multiselect(label="Tags", options=['structural', 'stratigraphic', 'sedimentology', 'reservoir', 'fluids', 'production'])
    category = st.selectbox(label="Category", index=category_index, options=sorted(categories))
    status = st.selectbox(label="Status", options=["Validated", "Not Validated", "To be reviewed"], index=["Validated", "Not Validated", "To be reviewed"].index(status))
    st.form_submit_button(label="Save changes")