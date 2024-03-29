#################################################################################
## Copyright 2023 Petroglyphs NLP Consulting
## Author: Jerome MASSOT - jerome.massot.78@gmail.com
#################################################################################

from utils import (
    load_images_from_bucket, predict_category, label2desc, index2label, 
    image_size, crop_image, overwrite_figure, update_bigquery_table, 
    load_dataset_from_bq, load_dataset_from_dataframe, 
    reformat_cropped_coordinates, tags_list
)

from google.oauth2 import service_account

from streamlit_cropper import st_cropper
import tensorflow as tf
import streamlit as st
import pandas as pd

from collections import defaultdict
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


def reset_crop_state():
    st.session_state['crop_state'] = None


#################################################################################
# Streamlit App

## Sidebar

# bucket selector
st.sidebar.subheader("Dataset Selector")

bucket_name = st.sidebar.selectbox(
    label="Select a dataset bucket",
    options=[
        "daks", "exploration_development", "general", "geophysical", 
        "ifp", "lithium", "petroleum_geology", "petrology", "petrophysics", 
        "sedimentology", "sequential_stratigraphy", "structural", 
        "well_logs"
    ]
)
bucket_figures_name = f"{bucket_name}_figures"
bucket_pages_name = f"{bucket_name}_pages"

# figure status selector
st.sidebar.subheader("Figure Status Selector")
status_filter = st.sidebar.selectbox(
    label="Status filter", 
    options=["Not Validated", "Validated", "To be reviewed"]
)

# filter dataset button
filter_dataset = st.sidebar.button(label="Select dataset")

# model selector
st.sidebar.subheader("Model Selector")

model_name = st.sidebar.selectbox(
    label="Select a model",
    options=["EfficientNetB0_57k", "EfficientNetB7_57k"]
)

# database status
st.sidebar.subheader("Database status")

if 'database-status' not in st.session_state.keys():
    st.session_state['database-status'] = "Database is up to date"
st.sidebar.write(st.session_state['database-status'])

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

st.warning("Please wait for the app to load the model used for classification.\
            This may take a few seconds...")

# loading classifier model
model, selected_image_size = load_model(model_name)

# captioned Figures Validator
st.subheader("Dataset selector")

st.markdown("The captioned figures explorer allows you to validate the caption\
             and other metadata of each figure in the dataset.")

st.info("Please select a dataset bucket and a status filter in the sidebar and\
         click on the 'Select dataset' button.")

# load original dataset
if filter_dataset:
    dataset_df, storage_client = load_dataset_from_bq(
        bucket_figures_name, status=status_filter
    )

    st.session_state['current_df'] = dataset_df
    st.session_state['storage_client'] = storage_client
    st.session_state['database-status'] = "Database is up to date"

## figure navigation slider
if 'current_df' in st.session_state.keys() and st.session_state['current_df'] is not None:

    df = st.session_state['current_df']
    storage_client = st.session_state['storage_client']

    st.subheader("Captioned Figures Explorer")

    st.markdown("""
    For each figure in the dataset, you can display the figure and its original context (page).
    You can zoom out to see the context of the figure by using the zoom-out selector.
    """)

    col1, col2 = st.columns([8, 2])
    with col1:
        # images navigation slider
        slider_max = 1 if len(df)==1 else len(df)-1
        if len(df)==1==1:
            slider_disable=True
        else:
            slider_disable=False
        index = st.slider(
            label="Image index", 
            min_value=0, max_value=slider_max, 
            disabled=slider_disable, 
            on_change=reset_crop_state
        )
    with col2:
        expansion = st.number_input(
            label="Zoom-out", value=1.1, min_value=1.0, max_value=1.5, 
            step=0.1
        )

    # display individual caption image in its page context

    # extract the figure and page images information
    data_dict = load_dataset_from_dataframe(df, index)
    id_image = data_dict['id']
    page_index = data_dict['page_index']
    document = data_dict['document']
    caption = data_dict['caption']
    status = data_dict['status']
    raw_coords = data_dict['coords']
    category = data_dict['category']

    # extract the tags as a list (originaly a string using | as separator)
    tags = data_dict['tags'] if '|' not in data_dict['tags'] else data_dict['tags'].split('|')
    if tags=="None":
        tags=None

    # extract the figure and page images from blobs
    image_blob_path = df.at[index, 'url']
    page_blob_path = f"gs://{bucket_pages_name}/{str(df.at[index, 'document']).zfill(3)}/{df.at[index, 'page_index']}.jpg"
    image, page = load_images_from_bucket(image_blob_path, page_blob_path, storage_client)

    # crop the page image to focus on the figure area
    coords = raw_coords.split('|')
    coords = [float(coord) for coord in coords]
    page_cropped_coords = [coords[0], 1-coords[3], coords[2], 1-coords[1]]
    page_cropped = crop_image(page, page_cropped_coords, page.width, page.height, expansion)

    # two columns to display the figure and page area side-by-side
    with st.expander(label="Figure and Page Context"):
        col3, col4, col5 = st.columns([9, 1, 9])
        with col3:
            st.image(page_cropped)
            st.write("Page context")
        with col5:
            st.image(image)
            st.write("Captioned Image")

    if page:
        st.markdown("""
        If needed, you can overwrite the image by cropping the page to better fit the figure.
        Just drag the corners of the box to select the area of interest. 
        The updated figure will be displayed below.
        """)

        st.warning("""The updated figure will replace the original figure in the database.\
                    Do it only if needed.""")

        # two columns to page the new figure side-by-side
        with st.expander(label="Figure Update"):
   
            # cropped_box as its left and top coordinates as well as its width and height.
            cropped_box = st_cropper(page, box_color='black', realtime_update=False, return_type='box')
            crop_coords = reformat_cropped_coordinates(cropped_box, page.width, page.height)
            cropped_img = crop_image(page, crop_coords, page.width, page.height)
            updated_coords = '|'.join([str(coord) for coord in crop_coords])

            if cropped_img:
                st.image(cropped_img)
                overwrite_button = st.button(label="Overwrite Figure")

                if overwrite_button:
                    # find the index of the figure in the bigquery table
                    filters = {
                        'id': id_image, 'page_index': page_index, 
                        'document': document, 'coords': raw_coords
                    }

                    # overwrite the figure in the bucket
                    backup_url = overwrite_figure(cropped_img, image, image_blob_path, storage_client)

                    # update the figure url backup and the the new figure coords in the bigquery table
                    url_box_log_dict = {'Backup url': 'update failed', 'Box coords': 'update failed'}
                    msg = update_bigquery_table(filters, 'backup_url', backup_url)
                    if msg.startswith('1'):
                        url_box_log_dict['Backup url'] = 'update successful'
                    msg = update_bigquery_table(filters, 'coords', updated_coords)
                    if msg.startswith('1'):
                        url_box_log_dict['Box coords'] = 'update successful'

                    # update the session crop_state with the updated box coordinates
                    st.session_state['crop_state'] = updated_coords

                    # display the messages
                    st.write(url_box_log_dict)

                st.write("Updated Figure")

        # predict the category of the figure if the figure has not category yet
        if category=='None':
            updated_category = predict_category(
                image, model, selected_image_size, index2label, label2desc
            )
        else:
            updated_category = category

        # find the index of the category in the list of categories
        categories = sorted(label2desc.values())
        category_index = list(categories).index(updated_category)

        st.markdown("""You can now validate or modify the metadata attached to the figure""")
        st.info("""Click the Save button when done.""")

        with st.form(key="figure_validator"):

            # display the figure caption
            updated_caption = st.text_area(label="Caption", value=caption, height=100)

            # display the tags multiselect with the current tags selected
            updated_tags = st.multiselect(label="Tags", options=sorted(tags_list), default=tags)
            if not updated_tags or len(updated_tags)==0:
                updated_tags = 'None' # to be consistent with the database
            else:
                updated_tags = '|'.join(updated_tags)

            # display the category
            updated_category = st.selectbox(
                label="Category", index=category_index, options=sorted(categories) + ['Others']
            )

            # display the status
            updated_status = st.selectbox(
                label="Status", options=["Validated", "Not Validated", "To be reviewed"], 
                index=["Validated", "Not Validated", "To be reviewed"].index(status)
            )

            # save the changes
            saved_changes_button = st.form_submit_button(label="Save changes")

            # update the figure metadata in the database
            if saved_changes_button:

                # find the index of the figure in the bigquery table based on
                # the image id, the page index, the document id and the box coordinates
                if 'crop_state' in st.session_state.keys() and st.session_state['crop_state']:
                    filtering_coords = st.session_state['crop_state']
                else:
                    filtering_coords = raw_coords

                filters = {
                    'id': id_image, 
                    'page_index': page_index, 
                    'document': document, 
                    'coords': filtering_coords
                }

                # update the figure metadata in the bigquery table
                figure_log_dict = defaultdict(str)

                if updated_caption != caption:
                    msg = update_bigquery_table(filters, 'caption', updated_caption)
                    if msg.startswith('1'):
                        figure_log_dict['Caption'] = 'update successful'
                    else:
                        figure_log_dict['Caption'] = 'update failed'

                if updated_tags != tags:
                    msg = update_bigquery_table(filters, 'tags', updated_tags)
                    if msg.startswith('1'):
                        figure_log_dict['Tags'] = 'update successful'
                    else:
                        figure_log_dict['Tags'] = 'update failed'
                
                if updated_category != category:
                    msg = update_bigquery_table(filters, 'category', updated_category)
                    if msg.startswith('1'):
                        figure_log_dict['Category'] = 'update successful'
                    else:
                        figure_log_dict['Category'] = 'update failed'
                
                if updated_status != status:
                    msg = update_bigquery_table(filters, 'status', updated_status)
                    if msg.startswith('1'):
                        figure_log_dict['Status'] = 'update successful'
                    else:
                        figure_log_dict['Status'] = 'update failed'


                st.write(figure_log_dict)

                # update the database status
                st.session_state['database-status'] = "Database has been updated."
