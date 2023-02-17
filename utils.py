from tensorflow.keras.preprocessing import image
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np

#################################################################################
# Constants

# Images Classifier label to description dict
label2desc = {
    'PAD': 'Production Area Diagram',
    'DIS': 'Distributions as Bars',
    'HBD': 'Horizontal Bar Diagram',
    'HSD': 'Horizontal Bar Symmetrical',
    'VBD': 'Vertical Bar Diagram',
    'DTB': 'Distribution as Tukey Boxes',
    'VSD': 'Vertical Stacked Bar Diagram',
    'VBU': 'Vertical Bar Uncertainty',
    'GE2': 'Geosciences 2D',
    'GE3': 'Geosciences 3D',
    'SEM': 'Scanning Electronic Microscopy',
    'IN2': 'Installation Schema 2D',
    'IN3': 'Installation Schema 3D',
    'SAT': 'Satellite Imagery',
    'LEB': 'Colorbar Legend',
    'SCA': 'Scale Legend',
    'TXT': 'Text Legend',
    'ARE': 'Area Diagram',
    'LCO': 'Logs Correlation',
    'LIN': 'Logs Interpreted',
    'LIM': 'Logs Imagery',
    'LSE': 'Logs Seismic',
    'MAD': 'Map Administrative',
    'MGE': 'Map Geosciences',
    'MMO': 'Map Geomodel',
    'MSE': 'Map Seismic',
    'MEQ': 'Equation',
    'M2D': 'Model 2D',
    'M3D': 'Model 3D',
    'LOG': 'Logo',
    'SIG': 'Signature',
    'TAB': 'Table',
    'ORG': 'Organization',
    'COR': 'Cores',
    'MIC': 'Optical Microscopy',
    'CUT': 'Cuttings',
    'FOS': 'Fossil Macroscopic',
    'OUT': 'Outcrop',
    'SUR': 'Equipment Surface',
    'PIE': 'Pie Chart',
    'PLN': 'Project Diagram',
    'SII': 'Seismic with Interpretations',
    'SIA': 'Seismic with Attributes',
    'SIR': 'Seismic Raw',
    'STB': 'Stratigraphic Bar Chart',
    'STL': 'Litho-Stratigraphic Diagram',
    'STT': 'Stratigraphic Diagram',
    'TGN': 'Ternary Diagram',
    'TSM': 'Thin-Section Macroscopic',
    'TSI': 'Thin-Section Microscopic',
    'WDS': 'Well Design',
    'XPC': 'Geochemistry Plot',
    'XPV': 'Cross-Plot Curve',
    'DST': 'DST Plot',
    'XPM': 'Cross-Plot Points & Curve',
    'XPP': 'Cross-Plot Points',
    'XPR': 'Polar Plot',
}

# Image size for prediction
image_size = {
    'B0': {
        'IMG_SIZE': 224
      
    },
    'B7': {
        'IMG_SIZE': 600
    }
}

#################################################################################
# Functions

def load_images_from_bucket(image_path, page_path, storage_client):
    """
    Load image, page from GCS bucket
    :param image_path: path of the image as gcs blob (gs://)
    :param page_path: path of the page as gcs blob (gs://)
    :param storage_client: GCS client
    :return: Pillow Image objects as tuple (image, page)
    """

    image_path_components = image_path.split('/')
    bucket_image_name = image_path_components[2]
    image_name = image_path_components[3] + '/' + image_path_components[4]

    page_path_components = page_path.split('/')
    bucket_page_name = page_path_components[2]
    page_name = page_path_components[3] + '/' + page_path_components[4]

    blob_image_bytes = storage_client.bucket(bucket_image_name).get_blob(image_name)
    blob_page_bytes = storage_client.bucket(bucket_page_name).get_blob(page_name)

    return Image.open(BytesIO(blob_image_bytes.download_as_bytes())), Image.open(BytesIO(blob_page_bytes.download_as_bytes()))


def predict_category(
    original_image:Image, model:tf.keras.Model, target_size:int,
    categories_mapping_dict:dict=None, labels_to_descriptions_dict:dict=None
) -> str:
    """
    Predict image category
    :param original_image: Pillow Image object
    :param model: Keras model to be used
    :param target_size: target image size
    :param categories_mapping_dict: index to category labels dict
    :param labels_to_descriptions_dict: label to description dict
    :return: image category as string
    """
    
    # load the image with the target size
    img = original_image.resize((target_size, target_size))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # prediction index
    pred = model.predict(img_batch)
    
    # get category or description
    if categories_mapping_dict:
        pred_category = categories_mapping_dict[np.argmax(pred, axis=1)[0]]
        label = pred_category

        if label2desc:
            pred_description = labels_to_descriptions_dict[pred_category]
            label = pred_description
        
        return label
    else:
        return pred