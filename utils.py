from tensorflow.keras.preprocessing import image
import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np

#################################################################################
# Classifier parameters (labels, indexes, description, image size)

index2label_original = {
    0: 'ARE', 
    1: 'COR', 
    2: 'CUT', 
    3: 'DIS', 
    4: 'DST', 
    5: 'DTB', 
    6: 'FOS', 
    7: 'GE2', 
    8: 'GE3', 
    9: 'HBD', 
    10: 'HSD', 
    11: 'IN2', 
    12: 'IN3', 
    13: 'LCO', 
    14: 'LEB', 
    15: 'LIM', 
    16: 'LIN', 
    17: 'LOG', 
    18: 'LSE', 
    19: 'M2D', 
    20: 'M3D', 
    21: 'MAD', 
    22: 'MEQ', 
    23: 'MGE', 
    24: 'MIC', 
    25: 'MMO', 
    26: 'MSE', 
    27: 'ORG', 
    28: 'OUT', 
    29: 'PAD', 
    30: 'PIE', 
    31: 'PLN', 
    32: 'SAT', 
    33: 'SCA', 
    34: 'SEM', 
    35: 'SIA', 
    36: 'SIG', 
    37: 'SII', 
    38: 'SIR', 
    39: 'STB', 
    40: 'STL', 
    41: 'STT', 
    42: 'SUR', 
    43: 'TAB', 
    44: 'TGN', 
    45: 'TSI', 
    46: 'TSM', 
    47: 'TXT', 
    48: 'VBD', 
    49: 'VBU', 
    50: 'VSD', 
    51: 'WDS', 
    52: 'XPC', 
    53: 'XPM', 
    54: 'XPP', 
    55: 'XPR', 
    56: 'XPV'
}

# Images label -> description dict and index -> label dict
label2desc = {
    'ARE': 'Area Diagram',
    'COR': 'Cores',
    'CUT': 'Cuttings',
    'DIS': 'Distributions as Bars',
    'DST': 'DST Plot',
    'DTB': 'Distribution as Tukey Boxes',
    'FOS': 'Fossil Macroscopic',
    'GE2': 'Geosciences 2D',
    'GE3': 'Geosciences 3D',
    'HBD': 'Horizontal Bar Diagram',
    'HSD': 'Horizontal Bar Symmetrical',
    'IN2': 'Installation Schema 2D',
    'IN3': 'Installation Schema 3D',
    'LCO': 'Logs Correlation',
    'LEB': 'Colorbar Legend',
    'LIM': 'Logs Imagery',
    'LIN': 'Logs Interpreted',
    'LOG': 'Logo',
    'LSE': 'Logs Seismic',
    'M2D': 'Model 2D',
    'M3D': 'Model 3D',
    'MAD': 'Map Administrative',
    'MEQ': 'Equation',
    'MGE': 'Map Geosciences',
    'MIC': 'Optical Microscopy',
    'MMO': 'Map Geomodel',
    'MSE': 'Map Seismic',
    'ORG': 'Organization',
    'OUT': 'Outcrop',
    'PAD': 'Production Area Diagram',
    'PIE': 'Pie Chart',
    'PLN': 'Project Diagram',
    'SAT': 'Satellite Imagery',
    'SCA': 'Scale Legend',
    'SEM': 'Scanning Electronic Microscopy',
    'SIA': 'Seismic with Attributes',
    'SIG': 'Signature',
    'SII': 'Seismic with Interpretations',
    'SIR': 'Seismic Raw',
    'STB': 'Stratigraphic Bar Chart',
    'STL': 'Litho-Stratigraphic Diagram',
    'STT': 'Stratigraphic Diagram',
    'SUR': 'Equipment Surface',
    'TAB': 'Table',
    'TGN': 'Ternary Diagram',
    'TSI': 'Thin-Section Microscopic',
    'TSM': 'Thin-Section Macroscopic',
    'TXT': 'Text Legend',
    'VBD': 'Vertical Bar Diagram',
    'VBU': 'Vertical Bar Uncertainty',
    'VSD': 'Vertical Stacked Bar Diagram',
    'WDS': 'Well Design',
    'XPC': 'Geochemistry Plot',
    'XPM': 'Cross-Plot Points & Curve',
    'XPP': 'Cross-Plot Points',
    'XPR': 'Polar Plot',
    'XPV': 'Cross-Plot Curve'
}
index2label = {k: v for k, v in enumerate(label2desc.keys())}

# Image size for prediction
image_size = {
    'B0': {'IMG_SIZE': 224},
    'B7': {'IMG_SIZE': 600}
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
    

def crop_image(image:Image, coords:list, w:int, h:int, expansion:float=1.) -> Image:
    """
    Crop image
    :param image: Pillow Image object
    :param coords: coordinates of the crop
    :param w: image width
    :param h: image height
    :param expansion: expansion factor
    :return: cropped image as Pillow Image object
    """

    coords[0] = coords[0]*expansion
    coords[1] = coords[1]*expansion
    coords[2] = coords[2]/expansion
    coords[3] = coords[3]/expansion

    image_crop = image.crop(
        (int(coords[2]*w), int((1-coords[1])*h), int(coords[0]*w), int((1-coords[3])*h))
    )
    return image_crop