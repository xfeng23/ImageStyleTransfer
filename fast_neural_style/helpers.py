import os, uuid, requests, time, base64, urllib, sys
from io import StringIO
from PIL import Image
from io import BytesIO
from fast_neural_style.neural_style import utils
from fast_neural_style.BlobServiceSingleton import BlobServiceSingleton
from azure.storage.blob import ContentSettings
from fast_neural_style.config import *
from fast_neural_style.django.manage import *
from datetime import datetime
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest, HttpResponseNotFound, HttpResponseServerError

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'production_models')
MAX_DIM = 512
USE_GPU = True
RESIZE_METHOD = Image.ANTIALIAS
BlobService = BlobServiceSingleton()

"""
Takes in an image and blob_uuid. Chooses random style from models
directory. Stylizes image with chosen style, and then uploads image to blob storage.
====================================
Parameters:
    - img: PIL Image to stylize
    - blob_uuid: uuid corresponding to stylized image.
Return:
    N/A
"""
def stylize_and_upload(img, blob_uuid):
    img = preprocess_image(img, MAX_DIM)
    
    style_path = os.path.join(MODELS_DIR, random.choice(os.listdir(MODELS_DIR)))
    img = utils.stylize(img, style_path)

    upload_image_to_blob(img, blob_uuid)

def get_style_from_disk():
    # list the model files in the disk
    if not os.path.exists(MODELS_DIR):
        return []
    else:
        return [filename[0:filename.find('.')] for filename in os.listdir(MODELS_DIR) if filename.endswith('.pth')]


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def loggingIP(request):
    log_ip = open(CLIENTIP_LOG, 'a')
    log_ip.write("{}\tclient ip: {}\n".format(str(datetime.now()), get_client_ip(request)))
    log_ip.close()

"""
Takes in the path of style models and check if it is matching with the models in blob storage.
====================================
Parameters:
    - path_to_model: The path where model ".pth" file saved.
Return:
    - downloads: List of model filenames that need to be downloaded from blob storage.
    - deletes: List of model filenames that need to be deleted from disk.

"""
def check_new_models(path_to_model):
    # 10 model for default setting (keep the first 10 lines in json file same as these default names)
    default_model = ['Bots.pth', 'Candy.pth', 'Gothic.pth', 'Mosaic.pth', 'Scream.pth',
                     'Cabin.pth', 'Composition.pth', 'Graffiti.pth', 'Rain Princess.pth','Trippy.pth']
    try: 
        # list the model files from blobs
        blob_models = []
        generator = BlobService.get_service().list_blobs(CONTAINER_NAME)
        for blob in generator: blob_models.append(blob.name)
        # corner case: when no model presents in blob
        if blob_models == []:
            return [], []
        # list the model files in the disk
        if not os.path.exists(path_to_model):
            return [], []
        disk_models = [filename for filename in os.listdir(path_to_model) if filename.endswith('.pth')]
        #print disk_models
    
        # download files
        downloads = []
        for filename in blob_models:
            if filename not in disk_models:
                downloads.append(filename)
            else:
                disk_models.remove(filename)
        deletes = disk_models
        # corner case: when deleting the default model    
        for item in deletes:
            if item in default_model:
                deletes.remove(item)
    except Exception as e:
        print(e)
        downloads = []
        deletes = []
    finally:
        return downloads, deletes 

"""
Takes in a list of strings and download the pytorch models from Azure blob storage.
=======================================
Parameters: 
    - downloads: List of model filenames need to be downloaded.
    - download_path: The model download path.
Return:
    N/A
"""
def download_models(downloads, download_path):
    # check existence of 'download_path'
    if not os.path.exists(download_path):
        return
    try: 
        for model in downloads:
            path_on_disk = os.path.join(download_path, model)
            # download file from blob
            BlobService.get_service().get_blob_to_path(CONTAINER_NAME, model, path_on_disk)
            print("downloading {}.".format(model))
    except Exception as e:
        print(e)
        return

"""
Takes in a list of strings and delete the pytorch models in the list.
=======================================
Parameters:
    - deletes: List of model filenames need to be deleted.
    - delete_path: The path of models on disk.
Return:
    N/A
"""
def delete_models(deletes, delete_path):
    try:
        for model in deletes:
            path_on_disk = os.path.join(delete_path, model)
            # check existence
            if os.path.exists(path_on_disk):
                os.remove(path_on_disk)
                print("deleting {}.".format(model))
    except Exception as e:
        print(e)
        return


"""
Takes in an image and blob_uuid and uploads image to blob storage.
====================================
Parameters:
    - img: PIL Image to upload to blob storage.
    - blob_uuid: uuid corresponding to stylized image.
Return:
    - str of url in blob storage
"""
def upload_image_to_blob(img, blob_uuid=None):
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf = buf.getvalue() 

    if blob_uuid is None:
        blob_uuid = str(uuid.uuid4())
    suffix = '.jpg'

    BlobService.get_service().create_blob_from_bytes(CONTAINER_NAME, blob_uuid + suffix, buf, 0, None, ContentSettings(content_type='image/jpeg'))
    return make_blob_url(blob_uuid + suffix)

def load_image_from_url(url, tc):
    """
    Called in views.py. Takes in url string and returns PIL Image corresponding to url.  

    Inputs:
    - url: Type 'str'. url of jpg or png image.
    - tc: Type 'appinsights telemetry'. used for logging to application insights.
    Returns:
    - img: PIL Image corresponding to image in url. Returns None on error.
    """
    try:
        r = requests.get(url)
        byte_io = BytesIO(r.content)
        img = Image.open(byte_io)       
    except Exception as e:
        #print("Error caught in load_image_from_url: {}".format(e))
        tc.track_exception()
        return None
    return img

def image_to_memory_stream(img, tc):
    """
    Convert PIL image to memory stream.
    
    Inputs:
    img(PIL object): PIL image.
    tc(appinsights telemetry object): used for logging to application insights.

    Outputs:
    [0](string): the string contain image memory stream and image tag.

    """
    try:
        imagefile = BytesIO()
        img.save(imagefile, format='JPEG')
        imageStream = imagefile.getvalue()
        data_uri = base64.b64encode(imageStream).decode('utf-8').replace('\n', '')
        img_tag = "data:image/jpeg;base64,{0}".format(data_uri)
    except:
        tc.track_exception()
        img_tag = None
    return img_tag




def preprocess_image(img, max_dim=None):
    """
    Takes in PIL and does some simple preprocessing, in preparation
    to stylize.
    
    Inputs:
        - img: PIL 'Image'. Image to preprocess
        - max_dim: Type 'str'. Desired style to transfer onto the image.
    Return:
        - img: PIL 'Image'. Image is has 3 channels 'RGB' and is possibly scaled down.
    """
    if img.mode == 'RGBA':
        img = utils.alpha_mask(img)

    # Model is trained on rgb images. Model takes in 3 channel input.
    if img.mode != 'RGB':
        img = img.convert('RGB')


    # downscale large images to ensure quick runtime
    # img.thumbnail scales the image to preserve image aspect, and be no larger than input dimensions
    if max_dim is not None:
        img.thumbnail((max_dim, max_dim), RESIZE_METHOD)
    return img

def make_blob_url(blob_url):
    return os.path.join(ENDPOINT, CONTAINER_NAME, blob_url)

