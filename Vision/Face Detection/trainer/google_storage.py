from google.cloud import storage
from google.cloud.storage import Blob

import os

BUCKET_NAME = 'mrrdt-object-detector-mlengine'
DATA_FOLDER_NAME = 'data'

def downloadIfAvailable(filePath):
    fileName = os.path.basename(filePath)
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(os.path.join(DATA_FOLDER_NAME, fileName))
    foundFile = blob is not None

    if foundFile:
        if not os.path.isdir(DATA_FOLDER_NAME):
            os.mkdir(DATA_FOLDER_NAME)

        blob.download_to_filename(filePath)
    
    return foundFile

def upload(filePath):
    fileName = os.path.basename(filePath)
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = Blob(os.path.join(DATA_FOLDER_NAME, fileName), bucket)
    blob.upload_from_filename(filePath)