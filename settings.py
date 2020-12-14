from common.utils import get_env

DEBUG = get_env('FLASK_DEBUG', True)
ENV = get_env('FLASK_ENV', 'development')

HOST = get_env('HOST', '0.0.0.0')
PORT = get_env('PORT', 5001)

CONTACT_TYPE_MODEL_PATH = get_env('MASK_MODEL_PATH', './models/pretrained/contact_type_model/final-model.pt')
CONTACT_TYPE_MODEL_S3_FILE_URL = get_env('CONTACT_TYPE_MODEL_S3_FILE_URL',
                                         'https://hm4kinci.s3-eu-west-1.amazonaws.com/task.processing/service.classification/contact_type/final-model.pt')