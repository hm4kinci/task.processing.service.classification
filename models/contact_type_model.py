from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence
from settings import CONTACT_TYPE_MODEL_PATH, CONTACT_TYPE_MODEL_S3_FILE_URL
import logging
import os
import requests


class ContactTypeModel:
    _model = None

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_model():
        if ContactTypeModel._model is not None:
            return
        logging.info('model loading')
        if os.path.isfile(CONTACT_TYPE_MODEL_PATH) is not True:
            logging.info('model downloading')
            r = requests.get(CONTACT_TYPE_MODEL_S3_FILE_URL)
            logging.info('model downloaded')
            with open(CONTACT_TYPE_MODEL_PATH, 'wb') as f:
                logging.info('model saving')
                f.write(r.content)
                logging.info('model saved')

        ContactTypeModel._model = TARSClassifier.load(CONTACT_TYPE_MODEL_PATH)
        logging.info('model loaded')

    @staticmethod
    def predict(query: str):

        if query is None or len(query) == 0:
            return {'success': False, 'message': 'query is required'}

        try:
            ContactTypeModel.load_model()
            sentence = Sentence(query)
            ContactTypeModel._model.predict(sentence)
            return {'success': True, 'data': [label.to_dict() for label in sentence.get_labels()]}
        except RuntimeError as e:
            logging.error(e, exc_info=True)
            return {'success': False, 'message': "Runtime Error: {0}".format(e)}
        except Exception as e:
            logging.error(e, exc_info=True)
            return {'success': False, 'message': 'exception occurred'}

