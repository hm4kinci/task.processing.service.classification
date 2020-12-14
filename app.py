from flask import Flask, request
from models.contact_type_model import ContactTypeModel
import settings

app = Flask(__name__)
app.config.from_object(settings)

ContactTypeModel.load_model()


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    query = request.args.get('query')
    return ContactTypeModel.predict(query)


if __name__ == '__main__':
    app.run(
        host=app.config['HOST'],
        port=int(app.config['PORT']),
        debug=app.config['DEBUG']
    )
