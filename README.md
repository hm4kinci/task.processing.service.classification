### CONTACT TYPE CLASSIFICATION

This service gets a string value as an input and tries to predict the type of the contact. 

Class Names: email, meeting, call

### TRAINING ###

Flair `TARSCLASSIFIER` has been trained further with a limited amount of training data.
Training script: `/models/train/contact_type_model.py`
please refer to: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md

### PREDICTION ###

Prediction method is served as an api endpoint `/predict`

### RUNNING ###

- uses Python >= 3.8

`Development/Virtual Environment`: 
- $pip install -r requirements.txt
- refer settings.py file if configuration is needed
- $python app.py
 
`Docker`
- $docker-compose build
- $docker-compose up
- refer docker-compose.yaml if configuration needed. 
- requires **task-network** docker network to communicate with other services. 
    - **to create**: $docker network create task-network
    - **to remove**: $docker network rm my-net