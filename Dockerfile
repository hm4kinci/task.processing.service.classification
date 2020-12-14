FROM python:3.8.6
RUN python -m pip install --upgrade pip 
COPY . /service.classification
WORKDIR /service.classification
RUN pip install -r requirements.txt
EXPOSE 5001
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]