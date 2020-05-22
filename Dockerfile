FROM python:3.6-slim-stretch
ADD ./requirements.txt $APP_HOME
RUN pip install -r requirements.txt
RUN rm -rf requirements.txt