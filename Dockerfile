FROM google/cloud-sdk:alpine as gcloud
WORKDIR /app
COPY . /app
# my_key.json has to be created with GCP project service account credentials to 
# successfully authenticate
RUN gcloud auth activate-service-account --key-file=my_key.json
    
# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
#COPY commands.sh .
COPY . /app

# Creates a non-root user and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN useradd appuser && chown -R appuser /app
USER appuser

WORKDIR /app/src

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["sh", "commands.sh"]

