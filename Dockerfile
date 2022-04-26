FROM python:3.8-slim as build

WORKDIR /usr/src/app

RUN apt-get update -y && apt-get install git -y --no-install-recommends build-essential gcc libsndfile1
COPY . .
RUN pip install -r requirements.txt
RUN pip install --no-deps rpunct~=1.0.2
ENTRYPOINT ["python", "-t", "-i", "-u", "./execute.py"]
