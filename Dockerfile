FROM python:3.8-slim as build

WORKDIR /usr/src/app

COPY requirements.txt .
COPY . /usr/src/app

ENV PYTHONPATH /usr/src/app
RUN apt-get update -y && apt-get install git -y --no-install-recommends build-essential gcc libsndfile1

RUN pip install -r requirements.txt
RUN pip install --no-deps rpunct~=1.0.2
#RUN pip install https://github.com/kpu/kenlm/archive/master.zip
ENTRYPOINT ["python", "-u", "./execute.py"]

# docker build -t [Name of the image] .
# e.g docker build -t cap .
# docker run -v ${pwd}:/usr/src/app [Name of the image] [arg1] [arg2] [arg3]
# e.g. docker run -v ${pwd}:/usr/src/app cap -t Atest.wav asr1 dia1
