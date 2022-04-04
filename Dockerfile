FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt .
COPY . /usr/src/app

ENV PYTHONPATH /usr/src/app

RUN pip install -r requirements.txt
# Below, installing pyannote
RUN pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1
#
ENTRYPOINT ["python", "-u", "./execute.py"]
#CMD ["./Combine/CombineFeatures.py"]

#ENV NGIXN_WORKER_PROCESSES auto
#EXPOSE 8000
# CMD ["python", "./main.py"]

# docker build -t cap .
# docker run cap testing (testing is goig to be the argv[1])
# docker run -v ${pwd}:/usr/src/app asr testing
