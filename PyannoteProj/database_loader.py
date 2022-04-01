import os
from pyannote.database import FileFinder
from pathlib import Path


def CreateDatabase(filename):
    pass


def DataLoader():
    # tell pyannote.database where to find partition, reference, and wav files
    sub_folder = '/data_preparation/TrainingData/AMI_set/database.yml'
    folder = str(Path(sub_folder).absolute())
    os.environ["PYANNOTE_DATABASE_CONFIG"] = folder
    print(os.environ["PYANNOTE_DATABASE_CONFIG"])

    # used to automatically find paths to wav files
    preprocessors = {'audio': FileFinder()}

    # initialize 'only_words' experimental protocol
    from pyannote.database import get_protocol
    only_words = get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)

    print("Load_AMI done")
    return only_words


if __name__ == '__main__':
    ami = DataLoader()
    print(type(ami))
