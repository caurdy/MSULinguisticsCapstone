import os
from pyannote.database import FileFinder

def DataLoader():
    # tell pyannote.database where to find partition, reference, and wav files
    os.environ["PYANNOTE_DATABASE_CONFIG"] = 'E:\MSULinguisticsCapstone\PyannoteProj\data_preparation\TrainingData\AMI_set\database.yml'
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
