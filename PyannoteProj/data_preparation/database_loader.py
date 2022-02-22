import os


def DataLoader():
    # tell pyannote.database where to find partition, reference, and wav files
    os.environ["PYANNOTE_DATABASE_CONFIG"] = 'PyannoteProj/data_preparation/database.yml'

    # used to automatically find paths to wav files
    from pyannote.database import FileFinder
    preprocessors = {'audio': FileFinder()}

    # initialize 'only_words' experimental protocol
    from pyannote.database import get_protocol
    only_words = get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)

    print("Load_AMI done")
    return only_words


if __name__ == '__main__':
    ami = DataLoader()
    print(type(ami))
