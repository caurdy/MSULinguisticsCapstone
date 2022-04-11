import os
from pyannote.database import FileFinder
from pathlib import Path

import yaml


def CreateDatabase(file_name: str, split=0.2, validation = True):
    dataset_path = 'TrainingData' + '/' + file_name
    if not os.path.exists(dataset_path):
        raise KeyError('Cannot find indicated file directory')

    data_names_lst = list(os.listdir(dataset_path))
    if "WAV_set" not in data_names_lst or \
            "RTTM_set" not in data_names_lst or \
            "UEM_set" not in data_names_lst:
        raise KeyError('Cannot find indicated file directory')

    os.makedirs(dataset_path + '/LIST_set')

    """
        Creating the txt file list
    """
    wav_lst = list(os.listdir(dataset_path + "/WAV_set"))
    len_train = int(len(wav_lst) * (1 - (split * 2)))
    with open(dataset_path + '/LIST_set/train.txt', 'w') as train_file:
        for idx in range(len_train):
            train_file.write(wav_lst[idx].split('.')[0])
            train_file.write('\n')

    if validation is True:
        len_val = int(len(wav_lst) - len_train) // 2
        len_test = len(wav_lst) - len_train - len_val

        with open(dataset_path + '/LIST_set/dev.txt', 'w') as train_file:
            for idx in range(len_train, len_train + len_val):
                train_file.write(wav_lst[idx].split('.')[0])
                train_file.write('\n')

        with open(dataset_path + '/LIST_set/test.txt', 'w') as train_file:
            for idx in range(len_train + len_val, len_train + len_val + len_test):
                train_file.write(wav_lst[idx].split('.')[0])
                train_file.write('\n')

    """
        Creating yml database config
    """

    with open('../database.yml', 'r') as config:
        data = yaml.load(config, Loader=yaml.FullLoader)

        data['Databases'][file_name] = dataset_path + "/WAV_set/{uri}.wav"
        data['Protocols'][file_name] = {}
        data['Protocols'][file_name]['SpeakerDiarization'] = {}
        data['Protocols'][file_name]['SpeakerDiarization']['only_words'] = {
            'train': {'uri': str(dataset_path + '/LIST_set/train.txt'),
                      'annotation': str(dataset_path + '/RTTM_set/{uri}.rttm'),
                      'annotated': str(dataset_path + '/UEM_set/{uri}.uem')},
            'development': {'uri': str(dataset_path + '/LIST_set/dev.txt'),
                            'annotation': str(dataset_path + '/RTTM_set/{uri}.rttm'),
                            'annotated': str(dataset_path + '/UEM_set/{uri}.uem')},
            'test': {'uri': str(dataset_path + '/LIST_set/test.txt'),
                     'annotation': str(dataset_path + '/RTTM_set/{uri}.rttm'),
                     'annotated': str(dataset_path + '/UEM_set/{uri}.uem')}}
    config.close()

    with open('../database.yml', 'w') as fp:
        yaml.dump(data, fp)
    fp.close()


def DataLoader(file_name: str):
    # tell pyannote.database where to find partition, reference, and wav files
    sub_folder = 'assets/database.yml'
    folder = str(Path(sub_folder).absolute())
    os.environ["PYANNOTE_DATABASE_CONFIG"] = folder
    print(os.environ["PYANNOTE_DATABASE_CONFIG"])

    # used to automatically find paths to wav files
    preprocessors = {'audio': FileFinder()}

    # initialize 'only_words' experimental protocol
    from pyannote.database import get_protocol
    only_words = get_protocol('{}.SpeakerDiarization.only_words'.format(file_name), preprocessors=preprocessors)

    print("Load_{} done".format(file_name))
    return only_words


if __name__ == '__main__':
    CreateDatabase('Talkbank', split=0.2, validation=True)
    data_names_lst = list(os.listdir('TrainingData/Talkbank'))
    print(data_names_lst)
    with open('assets/database.yml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        print(data)
    # DataLoader('Talkbank')
