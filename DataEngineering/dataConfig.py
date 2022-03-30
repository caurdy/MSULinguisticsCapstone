"""
Data Engineering dataConfig.py
Functions for cleaning and prepping the data for training
"""

import os
import re
from librosa.core import load, get_duration
import pandas as pd
import datasets
from typing import Dict, TypeVar

# ToDo
#  Add parameter for file size convertToDataframe

# Notes
# Special Characters in transcripts:
#   How do we handle numeric characters? For now I'm removing them but how does the transcript handle them?
#   what about +,-,/,*?
#

chars_to_ignore_regex = r'[\?\!\.\,\d\-\;\:\"\}\{\©\[\]\)\(\+\-\*\¼\%]'
chars_to_replace_with_space_regex = r'[\?\!\.\,]'


def convertToDataframe(transcript_dir: str = '../Data/Transcripts/', audio_dir: str = '../Data/wav/') -> pd.DataFrame():
    """
    Convert a directory of transcripts and audios to a single pandas dataframe
    This version currently loads corrected transcripts only
    columns are [file, text, audio, sampling_rate]
    Assumptions:
        The transcript directory has headers seperated by tabs in the order:
            Speaker, Header2, Chunk_start, Chunk_End, Chunk
        The transcript and corresponding audio file have the same filename (only differentiated by filetype)
    :param transcript_dir: directory holding transcripts
    :param audio_dir: directory holding audio files
    :return: Pandas Dataframe
    """
    df = pd.DataFrame(columns=['file', 'text', 'audio', 'sampling_rate'])

    # Read transcripts into dataframe
    for filename in os.listdir(transcript_dir):
        if 'corrected' not in filename:
            continue

        path = os.path.join(transcript_dir, filename)
        if os.path.getsize(path) > 1000:  # skip files over 10KB
            continue

        data = {'file': filename}
        with open(path) as fp:
            fp.readline()  # skip past header
            transcript = ""
            for line in fp.readlines():
                transcript += line.split('\t')[-1].replace('\n', ' ')  # get the last split assumed to be the text
            data['text'] = transcript

        filename = filename.replace('-corrected', '')
        filename = filename.split('.')[0] + '.wav'
        audio_path = os.path.join(audio_dir, filename)
        audio_array, sampling_rate = load(audio_path, sr=16000)  # files loaded at 22050 SR for some reason by default
        data['audio'] = audio_array
        data['sampling_rate'] = sampling_rate

        df.loc[len(df.index)] = data

    return df


def convertToDataset(transcript_dir: str = '../Data/Transcripts/',
                     audio_dir: str = '../Data/wav/', ) -> datasets.Dataset:
    """
    Convert a directory of transcripts and audios to a HuggingFace Dataset
    """
    dataset = datasets.Dataset()
    dataset.features = {'audio', datasets.features.audio}
    print(dataset)

    return dataset


def remove_special_characters(batch):
    return re.sub(chars_to_ignore_regex, '', batch).lower() + " "


def extract_all_chars(batch):
    all_text = " ".join(batch)
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def createVocabulary(series: pd.Series(str)) -> Dict[str, int]:
    text_list = series.tolist()
    textBlob = " ".join(text_list)
    vocabList = set(textBlob)
    vocab_dict = {v: k for k, v in enumerate(vocabList)}
    vocab_dict['|'] = vocab_dict[' ']
    del vocab_dict[' ']
    try:
        del vocab_dict['ã']  # where tf is this coming from?
    except KeyError as e:
        print('Key error')
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


def prepare_dataset(batch):
    audio = batch["audio"]
    processor = None
    # batched output is "un-batched" to ensure mapping is correct
    # print(processor(audio, sampling_rate=batch['sampling_rate']).input_values[0])
    batch["input_values"] = processor(audio, sampling_rate=batch['sampling_rate']).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def convertDataToNemoManifest(transcript_dir: str = '../Data/Transcripts/', audio_dir: str = '../Data/wav/',
                              output_directory: str = '../Data/', train_size: float = None,
                              file_size_limit: int = None) -> None:
    """
    Ref: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/datasets.html#preparing-custom-asr-data
    Takes in two paths to directories, one containing audio .wav files and the other transcript .txt files
    Converts directories to Manifest.json files where each line of the manifest follows the format:
        {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}
    These manifest files are the common format for datasets serving Nemo_asr models
    This function will create two manifest files if train_size is specified, one for training and one for testing
    and write them out to output_directory
    If train_size is None, all data will be written to manifest_train.json
        train_size percentage will divide train/test based on duration of audio files (NOT on count)
        the first n samples loaded in which fit under the limit will be transferred to the test set
    If file_size_limit is None, all files will be used.
    NOTE: The training and audio files must have the same name and only differ by a -corrected.txt tag (for transcripts)
            e.x. 'audio1.wav' corresponds to 'audio1-corrected.txt'
          Any transcripts filenames not containing a '-corrected' tag will be ignored
    :param file_size_limit: Limit of filesize to include in dataset, in MB for transcript (.txt) files
    :param output_directory: Directory to write manifest_train.json and manifest_test.json to
    :param transcript_dir: Directory containing .txt transcripts
    :param audio_dir: Directory containing audio .wav files
    :param train_size: Fraction of audio data to put in training manifest
    """
    # ToDO
    #  implement ctc segmentation on audio files > 16.5 seconds
    NEMO_CTC_DURATION_LIMIT = 16.5  # Audio files need to be less than 16.5 seconds for Nemo training
    df_train = pd.DataFrame(columns=['audio_filepath', 'text', 'duration'])
    df_test = pd.DataFrame(columns=['audio_filepath', 'text', 'duration'])
    # Read transcripts into dataframe
    for filename in os.listdir(transcript_dir):
        if '-corrected' not in filename:    # make sure the files have corrections
            continue

        path = os.path.join(transcript_dir, filename)
        if file_size_limit and os.path.getsize(path) > file_size_limit:  # skip files
            continue

        data = {}
        with open(path) as fp:
            fp.readline()  # skip past header
            transcript = ""
            for line in fp.readlines():
                transcript += line.split('\t')[-1].replace('\n', ' ')  # get the last split assumed to be the text
            transcript = " " + re.sub(chars_to_ignore_regex, '', transcript).lower() + " "  # clean transcript
            data['text'] = transcript

        audio_filename = filename.replace('-corrected', '')
        audio_filename = audio_filename.split('.')[0] + '.wav'
        audio_path = os.path.join(audio_dir, audio_filename)
        data['audio_filepath'] = os.path.abspath(audio_path)
        duration = round(get_duration(load(audio_path, sr=16000)[0]), 5)
        data['duration'] = duration
        if duration <= NEMO_CTC_DURATION_LIMIT:
            df_train.loc[len(df_train.index)] = data

    if train_size:
        test_size = df_train['duration'].sum() * (1 - train_size)
        duration = 0
        # Allocating examples to test_size
        for i, row in df_train.iterrows():
            duration += row['duration']
            df_test.loc[len(df_test.index)] = row
            df_train.drop(i, axis=0, inplace=True)
            if duration >= test_size:
                break

        df_test.to_json(os.path.join(output_directory, 'manifest_test_small.json'), orient='records', lines=True)
    df_train.to_json(os.path.join(output_directory, 'manifest_train_small.json'), orient='records', lines=True)


def createCleanedDataFrame():
    """
    Probably not working, supposed to create a dataframe and then clean the text in it for training ASR
    """
    processor = None
    df = convertToDataframe()
    df['text'] = df['text'].apply(remove_special_characters)
    vocab = createVocabulary(df['text'])
    # prepare dataset
    # remove audio from dataset
    df.to_json('../Data/corrected.json')


if __name__ == '__main__':
    convertDataToNemoManifest(train_size=0.80)
