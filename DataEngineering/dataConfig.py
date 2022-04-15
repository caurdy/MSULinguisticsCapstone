"""
Data Engineering dataConfig.py
Functions for cleaning and prepping the data for training
"""

import os
import re
from librosa.core import load, get_duration
import pandas as pd
import datasets

# Notes
# Special Characters in transcripts:
#   How do we handle numeric characters? For now I'm removing them but how does the transcript handle them?
#   what about +,-,/,*?
#

chars_to_ignore_regex = r'[\d\,\?\.\!\-\;\:\"\}\{\©\[\]\)\(\+\-\*\¼\%]'


def remove_special_characters(batch):
    return re.sub(chars_to_ignore_regex, '', batch).lower() + " "


def convertToDataframe(transcript_dir: str = '../Data/Transcripts/', audio_dir: str = '../Data/wav/',
                       filesize_limit: float = None, output_dir: str = "../Data/",
                       train_size: float = None) -> pd.DataFrame():
    """
    Convert a directory of transcripts and audios to a single pandas dataframe
    This version currently loads corrected transcripts only
    columns are [file, text, audio, sampling_rate]
    Assumptions:
        The transcript directory has headers seperated by tabs in the order:
            Speaker, Header2, Chunk_start, Chunk_End, Chunk
        The transcript and corresponding audio file have the same filename (only differentiated by filetype)
    :param train_size:
    :param output_dir:
    :param filesize_limit:
    :param transcript_dir: directory holding transcripts
    :param audio_dir: directory holding audio files
    :return: Pandas Dataframe
    """
    df_train = pd.DataFrame(columns=['file', 'text', 'audio', 'sampling_rate', 'duration'])
    df_test = pd.DataFrame(columns=['file', 'text', 'audio', 'sampling_rate', 'duration'])

    # Read transcripts into dataframe
    for filename in os.listdir(transcript_dir):
        if 'corrected' not in filename:
            continue

        path = os.path.join(transcript_dir, filename)
        if filesize_limit and os.path.getsize(path) > filesize_limit:  # skip files over 10KB
            continue

        data = {'file': filename}
        with open(path) as fp:
            fp.readline()  # skip past header
            transcript = ""
            for line in fp.readlines():
                transcript += line.split('\t')[-1].replace('\n', ' ')  # get the last split assumed to be the text
            transcript = " " + re.sub(chars_to_ignore_regex, '', transcript).lower() + " "  # clean transcript
            data['text'] = transcript

        filename = filename.replace('-corrected', '')
        filename = filename.split('.')[0] + '.wav'
        audio_path = os.path.join(audio_dir, filename)
        # floating points take up lots of memory we should look into using fp32/16 or something smaller for these arrays
        audio_array, sampling_rate = load(audio_path, sr=16000)  # files loaded at 22050 SR for some reason by default
        data['audio'] = audio_array
        data['sampling_rate'] = sampling_rate
        duration = round(get_duration(load(audio_path, sr=16000)[0]), 5)
        data['duration'] = duration
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

    filename_train = "wav2vec2trainUnder{}KB.json".format(str(filesize_limit)) if filesize_limit else "wav2vec2train.json"
    filename_test = "wav2vec2testUnder{}KB.json".format(str(filesize_limit)) if filesize_limit else "wav2vec2test.json"

    df_train.to_json(os.path.join(output_dir, filename_train))
    df_test.to_json(os.path.join(output_dir, filename_test))
    return df_train, df_test


def convertToDataset(transcript_dir: str = '../Data/Transcripts/',
                     audio_dir: str = '../Data/wav/', ) -> datasets.Dataset:
    """
    Convert a directory of transcripts and audios to a HuggingFace Dataset
    """
    dataset = datasets.Dataset()
    dataset.features = {'audio', datasets.features.audio}
    print(dataset)

    return dataset


def extract_all_chars(batch):
    all_text = " ".join(batch)
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def createVocabulary(series: pd.Series) -> dict[str: int]:
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


if __name__ == '__main__':
    convertToDataframe('../Data/Transcripts', '../Data/wav', filesize_limit=10000)
