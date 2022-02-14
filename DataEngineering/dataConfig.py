"""
Data Engineering dataConfig.py
Functions for cleaning and prepping the data for training
"""

import os
import re
from librosa.core import load
import pandas as pd

# ToDo
#  Add parameter for file size convertToDataframe

# Notes
# Special Characters in transcripts:
#   How do we handle numeric characters? For now I'm removing them but how does the transcript handle them?
#   what about +,-,/,*?
#

chars_to_ignore_regex = r'[\d\,\?\.\!\-\;\:\"\}\{\©\[\]\)\(\+\-\*\¼\%]'


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
        # if os.path.getsize(path) > 1000:  # skip files over 10KB
        #     continue

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


def remove_special_characters(batch):
    return re.sub(chars_to_ignore_regex, '', batch).lower() + " "


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


if __name__ == '__main__':
    df = convertToDataframe()
    df['text'] = df['text'].apply(remove_special_characters)
    # vocab = createVocabulary(df['text'])
    df.to_json('../Data/corrected.json')
