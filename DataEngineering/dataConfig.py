"""
Data Engineering dataConfig.py
Functions for cleaning and prepping the data for training
"""

import os
import sys
import random
import re
from librosa.core import load, get_duration
import pandas as pd
import datasets
from transformers import Wav2Vec2Processor

# Notes
# Special Characters in transcripts:
#   How do we handle numeric characters? For now I'm removing them but how does the transcript handle them?
#   what about +,-,/,*?
#

chars_to_ignore_regex = r'[\d\,\?\.\!\-\;\:\"\}\{\©\[\]\)\(\+\-\*\¼\%]'
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


def remove_special_characters(batch):
    return re.sub(chars_to_ignore_regex, '', batch).lower() + " "


def convertToDataframe(transcript_dir: str = '../Data/Transcripts/', audio_dir: str = '../Data/wav/',
                       min_filesize_limit: float = None, max_filesize_limit: float = None, output_dir: str = "../Data/",
                       train_size: float = None, totalDuration=None) -> pd.DataFrame():
    """
    Convert a directory of transcripts and audios to a single pandas dataframe
    This version currently loads corrected transcripts only
    columns are [file, text, audio, sampling_rate]
    Assumptions:
        The transcript directory has headers seperated by tabs in the order:
            Speaker, Header2, Chunk_start, Chunk_End, Text
        The transcript and corresponding audio file have the same filename (only differentiated by filetype and '-corrected' in the name)
    :param min_filesize_limit: minimum size of audio file in KB
    :param max_filesize_limit: maximum size of audio file in KB
    :param train_size:
    :param totalDuration: total duration to include in the datasets (in seconds)
    :param output_dir:
    :param transcript_dir: directory holding transcripts
    :param audio_dir: directory holding audio files
    :return: Pandas Dataframe
    """
    df_train = pd.DataFrame(columns=['file', 'text', 'audio', 'sampling_rate', 'duration'])
    df_test = pd.DataFrame(columns=['file', 'text', 'audio', 'sampling_rate', 'duration'])

    culminativeDuration = 0
    # Read transcripts into dataframe
    for filename in os.listdir(transcript_dir):
        data = {'file': filename}
        transcript_path = os.path.join(transcript_dir, filename)
        if 'corrected' not in filename:
            continue
        if totalDuration and culminativeDuration >= totalDuration:
            break

        # convert transcript filename to audio filename
        filename = filename.replace('-corrected', '')
        filename = filename.split('.')[0] + '.wav'

        audio_path = os.path.join(audio_dir, filename)
        filesize_KB = os.path.getsize(audio_path) / 1000
        if max_filesize_limit and filesize_KB > max_filesize_limit:
            continue
        if min_filesize_limit and filesize_KB < min_filesize_limit:
            continue

        # floating points take up lots of memory we should look into using a smaller float
        audio_array, sampling_rate = load(audio_path, sr=16000)  # files loaded at 22050 SR for some reason by default

        # Open audio file up,
        # split audio file into json objects with duration of at most 30 seconds using chunk start/end
        with open(transcript_path) as fp:
            fp.readline()  # skip past header
            transcript = ""
            chunk_start = 0
            lines = fp.readlines()
            for i, line in enumerate(lines):
                line_list = line.split('\t')
                chunk_end = float(line_list[-2])
                transcript += line_list[-1].replace('\n', ' ')  # get the last split assumed to be the text
                # if the duration is over 15 sec, or this is the last transcript chunk of the file, write to dataframe
                if chunk_end - chunk_start >= 15 or i == len(lines) - 1:
                    # write this chunk to the df
                    transcript = " " + re.sub(chars_to_ignore_regex, '', transcript).upper() + " "  # clean transcript
                    data['text'] = transcript
                    transcript = ""
                    # get the slice of the audio array which corresponds to this transcript chunk
                    start_frame = int(sampling_rate * chunk_start)
                    end_frame = int(sampling_rate * chunk_end)
                    section = audio_array[start_frame: end_frame]
                    data['audio'] = section
                    data['sampling_rate'] = sampling_rate
                    duration = chunk_end - chunk_start
                    data['duration'] = duration
                    culminativeDuration += duration
                    df_train.loc[len(df_train.index)] = data
                    chunk_start = chunk_end

    # Allocating examples to test dataframe
    if train_size is not None:
        test_size = df_train['duration'].sum() * (1 - train_size)
        duration = 0
        while duration < test_size:
            index = random.randint(0, len(df_train.index) - 1)
            row = df_train.loc[index]
            duration += row['duration']
            df_test.loc[index] = row
            df_train.drop(index, axis=0, inplace=True)

    filename_train = "wav2vec2trainUnder{}KB.json".format(
        str(max_filesize_limit)) if max_filesize_limit else "wav2vec2train.json"
    filename_test = "wav2vec2testUnder{}KB.json".format(
        str(max_filesize_limit)) if max_filesize_limit else "wav2vec2test.json"

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


def createVocabulary(series: pd.Series):
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
    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=batch['sampling_rate']).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


if __name__ == '__main__':
    """
    Arguments:
        transcript dir
        audio dir
        
    Options:
        -m: max filesize (in KB)
        -t: train size proportion
    """
    # convertToDataframe('../Data/Transcripts', '../Data/wav',
    #                    min_filesize_limit=1000, max_filesize_limit=12000, train_size=0, totalDuration=3600)

    if len(sys.argv) < 3:
        raise Exception("Too few arguments supplied. Please supply at least a transcript and audio directory")

    transcriptDir = sys.argv[1]
    audioDir = sys.argv[2]

    if not os.path.isdir(transcriptDir):
        raise Exception('Transcript Directory is invalid: ', transcriptDir)
    elif not os.path.isdir(audioDir):
        raise Exception('Audio Directory is invalid: ', audioDir)

    train_size = 0.90
    if '-t' in sys.argv:
        train_size = float(sys.argv[sys.argv.index("-t")+1])

    max_filesize = None
    if '-m' in sys.argv:
        max_filesize = int(sys.argv[sys.argv.index("-m")+1])

    convertToDataframe(transcriptDir, audioDir, train_size=train_size, max_filesize_limit=max_filesize)
