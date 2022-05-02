"""
Running analysis on transcript file from directory 'MCD transcripts - tsv 2022-01-17-1'
"""
import os
import shutil
from librosa import load, get_duration

oldDir = '../Data/MCD transcripts - tsv 2022-01-17-1'
newDir = '../Data/Transcripts'
newDir = '../../CapstoneCode/Data/wav'


def cleanTranscripts():
    for filename in os.listdir(oldDir):
        if filename[0] != '.':
            shutil.copy(os.path.join(oldDir, filename), newDir)


def transcriptStatistics():
    count = 0
    for filename in os.listdir(newDir):
        path = os.path.join(newDir, filename)
        size = os.path.getsize(path)
        if 'correct' in filename and size <= 10000:
            count += 1
    print('Number of corrected transcripts ', count, 'less than 10 kb')


def sizeStatistics():
    count = 0
    total_size = 0
    for filename in os.listdir(newDir):
        path = os.path.join(newDir, filename)
        size = os.path.getsize(path)
        if 'correct' in filename:
            total_size += size
            count += 1
    print('Average size (bytes): ', total_size / count)


def durationStatistics():
    count = 0
    filenames = []
    durationTotal = 0
    for filename in os.listdir(newDir):
        path = os.path.join(newDir, filename)
        size_kb = os.path.getsize(path) / 1000
        if size_kb:
            durationTotal += round(get_duration(load(path, sr=16000)[0]), 3)
            count += 1

    print('Total duration of files btwn 500 and 20000 KB', durationTotal, ' num:', count)


if __name__ == "__main__":
    durationStatistics()
