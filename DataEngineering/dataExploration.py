"""
Running analysis on transcript file from directory 'MCD transcripts - tsv 2022-01-17-1'
"""
import os
import shutil

oldDir = '../Data/MCD transcripts - tsv 2022-01-17-1'
newDir = '../Data/Transcripts'

def cleanTranscripts():
    for filename in os.listdir(oldDir):
        if filename[0] != '.':
            shutil.copy(os.path.join(oldDir, filename), newDir)


def transcriptStatistics():
    count = 0
    for filename in os.listdir(newDir):
        if 'correct' in filename:
            count += 1
    print('Number of corrected transcripts', count)


def sizeStatistics():
    count = 0
    total_size = 0
    for filename in os.listdir(newDir):
        path = os.path.join(newDir, filename)
        size = os.path.getsize(path)
        if 'correct' in filename:
            total_size += size
            count += 1
    print('Average size (bytes): ', total_size/count)


if __name__ == "__main__":
    sizeStatistics()
