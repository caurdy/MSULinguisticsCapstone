"""
Running analysis on transcript file from directory 'MCD transcripts - tsv 2022-01-17-1'
"""
import os
import shutil
from librosa import load, get_duration

oldDir = '../Data/MCD transcripts - tsv 2022-01-17-1'
newDir = '../Data/Transcripts'
newDir = '../../../CapstoneCode/Data/Transcripts'
audioDir = '../../../CapstoneCode/Data/wav'


def cleanTranscripts():
    for filename in os.listdir(oldDir):
        if filename[0] != '.':
            shutil.copy(os.path.join(oldDir, filename), newDir)


def transcriptStatistics():
    count = 0
    total = 0
    for filename in os.listdir(newDir):
        if 'correct' in filename:
            audioFilename = filename.replace('-corrected.txt', '.wav')
            path = os.path.join(audioDir, audioFilename)
            size = os.path.getsize(path) / 1000
            if size <= 10000:
                duration = get_duration(load(path)[0])
                print(duration)
                count += 1
                total += duration
    print('Number of corrected transcripts ', count, 'less than 10 kb', '\nTotal duration', total)


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
    durationTotal = 0
    for filename in os.listdir(audioDir):
        path = os.path.join(audioDir, filename)
        if 1000 <= os.path.getsize(path)/1000 <= 20000:
            durationTotal += get_duration(filename=path)

    print('Total duration of all audio files (in hours)', durationTotal / 3600)


if __name__ == "__main__":
    durationStatistics()
