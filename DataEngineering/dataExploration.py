"""
Running analysis on datasets from directory 'MCD transcripts - tsv 2022-01-17-1'
"""
import os
import shutil
import librosa

oldDir = '../Data/MCD transcripts - tsv 2022-01-17-1'
newDir = '../Data/Transcripts'
audioDir = '../Data/wav'

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


def lengthStatistics():
    SAMPLE_RATE = 16000
    count = 0
    total_len = 0
    for filename in os.listdir(newDir):
        filename_audio = filename.replace('-corrected', '')
        filename_audio = filename_audio.split('.')[0] + '.wav'
        path = os.path.join(audioDir, filename_audio)
        if 'correct' in filename:
            length = librosa.get_duration(librosa.load(path, sr=SAMPLE_RATE)[0], sr=SAMPLE_RATE)
            print('Filename', path, 'Length:', length, 'seconds')
            total_len += length
            count += 1
    print('Total length (minutes): ', total_len / 60, '# files=', count)


if __name__ == "__main__":
    lengthStatistics()
