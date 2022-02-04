"""
Running analysis on transcript file from directory 'MCD transcripts - tsv 2022-01-17-1'
"""
import os
import shutil


if __name__ == "__main__":
    oldDir = '../Data/MCD transcripts - tsv 2022-01-17-1'
    newDir = '../Data/Transcripts/'
    for filename in os.listdir(oldDir):
        if filename[0] != '.':
            shutil.copy(os.path.join(oldDir, filename), newDir)

