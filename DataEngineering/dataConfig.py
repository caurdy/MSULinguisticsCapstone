"""
Reconfiguring
"""

import pandas as pd
import os
import librosa


# ToDo
# Account for corrected data transcripts
#

def convertToDataframe(transcript_dir: str = '../Data/Transcripts', audio_dir: str = '../Data/wav') -> pd.DataFrame():
    """
    Convert a directory of transcripts and audios to a single pandas dataframe
    columns are 'index, id (filename), transcript, audio_file_path
    Assumptions:
        The transcript directory has headers seperated by tabs in the order:
            Speaker, Header2, Chunk_start, Chunk_End, Chunk
        The transcript and corresponding audio file have the same filename (only differentiated by filetype)
    :param transcript_dir: directory holding transcripts
    :param audio_dir: directory holding audio files
    :return: Pandas Dataframe
    """

    df = pd.DataFrame()

    # Read transcripts into dataframe
    for filename in os.listdir(transcript_dir):
        data = {'file': filename}
        with open(os.path.join(transcript_dir, filename)) as fp:
            fp.readline()  # skip past header
            for line in fp.readlines():
                transcript = line.split('\t')[-1]  # get the last split assumed to be the text
                data['text'] = transcript

        filename = filename.split('.')[0] + '.wav'
        try:
            with open(os.path.join(audio_dir, filename)) as fp:
                data['audio'] = None
                pass  # do stuff with audio file

        except FileNotFoundError as e:
            pass


if __name__ == '__main__':
    convertToDataframe()
