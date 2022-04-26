#!/usr/bin/env python
import sys

from SpeechToTextHF import Wav2Vec2ASR

if __name__ == "__main__":
    """
    Arguments
        directory of models
        test set json
    """

    if len(sys.argv) < 3:
        raise Exception('Too few arguments supplied.', sys.argv, '\n Enter at least a directory path and a test set.')


