#!/usr/bin/env python
import os
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

    modelDir = sys.argv[1]
    testSet = sys.argv[2]

    if not os.path.isdir(modelDir):
        raise Exception('Model Directory supplied is invalid: ', modelDir)

    if '.json' not in testSet:
        raise Exception('Evaluation Set is not a json file', testSet)

    model = Wav2Vec2ASR(use_cuda=True)
    for dir in os.listdir(modelDir):
        if os.path.isdir(dir):
            model.loadModel(dir)
            model.evaluate(testSet)


