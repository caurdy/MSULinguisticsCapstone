#!/usr/bin/env python
import os
import sys

from SpeechToTextHF import Wav2Vec2ASR

if __name__ == "__main__":
    """
    Arguments
        directory of models
        test set json
        
        e.x. python evaluateASR.py ./Data/Models/ ../Data/Wav2vec2trainUnder10000KB.json
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
        path = os.path.join(modelDir, dir)
        if os.path.isdir(path):
            try:
                model.loadModel(path)
                wer = model.evaluate(testSet)
                print('WER for ', path, ':', round(wer * 100, 2), '%')
            except RuntimeError as e:
                raise Warning('Invalid model directory detected', str(e))


