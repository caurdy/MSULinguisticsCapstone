#!/usr/bin/env python

import os
import sys
from SpeechToTextHF import Wav2Vec2ASR

os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

if __name__ == "__main__":
    """
    Parameters
    This file defaults to use_cuda=True, but wont use it if its unavailable

    Train
        model_path
        train_set json
        test_set json
        modelDir
        
    NOTE: If you have more than 8 gpus available uncomment line 8
    """
    if len(sys.argv) < 5:
        raise Exception('Too few arguments supplied.', sys.argv,
                        '\n Please enter at least a model path, train set .json file,'
                        ' test set .json file and directory to save the model & checkpoints in!')
    asr_model = Wav2Vec2ASR(use_cuda=True)
    model_path = sys.argv[1]
    asr_model.loadModel(model_path)

    train_set = sys.argv[2]
    test_set = sys.argv[3]
    if '.json' not in train_set:
        raise Exception('Train set is not a json file')
    elif '.json' not in test_set:
        raise Exception('Test set is not a json file')

    modelDir = sys.argv[4]
    asr_model.train(train_set, test_set, modelDir, 10)
    asr_model.saveModel(modelDir)
