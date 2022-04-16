import sys
from Combine.transcription import combineFeatures
from Combine.TimeAlligned_alter import ASRTimeAligner
from DataScience.SpeechToTextHF import Wav2Vec2ASR
from PyannoteProj.TestPipline import SpeakerDiaImplement
from os.path import isfile, isdir
import os
import pandas as pd
import torch.cuda


def run_cuda_setup():
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')
        print('Set device to', torch.cuda.current_device())
        print('Current memory stats\n', torch.cuda.memory_summary(abbreviated=True))
        # for i in range(1, 3):
        #     print(torch.cuda.memory_summary('cuda:' + str(i), abbreviated=True))
    else:
        print("CUDA not available, defaulting to CPU")

def main():

    # create transcript

    # options
    # -t = create transcript.
    # follow by the name of the file to transcribe and name of the two models
    # e.g. "-t" "./Data/Audio/Atest.wav" "facebook/wav2vec2-large-960h-lv60-self" "./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52"
    if sys.argv[1] == "-t":
        audio_file = sys.argv[2]
        audio_name = audio_file.split('/')[-1][:-4]
        if not isfile(audio_file):
            print("Error: File doesn't exist. Please check if the file name is correct, or the file is in the Data folder.")
            return

        # call combine feature
        # combineFeatures(audio_file, audio_name, sys.argv[3], sys.argv[4])
        run_cuda_setup()
        if len(sys.argv) == 3:
            timeAligner = ASRTimeAligner(useCuda=False)
        elif len(sys.argv) == 4:
            timeAligner = ASRTimeAligner(asrModel=sys.argv[3], useCuda=False)
        else:
            timeAligner = ASRTimeAligner(asrModel=sys.argv[3], diarizationModelPath=sys.argv[4], useCuda=False)

        timeAligner.timeAlign(audio_file, audio_name)

        print("transcript is done.")

    # create new models

    # 1. -m: option to enter creating new model option
    # 2. -a/-d: choose either automatic speech recognition model or speaker diarization model to retrain.
    # 3. Directory: the directory of a file/folder that has data to retrain the model
        # 3.1. Directory: if -a is chosen, then need to provide a directory for testing data.
    # 4. Model Name: the model to retrain.
        # 4.1. Model Name: if -a is chosen, then need to provide the new name of the retrained model.
    # 5. Epoch: the user can provide the epoch number. Default is 30.
    # e.g. -m -a ./Data/correctedShort.json ./Data/correctedShort.json facebook/wav2vec2-large-960h-lv60-self facebook_fineTune_test 30
    # e.g. -m -d ./PyannoteProj/data_preparation/TrainingData/Talkbank/ ./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52 30 ./Data/Models/Diarization
    elif sys.argv[1] == "-m":
        if not isdir(sys.argv[3]):
            print("Error: This folder path doesn't exit. Please check if the path is right.")
        # create new asr huggingface model
        elif sys.argv[2] == "-a":
            print("Creating new ASR Huggingface model")
            asr_model = Wav2Vec2ASR()
            asr_model.loadModel(sys.argv[4])
            print(sys.argv[3])
            epo = 30
            if len(sys.argv) > 6:
                epo = int(sys.argv[6])
            asr_model.train(sys.argv[3], sys.argv[4], '../Data/', num_epochs=epo)
            asr_model.saveModel(f"./Data/Models/ASR/{sys.argv[5]}")
            print("New asr model is created")

        # create new diarization model
        elif sys.argv[2] == "-d":
            print("Creating new diarization model.")
            dia_pipeline = SpeakerDiaImplement()
            dia_pipeline.AddPipeline(model_name=f"{sys.argv[4]}/seg_model.ckpt",
                                     parameter_name=f"{sys.argv[4]}/hyper_parameter.json")
            epo = 30
            if len(sys.argv) > 5:
                epo = int(sys.argv[5])

            if len(sys.argv) > 6:
                dia_pipeline.TrainData(sys.argv[3], sys.argv[6], epoch_num=epo)
                return

            dia_pipeline.TrainData(sys.argv[3], epoch_num=epo)

        else:
            print("Error: Please choose the type of model you want to retrain. Choose from -a, -d.")
    else:
        print("Error: The first input needs to be either -t or -m. Try it again.")


if __name__ == "__main__":
    main()
