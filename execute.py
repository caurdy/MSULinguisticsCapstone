import sys
from Combine.transcription import combineFeatures
from DataScience.SpeechToTextHF import Wav2Vec2ASR
from PyannoteProj.TestPipline import SpeakerDiaImplement
from os.path import isfile
import os
import pandas as pd


def main():
    # options
    # -t = create transcript.
    # follow by the name of the file to transcribe and name of the two models
    # -n for nemo model
    # -h for huggingface model
    # e.g. -t Atest.wav -h asr1 dia1

    # -m = create new models.
    # -h for HF model, -d for diarization model, -n for nemo model
    # follow by the directory of datas to retrain the model,
    # model you want to retrain,
    # and the name of the new model.
    # e.g. -m -h ./Data/forHF/ old_hf new_hf
    # e.g. -m -d ./Data/forDia/ old_dia new_dia
    # e.g. -m -n ./Data/forNemo/ old_nemo new_nemo

    # create transcript
    if sys.argv[1] == "-t":
        audio_file = "./PyannoteProj/Data/" + sys.argv[2]
        if not isfile(audio_file):
            print("Error: File doesn't exist. Please check if the file name is correct, or the \
                        file is in the Data folder.")
            return

        print("transcript option is selected")
        if sys.argv[3] != "-h" and sys.argv[3] != "-n":
            print("Error: Need to choose either -h or -n for the ASR model.")
            return

        # call combine feature
        combineFeatures(audio_file, sys.argv[2] + "_transcription", sys.argv[3], \
                        sys.argv[4], sys.argv[5])
        print("transcript is done.")

    # create new models
    elif sys.argv[1] == "-m":
        if not isfile(sys.argv[3]):
            print("Error: This folder path doesn't exit. Please check if the path is right.")
        # create new asr huggingface model
        elif sys.argv[2] == "-h":
            print("Creating new ASR Huggingface model")
            asr_model = Wav2Vec2ASR()
            asr_model.loadModel(sys.argv[4])
            print(sys.argv[3])
            asr_model.train("./Data/corrected.json", "./Data/corrected.json", '../Data/')
            asr_model.saveModel(f"./Data/Models/HF/{sys.argv[5]}.json")
            print("New asr model is created")

        # create new asr nemo model.
        elif sys.argv[2] == "-n":
            print("Creating new ASR Nemo model.")
            # nemo train
            # manifest test (dataset), config file(YML file)
        # create new diarization model
        elif sys.argv[2] == "-d":
            print("Creating new diarization model.")
            dia_pipeline = SpeakerDiaImplement()
            dia_pipeline.AddPipeline(
                model_name=f"./PyannoteProj/data_preparation/saved_model/{sys.argv[4]}/seg_model.ckpt",
                parameter_name=f"./PyannoteProj/data_preparation/saved_model/{sys.argv[4]}/hyper_parameter.json")
            dia_pipeline.TrainData("./PyannoteProj/data_preparation/TrainingData/Talkbank", epoch_num=1)

        else:
            print("Error: Please choose the type of model you want to retrain. \
            Choose from -h, -n, -d.")
    else:
        print("Error: The first input needs to be either -t or -m. Try it again.")


if __name__ == "__main__":
    import sys

    # print(sys.argv)
    main()
