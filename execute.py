import sys
from Combine.CombineFeatures import combineFeatures
from DataScience.SpeechToTextHF import Wav2Vec2ASR

def main():
# options
# -t = create transcript.
# follow by the name of the file to transcribe and name of the two models
# e.g. -t Atest.wav asr1 dia1

# -m = create new models.
# -a for asr model, -d for diarization model
# follow by the directory of datas to retrain the model,
# model you want to retrain,
# and the name of the new model.
# e.g. -m -a ./Data/forASR/ old_asr new_asr
# e.g. -m -d ./Data/forDia/ old_dia new_dia

    # create transcript
    if sys.argv[1] == "-t":
        audio_file = "./PyannoteProj/Data/" + sys.argv[2]
        print("transcript option is selected")
        # call combine feature
        combineFeatures(audio_file, sys.argv[2]+"_transcription")
        print("transcript is done.")
    # create new models
    elif sys.argv[1] == "-m":
        # create new asr model
        if sys.argv[2] == "-a":
            print("Creating new asr model")
            asr_model = Wav2Vec2ASR()
            asr_model.loadModel(sys.argv[4])
            asr_model.train(f"{sys.argv[5]}.json", '../Data/')
            asr_model.saveModel(f"./Data/Models/HF/{sys.argv[5]}.json")
            print("New asr model is created")
        # create new diarization model
        elif sys.argv[2] == "-m":
            print("Creating new diarization model")
    else:
        print("Error")


if __name__ == "__main__":
    import sys
    # print(sys.argv)
    main()
