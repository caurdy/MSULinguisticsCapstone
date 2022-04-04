import sys
from Combine.CombineFeatures import combineFeatures

def main():
# options
# -t = create transcript.
# follow by the name of the file to transcribe and name of the two models
# e.g. -t Atest.wav asr1 dia1

# -m = create new models.
# -a for asr model, -d for diarization model
# follow by the directory of datas to retrain the model.
# e.g. -m -a ./Data/forASR/
# e.g. -m -d ./Data/forDia

    # create transcript
    if sys.argv[1] == "-t":
        audio_file = "./PyannoteProj/Data/" + sys.argv[2]
        print("transcript option is selected")
        # call combine feature
        combineFeatures(audio_file, sys.argv[2]+"_transcription")
    # create new models
    elif sys.argv[1] == "-m":
        # create new asr model
        if sys.argv[2] == "-a":
            print("Creating new asr model")
        # create new diarization model
        elif sys.argv[2] == "-m":
            print("Creating new diarization model")


if __name__ == "__main__":
    main()
