import sys
from Combine.generateTranscription import ASRTimeAligner
from DataScience.SpeechToTextHF import Wav2Vec2ASR
from PyannoteProj.TestPipline import SpeakerDiaImplement
from os.path import isfile, isdir
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

    elif sys.argv[1] == "-m":
        # create new asr huggingface model
        if sys.argv[2] == "-a":
            if not isfile(sys.argv[3]) or not isfile(sys.argv[4]):
                print("Error: This folder path doesn't exit. Please check if the path is right.")
                return
            print("Creating new ASR Huggingface model")
            asr_model = Wav2Vec2ASR()
            asr_model.loadModel(sys.argv[5])
            epo = 30
            if len(sys.argv) > 7:
                epo = int(sys.argv[7])
            asr_model.train(sys.argv[3], sys.argv[4], '../Data/', num_epochs=epo)
            asr_model.saveModel(f"./Data/Models/ASR/{sys.argv[6]}")
            print("New asr model is created")

        # create new diarization model
        elif sys.argv[2] == "-d":
            print("Creating new diarization model.")
            dia_pipeline = SpeakerDiaImplement()
            dia_pipeline.AddPipeline(model_name=f"{sys.argv[3]}/seg_model.ckpt",
                                     parameter_name=f"{sys.argv[3]}/hyper_parameter.json")
            epo = 30
            if len(sys.argv) > 4:
                epo = int(sys.argv[4])

            if len(sys.argv) > 5:
                dia_pipeline.TrainData(save_folder=sys.argv[5], epoch_num=epo)
            else:
                dia_pipeline.TrainData(epoch_num=epo)

        else:
            print("Error: Please choose the type of model you want to retrain. Choose from -a, -d.")
    else:
        print("Error: The first input needs to be either -t or -m. Try it again.")


if __name__ == "__main__":
    main()
