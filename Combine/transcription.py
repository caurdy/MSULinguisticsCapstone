import os
import sys
import inspect

from DataScience.SpeechToTextHF import Wav2Vec2ASR
from NamedEntityRecognition.ner import ner
from PyannoteProj.TestPipline import SpeakerDiaImplement
from fastpunct import FastPunct
from rpunct import RestorePuncts

import json
import librosa
import time
import pandas as pd
import torch.cuda

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # comment this out and pass use_cuda=True to enable gpus
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def run_cuda_setup():
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')
        print('Set device to', torch.cuda.current_device())
        print('Current memory stats\n', torch.cuda.memory_summary(abbreviated=True))
        # for i in range(1, 3):
        #     print(torch.cuda.memory_summary('cuda:' + str(i), abbreviated=True))
    else:
        print("CUDA not available, defaulting to CPU")


# audio: the directory of the audio file
def combineFeatures(audio_path: str, transcript_name: str, asr_model: str, dia_model: str):
    # Initialize Diarization Class Object and set model.
    dia_pipeline = SpeakerDiaImplement()
    dia_pipeline.AddPipeline(model_name=f"./PyannoteProj/data_preparation/saved_model/{dia_model}/seg_model.ckpt",
                             parameter_name=f"./PyannoteProj/data_preparation/saved_model/{dia_model}/hyper_parameter.json")
    # Create Diarization file using the audio_path file provided
    diarization_time1 = time.perf_counter()
    diarization_result = dia_pipeline.Diarization(audio_path)
    diarization_time2 = time.perf_counter()

    # Convert rttm file to csv
    dair_csv = pd.read_csv(diarization_result, delimiter=' ', header=None)
    dair_csv.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']
    os.remove(f'./Data/Audio/{transcript_name}.rttm')

    # Read Audio file
    data, sample_rate = librosa.core.load(audio_path, sr=16000)
    transcriptions = []
    asr_m = Wav2Vec2ASR()
    asr_m.loadModel(asr_model)

    # Loop through all the rows in diarization file which will give us start/end times for audio section to transcribe
    total_conf = 0
    process_begin_time = time.perf_counter()
    for index, row in dair_csv.iterrows():
        start_t = row['Start Time']
        end_t = start_t + row['Duration']
        start_frame = int(sample_rate * start_t)
        end_frame = int(sample_rate * end_t)
        section = data[start_frame: end_frame]
        transcript, _ = asr_m.predict(audioArray=section)
        # total_conf += confidence
        # namedEntity = ner(transcript)
        transcriptions.append({"Start (sec.)": str(start_t),
                               "End (sec.)": str(end_t),
                               "Speaker": str(row['ID']),
                               "Transcript": str(transcript)})
                               # "Confidence": str(confidence),})
                               # "Named Entity": str(namedEntity)})

    process_end_time = time.perf_counter()
    with open(f"./Data/Transcripts/{transcript_name}_transcript.json", "w") as jsonFile:
        json.dump(transcriptions, jsonFile)

    diarization_time = round(diarization_time2 - diarization_time1, 3)
    transcription_time = round(process_end_time - process_begin_time, 3)
    # avg_confidence = total_conf / len(dair_csv)

    # return transcriptions, diarization_time, transcription_time, avg_confidence


if __name__ == "__main__":
    audioname = "../PyannoteProj/Data/test.wav"
    combineFeatures(audioname)
