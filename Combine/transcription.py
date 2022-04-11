import pandas as pd
from DataScience.SpeechToTextHF import Wav2Vec2ASR
from NamedEntityRecognition.ner import ner
from PyannoteProj.TestPipline import SpeakerDiaImplement
# import nemo.collections.asr as nemo_asr
import json
import librosa
from pyannote.audio import pipelines
from pyannote.audio import Model
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import os
import time

# TOKENIZER = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
# MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
# FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
#                                              do_normalize=True, return_attention_mask=False)
# PROCESSOR = Wav2Vec2Processor(feature_extractor=FEATURE_EXTRACTOR, tokenizer=TOKENIZER)

# test using test.rttm
#
# Steps
# 1. Call diarization function and create rttm file
# 2. Convert rttm file to csv
# 3. Loop through csv file and segment audio file using the start and end time frame
# 4. Call Speech to text for each segmented audio file
# 5. Record the result and save it into a text file with start, end time, and speaker ID

# example file_info list of dictionaries to fill out
final_info = [
    {"start": "01:003", "end": "02.035",
     "speaker": "1", "transcript": ""},
    {"start": "02:045", "end": "023035",
     "speaker": "2", "transcript": ""}
]


# audio: the directory of the audio file
def combineFeatures(audio_path: str, transcript_name: str, model: str, asr_model: str, dia_model: str):
    # Initialize Diarization Class Object and set model.
    dia_pipeline = SpeakerDiaImplement()
    dia_pipeline.AddPipeline(model_name=f"./PyannoteProj/data_preparation/saved_model/{dia_model}/seg_model.ckpt",
                             parameter_name=f"./PyannoteProj/data_preparation/saved_model/{dia_model}/hyper_parameter.json")
    # Create Diarization file using the audio_path file provided
    diarization_time1 = time.perf_counter()
    diarization_result = dia_pipeline.Diarization(audio_path)
    diarization_time2 = time.perf_counter()

    # Convert rttm file to csv
    only_name = audio_path.split('.')[0]
    dair_csv = pd.read_csv(f'./PyannoteProj/OutputSet/{only_name}.rttm', delimiter=' ', header=None)
    dair_csv.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']
    os.remove(f'./PyannoteProj/OutputSet/{only_name}.rttm')

    # Read Audio file
    data, sample_rate = librosa.core.load(audio_path, sr=16000)
    transcriptions = []
    asr_model = None
    if model == "-n":
        print("nemo")
        return
        # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)
    elif model == "-h":
        asr_model = Wav2Vec2ASR()
        asr_model.loadModel(asr_model)

    # Loop through all the rows in diarization file which will give us start/end times for audio section to transcribe
    total_conf = 0
    process_begin_time = time.perf_counter()
    for index, row in dair_csv.iterrows():
        start_t = row['Start Time']
        end_t = start_t + row['Duration']
        start_frame = int(sample_rate * start_t)
        end_frame = int(sample_rate * end_t)
        section = data[start_frame: end_frame]
        transcript = ""
        if model == "-n":
            return
            # save audio file temporally and delete it after transcript is written.
            # librosa.output.write_wav('./Data/audio_for_nemo_temp.wav', data, sample_rate)
            # transcript = asr_model.transcribe(paths2audio_files='./Data/audio_for_nemo_temp.wav')[0]
            # os.remove('./Data/audio_for_nemo_temp.wav')
        elif model == "-h":
            transcript = asr_model.predict_segment(data)
        # transcript, confidence = getTranscript(section, model=MODEL, processor=PROCESSOR)
        # total_conf += confidence
        #namedEntity = ner(transcript)
        transcriptions.append({"Start (sec.)": str(start_t),
                               "End (sec.)": str(end_t),
                               "Speaker": str(row['ID']),
                               "Transcript": str(transcript)})
                               # "Confidence": str(confidence),})
                               #"Named Entity": str(namedEntity)})

    process_end_time = time.perf_counter()
    with open(f"./Transcriptions/{transcript_name}.json", "w") as jsonFile:
        json.dump(transcriptions, jsonFile)

    diarization_time = round(diarization_time2 - diarization_time1, 3)
    transcription_time = round(process_end_time - process_begin_time, 3)
    # avg_confidence = total_conf / len(dair_csv)

    # return transcriptions, diarization_time, transcription_time, avg_confidence


if __name__ == "__main__":
    audioname = "../PyannoteProj/Data/test.wav"
    combineFeatures(audioname)
