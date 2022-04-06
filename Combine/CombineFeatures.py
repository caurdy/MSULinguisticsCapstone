import pandas as pd
from DataScience.SpeechToTextHF import Wav2Vec2ASR
from NamedEntityRecognition.ner import ner
import json
from scipy.io import wavfile
from pyannote.audio import pipelines
from pyannote.audio import Model
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import os
import datetime
import time
TOKENIZER = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                             do_normalize=True, return_attention_mask=False)
PROCESSOR = Wav2Vec2Processor(feature_extractor=FEATURE_EXTRACTOR, tokenizer=TOKENIZER)

# test using test.rttm
#
# Steps
# 1. Call diarization function and create rttm file
# 2. Convert rttm file to csv
# 3. Loop through csv file and segment audio file using the start and end time frame
# 4. Call Speech to text for each segmented audio file
# 5. Record the result and save it into a text file with start, end time, and speaker ID


list_audios = []
# example file_info list of dictionaries to fill out
final_info = [
    {"start": "01:003", "end": "02.035",
     "speaker": "1", "transcript": ""},
    {"start": "02:045", "end": "023035",
     "speaker": "2", "transcript": ""}
]

# base_file_name = "transcript.csv"
# transcript_file = open(base_file_name, "w")
# fieldnames = ["start", "end", "speaker", "transcript"]
# writer = csv.DictWriter(transcript_file, fieldnames=fieldnames)
# writer.writeheader()
# for i in range(len(list_audios)):
#     audio = list_audios[1]
#     current_entry = final_info[i]
#     transcript = getTranscript(audio)
#     current_entry["transcript"] = transcript
#
# writer.writerows(final_info)
# transcript_file.close()

# --------------------------------------------------------------------

# Speech Diarization Configuration
sad_scores = Model.from_pretrained("pyannote/segmentation")
emb_scores = Model.from_pretrained("pyannote/embedding")

pipeline = pipelines.SpeakerDiarization(segmentation=sad_scores,
                                        embedding=emb_scores,
                                        embedding_batch_size=32)
initial_params = {
    "onset": 0.810,
    "offset": 0.481,
    "min_duration_on": 0.055,
    "min_duration_off": 0.098,
    "min_activity": 6.073,
    "stitch_threshold": 0.040,
    "clustering": {"method": "average", "threshold": 0.595},
}
pipeline.instantiate(initial_params)


# audio: the directory of the audio file
def combineFeatures(audio, filename="transcript"):

    # Create Diarization file using the audio file provided
    diarization_time1 = time.perf_counter()
    diarization_result = pipeline(audio)
    diarization_time2 = time.perf_counter()
    with open('diarization.rttm', 'w') as file:
        diarization_result.write_rttm(file)

    # Convert rttm file to csv
    dair_csv = pd.read_csv('diarization.rttm', delimiter=' ', header=None)
    dair_csv.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']
    # test_file.to_csv('../Combine/test.csv', index=None)
    os.remove("diarization.rttm")

    # Read Audio file
    rate, data = wavfile.read(audio)
    data = data.astype('float32')

    # Create directory to save transcript and other information
    dict = {}

    # Loop through all the rows in diarization csv
    total_conf = 0
    process_begin_time = time.perf_counter()

    # Initialize ASR model
    asr_model = Wav2Vec2ASR()
    asr_model.loadModel("patrickvonplaten/wav2vec2-base-100h-with-lm")

    for index, row in dair_csv.iterrows():
        start_t = row['Start Time']
        end_t = start_t + row['Duration']
        start_frame = int(rate * start_t)
        end_frame = int(rate * end_t)
        # Sectioned audio data
        section = data[start_frame: end_frame]
        # transcript, confidence = getTranscript(section, model=MODEL, processor=PROCESSOR)
        # total_conf += confidence
        transcript = asr_model.predict_segment(section)
        namedEntity = ner(transcript)
        dict[index] = {"Start (sec.)": str(start_t),
                     "End (sec.)": str(end_t),
                     "Speaker": str(row['ID']),
                     "Transcript": str(transcript),
                     # "Confidence": str(confidence),
                     "Named Entity": str(namedEntity)}
        # sentence = [{"start": str(datetime.timedelta(seconds=round(start_t, 3))),
        #              "end": str(datetime.timedelta(seconds=round(end_t, 3))),
        #              "speaker": str(row['ID']),
        #              "transcript": str(transcript)}]
        # writer.writerows(sentence)
    process_end_time = time.perf_counter()
    # transcript_file.close()
    avg_confidence = total_conf/len(dair_csv)
    with open(f"./Transcriptions/{filename}.json", "w") as jsonFile:
        json.dump(dict, jsonFile)
    return round(diarization_time2 - diarization_time1, 3), round(process_end_time - process_begin_time, 3), avg_confidence

if __name__ == "__main__":
    audioname = "../PyannoteProj/Data/test.wav"
    combineFeatures(audioname)
