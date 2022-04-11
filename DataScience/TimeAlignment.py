"""
Culmination of all the bullcrap, lets get it
"""
import os
import time
import json
import librosa
import pandas as pd
from DataScience.SpeechToTextHF import Wav2Vec2ASR
from PyannoteProj.TestPipline import SpeakerDiaImplement
from NamedEntityRecognition.ner import ner
from fastpunct import FastPunct


class ASRTimeAligner:
    """
    Creates time aligned transcripts from audio file with ability for punctuation restoration and NER tagging
    """
    __slots__ = ['asrModel', 'diarizationModel', 'punctuationModel', 'nerModel', 'transcripts']

    def __init__(self, asrModel="facebook/wav2vec2-large-960h-lv60-self",
                 diarizationModelPath="../PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52",
                 punctuationModel=None,
                 nerModel=None):
        """
        :param asrModel: Wav2Vec2ASR model path
        :param diarizationModelPath: Directory name of saved diarization model
        :param punctuationModel: Punctuation Model
        :param nerModel: NER model
        """
        self.asrModel = Wav2Vec2ASR()
        self.asrModel.loadModel(asrModel)
        self.diarizationModel = SpeakerDiaImplement()
        self.diarizationModel.AddPipeline(model_name="{}/seg_model.ckpt".format(diarizationModelPath),
                                          parameter_name="{}/hyper_parameter.json".format(diarizationModelPath))
        if punctuationModel:
            self.punctuationModel = punctuationModel
        else:
            self.punctuationModel = FastPunct()

        if nerModel:
            self.nerModel = nerModel
        else:
            self.nerModel = ner

        self.transcripts = []

    def timeAlign(self, audioPath: str, outputDir: str = '.'):
        """
        Writes a json output of time aligned output to outputDir
        Name of output is {audioPath}.json
        :param outputDir: output directory
        :param audioPath: str path to
        :return:
        """
        diarization_time1 = time.perf_counter()
        rttm_path = self.diarizationModel.Diarization(audioPath)
        diarization_time2 = time.perf_counter()

        # Convert rttm file to csv
        dair_csv = pd.read_csv(rttm_path, delimiter=' ', header=None)
        dair_csv.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']

        # Read Audio file
        data, sample_rate = librosa.core.load(audioPath, sr=16000)
        transcriptions = []

        # Loop through all the rows in diarization file which will give us
        # start/end times for audio section to transcribe
        total_conf = 0
        process_begin_time = time.perf_counter()
        for index, row in dair_csv.iterrows():
            start_t = row['Start Time']
            end_t = start_t + row['Duration']
            start_frame = int(sample_rate * start_t)
            end_frame = int(sample_rate * end_t)
            section = data[start_frame: end_frame]
            transcript, confidence = self.asrModel.predict(audioArray=section)
            total_conf += confidence
            transcriptions.append({"Start (sec.)": str(start_t),
                                   "End (sec.)": str(end_t),
                                   "Speaker": str(row['ID']),
                                   "Transcript": str(transcript),
                                   "Confidence": str(confidence), })

        process_end_time = time.perf_counter()
        transcript_path = audioPath.replace('.wav', '.json')
        with open(transcript_path, "w") as jsonFile:
            json.dump(transcriptions, jsonFile)

        diarization_time = round(diarization_time2 - diarization_time1, 3)
        transcription_time = round(process_end_time - process_begin_time, 3)
        avg_confidence = total_conf / len(dair_csv)
        self.transcripts.append(transcriptions)
        return transcriptions, diarization_time, transcription_time, avg_confidence

    def getEntitiesLastTranscript(self):
        transcript = self.transcripts[-1]
        punc_time = 0
        ner_time = 0
        for entry in transcript:
            sentence = entry["Transcript"].lower()
            time1 = time.perf_counter()
            punc_restored = self.punctuationModel.punct(sentence)
            time2 = time.perf_counter()
            namedEntity = ner(punc_restored)
            time3 = time.perf_counter()
            entry["Transcript"] = punc_restored
            entry["Named Entities"] = str(namedEntity)
            punc_time += time2 - time1
            ner_time += time3 - time2

        self.transcripts[-1] = transcript
        return transcript, punc_time, ner_time


if __name__ == '__main__':
    file = '../assets/short_audio3.wav'
    timeAligner = ASRTimeAligner()
    # timeAligner.timeAlign(file)
    timeAligner.getEntitiesLastTranscript()
