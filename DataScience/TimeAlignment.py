"""
Combines ASR, Diarization, PuncRestoration and NER into one class
If rpunct is throwing use_cuda issues (since you are running on cpu)
change line 17 in punctuate.py to
    self.model = NERModel("bert", "felflare/bert-restore-punctuation", labels=self.valid_labels, use_cuda=False,
                            args={"silent": True, "max_seq_length": 512})
"""
import os
import sys
import inspect

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # comment this out and pass use_cuda=True to enable gpus
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import time
import json
import librosa
import pandas as pd
import torch.cuda

from DataScience.SpeechToTextHF import Wav2Vec2ASR
from PyannoteProj.TestPipline import SpeakerDiaImplement
from NamedEntityRecognition.ner import ner
from fastpunct import FastPunct
from rpunct import RestorePuncts


def run_cuda_setup():
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')
        print('Set device to', torch.cuda.current_device())
        print('Current memory stats\n', torch.cuda.memory_summary(abbreviated=True))
        # for i in range(1, 3):
        #     print(torch.cuda.memory_summary('cuda:' + str(i), abbreviated=True))
    else:
        print("CUDA not available, defaulting to CPU")


class ASRTimeAligner:
    """
    Creates time aligned transcripts from audio file with ability for punctuation restoration and NER tagging
    """
    __slots__ = ['asrModel', 'diarizationModel', 'punctuationModel', 'nerModel', 'transcripts', 'useCuda']

    def __init__(self, asrModel="facebook/wav2vec2-large-960h-lv60-self",
                 diarizationModelPath="../PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52",
                 punctuationModel=None,
                 nerModel=None,
                 useCuda=True):
        """
        :param asrModel: Wav2Vec2ASR model path
        :param diarizationModelPath: Directory name of saved diarization model
        :param punctuationModel: Punctuation Model
        :param nerModel: NER model
        :param use_cuda: Whether to use cuda for inference on all models
                            (diarization will use gpu if it exists, have to edit source to change)
        """
        self.useCuda = useCuda
        self.asrModel = Wav2Vec2ASR(useCuda)
        self.asrModel.loadModel(asrModel)
        self.diarizationModel = SpeakerDiaImplement()
        self.diarizationModel.AddPipeline(model_name="{}/seg_model.ckpt".format(diarizationModelPath),
                                          parameter_name="{}/hyper_parameter.json".format(diarizationModelPath))
        if punctuationModel:
            self.punctuationModel = punctuationModel
        else:
            if torch.cuda.is_available() and useCuda:
                self.punctuationModel = RestorePuncts()
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
        :return: List of json objects, diarization runtime, ASR runtime, Avg. ASR confidence
        """
        # print('Before diarization\n', torch.cuda.memory_summary(abbreviated=True))
        # print(torch.cuda.memory_summary('cuda:1', abbreviated=True))
        diarization_time1 = time.perf_counter()
        rttm_path = self.diarizationModel.Diarization(audioPath)
        diarization_time2 = time.perf_counter()
        # print('After diarization\n', torch.cuda.memory_summary(abbreviated=True))
        # print(torch.cuda.memory_summary('cuda:1', abbreviated=True))

        # Convert rttm file to csv
        dair_csv = pd.read_csv(rttm_path, delimiter=' ', header=None)
        # print('Diarization DF\n', dair_csv)
        dair_csv.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']

        # Read Audio file
        data, sample_rate = librosa.core.load(audioPath, sr=16000)
        transcriptions = []

        # print('Before time alignment\n', torch.cuda.memory_summary(abbreviated=True))
        # print(torch.cuda.memory_summary('cuda:1', abbreviated=True))
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

        # print('After time alignment\n', torch.cuda.memory_summary(abbreviated=True))
        # print(torch.cuda.memory_summary('cuda:1', abbreviated=True))
        transcript_path = audioPath.replace('.wav', '.json')
        with open(transcript_path, "w") as jsonFile:
            json.dump(transcriptions, jsonFile)

        diarization_time = round(diarization_time2 - diarization_time1, 3)
        transcription_time = round(process_end_time - process_begin_time, 3)
        avg_confidence = total_conf / len(dair_csv)
        self.transcripts.append(transcriptions)
        return transcriptions, diarization_time, transcription_time, avg_confidence

    def getEntitiesLastTranscript(self):
        """
        Punctuate and run NER on last transcript, return and store results in transcripts
        """
        transcript = self.transcripts[-1]

        punc_time = 0
        ner_time = 0
        for entry in transcript:
            if entry['Transcript'] == "":
                transcript.remove(entry)
                continue
            sentence = entry["Transcript"].lower()
            time1 = time.perf_counter()
            punc_restored = self.punctuate(sentence)
            time2 = time.perf_counter()
            namedEntity = ner(punc_restored)
            time3 = time.perf_counter()
            entry["Transcript"] = punc_restored
            entry["Named Entities"] = str(namedEntity)
            punc_time += time2 - time1
            ner_time += time3 - time2

        # with open('../assets/AbbottCostelloWhosonFirstCorrect.json', "w") as jsonFile:
        #     json.dump(transcript, jsonFile)
        self.transcripts[-1] = transcript
        return transcript, punc_time, ner_time

    def punctuate(self, text):
        """
        Wrapper for punctuation model inference since their APIS differ
        """
        if torch.cuda.is_available() and self.useCuda:
            punc_restored = self.punctuationModel.punctuate(text, lang="en")
        else:
            word_list = text.split()
            restored = []
            # punctuate in 50 word segments, since FastPunct breaks on long texts
            if len(word_list) > 50:
                i = 0
                while i <= len(word_list):
                    if i + 50 > len(word_list):
                        restored.append(self.punctuationModel.punct(" ".join(word_list[i:])))
                    else:
                        restored.append(self.punctuationModel.punct(" ".join(word_list[i:i + 50])))
                    i += 50
                punc_restored = " ".join(restored)
            else:
                punc_restored = self.punctuationModel.punct(text)

        return punc_restored


if __name__ == '__main__':
    run_cuda_setup()
    file = '../assets/0short_audio.wav'
    timeAligner = ASRTimeAligner(useCuda=True)
    # print('After intialization\n', torch.cuda.memory_summary(abbreviated=True))
    # print(torch.cuda.memory_summary('cuda:1', abbreviated=True))
    dtt, ttt, t1t, t2t = 0, 0, 0, 0

    with open('../Data/minuteFiles.txt', 'r') as fp:
        for file in fp.readlines():
            file = os.path.join('../Data/wav', file.strip())
            print('Processing', file)
            transcriptions, dt, tt, avg_confidence = timeAligner.timeAlign(file)
            print(dt, tt, '\n', transcriptions)
            transcript, t1, t2 = timeAligner.getEntitiesLastTranscript()
            print(t1, t2, '\n', transcript)
            dtt += dt
            ttt += tt
            t1t = t1
            t2t = t2
    print('Average runtimes on files btwn 55 and 65 seconds')
    print('Average diarization runtime:', dtt/24)
    print('Average asr runtime:', ttt / 24)
    print('Average punc time:', t1t/24)
    print('Average ner time:', t2t/24)
    # for i in range(3):
    # print(torch.cuda.memory_summary('cuda:'+str(i), abbreviated=True))
    # print(torch.cuda.memory_summary(abbreviated=True))
