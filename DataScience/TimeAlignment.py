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
# Setting path for local imports OS/IDE-agnostic
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import time
import json
import librosa
import pandas as pd
import torch.cuda
from fastpunct import FastPunct
from rpunct import RestorePuncts

from DataScience.SpeechToTextHF import Wav2Vec2ASR
from PyannoteProj.TestPipline import SpeakerDiaImplement
from NamedEntityRecognition.ner import ner


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
        :param punctuationModel: Punctuation Model, for now THIS SHOULD NOT BE USED
        :param nerModel: NER model, for now, THIS SHOULD NOT BE USED
        :param use_cuda: Whether to use cuda for inference on all models
                            (diarization will use gpu if it exists, have to edit source to change)
        """
        self.useCuda = useCuda
        if torch.cuda.is_available() and self.useCuda:
            self.useCuda = True
        else:
            self.useCuda = False
        self.run_cuda_setup()

        self.asrModel = Wav2Vec2ASR(useCuda)
        self.asrModel.loadModel(asrModel)
        self.diarizationModel = SpeakerDiaImplement()
        # self.diarizationModel.AddPipeline(model_name="{}/seg_model.ckpt".format(diarizationModelPath),
        #                                   parameter_name="{}/hyper_parameter.json".format(diarizationModelPath))
        if punctuationModel:
            self.punctuationModel = punctuationModel
        else:
            if self.useCuda:
                self.punctuationModel = RestorePuncts()
            else:
                self.punctuationModel = FastPunct()

        if nerModel:
            self.nerModel = nerModel
        else:
            self.nerModel = ner

        self.transcripts = []

    def timeAlign(self, audioPath: str, outputDir: str = '.', writeOutput: bool = True):
        """
        Writes a json output of time aligned output to outputDir
        Name of output is {audioPath}.json
        :param outputDir: output directory
        :param audioPath: str path to
        :return: List of json objects, diarization runtime, ASR runtime, Avg. ASR confidence
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

        if writeOutput:
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
        if self.useCuda:
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

    def run_cuda_setup(self):
        if self.useCuda:
            import tensorflow
            torch.cuda.device('cuda')
            print('Set device to', torch.cuda.current_device())
            # Torch can only run on 8 gpus max, in parallel
            if torch.cuda.device_count() > 8:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

            physical_devices = tensorflow.config.list_physical_devices('GPU')
            try:
                tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                physical_devices = tensorflow.config.PhysicalDevice("CPU")
        else:
            if not torch.cuda.is_available():
                print("GPU not available, defaulting to CPU")
            else:
                print("CPU Execution Selected")


if __name__ == '__main__':
    """
    Parameters: useCuda, dir
    """
    # input processing
    if len(sys.argv) < 3:
        raise UserWarning("Not enough parameters entered. Enter an use_cuda and directory parameter. "
                          "E.x. 'python TimeAlignment.py true demo'")

    if sys.argv[1].lower() == 'true':
        useCuda = True
    elif sys.argv[1].lower() == 'false':
        useCuda = False
    else:
        raise UserWarning("Invalid use_cuda parameters, options include [false, true]")

    if sys.argv[2].lower() == 'demo':
        audioDir = '../Data/demo'
    elif os.path.exists(sys.argv[2]) and os.path.isdir(sys.argv[2]):
        audioDir = sys.argv[2]
    else:
        raise UserWarning("Invalid path input, please provide a legal path to a directory")

    print('Arguments', str(sys.argv))

    # make a directory for the transcripts
    try:
        transcriptDir = os.path.join(audioDir, "Transcriptions")
        os.mkdir(transcriptDir)
    except FileExistsError as e:
        raise FileExistsError("Delete the old Transcriptions folder in ", audioDir, ".")

    # transcribe the files in the directory
    timeAligner = ASRTimeAligner(useCuda=useCuda)
    for filename in os.listdir(audioDir):
        if '.wav' in filename:
            audioPath = os.path.join(audioDir, filename)
            transcriptRough, dt, tt, avg_confidence = timeAligner.timeAlign(audioPath, writeOutput=False)
            transcriptPunctuated, pt, nt = timeAligner.getEntitiesLastTranscript()
            transcript_path = os.path.join(transcriptDir, filename.replace('.wav', '.json'))
            with open(transcript_path, "w") as jsonFile:
                json.dump(transcriptPunctuated, jsonFile)
                print('Processed', transcript_path)

    dtt, ttt, t1t, t2t = 0, 0, 0, 0
    # with open('../Data/minuteFiles.txt', 'r') as fp:
    #     for file in fp.readlines():
    #         file = os.path.join('../Data/wav', file.strip())
    #         print('Processing', file)
    #         transcriptions, dt, tt, avg_confidence = timeAligner.timeAlign(file)
    #         print(dt, tt, '\n', transcriptions)
    #         transcript, t1, t2 = timeAligner.getEntitiesLastTranscript()
    #         print(t1, t2, '\n', transcript)
    #         dtt += dt
    #         ttt += tt
    #         t1t = t1
    #         t2t = t2
    # print('Average runtimes on files btwn 55 and 65 seconds')
    # print('Average diarization runtime:', dtt / 24)
    # print('Average asr runtime:', ttt / 24)
    # print('Average punc time:', t1t / 24)
    # print('Average ner time:', t2t / 24)
