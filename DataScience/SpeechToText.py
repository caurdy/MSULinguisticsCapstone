"""
ASR predictions using non fine-tuned Wav2Vec2 model
"""
from typing import Dict, Tuple

import soundfile as sf
import librosa
import torch
from datasets import load_metric
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer, Wav2Vec2Model, Wav2Vec2Processor
from DataEngineering.CleanTranscript import cleanFile
from transformers import AutoProcessor, AutoModelForCTC
import textract
from rpunct import RestorePuncts
import pandas as pd
import nemo.collections.asr as nemo_asr
import numpy as np

ASR_MODEL = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)
RPUNCT = RestorePuncts()    # Had to manually set use_cuda to false in source code


def getTranscript(audio: str):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('../Data/model_corrected_lessthan2_5KB/')
    # tokenizer = Wav2Vec2CTCTokenizer('../Data/vocab.json', unk_token='[UNK]',
    #                                 pad_token='[PAD]', word_delimiter_token='|')

    model = Wav2Vec2ForCTC.from_pretrained("../Data/model_corrected_lessthan2_5KB/")
    input_audio, _ = librosa.load(audio, sr=16000)

    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print(transcription)
    return transcription


def getTranscript2(audio: str):
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    input_audio, _ = librosa.load(audio, sr=16000)
    # tokenize
    input_values = processor(input_audio, return_tensors="pt", padding="longest").input_values  # Batch size 1
    # retrieve logits
    logits = model(input_values).logits
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription


def testWER(transcription: str, label_file: str):
    target = cleanFile(label_file)
    target = target.upper()
    wer = jiwer.wer(target, transcription)
    return target, wer


# softmax implementation in NumPy
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


def getTimestamps() -> Tuple[float, float, str]:
    pass


def produceTimeAlignedTranscript(audio_file: str, speaker_file: str) -> None:
    """
    Takes in an audio file path and a .rttm file path (the product of speaker diarization) to produce
    a series of json objects representing sentences with attributes
        (start, end, content, speaker, wordcount, confidence)
    :param audio_file: Path to audio file
    :param speaker_file: Path to speaker diarization file
    :return: None
    """

    # load audio signal with librosa
    signal, sample_rate = librosa.load(audio_file, sr=16000)
    transcript = ASR_MODEL.transcribe(paths2audio_files=[audio_file])[0] # get txt transcript
    transcript = RPUNCT.punctuate(transcript)
    logits = ASR_MODEL.transcribe([audio_file], logprobs=True)[0] # get vector of probabilities for word predictions
    probs = softmax(logits)
    # 20ms is duration of a timestep at output of the model
    time_stride = 0.02
    # get model's alphabet
    labels = list(ASR_MODEL.decoder.vocabulary) + ['blank']
    labels[0] = 'space'
    # get timestamps for space symbols
    spaces = []
    state = ''
    idx_state = 0

    if np.argmax(probs[0]) == 0:
        state = 'space'
    for idx in range(1, probs.shape[0]):
        current_char_idx = np.argmax(probs[idx])
        if state == 'space' and current_char_idx != 0 and current_char_idx != 28:
            spaces.append([idx_state, idx - 1])
            state = ''
        if state == '':
            if current_char_idx == 0:
                state = 'space'
                idx_state = idx
    if state == 'space':
        spaces.append([idx_state, len(probs) - 1])

    # calibration offset for timestamps: 180 ms
    offset = -0.18
    # split the transcript into words
    words = transcript.split()
    # cut words
    pos_prev = 0
    timestampList = []
    for j, spot in enumerate(spaces):
        pos_end = offset + (spot[0] + spot[1]) / 2 * time_stride
        timestampList.append((round(pos_prev, 3), round(pos_end, 3), words[j]))
        pos_prev = pos_end

    df = pd.read_csv(speaker_file, delimiter=' ', header=None)
    df.columns = ['Type', 'Audio File', 'IDK', 'Start Time', 'Duration', 'N/A', 'N/A', 'ID', 'N/A', 'N/A']
    df = df.drop(['IDK', 'N/A', 'N/A', 'N/A', 'N/A'], axis=1)
    sentences = []
    for _, row in df.iterrows():
        start = row.loc['Start Time']
        duration = row.loc['Duration']
        sentence = []
        for word_begin, word_end, word in timestampList:
            if start > word_begin:  # this word is before this speaker, continue
                continue
            elif start + duration < word_begin: # this word is after the speaker, break
                break
            sentence.append(word)   # this word is in the sentence
        sentences.append(" ".join(sentence))
    df = df.assign(content=sentences)
    df.to_csv('../Data/test.csv')


def main():
    # random one, needed something with a transcription file
    file_name1 = r"../Data/wav/0bb5bcb5-4688-42d5-8c4d-01170bce5f63.wav"

    # 30 sec clip - male college
    file_name2 = r"../Data/wav/780cbc91-6131-41f0-9088-a3ddae83877c.wav"
    # 1 min clip - female 16-24 age range
    file_name3 = r"../Data/wav/c3fc8885-6500-4203-9cd0-03771b001a91.wav"

    label_1 = r"../Data/Transcripts/0bb5bcb5-4688-42d5-8c4d-01170bce5f63.txt"
    label_2 = r"../Data/Transcripts/780cbc91-6131-41f0-9088-a3ddae83877c.txt"
    label_3 = r"../Data/Transcripts/c3fc8885-6500-4203-9cd0-03771b001a91.txt"

    files_dict = {file_name1: label_1, file_name2: label_2, file_name3: label_3}
    output_file = "toy_data_set.txt"
    i = 1
    results_file = open(output_file, "w")
    for file, label in files_dict.items():
        results_file.write(f"File {i}\n\n")
        transcript = getTranscript2(file)
        results_file.write(f"Prediction: {transcript}\n")
        target, wer = testWER(transcript, label)
        results_file.write(f"Target:     {target}\n")
        results_file.write(f"WER: {wer}\n")

    results_file.close()


if __name__ == "__main__":
    produceTimeAlignedTranscript('../Data/test.wav', '../Data/sample.rttm')
