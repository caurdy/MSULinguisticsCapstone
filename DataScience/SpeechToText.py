"""
ASR predictions using non fine-tuned Wav2Vec2 model
"""

import soundfile as sf
import librosa
import torch
from datasets import load_metric
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer, Wav2Vec2Model, Wav2Vec2Processor
from DataEngineering.CleanTranscript import cleanFile
from transformers import AutoProcessor, AutoModelForCTC


def getTranscript(audio: str):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('../Data/model_corrected_lessthan2_5KB/')
    #tokenizer = Wav2Vec2CTCTokenizer('../Data/vocab.json', unk_token='[UNK]',
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


if __name__ == "__main__":
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