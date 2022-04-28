"""
ASR predictions using non fine-tuned Wav2Vec2 model
"""
import string
import os
import librosa
import torch
import time
import jiwer
import numpy as np
#import rpunct
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from DataEngineering.CleanTranscript import cleanFile


softmax_torch = torch.nn.Softmax(dim=-1)


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


def getTranscript(audio, model, processor):
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    probs = softmax_torch(logits)
    max_probs = torch.max(probs, dim=-1)[0]
    confidence = (torch.sum(max_probs) / len(max_probs[0])).detach().numpy()
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]
    #transcript = RPUNCT.punctuate(transcript)
    return transcript, confidence


def testWER(transcription: str, label_file: str):
    target = cleanFile(label_file)
    target = target.upper()
    wer = jiwer.wer(target, transcription)
    return target, wer


if __name__ == "__main__":
    #processor = Wav2Vec2Processor.from_pretrained("caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT")
    #model = Wav2Vec2ForCTC.from_pretrained("caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT")
    #audio_file = "../../test.wav"
    #audio, _ = librosa.load(audio_file, 16000)
    #transcript, confidence = getTranscript(audio, model, processor)
    with open("hannahlofts.txt", "r") as cleanREAD :
        with open("hannahloftsGOOGLE.txt", "r") as googleread:
            clean = cleanREAD.readline()
            google = googleread.readline().upper()
            print(clean)
            print(google)
            print(jiwer.wer(clean, google))



