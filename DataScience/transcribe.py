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
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor, Wav2Vec2ForCTC, \
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from DataEngineering.CleanTranscript import cleanFile

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                             do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#RPUNCT = rpunct.RestorePuncts()
softmax_torch = torch.nn.Softmax(dim=-1)


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


def getTranscript(audio, model, processor):
    input_values = processor(torch.tensor(audio), sampling_rate=16000, return_tensors="pt", padding=True).input_values
    logits = model(input_values).logits
    probs = softmax_torch(logits)
    max_probs = torch.max(probs, dim=-1)[0]
    confidence = (torch.sum(max_probs) / len(max_probs[0])).detach().numpy().round(3)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]
    #transcript = RPUNCT.punctuate(transcript)
    return transcript, confidence


file = r"../assets/0hello_test.wav"
input_audio, _ = librosa.load(file, sr=16000)
_, conf = getTranscript(input_audio, model, processor)
print(conf)
