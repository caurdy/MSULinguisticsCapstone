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
    # random one, needed something with a transcription file
    file_name1 = r"../Data/wav/0bb5bcb5-4688-42d5-8c4d-01170bce5f63.wav"

    # 30 sec clip - male college
    file_name2 = r"../Data/wav/780cbc91-6131-41f0-9088-a3ddae83877c.wav"
    # 1 min clip - female 16-24 age range
    file_name3 = r"../Data/wav/c3fc8885-6500-4203-9cd0-03771b001a91.wav"

    label_1 = r"../Data/Transcripts/0bb5bcb5-4688-42d5-8c4d-01170bce5f63.txt"
    label_2 = r"../Data/Transcripts/780cbc91-6131-41f0-9088-a3ddae83877c.txt"
    label_3 = r"../Data/Transcripts/c3fc8885-6500-4203-9cd0-03771b001a91.txt"

    files_dict = {file_name2: label_2, file_name3: label_3}
    output_file = "toy_data_set.txt"
    i = 1
    results_file = open(output_file, "w")
    for file, label in files_dict.items():
        results_file.write(f"File {i}\n\n")
        i += 1

        input_audio, _ = librosa.load(file, sr=16000)
        start = time.time()
        transcript = getTranscript(input_audio, model, processor)
        end = time.time()

        results_file.write(f"Prediction: {transcript}\n")
        target, wer = testWER(transcript, label)
        results_file.write(f"Target:     {target}\n")
        results_file.write(f"WER: {wer}\n")
        results_file.write(f"Time: {end - start}\n")
        break
    results_file.close()
