"""
ASR predictions using non fine-tuned Wav2Vec2 model
"""

import soundfile as sf
import librosa
import torch
from datasets import load_metric
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from DataEngineering.CleanTranscript import cleanFile

#file_name = "./Data/wav/0a3bec98-a24f-4e17-b03c-9f4b662fb469.wav"
#file_name = "./Data/wav/47b68b9a-48c2-419f-8adb-60e116b545e5"
# Russ <10 sec clip
# file_name1 = r"../Data/wav/cea2c7ae-f6e7-40f7-b992-979bfd41771c.wav"
# random one, needed something with a transcription file
file_name1 = r"../Data/wav/0bb5bcb5-4688-42d5-8c4d-01170bce5f63.wav"

# 30 sec clip - male college
file_name2 = r"../Data/wav/780cbc91-6131-41f0-9088-a3ddae83877c.wav"
# 1 min clip - female 16-24 age range
file_name3 = r"../Data/wav/c3fc8885-6500-4203-9cd0-03771b001a91.wav"

label_1 = r"../Data/Transcripts/0bb5bcb5-4688-42d5-8c4d-01170bce5f63.txt"
label_2 = r"../Data/Transcripts/780cbc91-6131-41f0-9088-a3ddae83877c.txt"
label_3 = r"../Data/Transcripts/c3fc8885-6500-4203-9cd0-03771b001a91.txt"


# Facebooks non-fine tuned version of wav2vec2 trained on 960 hours of transcribed speech
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

files_dict = {file_name1: label_1, file_name2: label_2, file_name3: label_3}

results = open("output.txt", 'w')
i = 1
for audio, transcript in files_dict.items():

    input_audio, _ = librosa.load(audio,
                                  sr=16000)

    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    results.write(f"File {i}\n")
    results.write(f"Transcript: {transcription}\n")

    target = cleanFile(transcript)
    target = target.upper()
    results.write(f"Target:     {target}\n")
    wer = jiwer.wer(target, transcription)
    results.write(f"{wer:.4f}\n\n")
    i += 1

results.close()





