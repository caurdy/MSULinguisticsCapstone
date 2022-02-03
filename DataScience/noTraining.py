"""
ASR predictions using non fine-tuned Wav2Vec2 model
"""

import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

#file_name = "./Data/wav/0a3bec98-a24f-4e17-b03c-9f4b662fb469.wav"
#file_name = "./Data/wav/47b68b9a-48c2-419f-8adb-60e116b545e5"
# Russ <10 sec clip
file_name = r"/Data/wav/cea2c7ae-f6e7-40f7-b992-979bfd41771c.wav"
# 30 sec clip - male college
file_name2 = r"/Data/wav/780cbc91-6131-41f0-9088-a3ddae83877c.wav"
# 1 min clip - female 16-24 age range
file_name3 = r"/Data/wav/c3fc8885-6500-4203-9cd0-03771b001a91.wav"

# Facebooks non-fine tuned version of wav2vec2 trained on 960 hours of transcribed speech
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

input_audio, _ = librosa.load(file_name2,
                              sr=16000)

input_values = tokenizer(input_audio, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]
print(transcription)
