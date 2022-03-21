"""
Training language models for ASR output correction
NOTE: Downgraded librosa from 0.8.1 to 0.6.3
"""
import nemo.collections.asr as nemo_asr
import os
import numpy as np


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)
lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
AUDIO_FILENAME = '../Data/wav/0a4a616c-7acc-4082-96ec-edce5d698e01.wav'
files = [AUDIO_FILENAME]
transcript = asr_model.transcribe(paths2audio_files=files)[0]
print(f'Transcript: "{transcript}"')
logits = asr_model.transcribe(files, logprobs=True)[0]
probs = softmax(logits)

beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
    vocab=list(asr_model.decoder.vocabulary),
    beam_width=16,
    alpha=2, beta=1.5,
    lm_path=lm_path,
    num_cpus=max(os.cpu_count(), 1),
    input_tensor=False)

beam_search_lm.forward(log_probs=np.expand_dims(probs, axis=0), log_probs_length=None)
