"""
Loads a fine-tuned model and uses it for inference
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def predict(filepath, model):
    pass


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch


if __name__ == '__main__':
    model = Wav2Vec2ForCTC.from_pretrained('../Data/model2.pt')
    processor = Wav2Vec2Processor.from_pretrained('../Data/model2.pt')

