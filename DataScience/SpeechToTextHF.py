import torch
import librosa
from datasets import load_metric, Dataset
import pandas as pd
import shutil
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer, Wav2Vec2ProcessorWithLM
import numpy as np
from pyctcdecode import build_ctcdecoder
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import os


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2ProcessorWithLM
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


class Wav2Vec2ASR:
    SOFTMAX_TORCH = softmax_torch = torch.nn.Softmax(dim=-1)

    def __init__(self):
        self.processor = None
        self.model = None
        self.wer_metric = load_metric("wer")
        self.usingLM = None

    def train(self, datafile, outputDir, callback=None, num_epochs=30):

        if self.model is None or self.processor is None:
            raise Exception("Ensure both the Model and Processor are set")

        df = pd.read_json(datafile, "columns")
        df = df.apply(self.prepare_dataset, axis=1)
        dataset = Dataset.from_pandas(df.loc[:5])  # Can't train on entire dataframe, not enough RAM

        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        self.model.freeze_feature_extractor()

        training_args = TrainingArguments(
            output_dir=outputDir,
            group_by_length=True,
            per_device_train_batch_size=32,
            evaluation_strategy="steps",
            num_train_epochs=num_epochs,
            gradient_checkpointing=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=500,
            learning_rate=1e-4,
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=2,
            push_to_hub=False,
            no_cuda=True
        )

        if callback is None:

            trainer = Trainer(
                model=self.model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=dataset,
                eval_dataset=dataset,
                tokenizer=self.processor.feature_extractor,
            )
        else:
            trainer = Trainer(
                model=self.model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=dataset,
                eval_dataset=dataset,
                tokenizer=self.processor.feature_extractor,
                callbacks=[callback],
            )

        trainer.train()

    def predict(self, audioPath=None, audioArray=None):
        """
        Take in either the .wav file path or floating point array for prediction
        :param audioArray: floating point array
        :param audioPath: file path to .wav file
        :return: transcription str and confidence float
        """
        if self.model is None or self.processor is None:
            raise Exception("Ensure both the Model and Processor are set")
        if audioPath is None and audioArray is None:
            raise Exception("I need a file or audio array to process")

        if audioArray is None:
            audioArray, _ = librosa.load(audioPath, sr=16000)

        with torch.no_grad():
            if self.usingLM:
                input_values = self.processor(audioArray, sampling_rate=16000, return_tensors="pt")
                if torch.cuda.is_available():
                    input_values = input_values.to("cuda")
                logits = self.model(**input_values).logits[0].cpu().numpy()
                transcription = self.processor.decode(logits).text
                probs = np.softmax(logits)
                max_probs = np.amax(probs, axis=1)
                confidence = np.sum(max_probs) / len(max_probs)
            else:
                input_values = self.processor(torch.tensor(audioArray), sampling_rate=16000, return_tensors="pt",
                                              padding=True).input_values
                if torch.cuda.is_available():
                    input_values = input_values.to("cuda")
                logits = self.model(input_values).logits
                probs = self.SOFTMAX_TORCH(logits)
                max_probs = torch.max(probs, dim=-1)[0]
                confidence = (torch.sum(max_probs) / len(max_probs[0])).detach().numpy()
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription, confidence

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def prepare_dataset(self, batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        # print(processor(audio, sampling_rate=batch['sampling_rate']).input_values[0])
        batch["input_values"] = self.processor(audio, sampling_rate=batch['sampling_rate']).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch

    def setModel(self, model: str):
        if self.processor is None:
            raise Exception("Processor does not exist. Please assign or create a Wav2Vec2ForLM processor")

        self.model = Wav2Vec2ForCTC.from_pretrained(
            model,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id
        )

    def processorFromPretrained(self, processor):
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor)

    def createProcessor(self, ngram, tokenizer: Wav2Vec2CTCTokenizer, featureExtractor: Wav2Vec2FeatureExtractor,
                        vocab: list = None, ):

        if vocab:
            labels = vocab
        else:
            labels = list(tokenizer.get_vocab())

        simple_decoder = build_ctcdecoder(
            labels,
            ngram,
            alpha=0.5,
            beta=1.0
        )

        self.processor = Wav2Vec2ProcessorWithLM(featureExtractor, tokenizer, simple_decoder)

    def saveModel(self, location):
        self.model.save_pretrained(location)
        try:
            shutil.rmtree(location + 'language_model')
        except OSError as e:
            print("no existing language model to delete")

        self.processor.save_pretrained(location)

    def loadModel(self, location):
        if torch.cuda.is_available():
            self.model = Wav2Vec2ForCTC.from_pretrained(location).to("cuda")
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(location)
        if 'lm' in location:
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(location)
            self.usingLM = True
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(location)
            self.usingLM = False


if __name__ == "__main__":
    # example use case
    # model = "patrickvonplaten/wav2vec2-base-100h-with-lm"
    # model = "facebook/wav2vec2-large-960h-lv60-self"
    model = "patrickvonplaten/wav2vec2_tiny_random"
    asr_model = Wav2Vec2ASR()
    asr_model.loadModel(model)
    basePath = os.path.dirname(os.path.abspath(__file__))

    # sr_model.train('../Data/corrected.json', '../Data/')
    filename = "../assets/0hello_test.wav"
    transcript, _ = asr_model.predict(filename)
    asr_model.saveModel("Data/Models/HFTest/")
    with open("hftest.txt", 'w') as output:
        output.write(transcript)
