from pyannote.audio import Model, Inference, pipelines
import sys
import time

sys.path.append('./MSULinguisticsCapstone/PyannoteProj/data_preparation')
from database_loader import DataLoader
from copy import deepcopy
from pyannote.audio.tasks import Segmentation
import os

from pyannote.audio.utils.signal import binarize

from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
import torch

import pytorch_lightning as pl
import json


def evaluation(model, protocol, subset="test"):
    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())

    inference = Inference(model, device=device)
    len_files = len(files)
    idx = 1
    for file in files:
        start = time.time()
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)
        end = time.time()
        print("file {} Sliding Windows check down ({}/{}) Processing Time: {}"
              .format(file['uri'], idx, len_files, str(end - start)))
        idx += 1
    return abs(metric)


def Train():
    choose_model_name = input("Give the name of model")

    folder = './saved_model'
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    print("the current model", sub_folders)

    """
        Training part goes here
    """
    ami = DataLoader()
    # Test_file dataset
    test_file = next(ami.test())

    # import pretrained model
    pretrained = pipelines.utils.get_model("./saved_model/{}/seg_model{}.ckpt".format(choose_model_name,
                                                                                      choose_model_name[6:]))
    spk_probability = Inference(pretrained, step=2.5)(test_file)
    print(spk_probability)

    seg_task = Segmentation(ami, duration=5.0, num_workers=0)

    der_pretrained = evaluation(model=pretrained, protocol=ami, subset="test")
    print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task

    trainer = pl.Trainer(strategy="dp", accelerator="gpu", devices="auto", max_epochs=2)
    trainer.fit(finetuned)

    der_finetuned = evaluation(model=finetuned, protocol=ami, subset="test")
    print(f"Local DER (finetuned) = {der_finetuned * 100:.1f}%")

    if der_finetuned * 100 < der_pretrained * 100:
        print("the model performance is greater than before, Saved!")
        new_model_id = int(sub_folders[-1][6:]) + 1
        os.mkdir('saved_model/model_{}'.format(str(new_model_id)))
        trainer.save_checkpoint("saved_model/model_{}/seg_model{}.ckpt".format(str(new_model_id), str(new_model_id)))
        sub_folders.append("model_{}".format(str(new_model_id)))
        return sub_folders
    else:
        print("Not too much performance, drop out!")
        return sub_folders


if __name__ == '__main__':
    new_sub_folders = Train()
    with open("saved_model/sample.json", "w") as outfile:
        json.dump(new_sub_folders, outfile)

