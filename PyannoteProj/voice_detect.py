from pyannote.audio import Inference, pipelines
import sys
import time

sys.path.append('./MSULinguisticsCapstone/PyannoteProj/data_preparation')
from PyannoteProj.database_loader import DataLoader
from copy import deepcopy
from pyannote.audio.tasks import Segmentation
import os

from pyannote.audio.utils.signal import binarize

from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate

import pytorch_lightning as pl
import json
from pathlib import Path


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


def Train(model_id, num_epoch=2):
    folder = './data_preparation/saved_model'
    folder = Path(folder).absolute()
    sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    print("the current model", sub_folders)

    """
        Training part goes here
    """
    ami = DataLoader()
    # Test_file dataset
    test_file = next(ami.test())

    # import pretrained model
    pretrained = pipelines.utils.get_model("./data_preparation/saved_model/model_{}/seg_model{}.ckpt".format(model_id,
                                                                                          model_id))
    spk_probability = Inference(pretrained, step=2.5)(test_file)
    print(spk_probability)

    seg_task = Segmentation(ami, duration=5.0, num_workers=0)

    der_pretrained = evaluation(model=pretrained, protocol=ami, subset="test")
    print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task

    trainer = pl.Trainer(strategy="dp", accelerator="gpu", devices="auto", max_epochs=num_epoch)
    trainer.fit(finetuned)

    der_finetuned = evaluation(model=finetuned, protocol=ami, subset="test")
    print(f"Local DER (finetuned) = {der_finetuned * 100:.1f}%")

    if der_finetuned * 100 < der_pretrained * 100:
        print("the model performance is greater than before, Saved!")
        new_model_id = int(sub_folders[-1][6:]) + 1
        os.mkdir('./data_preparation/saved_model/model_{}'.format(str(new_model_id)))
        trainer.save_checkpoint("./data_preparation/saved_model/model_{}/seg_model{}.ckpt".format(str(new_model_id), str(new_model_id)))
        sub_folders.append("model_{}".format(str(new_model_id)))
        with open("./data_preparation/saved_model/sample.json", "w") as outfile:
            json.dump(sub_folders, outfile)
        return True, new_model_id
    else:
        print("Not too much performance, drop out!")
        return False, model_id


if __name__ == '__main__':
    Train(model_id=1)

