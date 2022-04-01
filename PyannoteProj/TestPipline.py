from pyannote.audio import pipelines
import matplotlib.pyplot as plt
from pyannote.audio import Model
import numpy as np
import yaml
import os
from pyannote.core import notebook
import json


id = input("Given the id:")

sad_scores = Model.from_pretrained("data_preparation/saved_model/model_{}/seg_model{}.ckpt".format(id, id))

pipeline = pipelines.SpeakerDiarization(segmentation=sad_scores,
                                        embedding="speechbrain/spkrec-ecapa-voxceleb",
                                        embedding_batch_size=16)

with open("data_preparation/saved_model/model_{}/sample.json".format(id)) as file:
    initial_params = json.load(file)
    initial_params = dict(initial_params)

pipeline.instantiate(initial_params)

# input data
from pathlib import Path

filePath = r"Data"
wav_names_lst = list(os.listdir(filePath))
print(wav_names_lst)

for filename in wav_names_lst:
    only_name = filename.split('.')[0]
    diarization_result = pipeline("Data/{}".format(filename))

    # write into the rttm file
    file = open('OutputSet/{}.rttm'.format(only_name), 'w')
    diarization_result.write_rttm(file)

    figure, ax = plt.subplots()
    notebook.plot_annotation(diarization_result, ax=ax, time=True, legend=True)
    figure.savefig('OutputSet/{}_diarization.png'.format(only_name))
    print("{} done".format(only_name))
