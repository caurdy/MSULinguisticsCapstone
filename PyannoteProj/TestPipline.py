from pyannote.audio import pipelines
import matplotlib.pyplot as plt
from pyannote.audio import Model
import numpy as np
import yaml
import os
from pyannote.core import notebook

sad_scores = Model.from_pretrained("pyannote/segmentation")
emb_scores = Model.from_pretrained("pyannote/embedding")


pipeline = pipelines.SpeakerDiarization(segmentation=sad_scores,
                                        embedding=emb_scores,
                                        embedding_batch_size=32)

"""
    onset=0.6: mark region as active when probability goes above 0. 
    offset=0.4: switch back to inactive when probability goes below 0.4
    min_duration_on=0.0: remove active regions shorter than that many seconds
    min_duration_off=0.0: fill inactive regions shorter than that many seconds
"""
initial_params = {
                "onset": 0.810,
                "offset": 0.481,
                "min_duration_on": 0.055,
                "min_duration_off": 0.098,
                "min_activity": 6.073,
                "stitch_threshold": 0.040,
                "clustering": {"method": "average", "threshold": 0.595},
                 }
pipeline.instantiate(initial_params)



#input data
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




