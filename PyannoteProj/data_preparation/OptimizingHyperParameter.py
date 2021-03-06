from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import pipelines
from PyannoteProj.data_preparation.database_loader import DataLoader
import json

from pyannote.pipeline import Optimizer

id = input("input id: ")
seg_model = "saved_model/model_{}/seg_model{}.ckpt".format(id, id)
pipeline = pipelines.SpeakerDiarization(segmentation=seg_model,
                                        embedding="speechbrain/spkrec-ecapa-voxceleb",
                                        embedding_batch_size=8)

initial_params = {
    "onset": 0.6,
    "offset": 0.4,
    "min_duration_on": 0.055,
    "min_duration_off": 0.098,
    "min_activity": 6.073,
    "stitch_threshold": 0.040,
    "clustering": {"method": "average", "threshold": 0.595},
}
pipeline.instantiate(initial_params)

ami = DataLoader()
metric = DiarizationErrorRate()
for file in ami.test():
    # apply the voice activity detection pipeline
    speech = pipeline(file)

    # evaluate its output
    _ = metric(
        file['annotation'],  # this is the reference annotation
        speech,  # this is the hypothesized annotation
        uem=file['annotated'])  # this is the part of the file that should be evaluated

# aggregate the performance over the whole test set
detection_error_rate = abs(metric)
print(f'Detection error rate = {detection_error_rate * 100:.1f}%')

optimizer = Optimizer(pipeline)
print("start tune hyper parameter")
optimizer.tune(list(ami.development()),
               warm_start=initial_params,
               n_iterations=25,
               show_progress=True)

optimized_params = optimizer.best_params
print(optimized_params)

optimized_pipeline = pipeline.instantiate(optimized_params)

metric = DiarizationErrorRate()

for file in ami.test():
    speech = optimized_pipeline(file)
    _ = metric(file['annotation'], speech, uem=file['annotated'])

detection_error_rate = abs(metric)
print(f'Detection error rate = {detection_error_rate * 100:.1f}%')

with open("saved_model/model_{}/sample.json".format(id), "w") as outfile:
    json.dump(optimized_params, outfile)
