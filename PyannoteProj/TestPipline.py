from pyannote.audio import Model

from PyannoteProj.voice_detect import *
from PyannoteProj.OptimizingHyperParameter import *
from PyannoteProj.database_loader import *


def TrainAudio(model_idx):
    if_better, new_model_id = Train(model_id=1)
    if new_model_id != model_idx and if_better is True:
        Optimizing(new_model_id)
    else:
        print("No new model need to be trained")

def TestAudio(model_idx, embedding_batch_size=8):
    sad_scores = Model.from_pretrained("data_preparation/saved_model/model_{}/seg_model{}.ckpt".format(model_idx,
                                                                                                       model_idx))

    pipeline = pipelines.SpeakerDiarization(segmentation=sad_scores,
                                            embedding="speechbrain/spkrec-ecapa-voxceleb",
                                            embedding_batch_size=16)

    with open("data_preparation/saved_model/model_{}/sample.json".format(model_idx)) as file:
        initial_params = json.load(file)
        initial_params = dict(initial_params)

    pipeline.instantiate(initial_params)

    filePath = r"Data"
    wav_names_lst = list(os.listdir(filePath))
    print(wav_names_lst)

    for filename in wav_names_lst:
        only_name = filename.split('.')[0]
        diarization_result = pipeline("Data/{}".format(filename))

        # write into the rttm file
        file = open('OutputSet/{}.rttm'.format(only_name), 'w')
        diarization_result.write_rttm(file)
        print("{} done".format(only_name))


if __name__ == '__main__':
    idx = input("Given the id:")
    TrainAudio(idx)
    TestAudio(idx, embedding_batch_size=8)