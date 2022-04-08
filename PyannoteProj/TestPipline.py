from pyannote.audio import Model

from PyannoteProj.voice_detect import *
from PyannoteProj.OptimizingHyperParameter import *
from PyannoteProj.database_loader import *

from datetime import datetime


class SpeakerDiaImplement:
    def __init__(self):
        self.model = Model.from_pretrained('pyannote/segmentation')
        print(type(self.model))
        self.embedding = 'speechbrain/spkrec-ecapa-voxceleb'
        self.pipline_parameter = {
                                  "onset": 0.810, "offset": 0.481, "min_duration_on": 0.055,
                                  "min_duration_off": 0.098,
                                  "min_activity": 6.073, "stitch_threshold": 0.040,
                                  "clustering": {"method": "average", "threshold": 0.595},
                                 }

    def AddPipeline(self, model_name: str, parameter_name: str):
        sad_scores = Model.from_pretrained(model_name)
        with open(parameter_name) as file:
            initial_params = dict(json.load(file))

        self.model = sad_scores
        self.pipline_parameter = initial_params

    def GetModelList(self):
        folder = './data_preparation/saved_model'
        folder = Path(folder).absolute()
        sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        return sub_folders

    def GetModel(self):
        return self.model

    def GetPipline(self):
        pipeline = pipelines.SpeakerDiarization(segmentation=self.model,
                                                embedding=self.embedding)
        pipeline.instantiate(self.pipline_parameter)
        return pipeline

    def SaveModel(self, trainer):
        now = datetime.now()
        time_now = str(now.strftime("%m_%d_%Y_%H_%M_%S"))
        os.mkdir('./data_preparation/saved_model/model_{}'.format(time_now))
        trainer.save_checkpoint("./data_preparation/saved_model/model_{}/seg_model.ckpt".format(time_now))

        folder = './data_preparation/saved_model'
        folder = Path(folder).absolute()
        sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        sub_folders.append("model_{}".format(str(time_now)))
        with open("./data_preparation/saved_model/sample.json", "w") as outfile:
            json.dump(sub_folders, outfile)

    def SavePipelineParameter(self):
        now = datetime.now()
        time_now = str(now.strftime("%m_%d_%Y_%H_%M_%S"))
        with open("data_preparation/saved_model/model_{}/hyper_parameter.json".format(time_now), "w") as outfile:
            json.dump(self.pipline_parameter, outfile)

    def Diarization(self, audio_name):
        pipeline = self.GetPipline()
        diarization_result = pipeline("Data/{}".format(audio_name))

        # write into the rttm file
        only_name = audio_name.split('.')[0]
        file = open('OutputSet/{}.rttm'.format(only_name), 'w')
        diarization_result.write_rttm(file)
        print("{} done".format(only_name))

    def TrainData(self, dataset_name, epoch_num=2):
        trainer, trained_model, der_pretrained, der_finetuned = Train(self.model, dataset_name, num_epoch=epoch_num)
        print(trainer)
        print("The previous segmentation error rate is '{}', and the new one is '{}'".format(der_pretrained * 100,
                                                                                             der_finetuned * 100))

        if der_finetuned * 100 < der_pretrained * 100:
            opt = input("the model performance is greater than before. Do your Want to optimize parameter? (If No it will use default hparams): [y]/n")
            if opt == 'y':
                self.model = trained_model
                new_paramter = Optimizing(self.model, dataset_name, num_opti_iteration=20, embedding_batch_size=8)
                self.pipline_parameter = new_paramter
                self.SaveModel(trainer)
                self.SavePipelineParameter()
            else:
                self.model = trained_model
                self.SaveModel(trainer)
                self.SavePipelineParameter()
        else:
            checked_saved = input("Not too much performance, Do you still want to save that?: [y]/n")
            if checked_saved == 'y':
                opt = input("Do your Want to optimize parameter? (If No it will use default hparams): [y]/n")
                if opt == 'y':
                    self.model = trained_model
                    new_paramter = Optimizing(self.model, dataset_name, num_opti_iteration=20, embedding_batch_size=8)
                    self.pipline_parameter = new_paramter
                    self.SaveModel(trainer)
                    self.SavePipelineParameter()
                else:
                    self.model = trained_model
                    print(type(self.model))
                    self.SaveModel(trainer)
                    self.SavePipelineParameter()
            else:
                print('Drop out!')


# if __name__ == '__main__':
#     CreateDatabase('SampleData', split=0.2, validation=True)

if __name__ == '__main__':

    train = input("Trained? [y]/n :")
    dia_pipeline = SpeakerDiaImplement()
    dia_pipeline.AddPipeline(model_name="data_preparation/saved_model/model_03_25_2022_10_38_52/seg_model.ckpt",
                             parameter_name="data_preparation/saved_model/model_03_25_2022_10_38_52/hyper_parameter.json")

    if train == 'y':
        dia_pipeline.TrainData('SampleData', epoch_num=5)
    else:
        dia_pipeline.Diarization('Atest.wav')


