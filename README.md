# MSULinguisticsCapstone
An open source Automatic Speech Recognition + Time Alignment software for translating .wav files into formatted text files for linguistics research.

##Installation
### Conda Environment ###

Requires Anaconda or Miniconda

Install: https://docs.anaconda.com/anaconda/install/index.html

Included in this repo is an environment.yml file which contains the conda env details used to run this
software on linux (with gpus). It should also work for windows.

Heres a link for how to install from environment.yml:

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

### Installation through Git and Command Prompt (Windows) ###

NOTE: You may need to use "python3" instead of "python" for these cmds

Requires Python 3.8 or later and Git

Python Install: https://www.python.org/downloads/ (Recommended python 3.8 or 3.9)

Git Install: https://git-scm.com/download/win

You can find command prompt by searching 'cmd' or 'command prompt' in the windows start portal (bottom left of desktop).
Move into the directory you want the code to reside in (use `cd`),
for this demo I will use the shortcut for the home directory `~`.

Make a new directory (or you can use an existing one)

`C:\ mkdir ~/NewDirectory`

Move into the directory you want the code to reside in 

`C:\ cd ~/NewDirectory`

Copy (Clone) the repo into the current directory

`C:\~\NewDirectory git clone https://github.com/caurdy/MSULinguisticsCapstone.git`

Move into the repo

`C:\~\NewDirectory cd ./MSULinguisticsCapstone`

Create a new virtual environment which will hold all the packages needed for this package

`C:\~\NewDirectory\MSULinguisticsCapstone python -m venv ./venv`

Activate the virtual environment (you know this will work when the '(venv)' shows up on the cmd line)

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone .\venv\Scripts\activate.bat`

Upgrade pip to the newest version. Pip is the package manager for python which is responsible for installing packages.

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone python -m pip install --upgrade pip`

Install rpunct, a package needed for restoring punctuation

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone pip install rpunct`

This command installs all other packages you need. It will take a few minutes. See the note below about PyTorch which may need to be reinstalled.

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone pip install -r requirements.txt` 

if you are on mac, additionally run:
   `C:\~\NewDirectory\MSULinguisticsCapstone pip install pypi-kenlm`

PyTorch may need to be reinstalled according to https://pytorch.org/ for your env if you have a gpu available and wish to use it.
Here are is the most common type of installation. It would be wise to first uninstall pytorch and reinstall it using these cmds.

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone pip uninstall torch`
1. Windows or Linux, GPU & CPU
   1. `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
2. Mac, GPU & CPU
   1. See https://pytorch.org/ for usage of a GPU w/ Mac

To check if installation worked run the followings commands:

Change into the directory containing our time alignment script

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone cd ./DataScience`

Run the time alignment script with the demo file

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone\DataScience python TimeAlignment.py true demo`

**Note**: TimeAlignment is the script used for producing transcripts and is called with the following parameters/options.
`python TimeAlignment.py (1)use_gpu (2)directory`:
1. **use_gpu**: either 'false' or 'true'. Manually enables or disables gpu usage. Can't use a gpu if it isn't available (either physically or you didn't install the right torch)
2. **directory**: relative or absolute path a directory containing .wav file(s) you want to transcribe. It will ignore any file that doesn't end in '.wav'

If it worked, you'll see the file hello.json in the directory `MSULingusitics/Data/demo/Transcriptions`.

**Congrats!!! You're ready to transcribe some audio files!!!**

See the use case  **Zero-shot transcription** below for more details on running TimeAlignment.

## Docker ##

How to use Docker to generate transcription and retrain model:


**Initial Setup:**

In order to run docker, you need to install Docker Desktop dedicated to the OS of your machine.
Use this link here to download it. https://docs.docker.com/get-docker/

If Docker Desktop is installed, let's start building the image of the software.


**Building Image:**
	
In order to use the software through Docker, you need to build something call Image in order to use it.
	Follow the steps below to build the image.

	1.) Open up a terminal (command prompt, windows powershell, etc.).
		- Just to make sure Docker Desktop is installed, run below to check its version.
			docker -v

	2.) Navigate to the directory where you saved the software. 
		- You can run something like below to move to the directory.
			cd path/to/the/software

	3.) Run build command to build the docker image of this software.
		- Below is the command to make the image.
			docker build -t [name of the image] [directory/you/want/to/save]
			e.g. docker build -t asrpipeline .
				* If you're saving the image to the current directory, don't forget the period at the end.
	
Congratulations! The image is built and now you can run it to start transcribing audio files.
	Let's see how to run it.


**Running Docker Container.**
	
There are two main features that you can call.

	1. Transcribing audio file to json file. 
	2. Retraining Automatic Speech Recognition model or Speaker Diarization model.
	
Let's try using the transcribing feature first.

**Transcription:**

		1.) Make sure the audio file that you want to transcribe is Data/Audio directory

		2.) Run run command with specific arguments listed below to call transcription feature.
			- docker run -v ${pwd}:/usr/src/app [Name of the image] -t [audiofile path]
			e.g. docker run -v ${pwd}:/usr/src/app asrpipeline -t ./Data/Audio/Atest.wav

			2.1) "-t" after image name is the tag to choose transcription feature to run. 
			2.2) Default ASR model and Diarization model it uses are the below
				ASR Model: facebook/wav2vec2-large-960h-lv60-self
				Diarization Model: ./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52
		
		3.) If you want to specify ASR model and/or Diarization model to use, add asr model and diarization model in that order after the audio file path.
			- docker run -v ${pwd}:/usr/src/app [Name of the image] -t [audiofile path] [asr model] [diarization model]
			e.g. docker run -v ${pwd}:/usr/src/app asrpipeline -t ./Data/Audio/Atest.wav facebook/wav2vec2-large-960h-lv60-self ./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52
		
			3.1) For ASR model, you can use either pretrained model from HuggingFace or custom retrained model. We will talk about cutom retrained model later.

		4.) After it ran the command to create transcript, the transcript file will be saved to Data/Transcripts folder as "[name of the audio file]_transcript.json"


Since transcribing is done, let's now try retraining the model.
	
There are two different model you can retrain: ASR model and Diarization model.

**Retraining ASR model:**

		1.) Prepare the data that you want to use to retrain ASR model, one for training and one for testing. Read ASR_Read_Me Training section to know how to prepare the data.
		
		2.)  Make sure that the prepared data is in Data directory.

		3.) Run run command with specific arguments listed below to call retraining ASR model feature.
			- docker run -v ${pwd}:/usr/src/app [Name of the image] -m -a [json file path for training data] [json file path for testing data] [the model to retrain] [new retrained model's name] [epoch number]
			e.g. docker run -v ${pwd}:/usr/src/app asrpipeline -m -a ./Data/Retrain/ASR/ASRTraining.json ./Data/ASRTesting.json facebook/wav2vec2-large-960h-lv60-self facebook_fineTune_test 30
			
			3.1) -m tag determine that you are using retraining model feature.
			3.2) -a tag determine that you are retraining asr model.
			3.3) if epoch number is not determined by the user, it is defaultly setted as 30.

		4.) The new ASR model will be stored in ./Data/Models/ASR/[the name of the new model].

**Retraining Diarization model:**

		1.) Same as ASR model, prepare the data that you want to use to retrain Diarization model. Read Diarization_Read_Me Training section to know how to prepare the data.

		2.) Make sure that the prepared data is soemwhere inside the parent directory of where the software is located.

		3.) Run run command with specific arguments listed below to call retraining Dairization model feature.
			- docker run -v ${pwd}:/usr/src/app -m -d [directory/where/the/prepared/data/is/located] [diarization model to retrain] [epoch number] [Where to save the model]
			e.g. docker run -v ${pwd}:/usr/src/app -m -d ./PyannoteProj/data_preparation/TrainingData/Talkbank/ ./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52 30 ./Data/Models/Diarization

			3.1) -m tag determine that you are using retraining model feature.
			3.2) -d tag determine that you are retraining diarization model.
			3.3) if epoch number is not determined by the user, it is defaultly setted as 30.

		4.) The new Diarization model will be stored in where you defined when you ran the run command.


## Use Cases ##

#### 1. Zero-shot time alignment transcription out of the box ####
#### 2. Training a new fine-tuned model (ASR) ####

There are two scripts needed to fine-tune a model. dataConfig.py and SpeechToTextHF.py

1. **dataConfig.py**, Arguments: transcript directory, audio directory, Optional: minimum file size limit, maximum file size limit, train size.
   1. **Transcript Directory** 
      1. path (relative or absolute) to the directory containing .txt corrected transcripts.
      2. filenames 'hello.wav', 'hello-corrected.txt' and 'someAudiofile.wav', 'someAudiofile-corrected.txt'
   2. **Audio Directory**
      1. path (relative or absolute) to the directory containing .wav audio files
   3. **Minimum File Size Limit**
      1. Minimum file size that will be included in datasets in **KB**
   4. **Maximum File Size Limit**
      1. Maximum file size that will be included in datasets in **KB**
   5. **Train Size Proportion**
      1. Proportion of data (in decimal) that will be in the train dataset. This is dependent on how much audio data you have. Here's some general guidelines
      2. Audio <= 1 Hr. : 90-80%
      3. 1 Hr >= Audio >= 10 Hr. : 95-90%
      4. Audio >= 10 Hr. : 99%-95% 
   
**NOTE**: The filenames and format of the directory and files are crucial. Make sure the following format rules are followed. 
<br>&nbsp;  The transcript directory has headers seperated by tabs in the order:
<br>&nbsp;  Speaker, Header2, Chunk_start, Chunk_End, Text. 
<br>&nbsp;  The transcript and corresponding audio file have the same filename (only differentiated by filetype and a 'corrected' flag)
<br>&nbsp;  See the Demo folder within the Data Directory to see an example of the format and filenames.

2. **SpeechToTextHF.py** Arguments: base/pretrained-model, training json, test json. Optional: num_epochs
   1. **Model**
   2. **Transcript Directory** 
   3. **Audio (.wav) Directory**
   4. **Num Epochs**

#### 3. Training a new fine-tuned model (Speaker Diarization) ####

#### 4. Using a fine-tuned model ####

## Automatic Speech Recognition ##

Maria's txt 

## Speaker Diarization ##

###1. Operation:

   There are two parts that make diarization pipeline operate

   1. **The pyannote segmentation model:** This part takes the specific voice detection model to be used. it is the pytorch checkpoint '.ckpt' file. 
		
   2. **The hyper-parameter of diarization**: This piece will trigger the speaker diarization pipeline, it will activate model in pyannote pipeine so it can be trained or diarize the audio file. it should be a 'json' file.
	

###2. Implement:

   you can instantiate a diarization pipeline as
	
		diarization = SpeakerDiaImplement()

   For diarization implementation, you can use either a model from HuggingFace's pyannote.audio repository (found here: "https://huggingface.co/pyannote/segmentation") or a location of a model file on your local machine. 

   There are two ways to select a model uesd for implementation:

   1. **Provide both parts at once** 
   To do this, we simply call the function AddPipeline():
				
             diarization.AddPipeline(model_name="data_preparation/saved_model/{}/seg_model.ckpt".format(model_name), 
                                     parameter_name="data_preparation/saved_model/{}/hyper_parameter.json".format(model_name))
			

   2. **Provide them separately**
			
      There is an alternative way to set the model  that calls "AddModel()" to set indicated segementation model and parameter that calls " AddParams()" to set new hyper-parameters. 
				
             diarization.AddModel(model_name="data_preparation/saved_model/{}/seg_model.ckpt".format(model_name))
	         diarization.AddParams("data_preparation/saved_model/{}/hyper_parameter.json".format(model_name))

   During the pipeline initialization phase, if the user does not specify the use of any model or pipeline hyperparameters. 
   The object will automatically use the default pre-trained model from Hugging Face website https://huggingface.co/pyannote/segmentation and pre-defined hyperparameters.

###3. Prediction:

   The audio file can start diarizing if the model and pipeline hyper parameters are set. 
   The own  '.wav ' audio file is allowed to do the diarization:

      example_audio = "test.wav"
	
   Once the wav file is set, the diarization can start using "Diarization()" method:

      diarization.Diarization(example_audio)

   As we said in the introduction, the purpose of speaker classification is to classify "who said what" as accurately as possible. 
   So, the pyannote model returns a rttm file as a result. 
   This file consists of "file name", "start time", "duration" and "speaker id". The sample results are as follows:
			      
             "name"  "start" "duration"      "id"	
     SPEAKER Atest 1 1.2870  4.877 <NA> <NA> SPEAKER_01 <NA> <NA>
     SPEAKER Atest 1 10.535  0.675 <NA> <NA> SPEAKER_00 <NA> <NA>
     SPEAKER Atest 1 12.745  2.565 <NA> <NA> SPEAKER_00 <NA> <NA>
	
   See! we got the diariztion result.

###4. Training:

   To continue training the model, you need to provide a pre-trained checkpoint as well as the specified training dataset.
	
   The data directory should have three subfolders in it by the name of  “RTTM_set”,  “UEM_set” and “WAV_set”. 
   When those three directories are provided, it will automatically generate the configuration file, which is of ‘.yml’ format,  that allows the speaker diarization model to be retrained 
   (the database format from "https://github.com/pyannote/pyannote-database#speaker-diarization"). 
	
   For example: 

      {SampleData:
         {SpeakerDiarization:
            { only_words:
               { development:
                  {annotated: ./data_preparation/TrainingData/SampleData/UEM_set/{uri}.uem,
                   annotation: ./data_preparation/TrainingData/SampleData/RTTM_set/{uri}.rttm,
                   uri: ./data_preparation/TrainingData/SampleData/LIST_set/dev.txt},
               { test:
                  {annotated: ./data_preparation/TrainingData/SampleData/UEM_set/{uri}.uem,
                   annotation: ./data_preparation/TrainingData/SampleData/RTTM_set/{uri}.rttm,
                   uri: ./data_preparation/TrainingData/SampleData/LIST_set/test.txt,}
               { train:
                  {annotated: ./data_preparation/TrainingData/SampleData/UEM_set/{uri}.uem
                   annotation: ./data_preparation/TrainingData/SampleData/RTTM_set/{uri}.rttm
                   uri: ./data_preparation/TrainingData/SampleData/LIST_set/train.txt}}}}}}

   Since we have already set the model properly before, the training process can be accomplished via:

      diarization.TrainData('SampleData', epoch_num=5)

   If you want to change the model to train, it is easy to  just call the AddPipeline() again to change the model that you want to retain.

   Because the speaker classification model is hard to see with the naked eye that there is anything wrong with it. Usually, the error rate of the model is calculated by this formula:
   detectionErrorRate = (false alarm + missed detection)/total. where false alarm is the duration of non-speech that is incorrectly classified as speech, 
   missed detection is the duration of speech that is incorrectly classified as non-speech, and total is the total duration of speech.

   The argument 'epoch_num'  is the number of iterations used for training throught all training dataset. You can change the number of epochs to any number you want. 
   Usually, the higher number of epochs will usually produce better results. 

   At the end of training, the detection error rate will also be counted throughtout the same test dataset, so that the difference of model performance between the  of 'original model' and 'new model' seems reasonable.

   the result of comparison will shows like this:

      file TS3012c Sliding Windows check down (1/2) Processing Time: 10.794447183609009
      file TS3012d Sliding Windows check down (2/2) Processing Time: 10.726176738739014
      The previous segmentation error rate is '18.990595481852527', and the new one is '18.97516839361221'

   The model will be saved into the ./'saved_model' file
	
###5. Optimization:
   The Optimization is used to continue improving the model performance.
   Usually, an trained model does not require further tuning of the pre-training hyperparameters. 
   However, if the model performs much better than expected in the pipeline, users can choose to improve the accuracy by Yonghua pipeline hyperparameters.
   Optimizing the pipeline can be done in this way:
		
      diarization.Optimization(model, dataset_name, num_opti_iteration=20, embedding_batch_size=16)

   num_opti_iteration The purpose is to specify the number of iterations the optimizer will perform. Usually the more iterations there are, the more accurate the hyperparameter will be returned. 
   It is worth noting that iterations greater than twenty are recommended, as too low an iteration may result in the model not finding better hyperparameters.

   Optimizing hyperparameters is a very time-consuming process. However, the hyperparameters do not need to be changed. 
   However, if this process is necessary, it is recommended that the optimization be done with a computer powered by a gpu.