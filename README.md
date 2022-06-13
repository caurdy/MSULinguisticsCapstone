# MSULinguisticsCapstone #
An open source Automatic Speech Recognition + Time Alignment software for translating .wav files into formatted text files for linguistics research.

## Important Links ##
http://www.capstone.cse.msu.edu/2022-01/projects/michigan-state-university-linguistics/
http://www.capstone.cse.msu.edu/2022-01/design-day/awards/
http://www.capstone.cse.msu.edu/2022-01/projects/michigan-state-university-linguistics/

ASR Model:
https://huggingface.co/caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT

## Installation ##
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
				ASR Model: caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT
				Diarization Model: ./Data/Models/Diarization/model_03_25_2022_10_38_52
		
		3.) If you want to specify ASR model and/or Diarization model to use, add asr model and diarization model in that order after the audio file path.
			- docker run -v ${pwd}:/usr/src/app [Name of the image] -t [audiofile path] [asr model] [diarization model]
			e.g. docker run -v ${pwd}:/usr/src/app asrpipeline -t ./Data/Audio/Atest.wav caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT ./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52
		
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
			- docker run -v ${pwd}:/usr/src/app -m -d [diarization model to retrain] [epoch number] [Where to save the model]
			e.g. docker run -v ${pwd}:/usr/src/app -m -d ./Data/Models/Diarization/model_03_25_2022_10_38_52 30 ./Data/Models/Diarization

			3.1) -m tag determine that you are using retraining model feature.
			3.2) -d tag determine that you are retraining diarization model.
			3.3) if epoch number is not determined by the user, it is defaultly set to 30.

		4.) The new Diarization model will be stored in where you defined when you ran the run command.


## Use Cases ##

#### 1. Zero-shot time alignment transcription out of the box ####
***Make sure that Docker is installed.***
1. Open Terminal and navigate to the directory where you save this software using cd command.
   1. e.g. cd directory/where/you/saved/this
2. Call build command to build docker image.
   1. docker build -t asrpipeline .
3. Call run command to run docker image and transcribe an audio file.
   1. docker run -v ${pwd}:/usr/src/app asrpipeline -t ./Data/Audio/Atest.wav

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

#### 3. Training a new fine-tuned model ####
***Make sure that Docker is installed.***
1. Open Terminal and navigate to the directory where you save this software using cd command.
   1. e.g. cd directory/where/you/saved/this
2. Call build command to build docker image.
   1. docker build -t asrpipeline .
      1. If docker is already built, you can skip this.
3. Call run command to run docker image and retrain the model.
   1. For ASR model
      1. docker run -v ${pwd}:/usr/src/app asrpipeline -m -a ./Data/Retrain/ASR/ASRTraining.json ./Data/ASRTesting.json facebook/wav2vec2-large-960h-lv60-self facebook_fineTune_test 30
   2. For Diarization model
      1. docker run -v ${pwd}:/usr/src/app asrpipeline -m -d ./Data/Models/Diarization/model_03_25_2022_10_38_52 30 ./Data/Models/Diarization
For more detail, please check Docker section above.

#### 4. Using a fine-tuned model ####
***Make sure that Docker is installed.***
1. Open Terminal and navigate to the directory where you save this software using cd command.
   1. e.g. cd directory/where/you/saved/this
2. Call build command to build docker image.
   1. docker build -t asrpipeline .
      1. If docker is already built, you can skip this.
3. Call run command to run docker image and transcribe an audio file using the new model.
   1. docker run -v ${pwd}:/usr/src/app asrpipeline -t ./Data/Audio/Atest.wav caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT ./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52
      1. "caurdy/wav2vec2-large-960h-lv60-self_MIDIARIES_72H_FT" can be replaced with other ASR models.
      2. "./PyannoteProj/data_preparation/saved_model/model_03_25_2022_10_38_52" can be replaced with other Diarization models.
   

## Automatic Speech Recognition ##

### Setup ###

This model requires two pieces to operate

1. **The Wav2Vec2ForCTC model:** This piece takes in the audio data from a wav file and predicts what sounds it could be according to information it's stored from training.
	
		
2. **The Wav2Vec2ProcessorWithLM:** This piece both extracts the features necessary for prediction from your audio sample as well as decodes the output that the model predicts from a series of tokens into actual letters.

These two pieces can be set in a number of ways on an instantiated Wav2Vec2ASR object.

Let's say that we have instantiated our model as
	
	example_model = Wav2Vec2ASR()


We can set the above pieces in one of three ways. For each option you can use either a model from HuggingFace's repository 
(found [here](URL "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads&search=lm")) or a location of a file on your personal machine

1. **Provide both pieces at once**
		
    To do this, we simply call the function loadModel()
    
    		example_model.loadModel("patrickvonplaten/wav2vec2-base-100h-with-lm")
		
2. **Provide them separately with a pretrained processor**
	To set the model part we can use setModel() and then we can use processorFromPretrained() to set the processor from a different location. 
	This allows us to use more base models that don't have a Wav2Vec2ProcessorWithLM trained with them.
	
		example_model.setModel("facebook/wav2vec2-base-960h")
		example_model.processorFromPretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
			

3. **Provide them separately with a language model base.**
		
	Should you have an n-gram model for your language of choice that you'd like to create a processor from, you can do so using the createProcessor() method
		
	For this method, you will need to create a Wav2Vec2CTCTokenizer, a Wav2Vec2FeatureExtractor, and an n-gram Model
	
		example_tokenizer = Wav2Vec2CTCTokenizer('vocab.json', unk_token='[UNK]', pad_token='[PAD]',word_delimeter_token='|')
		example_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
		example_ngram = "local_n_gram"
		example_model.createProcessor(example_ngram, example_tokenizer, example_extractor)
		
	You can still set the model via:
		example_model.setModel("facebook/wav2vec2-base-960h")
		
**Some questions you may still have:**
			
1. What is an n-gram model?

    An n-gram model is a language model trained on a data set that helps predict what patterns are associated with what sounds. For more instructions on how to make your own, here are some external resources: [link 1](URL "https://huggingface.co/blog/wav2vec2-with-ngram#3-build-an-n-gram-with-kenlm")  [link 2](URL "https://masatohagiwara.net/training-an-n-gram-language-model-and-estimating-sentence-probability.html")
	
2. What is the Wav2Vec2CTCTokenizer?
				
	The tokenizer object turns the audio input into a series of tokens for later translation. 
	In order to do this, it requires:
		- A vocabulary file of all the characters it should recognize. 
		- The "unk_token" is the unknown token, or what to set something to if it doesn't recognize it. 
		- The "pad_token" is the padding token, or what it should use to pad things of mismatched length 

	For more insight on generating these things, please look [here](URL "https://huggingface.co/blog/fine-tune-wav2vec2-english#create-wav2vec2ctctokenizer") 
		
3. What is the Wav2Vec2FeatureExtractor?

	The feature extractor object extracts all the necessary information from each audio sample that it can then use for prediction. There are no external files, but to learn more about what each of the parameters do you can take a look [here](URL "https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor") 

### Prediction: ###

Once the model and processor are set, you can begin to transcribe your audio! To do so you need your audio file to be a .wav file
	
	example_audio = "test.wav"`

After this, all you have to do is call the predict() method
	
	transcription, confidence = example_model.predict(audioPath=example_audio)`

If you have already gotten your audio array via librosa.load() (found [here](URL "https://librosa.org/doc/latest/generated/librosa.load.html"))

	example_audio = librosa.load(example_audio, sr=16000)
	transcription, confidence = example_model.predict(audioArray=example_audio)

The "transcription" is the string text that resulted from the speech input, and the confidence is the confidence interval the model has that the transcription is correct.

### Training: ###

To train this model, you need to set the base model and processor using the above instructions. After this, you will need a training set, a testing set, and an output directory.
	
For both the training set and the testing set, you will need two .json files with identical format
	
For every audio input in the dataset, there should be the processed wav file (the output from librosa.load() ) under the tag “audio” and the labeled transcript under the label “text”. 
The transcript should be lowercase with all punctuation except apostrophes. 
For example (note: this example file does not contain valid data):
		
		{“text”:{“0”: "According to all known laws of aviation, there is no way a bee should be able to fly."},
    		“audio”:{“0”: [0.0,0.0,0.0,0.0,-0.0000610352,0.0,-0.0000610352,0.0000305176,-0.0001525879,0.0001220703,
		-0.000213623,0.0002441406,-0.0003967285,0.0005187988,-0.0010070801,0.0036621094,0.0098266602,
		0.0093078613,0.009765625,0.008605957,0.0099182129,0.0062561035,0.0065307617,0.0067443848,0.00390625,
		0.0025939941,0.003692627,-0.0018615723,0.005279541,]}}

After this, training can be accomplished via

	example_model.train("train_set.json", "test_set.json", "/testOutput/", epochs=30)

The number of epochs is the number of layers for training. The higher numbers of epochs generally produce better results. However, if you would like training to be less heavy, you can change the amount of epochs to whatever you would like.
	
After training is complete, the model and processor pieces of your example_odel object will have changed. To save these changes, we call the saveModel() method with the location where we'd like to save.

	example_model.saveModel("outputlocation/")

This way you don't lose the results of training and can use them for future predictions.

## Speaker Diarization ##

### 1. Operation: ###

   There are two parts that make diarization pipeline operate

   1. **The pyannote segmentation model:** This part takes the specific voice detection model to be used. it is the pytorch checkpoint '.ckpt' file. 
		
   2. **The hyper-parameter of diarization**: This piece will trigger the speaker diarization pipeline, it will activate model in pyannote pipeine so it can be trained or diarize the audio file. it should be a 'json' file.
	

### 2. Implement: ###

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

### 3. Prediction: ###

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

### 4. Training: ###

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
	
### 5. Optimization: ###
   The Optimization is used to continue improving the model performance.
   Usually, an trained model does not require further tuning of the pre-training hyperparameters. 
   However, if the model performs much better than expected in the pipeline, users can choose to improve the accuracy by Yonghua pipeline hyperparameters.
   Optimizing the pipeline can be done in this way:
		
      diarization.Optimization(model, dataset_name, num_opti_iteration=20, embedding_batch_size=16)

   num_opti_iteration The purpose is to specify the number of iterations the optimizer will perform. Usually the more iterations there are, the more accurate the hyperparameter will be returned. 
   It is worth noting that iterations greater than twenty are recommended, as too low an iteration may result in the model not finding better hyperparameters.

   Optimizing hyperparameters is a very time-consuming process. However, the hyperparameters do not need to be changed. 
   However, if this process is necessary, it is recommended that the optimization be done with a computer powered by a gpu.


## Named Entity Recognition ##
**What is Named Entity Recognition?**

- It is a task of identifying information in a text, especially proper nouns, and categorizing them into multiple different types.

**How to use**
- Import ner from NamedEntityRecognition.ner
  - ner.py is a python file under NamedEntityRecognition directory.
- Whenever you want to find Named Entity in a text, call the function "ner(String Object)".

      e.g. nerOutput = ner("My name is Johnson");

  - Currently, the function ner is using a model called "dslim/bert-base-NER"
    - If you want to use different model, please follow below to initialize the model and use it.

            from transformers import AutoTokenizer, AutoModelForTokenClassification
            from transformers import pipeline

            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

            nlp = pipeline("ner", model=model, tokenizer=tokenizer)
            example = "My name is Wolfgang and I live in Berlin"

            ner_results = nlp(example)
            print(ner_results)
      More Information on this link https://huggingface.co/dslim/bert-base-NER.


- The output of the function would look something like this

      entity': 'B-PER', 'score': 0.9987759, 'index': 5, 'word': 'Johnson', 'start': 10, 'end': 17
- Table below shows the entity categories, describing what each abbreviation stands for.

| Abbreviation | Description                                                                  |
|--------------|------------------------------------------------------------------------------|
| O            | Outside of a named entity                                                    |
| B-MIS        | Beginning of a miscellaneous entity right after another miscellaneous entity |
| I-MIS        | Miscellaneous entity                                                         |
| B-PER        | Beginning of a person’s name right after another person’s name               |
| I-PER        | Person’s name                                                                |
| B-ORG        | Beginning of an organization right after another organization                |
| I-ORG        | Organization                                                                 |
| B-LOC        | Beginning of a location right after another location                         |
| I-LOC        | Location                                                                     |


Named Entity Recognition is implemented when transcribing the audio file by default, but if you want to not use it, please follow the direction below.
  1. Open generateTranscription.py in Combine directory.
  2. Go to line 122 where it says "namedEntity = ner(punc_restore)"
  3. Delete that line.
  4. Go to line 128 (line 129 before deleting the line above) where it says ""Named Entity": str(namedEntity)".
  5. Delete the line and delete the comma on line 127.
  6. Now, your transcription won't have a named entity recognition result.
