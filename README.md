# MSULinguisticsCapstone
An open source Automatic Speech Recognition + Time Alignment docker container for translating .wav files into formatted text files for acoustic analysis 

##Installation
### Conda Environment ###
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

`C:\ mkdir ~/NewDirectory`

`C:\ cd ~/NewDirectory`

`C:\~\NewDirectory git clone https://github.com/caurdy/MSULinguisticsCapstone.git`

`C:\~\NewDirectory cd ./MSULinguisticsCapstone`

`C:\~\NewDirectory\MSULinguisticsCapstone python -m venv ./venv`

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone .\venv\Scripts\activate.bat`

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone python -m pip install --upgrade pip`

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone pip install rpunct`

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone pip install -r requirements.txt` (this will take a few minutes)

if you are on mac, additionally run:
   `C:\~\NewDirectory\MSULinguisticsCapstone pip install pypi-kenlm`

PyTorch needs to be reinstalled according to https://pytorch.org/ for your env if you have a gpu available and wish to use it.
Here are is most common type of installation. It would be wise to first uninstall pytorch and reinstall it using these cmds.

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone pip uninstall torch`
1. Windows or Linux, GPU & CPU
   1. `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
2. Mac, GPU & CPU
   1. See https://pytorch.org/ for usage of a GPU w/ Mac

To check if installation worked run the followings commands:

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone cd ./DataScience`

`(venv) C:\~\NewDirectory\MSULinguisticsCapstone\DataScience python TimeAlignment.py true demo`

**Note**: TimeAlignment is the script used for producing transcripts and is called with the following parameters/options.
`python TimeAlignment.py (1)use_gpu (2)directory`:
1. **use_gpu**: either 'false' or 'true'. Manually enables or disables gpu usage. Can't use a gpu if it isn't available (either physically or you didn't install the right torch)
2. **directory**: relative or absolute path a directory containing .wav file(s) you want to transcribe. It will ignore any file that doesn't end in '.wav'

If it worked, you'll see the file hello.json in the directory `MSULingusitics/Data/demo/Transcriptions`.

**Congrats!!! You're ready to transcribe some audio files!!!**

See the use case  **Zero-shot transcription** below for more details on running TimeAlignment.

## Docker ##

Eden fill in

## Use Cases ##

#### 1. Zero-shot transcription out of the box ####
#### 2. Training a new fine-tuned model (ASR) ####

There are two scripts needed to fine-tune a model. dataConfig.py and SpeechToTextHF.py

1. **dataConfig.py**, arguments: transcript directory, audio directory, Optional: minimum file size limit, maximum file size limit, train size.
   1. **Transcript Directory** 
      1. path (relative or absolute) to the directory containing .txt corrected transcripts.
      2. Fileames
   2. **Audio (.wav) Directory**
      1. 1. path (relative or absolute) to the directory containing .txt corrected transcripts
   3. **Minimum File Size Limit**
   4. **Maximum File Size Limit**
   5. **Train Size Proportion**
   
   &nbsp;**NOTE**: The filenames and format of the directory and files are crucial.<br/> Make sure the following format rules are followed. 
<br>The transcript directory has headers seperated by tabs in the order:
<br>Speaker, Header2, Chunk_start, Chunk_End, Text. 
<br>The transcript and corresponding audio file have the same filename (only differentiated by filetype and a 'corrected' flag)
<br> See the Demo folder within the Data Directory to see an example of the format and filenames.


2. **SpeechToTextHF.py** arguments: base/pretrained-model, training json, test json, num_epochs
    1. **Transcript Directory** 
    2. **Audio (.wav) Directory**
    3. **Minimum File Size Limit**
    4. **Maximum File Size Limit**
   
Setup train set, test set

Training params: model, train set, test set, num_epochs, 

#### 3. Training a new fine-tuned model (Speaker Diarization) ####

#### 4. Using a fine-tuned model ####

## Automatic Speech Recognition ##

Maria's txt 

## Speaker Diarization ##

Yichen's txt