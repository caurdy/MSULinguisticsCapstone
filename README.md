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

Eden fill in

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

Yichen's txt