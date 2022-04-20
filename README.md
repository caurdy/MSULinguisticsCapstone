# MSULinguisticsCapstone
Working towards an open source Automatic Speech Recognition + Time Alignment docker container for translating .wav files into formatted text files for acoustic analysis 


## Installation through Git and Command Prompt (Windows) ##

NOTE: You may need to use "python3" instead of "python" for these cmds

Requires Python 3.8 or later

Move into the directory you want the code to reside in

mkdir ~/NewDirectory
cd ~/NewDirectory

git clone https://github.com/caurdy/MSULinguisticsCapstone.git

cd ./MSULinguisticsCapstone

python -m venv ./venv
	
.\venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install rpunct

pip install -r requirements.txt (this will take a minute)

??? --upgrade fastpunct?


PyTorch needs to be reinstalled according to https://pytorch.org/ for your env if you have a gpu available and wish to use it.
Here are is most common type of installation. It would be wise to first uninstall pytorch and reinstall it using these cmds.

pip uninstall torch
1. Windows or Linux, GPU & CPU
   1. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
2. Mac, GPU & CPU
   1. See https://pytorch.org/ for usage of a GPU w/ Mac

   
## Docker ##

Eden fill in

## Use Cases ##

#### 1. Zero-shot transcription out of the box ####
#### 2. Training a new fine-tuned model ####
#### 3. Using a fine-tuned model ####

