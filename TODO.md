For whoever takes on this project, or attempts to better it. Some suggestions.

The current loading process of datasets for training is slow and can be improved. Currently, the workflow is 
Audio/Transcript directories -> ConvertToDataFrame (pandas df -> json) -> Training loading (json -> pandas df -> HF Dataset) -> Train.
Which is highly inefficent and can be improved to save time.

Training auto hyperparameter tuning. Pretty simple idea, just needs to be implemented.

Currently, if diarization doesn't classify part of the audio as speech it won't be transcribed.
Audio with background noise or just unclear audio in general will suffer from this and not be transcribed.
This can be fixed by using Nemo's method of time alignment as a backup to the current method. Nemo
transcribes the full audio file and then will get timestamps for all individual words which can
be used to supplement the current time alignment algorithm to avoid missing the transcription of
long parts of audio which are poor quality.


