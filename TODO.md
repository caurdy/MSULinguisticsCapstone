For whoever takes on this project, or attempts to better it. Some suggestions.

The current loading process of datasets for training is slow and can be improved. Currently, the workflow is 
Audio/Transcript directories -> ConvertToDataFrame (pandas df -> json) -> Training loading (json -> pandas df -> HF Dataset) -> Train.
Which is highly inefficent and can be improved to save time.

Training auto hyperparameter tuning. Pretty simple idea, just needs to be implemented.

