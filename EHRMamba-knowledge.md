# Use of this file

We're going to use this file to share any knowledge about the EHRMamba architecture

## What script are used for loading and processing the dataset?

Here are some key scripts responsible for loading and processing the dataset in the odyssey/data directory:

- [processor.py](https://github.com/VectorInstitute/odyssey/blob/main/odyssey/data/processor.py): Processes patient sequences based on tasks and splits them into train-test-finetune datasets.
- [tokenizer.py](https://github.com/VectorInstitute/odyssey/blob/main/odyssey/data/tokenizer.py): Tokenizes event concepts using the HuggingFace library.
- [seq/_tokens.py](https://github.com/VectorInstitute/odyssey/blob/main/odyssey/data/seq/_tokens.py): Generates tokens for patient sequences.
- [seq/_events.py](https://github.com/VectorInstitute/odyssey/blob/main/odyssey/data/seq/_events.py): Processes events for patient sequence generation.


## How phisionet is structured?

The PhysioNet 2012 dataset is structured as follows:
- RecordID -> no embeddings, since we're treating each visit as a unique patient
- age -> 
