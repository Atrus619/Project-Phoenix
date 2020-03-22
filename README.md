# Project-Phoenix
Chat bot to assist users in analyzing specific slices of the job market

## Recommend using a virtual env
1. Set up virtual env
2. Add https://pypi.python.org/simple and https://github.com/NVIDIA/apex/ to repositories
3. Install requirements: `pip install -r requirements.txt`

## Download Apex
https://github.com/NVIDIA/apex
* Follow repo instructions to set up (2-3 steps)
* Likely will be able to skip this step if you followed the first step correctly

## Download Bert as Service model
https://github.com/hanxiao/bert-as-service
* Download model of choice from the repo and unzip into gitignored downloads directory

## Download stanford NER model
https://nlp.stanford.edu/software/CRF-NER.shtml
* Make sure to place stanford-ner.jar file into gitignored downloads directory

## Be sure to set up SmallTalk model first time (downloading can take a bit of time)
`import src.models.SmallTalk.utils as stu`

`stu.download_pretrained_small_talk_model()`
## Set up .env file
See .sample_env for a list of entries
