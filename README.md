# Project-Phoenix
Chat bot to assist users in analyzing specific slices of the job market

## Download Apex
https://github.com/NVIDIA/apex
* Follow repo instructions to set up (2-3 steps)

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
See .sample_env for getting started
