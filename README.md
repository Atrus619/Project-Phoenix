# Project-Phoenix
Chat bot to assist users in analyzing specific slices of the job market

## Download Apex
https://github.com/NVIDIA/apex

## Download Bert as Service model
https://github.com/hanxiao/bert-as-service

## Download stanford NER model
https://nlp.stanford.edu/software/CRF-NER.shtml

## Be sure to set up SmallTalk model first time (downloading can take a bit of time)
`import src.models.SmallTalk.utils as stu`

`stu.download_pretrained_small_talk_model()`
## Set up .env file
Needs the following:
sudo_password=???
ipvanish_password=??? (optional)
ip=???

chatbot_host=localhost
chatbot_port=3000

bert_model_dir=path_to_bert_download

GCP_API_KEY=enter_gcp_api_key_here
