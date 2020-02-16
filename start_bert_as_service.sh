source .env
bert-serving-start -model_dir $bert_model_dir -num_worker=1 -max_batch_size 1024