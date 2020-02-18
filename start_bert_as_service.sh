source .env
trap  "python3 clear_tmps.py" SIGINT
bert-serving-start -model_dir $bert_model_dir -num_worker 1 -max_batch_size 8192 -max_seq_len 8 -prefetch_size 1024
