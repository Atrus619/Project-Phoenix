# Adding cpu as first arg will run this on cpu
source .env
trap  "python3 clear_tmps.py" SIGINT

if [ "$1" == "cpu" ]
then
  echo "Starting bert on cpu..."
  bert-serving-start -model_dir $bert_model_dir -num_worker 1 -max_batch_size 8192 -max_seq_len 8 -prefetch_size 1024 -cpu
else
  echo "Starting bert on gpu..."
  bert-serving-start -model_dir $bert_model_dir -num_worker 1 -max_batch_size 8192 -max_seq_len 8 -prefetch_size 1024
fi