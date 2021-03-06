import os

basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    GCP_API_KEY = os.environ.get('GCP_API_KEY') or 'get-your-own-gcp-api-key'

    # MongoDB
    db = 'phoenixdb'
    collection = 'post'

    # Chatbot server
    chatbot_host = os.environ.get('chatbot_host') or 'localhost'
    chatbot_port = os.environ.get('chatbot_port') or 8765

    # Secrets
    ipvanish_password = os.environ.get('ipvanish_password') or 'you-will-never-guess'
    sudo_password = os.environ.get('sudo_password') or 'good-luck'
    ip = os.environ.get('ip') or '127.0.0.1'

    # Scraper
    min_pause = 0.5  # in seconds
    max_pause = 2

    jobs = [
        "machine learning engineer",
        "product development",
        "data scientist",
        "strategy and operations",
        "AI scientist"
    ]

    cities = [
        "Chicago, IL",
        "New York, NY",
        "San Francisco, CA",
        "Boston, MA",
        "Denver, CO",
        "San Diego, CA"
    ]

    sources = {
        'indeed': 5,
        'monster': 2
    }

    log_folder = 'logs'
    validation_log_folder = 'logs/validations'
    tb_log_folder = 'logs/tensorboard'
    checkpoint_log_folder = 'logs/checkpoints'
    pickle_log_folder = 'logs/pickles'

    [os.makedirs(folder, exist_ok=True) for folder in (log_folder, validation_log_folder, tb_log_folder, checkpoint_log_folder)]

    scrape_log_name = 'scrape_log'
    scrape_error_log_name = 'scrape_error_log'

    job_description_link_fail_msg = 'Job description unavailable'
    job_description_parse_fail_msg = 'Failed to parse.'

    fail_wait_time = 60  # seconds
    max_retry_attempts = 3  # attempts

    # Chatbot
    chatbot_training_log_name = 'chatbot_training'
    ner_and_intent_training_data_path = 'src/data/intent_and_ner/Intent Training Examples_JL.xlsx'
    bert_dir = os.environ.get('bert_model_dir')
    bert_port = 5555
    ner_jar_path = 'downloads/stanford-ner.jar'
    serialized_model_dir = 'src/pipeline/serialized_models'
    default_model_name = 'model'  # NER file will end in _ner.ser.gz, and interpreter dict will end in _interpreter_dict.pkl

    # NER Training - see docstring in src/models/ner/train.py
    ner_prop_path = 'src/models/ner/config.prop'
    ner_training_num_cv = 5
    ner_training_folder = 'logs/ner/cv'
    ner_full_train_path = os.path.join(ner_training_folder, 'full_train.tsv')

    # Intent Training - see docstring in src/models/intent/train.py
    intent_training_num_cv = 5
    intent_follow_up_training_num_cv = 3

    # Smalltalk
    interact_config = {
        'max_history': 2,
        'max_length': 20,
        'min_length': 1,
        'temperature': 0.7,
        'top_k': 0,
        'top_p': 0.9,
        'no_sample': False,
        'random_pause': None
    }

    # Redis
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'

    templates_folder = 'app/templates'
    user_output_folder = 'app/static/users'
