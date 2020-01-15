import os

basedir = os.path.abspath(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(basedir, '.env'))

except ModuleNotFoundError:
    pass


class Config:
    # MongoDB
    db = 'phoenixdb'
    collection = 'post'

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

    fail_wait_time = 60  # seconds
    max_retry_attempts = 3  # attempts

    # Chatbot
    chat_bot_training_log_name = 'chat_bot_training'
    ner_and_intent_training_data_path = 'src/data/intent_and_ner/Intent Training Examples_JL.xlsx'
    bert_dir = 'downloads/cased_L-24_H-1024_A-16/'
    ner_jar_path = 'downloads/stanford-ner.jar'

    # NER Training - see docstring in src/models/ner/train.py
    ner_prop_path = 'src/models/ner/config.prop'
    ner_model_path = 'src/pipeline/serialized_models/ner-model.ser.gz'
    ner_training_num_cv = 5
    ner_training_folder = 'logs/ner/cv'
    ner_full_train_path = os.path.join(ner_training_folder, 'full_train.tsv')

    # Intent Training - see docstring in src/models/intent/train.py
    intent_training_num_cv = 5
    intent_follow_up_training_num_cv = 3
    # ner_model_path = 'src/models/ner/ner-model.ser.gz'  # Defined above in NER Training section
    # ner_jar_path = 'logs/ner/stanford-ner-2018-10-16/stanford-ner.jar'  # Defined above in NER Training section
    default_interpreter_dict_output_path = 'src/pipeline/serialized_models/interpreter_dict.pkl'
    valid_intents = {
        'small_talk': 'Smalltalk',
        'end_of_conversation': 'Conclusions',
        '[job]_in_[location]': '[JOB]_in_[LOCATION]',
        'skills_for_[job]': 'skills_for_[JOB]'
    }

    valid_follow_up_intents = {
        'Rejection',
        'Acceptance'
    }

    # Entities
    entities = {
        'J': 'job',
        'L': 'location'
    }

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
