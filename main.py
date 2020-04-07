from src.classes.ChatBot import ChatBot
from src.classes.Interpreter import Interpreter
from src.classes.Policy import Policy
from src.classes.SmallTalk import SmallTalk
import src.models.SmallTalk.utils as stu
from config import Config as cfg
import pickle as pkl

# Interpreter, pretrained elsewhere
with open(cfg.default_interpreter_output_path, 'rb') as f:
    interpreter = pkl.load(f)

# SmallTalk
dir = stu.download_pretrained_small_talk_model()
small_talk = SmallTalk(name='Test_SmallTalk_Pretrained', model_type='openai-gpt', model_name=dir, opt_level='O1')

# Policy
policy = Policy(small_talk=small_talk)
small_talk.print_personality(policy.small_talk_personality)

# ChatBot
chatbot = ChatBot(interpreter=interpreter,
                  policy=policy)

# Interact
chatbot.interact()

# Debugging
print(chatbot.conversation_history)
x = chatbot.conversation_history.get_list_of_conversation_latest_n_exchanges(4)
for y in x:
    print(y)

from src.classes.Scraper import Scraper
from src.visualization.visualize import create_wordcloud
from src.classes.JobPostingExtractor import JobPostingExtractor

scraper = Scraper()
scraped_jobs = scraper.scrape_page_indeed('Actuary', 'Chicago', page=1, vpn=True)
scraped_jobs.append(scraper.scrape_page_indeed('Consultant', 'Chicago', page=2, vpn=True))
print(scraped_jobs)

# create_wordcloud(scraped_jobs)
# create_wordcloud(scraped_jobs, type='job_title')

jpe = JobPostingExtractor(scraped_jobs[3])
jpe.set_encodings(8)
print(jpe.extract_salary())
print(jpe.extract_required_years_experience())
print(jpe.extract_required_degree())
print(jpe.extract_travel_percentage())
print(jpe)

# List of todos!
# TODO: Begin working on logic and things for the selecting results (May be able to just pass links directly in the original response instead of training a classifier to respond here)
"""
Plan for serving up content:
1. Look up node express routing CHECK
2. Build out process for node to give access to stored objects via get request CHECK
3. Chatbot sends message with links encoded (build function / class for this) CHECK
4. Server decodes message and turns links into actual links on the page CHECK / NEW ITEM TO DO
5. IFRAME to open up links inside DOM CHECK
6. Make sure I have a firm grip on what exactly the user will be getting back
NEXT TODO:
    Fix follow up request
    Make user / bot text more obvious - aligned, add tag for the sender, etc.
    Also retest local running of bot
    Other formats like table! make pretty and then make template for python
7. Clean up page to be more aesthetically pleasing / make sense CHECK
8. Add a header to website?
"""

# TODO: Containerize application?
# TODO: Single file to run app

# TODO: Allow user to cancel request while in progress to perform a different query
# TODO: May want to split state for seeking additional information into two (initially seeking new information and seeking information for the second + time). This will cut down on redundancy in the convo.
# TODO: Seems to get stuck if not already in vpn mode???
# TODO: Retain postings with salary information, and sort in order from highest to lowest (can show to user as a feature)
# TODO: Add extract benefits (will be more challenging to implement)

from src.classes.Scraper import Scraper
Scraper().rotate_ip()

import src.visualization.visualize as vizz
from src.classes.Visualizer import Visualizer
from src.classes.Extractions import Extractions
from src.classes.JobPostingExtractor import JobPostingExtractor
viz = Visualizer()
test = viz.process_job_in_location('Software Engineer', 'San Francisco')
viz.is_task_complete()

job, location = 'Software Engineer', 'San Francisco'
# vizz.setup_extractions_logger(job='Actuary', location='Chicago')
extractions = Extractions(required_years_experience=5, required_degree=5, travel_percentage=1, salary=5)
extractions.gather(job, location, ngram_size=8, ngram_stride=3, vpn=True)

jpe = JobPostingExtractor(extractions.scraped_jobs[-5])
jpe.set_encodings(8)
print(jpe.extract_salary())
print(jpe.extract_required_years_experience())
print(jpe.extract_required_degree())
print(jpe.extract_travel_percentage())
print(jpe)

import re
salary_pattern = r'(salary|compensation)'
match = re.findall(salary_pattern, str(jpe))
print(len(match))

similarities = jpe._get_similarities(reference_sentences=('education: bachelors degree, masters degree, phd or higher', ), threshold=0.89)

from src.classes.Scraper import Scraper
x = Scraper().scrape_page_indeed('Actuary', 'Chicago', 1, True)

# Running server
./start_bert_as_service.sh
rq worker extractor
python3 run_chatbot.py -n dajan2 -s
nodemon app/app.js
