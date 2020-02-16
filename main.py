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
# scraped_jobs.append(scraper.scrape_page_indeed('Actuary', 'Chicago', page=2, vpn=True))
# print(scraped_jobs)
#
# create_wordcloud(scraped_jobs)
# create_wordcloud(scraped_jobs, type='job_title')

jpe = JobPostingExtractor(scraped_jobs[8])
jpe.set_encodings(8)
print(jpe.extract_salary())
print(jpe.extract_required_years_experience(reference_sentence='5+ years of experience', threshold=0.9))
print(jpe)
print(jpe.extract_required_degree())
