from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
import os

os.system('src/models/SentenceEmbeddings/start_bert_as_service.sh')

bc = BertClient()
test = bc.encode(['Hello, can you please tell me about a software engineer in San Francisco?',
                 'Hi, would you mind giving me information on a software engineer in San Francisco?',
                 'I strongly dislike software engineers in San Francisco.'])

cosine_similarity(test[0].reshape(1, -1), test[1].reshape(1, -1))
cosine_similarity(test[0].reshape(1, -1), test[2].reshape(1, -1))
cosine_similarity(test[1].reshape(1, -1), test[2].reshape(1, -1))
