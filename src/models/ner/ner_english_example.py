import nltk
from nltk.tag.stanford import StanfordNERTagger

jar = 'logs/ner/stanford-ner-2018-10-16/stanford-ner.jar'

# Test newly trained model on some made up text
newly_trained_model = 'src/models/ner/ner-model.ser.gz'
ner_tagger = StanfordNERTagger(newly_trained_model, jar, encoding='utf8')

test_sentence = 'Please tell me about a computer scientist in Texas.'
words = nltk.word_tokenize(test_sentence)
print(ner_tagger.tag(words))
