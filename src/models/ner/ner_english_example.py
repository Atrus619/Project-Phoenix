import nltk
from nltk.tag.stanford import StanfordNERTagger

sentence = u"Twenty miles east of Reno, Nev., " \
    "where packs of wild mustangs roam free through " \
    "the parched landscape, Tesla Gigafactory 1 " \
    "sprawls near Interstate 80."

jar = 'logs/ner/stanford-ner-2018-10-16/stanford-ner.jar'
model = 'logs/ner/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'

# Prepare NER tagger with english model
ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

# Tokenize: Split sentence into words
words = nltk.word_tokenize(sentence)

# Run NER tagger on words
print(ner_tagger.tag(words))
