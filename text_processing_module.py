#install nltk
# pip install nltk

#browse the available packages
# import nltk #import nltk
# nltk.download()

#import all from book module
# from nltk.book import *

#import nltk
import nltk
#import word_tokenize from nltk
from nltk.tokenize import word_tokenize
#import sent_tokenize from nltk
from nltk.tokenize import sent_tokenize
#import nltk stopwords
from nltk.corpus import stopwords
#import PorterStemmer, for stemming words
from nltk.stem import PorterStemmer
#import wordnet lemmatizer, for lemmatization
from nltk.stem import WordNetLemmatizer
# Lemmatize with POS Tag
from nltk.corpus import wordnet
import pandas as pd

# tokenize words
def tokenize_words(text):
    """
    Tokenize the raw text passed as an arguments into tokens and returns a list of tokens

    Arguments:
    text: raw text, strings

    Returns:
    words: list of tokens
    """

    # tokenize words
    words = word_tokenize(text)
    
    return words

# tokenize sentence
def tokenize_sentence(text):
    """
    Tokenize the raw text passed as an arguments into sentence and returns a list of sentence


    Arguments:
    text: raw text, strings

    Returns:
    sentences: list of  sentence
    """

    #tokenize sentence
    sentences = sent_tokenize(text)

    return sentences

# lower casing
def lower_casing(text):
    """
    Obtained the lower case version and passed lower case version of text 

    Arguments:
    text: raw text, strings

    Returns:
    lower_text: string, representing lower case of raw text
    """

    # lower casing
    lower_text = text.lower()

    return lower_text

# stop words removal
def remove_stop_words(text):
    """
    Removes stop words from text passed as an argument and returns list of representing text 
    without stop words
    
    Arguments:
    text: raw text, strings

    Returns:
    tokens_without_sw: list of tokens without stopwords
    """
    # nltk stopwords
    stop_words = set(stopwords.words('english'))

    # tokenize words
    word_tokens = word_tokenize(text)

    # filter stopwords
    tokens_without_sw = [word for word in word_tokens if word not in stop_words]

    return tokens_without_sw

# stemming
def stemming(text):
    """
    - Stemming is a process of transforming a word to its root form. 
    - Input to the Stemming is tokenized words.
    - Accpets text as a inputs and returns list of stems for each token in the text
    
    Arguments:
    text: raw text, strings

    Returns:
    token_stem_list: list containing toke with its stem
    """
    # instantiate PorterStemmer 
    ps = PorterStemmer()

    # tokenize
    words = word_tokenize(text)

    # stemming
    token_stem_list = [(word, ps.stem(word))for word in words]

    return token_stem_list

# Lemmatization
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatization(text):
    """
    Find and returns the lemma for each token in the given text

    Arguments:
    text: raw text, strings

    Returns:
    token_lemma_list: list containing token with its lemma
    """

    # 1. Init Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # tokenize words
    tokens = word_tokenize(text)

    # 1. Init Lemmatizer
    lemmatizer = WordNetLemmatizer()

    token_lemma_list = [(token, lemmatizer.lemmatize(token, get_wordnet_pos(token))) for token in tokens]

    return token_lemma_list

# Named Entity Recognizer
def named_entity_recognizer(text, binary_entity=True):
    """
    Identify words which are named entities in a given text.

    Arguments:
    text: raw text, string
    binary_entity: Flag whther to obtains entity labels or not, 
                    True-> no entity labels, Default=True
                    False-> entity with labels

    Returns:
    entity_labels: list of tuples containing all the entities
    """
    # tokenize to words
    words = nltk.word_tokenize(text)

    # part of speech tagging
    pos_tags = nltk.pos_tag(words)

    # check nltk help for description of the tag
    # print(nltk.help.upenn_tagset("NNP"))
    
    # ne_chunk, Binary=True
    chunks = nltk.ne_chunk(pos_tags, binary=binary_entity) # either NE or not NE
    
    # obtain list of entities
    entities =[]
    labels =[]
    for chunk in chunks:
        if hasattr(chunk,'label'):
            #print(chunk)
            entities.append(' '.join(c[0] for c in chunk))
            labels.append(chunk.label())
            
    entities_labels = list(set(zip(entities, labels)))

    return entities_labels

# tokenize  words
sample_text = "Oh man, this is pretty cool. We will do more such things."
words = tokenize_words(sample_text)
# print tokens
print("\n**Word tokens**\n", words)

# tokenize sentence
sentences = tokenize_sentence(sample_text)
# print sentences
print("\n**Sentence tokens**\n", sentences)


#lower casing
print("\n**Lower Casing**\n", lower_casing(sample_text))

# stopwords removal
filtered_sentence = remove_stop_words(sample_text)
print("\n**filtered sentence without stop words:**\n", filtered_sentence)

# Stemming
sample_text = "The striped bats are hanging on their feet for best"
token_stem_list = stemming(sample_text)
#display
print("\n**Stemming**")
for token_stem in token_stem_list:
    print(token_stem[0], '-->', token_stem[1])

# Lemmatization
sample_text = "The striped bats are hanging on their feet for best"
token_lemma_list = lemmatization(sample_text)
#printing
print("\n**Lemmatization**")
for token_lemma in token_lemma_list:
    print(token_lemma[0], '-->', token_lemma[1])

#named entites recognition
text = "Apple acquired Zoom in China on Wednesday 6th May 2020.\
This news has made Apple and Google stock jump by 5% on Dow Jones Index in the \
United States of America"
#call named entity recognizer
entities_labels = named_entity_recognizer(text, False)
#display entities in tabular  format
entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities","Labels"]
print("\n**Displaying Entities in Tabular format:**\n", entities_df)




