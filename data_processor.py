import nltk
import re
import string

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as sw

lemmatizer, stopwords, spell = None, None, None


def init_nlp_dependencies():
    download_nltk_dependencies()
    global lemmatizer
    global stopwords
    global spell
    lemmatizer = WordNetLemmatizer()
    stopwords = sw.words('english')
    # spell = SpellChecker()


def download_nltk_dependencies():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_data(text):
    # remove punctuation and lowercase
    # remove stop words
    filtered_tokens = []
    for token in word_tokenize(text):
        token = re.sub(r'\b[0-9]+\b', '', token)
        token = token.translate(str.maketrans('', '', string.punctuation))
        token = token.lower()
        if len(token) > 0 and token not in stopwords:
            filtered_tokens.append(token)

    # spell checking
    # for i, token in enumerate(filtered_tokens):
    #     misspelled = self.spell.unknown(token)
    #     if misspelled:
    #         corrected = self.spell.correction(token)
    #         if corrected is not None:
    #             filtered_tokens[i] = corrected

    # lemmatization
    tagged_tokens = pos_tag(filtered_tokens)

    # Lemmatize based on POS tags
    lemmatized_words = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    processed_text = ' '.join(lemmatized_words)
    return processed_text
