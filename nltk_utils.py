import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer

# Inicializa o stemmer para português
stemmer = SnowballStemmer(language='portuguese')

def tokenize(sentence):
    """
    Divide uma sentença em palavras/tokens
    """
    return sentence.lower().split()


def stem(word):
    """
    Aplica stem (radicalização) na palavra
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Retorna um vetor de presença de palavras (Bag of Words)
    """
    # Aplica stem nas palavras da sentença
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Cria vetor com zeros para cada palavra do vocabulário
    bag = np.zeros(len(words), dtype=np.float32)

    # Marca 1 se a palavra está presente na sentença
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0

    return bag
