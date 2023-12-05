import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer(language='portuguese')
def tokenize(sentence):
    """
    dividir a frase em um conjunto de palavras/tokens
    um token pode ser uma palavra, um caractere de pontuação ou um número
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    derivação = encontre a forma raiz da palavra
    examples:
    words = ["organizar", "organiza", "organizando"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return snow_stemmer.stem(word)


def bag_of_words(tokenized_sentence, words):
    """
    retornar conjunto de palavras:
    1 para cada palavra conhecida que existe na frase, 0 caso contrário
    example:
    sentence = ["oi", "como", "voce", "esta"]
    words = ["oi", "ola", "eu", "voce", "tchau", "obrigado", "legal"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
