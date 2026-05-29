from nltk.tokenize import word_tokenize

def tokenize_text(text):

    try:
        return word_tokenize(text)

    except:
        return text.split()