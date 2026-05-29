from sklearn.feature_extraction.text import CountVectorizer

def generate_bow(text):

    vectorizer = CountVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform([text])

    return vectorizer, matrix