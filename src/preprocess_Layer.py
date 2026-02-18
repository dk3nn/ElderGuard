import string 
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('','', string.punction))
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    cleaned_text = ''.join(words)
    return cleaned_text