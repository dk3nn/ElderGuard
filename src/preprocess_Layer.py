import string 
import nltk
from nltk.corpus import stopwords


#nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Conversion to lowercase, remove punctuation, and remove stopwords
def clean_text(text):
    text = text.lower()
   
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    return ' '.join(words)