import nltk
from nltk.corpus import stopwords
from contractions import fix
from nltk.stem import PorterStemmer
import emoji

# Initialize resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to remove HTML tags
def remove_html_tags(text):
    import re
    return re.sub(r'<.*?>', '', text)

# Function to remove punctuation
def remove_punctuation(text):
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to convert text to lowercase
def lowercase_text(text):
    return text.lower()

# Function to remove extra whitespaces
def remove_extra_whitespace(text):
    return ' '.join(text.split())

# Function to expand contractions
def expand_contractions(text):
    return fix(text)

# Function to apply stemming
def apply_stemming(text):
    return ' '.join(stemmer.stem(word) for word in text.split())

# Function to replace emojis with text descriptions
def replace_emojis(text):
    return emoji.demojize(text, delimiters=("", " "))

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stop_words)

# Main cleaning function that applies all preprocessing steps
def clean_review(text):
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = lowercase_text(text)
    text = remove_extra_whitespace(text)
    text = expand_contractions(text)
    text = replace_emojis(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    return text
