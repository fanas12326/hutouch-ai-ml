import nltk
import spacy
import subprocess

nltk.download('punkt')
nltk.download('stopwords')
# Download spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])