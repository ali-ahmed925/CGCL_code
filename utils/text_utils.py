#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import os
import requests
import zipfile
from tqdm import tqdm


# In[2]:


# Load SpaCy model
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('[^a-zA-Z ]+', ' ', text)
    words = text.split()
    filtered = [lemmatizer.lemmatize(word) for word in words if word not in STOPWORDS]
    return ' '.join(filtered)


import os
import zipfile
import requests
from tqdm import tqdm

def download_glove(destination_folder='glove', dim=100):
    """
    Downloads and extracts GloVe embeddings in a Colab-compatible way.
    Returns the path to the specific .txt file (e.g., glove.6B.100d.txt).
    """
    os.makedirs(destination_folder, exist_ok=True)
    zip_path = os.path.join(destination_folder, 'glove.6B.zip')
    glove_file = os.path.join(destination_folder, f'glove.6B.{dim}d.txt')

    # Skip if already downloaded
    if os.path.exists(glove_file):
        print(f"GloVe file already exists at: {glove_file}")
        return glove_file

    url = 'http://nlp.stanford.edu/data/glove.6B.zip'

    print("Downloading GloVe embeddings...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        for data in tqdm(response.iter_content(1024), total=total_size // 1024, unit='KB'):
            f.write(data)

    print("Extracting embeddings...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    if os.path.exists(glove_file):
        print(f"GloVe downloaded and available at: {glove_file}")
        return glove_file
    else:
        raise FileNotFoundError("GloVe file was not found after extraction.")


# In[5]:


# Load GloVe Embeddings
def load_glove_embeddings(glove_file_path='glove.6B.100d.txt'):
    embeddings_index = {}
    with open(glove_file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index


# In[6]:


# Convert text to vector
def document_vector(text, embeddings_index, dim=100):
    words = text.split()
    vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


# In[7]:


# Save and load helper functions
def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# In[8]:


# Custom transformer for pipelines
class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embeddings_index, dim=100):
        self.embeddings_index = embeddings_index
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([document_vector(text, self.embeddings_index, self.dim) for text in X])
