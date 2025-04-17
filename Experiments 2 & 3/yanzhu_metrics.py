import os
import re
from nltk import ngrams
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import string
import random
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from tqdm import tqdm
import networkx as nx
from nltk.tokenize import sent_tokenize
import stanza
from sklearn.manifold import TSNE
from scipy.stats import zscore
import scipy.stats as stats
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


def lexical_diversity(lines, language="french"):
    # Set the random seed for reproducibility
    random.seed(42)

    # Initialize the tokenizer and other utilities
    tokenizer = WordPunctTokenizer()

    # Load stopwords and punctuation
    #stopwords = set(nltk.corpus.stopwords.words(language))
    punctuation = set(string.punctuation)

    # Function to compute Type-Token Ratio (TTR)
    def ttr(tokens):
        return len(set(tokens)) / len(tokens)

    tokens = []
    bi_grams = []
    tri_grams = []
    lines = [l.replace('<newline>', '\n') for l in lines if l.strip() != ""]

    # Tokenize the lines and extract n-grams
    for l in lines:
        token = tokenizer.tokenize(l)
        token = [t.lower() for t in token if t not in punctuation]
        tokens += token
        bi_grams += ngrams(token, 2)
        tri_grams += ngrams(token, 3)

    # Randomly sample tokens and n-grams (if there are enough tokens)
    n_tokens = 200000 #modify according to dataset
    tokens = random.sample(tokens, min(n_tokens, len(tokens)))
    bi_grams = random.sample(bi_grams, min(n_tokens, len(bi_grams)))
    tri_grams = random.sample(tri_grams, min(n_tokens, len(tri_grams)))

    # Calculate the TTR for unigrams, bigrams, and trigrams
    unique_1 = ttr(tokens)
    unique_2 = ttr(bi_grams)

    unique_3 = 0
    if len(tri_grams) > 0:
        unique_3 = ttr(tri_grams)

    # Compute the average uniqueness across n-grams
    average_uniqueness = (unique_1 + unique_2 + unique_3) / 3

    #
    return average_uniqueness
"""{"tokens": tokens,
            "bi_grams": bi_grams,
            "tri_grams": tri_grams,
            "unique_1": unique_1,
            "unique_2": unique_2,
            "unique_3": unique_3,
            "average_uniqueness": average_uniqueness
            }

"""


def semantic_diversity(lines, language="french"):

    # Function to calculate the mean confidence interval
    def mean_confidence_interval(data, confidence=0.95):
        data = np.array(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
        return mean, margin_of_error

    # Parameters
    #n = 3000  # Number of sentences to sample
    
    lines = [l.replace('<newline>', '\n') for l in lines if l.strip() != ""]

    # Tokenize sentences
    sentences = []
    for t in lines:
        sents = sent_tokenize(t)
        sentences += sents

    # Randomly sample sentences
    #random.seed(42)
    #sentences = random.sample(sentences, n)

    # Load the sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:1')
    
    # Encode sentences into embeddings
    sentence_embeddings = model.encode(sentences, batch_size=128, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
    
    # Convert embeddings to tensor and move to CUDA
    x = torch.tensor(sentence_embeddings).to(torch.float).to("cuda:0")

    # Clear memory by moving model and embeddings to CPU
    model.to('cpu')
    del model
    sentence_embeddings.to('cpu')
    del sentence_embeddings
    torch.cuda.empty_cache()

    # Compute cosine similarity matrix
    with torch.no_grad():
        x_cosine_similarity = torch.nn.functional.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)

    # Extract non-diagonal elements from the cosine similarity matrix
    mask = ~torch.eye(x_cosine_similarity.size(0), dtype=bool)
    non_diag_elements = x_cosine_similarity[mask]
    non_diag_array = non_diag_elements.cpu().numpy()
    
    # Rescale cosine similarity values to range [0, 1]
    res = (1 - non_diag_array) / 2
    
    # Calculate mean and confidence interval
    mean, error = mean_confidence_interval(res)

    return mean
#{"mean": mean, "error": error}