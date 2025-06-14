from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import nltk

# Download tokenizer model (only once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Sample data
sentences = [
    "I love learning about machine learning",
    "Natural language processing is a part of AI",
    "Word2Vec helps convert words to numbers",
    "Vectors can capture word meanings"
]

# Tokenize each sentence
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=3, min_count=1, workers=2)

# Display vector for a word
print("Vector for 'learning':")
print(model.wv['learning'])

# Find similar words
print("\nWords similar to 'learning':")
print(model.wv.most_similar('learning'))
