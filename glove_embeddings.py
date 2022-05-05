from keras.layers import TextVectorization
from keras.layers import Embedding
from keras.initializers import Constant
import tensorflow as tf
import numpy as np
import os
import pickle

# Load the GloVE embeddings
path_to_glove_file = "glove.6B.300d.txt"
embedding_dim = 300 ## 300 dimensions (as dictated by the glove file)

embeddings_index = {}
with open(path_to_glove_file) as f:
  for line in f:
    word, coefs = line.split(maxsplit=1)
    coefs = np.fromstring(coefs, "f", sep=" ")
    embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))

# Create text vectorizer
def create_text_vectorizer_for_train_samples(train_set):
    train_samples = [utterance for utterance in train_set['UTTERANCE']]

    vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=300)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128) ## Read batches of 128 samples
    vectorizer.adapt(text_ds)
    return vectorizer

def create_glove_embedding_matrix_for_vectorizer(vectorizer):
    # Create embedding matrix with our vocabulary indexed on glove
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    num_tokens = len(voc) 
    hits = 0 ## number of words that were found in the pretrained model
    misses = 0 ## number of words that were missing in the pretrained model

    # Prepare embedding matrix for our word list
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # Words not found in embedding index will be all-zeros.
          # This includes the representation for "padding" and "OOV"
          embedding_matrix[i] = embedding_vector
          hits += 1
        else:
          misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    
    return embedding_matrix

def create_glove_embedding_layer_for_matrix(embedding_matrix, num_tokens):
    glove_embedding_layer = Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix), trainable=False)
    return glove_embedding_layer

def create_and_pickle_glove_embeddings(iemocap_data, output_file):
    iemocap_prefix = "iemocap_"
    iemocap_split_prefix = "iemocap_split_"
    glove_embedding_prefix = "glove_embedding_"

    print("Creating vectorizer")
    text_vectorizer = create_text_vectorizer_for_train_samples(iemocap_data)
    
    print("Creating embedding matrix")
    embedding_matrix = create_glove_embedding_matrix_for_vectorizer(text_vectorizer)
    
    print("Dumping to pickle file")
    with open(output_file, 'wb') as f:
        pickle.dump(embedding_matrix, f)
    
    print("Success!")

# Load pickled glove embedding matrix
def load_glove_embedding_matrix(filename):
    glove_embedding_matrix = np.array(np.load(filename, allow_pickle=True))
    return glove_embedding_matrix
