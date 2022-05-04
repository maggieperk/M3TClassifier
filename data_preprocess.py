# Turn code into categorical data
# Create one hot embeddings of the label values (0, 1, 2)
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import numpy as np

# Full set of dialogue act labels
da_labels = ['s', 'q', 'o', 'ans', 'c', 'ag', 'b', 'dag', 'oth', 'a', 'ap', 'g']
da_encoder = preprocessing.LabelEncoder()
da_encoder.fit(da_labels)

emotion_labels = ['xxx', 'fru', 'neu', 'ang', 'sad', 'exc', 'hap', 'sur', 'fea', 'oth', 'dis']
emotion_encoder = preprocessing.LabelEncoder()
emotion_encoder.fit(emotion_labels)

def convert_da_labels_to_categorical(dialog_acts):
    num_labels = da_encoder.transform(dialog_acts)
    cat_labels = to_categorical(num_labels)
    return cat_labels

def convert_cat_da_to_string(cat_das):
    num_labels = np.argmax(cat_das, axis=-1)
    return da_encoder.inverse_transform(num_labels)

def convert_emot_labels_to_categorical(emot_data):
    num_labels = emotion_encoder.transform(emot_data)
    cat_labels = to_categorical(num_labels)
    return cat_labels

def convert_cat_emot_to_string(cat_emot_data):
    num_labels = np.argmax(cat_emot_data, axis=-1)
    return emotion_encoder.inverse_transform(num_labels)
