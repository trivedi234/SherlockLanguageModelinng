
##############data loading part#################

import re
import pandas as pd
import csv
import numpy as np
from numpy import array
from numpy import zeros
import nltk
from collections import Counter
from itertools import dropwhile
from nltk.tokenize import word_tokenize
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

nltk.download('punkt')


file = open("hound-train.txt", "r", encoding="utf8")
train_corpus = re.findall(r"(?<=I.  )(To.*)(?=End of the Project Gutenberg EBook)",file.read().replace('\n', ' ').replace('\r', ''))[0]
file = open("hound-test.txt", "r", encoding="utf8")
test_corpus = file.read()


#######data processing###############################################################
##normalize by removing symbols and convert to lower case 
train_text = re.sub("[^a-zA-Z0-9\.|\?|!]", " ", train_corpus.lower())
test_text =  re.sub("[^a-zA-Z0-9\.|\?|!]", " ", test_corpus.lower())

###substitue non-frequent words to <unk> and create a closed vocabulary by substituing unknown words in test_set by <unk> 
train_vocab = word_tokenize(train_text.lower())
train_dict = Counter(train_vocab)
for key, count in dropwhile(lambda key_count: key_count[1] >= 2, main_dict.most_common()):
    del main_dict[key]
frequent_words = list(main_dict.keys())
test_vocab = word_tokenize(test_text.lower())
to_remove = [l for l in train_vocab if l not in frequent_words]
unknowns = [l for l in test_vocab if l not in frequent_words]

def replace_unk(vocab,exceptions):
  new_set =[]
  for l in vocab:
    if l not in exceptions:
      new_set.append(l)
    else:
      new_set.append("<unk>")
  new_data= " ".join(new_set)
  return(new_data)
##prepare trainning and test corpus
train_text = replace_unk(list1,to_remove)
test_text = replace_unk(list2,unknowns)

#######preparation of trainning and test sets#################################################

tokenizer = Tokenizer()
tokenizer.fit_on_texts([train_text])
vocab_size = len(tokenizer.word_index) + 1

def seq_gen(data,max_length = None):
  seq =[]
  encodings = []
  sentences = re.split(r"\.|\?|!", data)
  for line in sentences:
    encoded = tokenizer.texts_to_sequences([line])[0]
    encodings.append(encoded)
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        seq.append(sequence)
  if max_length is None:
    max_length = max([len(s) for s in seq])
  seq = pad_sequences(seq, maxlen=max_length, padding='pre')
  return seq, encodings


train_data, train_encodings = seq_gen(train_text)
test_data, test_encodings = seq_gen(test_text,max_length=len(train_data[0]))

X, y = train_data[:,:-1],train_data[:,-1]
y = to_categorical(y, num_classes=vocab_size)
X_val, y_val = test_data[:,:-1],test_data[:,-1]
y_val = to_categorical(y_val, num_classes=vocab_size)


##upload preptrained word vectors
word_vec = pd.read_csv("glove.6B.50d.txt",sep=' ',quoting=csv.QUOTE_NONE,header=None, index_col =0)
word_vec["combined"] = word_vec.values.tolist()
vector_dict = word_vec["combined"].to_dict()

def get_embeddings(vectors, tokenizer):
  embedding_matrix = zeros((vocab_size, len(list(vector_dict.values())[0])))
  for word, i in tokenizer.word_index.items():
    embedding_vector = vectors.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  return embedding_matrix
  
#########define model and train##############################################################################
## single layer of lstm with 64 units takes a trainable embeddings as input
model = Sequential()
model.add(Embedding(vocab_size, 50,weights=[get_embeddings(vector_dict,tokenizer)], input_length=len(X[0])))
model.add(LSTM(64))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X, y, epochs=200, batch_size=128, verbose=2)# hyperparameters manually adjusted

#########evaluate model#####################################################################################
##get probabilities for all classes in trainning set 
y_pred = model.predict(X_val)
all_probab = pd.DataFrame(y_pred)

##extract probabilities of the tru classes
probabilities =[]
for i, prob in all_probab.iterrows():
  probabilities.append(prob[y_val[i]])
  
##calculate perplexity  
ent = 0
for p in probabilities:
  ent -= np.log(p)
perplexitiy = np.exp(ent)/(float(len(probabilities)))

##generate output text
intv_tokenizer = dict(map(reversed, tokenizer.word_index.items()))
predict_seq = loaded_model.predict_classes(X_val)

output_text = " ".join([intv_tokenizer.get(key) for key in predict_seq])  

start = "sherlock"
for i in range(1,50):
  input = tokenizer.texts_to_sequences([start])
  while len(input) <101:
    input = pad_sequences(input, maxlen=101, padding='pre')
    pred = intv_tokenizer[loaded_model.predict_classes(input)[0]]
    start = start + " " +pred
	