'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


#jaox imports
from gensim.models import KeyedVectors
import _pickle as cPickle
import sys, re
import xml.etree.ElementTree
from  collections import defaultdict
from keras.utils import np_utils


#keras
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Merge
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.layers import Input, Embedding, LSTM, Dense, merge

#functions---------------------------------------------------------------

def loadvecs(pathfile, vocab , binary=False):
	model = KeyedVectors.load_word2vec_format(pathfile, binary=binary)
	embeddings_index = {}
	for word in vocab:
		if word in model.vocab:
			embeddings_index[word] = model[word]
	del model

	print('Found %s word vectors.' % len(embeddings_index))
	return embeddings_index

def clean_str(string, TREC=False):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string) 
	string = re.sub(r",", " , ", string) 
	string = re.sub(r"!", " ! ", string) 
	string = re.sub(r"\(", " \( ", string) 
	string = re.sub(r"\)", " \) ", string) 
	string = re.sub(r"\?", " \? ", string) 
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip() if TREC else string.strip().lower()


def load_samples(pathfile, pathfiledev , pathfiletest, pathfiletestpolarity , cv=10, clean_string=True):
	e = xml.etree.ElementTree.parse(pathfile).getroot()
	revs = []
	vocab = defaultdict(float)

	for line in e.findall('tweet'):
		polarity = line.find('sentiment').find('polarity').find('value').text

		if polarity == 'N':
			polarity = 0
		elif polarity == 'P':
			polarity = 1
		elif polarity == 'NEU':
			polarity = 2
		else:
			polarity = 3

		rev = []
		rev.append(line.find('content').text)
		if clean_string:
			orig_rev = clean_str(" ".join(rev))
		else:
			orig_rev = " ".join(rev).lower()

		words = set(orig_rev.split())
		for word in words:
			vocab[word] += 1

		datum  = {
					"is_train": 1,
					"y":polarity,
					"text": orig_rev,
					"num_words": len(orig_rev.split()),
					"split": np.random.randint(0,cv)}

		revs.append(datum)


	#read the data dev
	e = xml.etree.ElementTree.parse(pathfiledev).getroot()

	for line in e.findall('tweet'):
		polarity = line.find('sentiment').find('polarity').find('value').text

		if polarity == 'N':
			polarity = 0
		elif polarity == 'P':
			polarity = 1
		elif polarity == 'NEU':
			polarity = 2
		else:
			polarity = 3

		rev = []
		rev.append(line.find('content').text)
		if clean_string:
			orig_rev = clean_str(" ".join(rev))
		else:
			orig_rev = " ".join(rev).lower()

		words = set(orig_rev.split())
		for word in words:
			vocab[word] += 1

		datum  = {
					"is_train": 1,
					"y":polarity,
					"text": orig_rev,
					"num_words": len(orig_rev.split()),
					"split": np.random.randint(0,cv)}

		revs.append(datum)


	#read the data set test
	f = open(pathfiletestpolarity)
	i = 0
	lst_polarity = []
	for polarity in f.read().split():
		if(i%2 != 0):
			if polarity == 'N':
				polarity = 0
			elif polarity == 'P':
				polarity = 1
			elif polarity == 'NEU':
				polarity = 2
			else:
				polarity = 3
			lst_polarity.append(polarity)
		i = i+1
	e = xml.etree.ElementTree.parse(pathfiletest).getroot()
	i = 0
	for line in e.findall('tweet'):
		rev = []
		rev.append(line.find('content').text)
		if clean_string:
			orig_rev = clean_str(" ".join(rev))
		else:
			orig_rev = " ".join(rev).lower()

		
		words = set(orig_rev.split())
		for word in words:
			vocab[word] += 1
		
		datum  = {
					"is_train": 0,
					"y":lst_polarity[i], 
					"text": orig_rev,
					"num_words": len(orig_rev.split()),
					"split": np.random.randint(0,cv)}

		revs.append(datum)
		i = i + 1

	return revs, vocab

def extract_label(texts , numbers_categories):
	text = []
	labels = []
	for doc in texts:
		sentence = doc['text']
		text.append(sentence)
		labels.append(doc['y'])
	labels = np_utils.to_categorical(labels, 4)
	return text, labels


def extract_cv(x, y , cv, index):
	
	index_cv = []

	x_train = []
	x_test = []
	y_test = []
	y_train = []


	for i in range(len(x)):
		index_cv.append(np.random.randint(0,cv))


	for i in range(len(x)):
		if index_cv[i] == index:
			x_test.append(x[i])
			y_test.append(y[i])
		else:
			x_train.append(x[i])
			y_train.append(y[i])


	return  np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


def format_data_aspect (x_train_text, y_train_text, embedding_matrix1, embedding_matrix2 , embedding_matrix3 , EMBEDDING_DIM, MAX_SEQUENCE_LENGTH ):

	x_train1= []
	x_train2 = []
	x_train3 = []


	for i in range ( len( x_train_text ) ):
		#for a, aspect in enumerate(x_train_aspect[i]):

		x_sentence1 = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
		x_sentence2 = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
		x_sentence3 = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))

		for j,index in enumerate(x_train_text[i]): #recorremos las palabras en la oracion
			x_sentence1[j] = embedding_matrix1 [ index ]
			x_sentence2[j] = embedding_matrix2 [ index ]
			x_sentence3[j] = embedding_matrix3 [ index ]

		x_train1.append(x_sentence1)
		x_train2.append(x_sentence2)
		x_train3.append(x_sentence3)

	return np.asarray(x_train1), np.asarray(x_train2), np.asarray(x_train3), np.asarray(y_train_text)


#end---------------------------------------------------------------------



#vars global-------------------------------------------------------------



MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 20000
num_validation_samples = 1899 
#end---------------------------------------------------------------------

#main -------------------------------------------------------------------

print('loading samples text tweets')
texts, vocab = load_samples('../datasets/intertass-train-tagged.xml','../datasets/intertass-development-tagged.xml', '../datasets/intertass-test.xml', '../datasets/intertass-sentiment.qrel.txt', cv=3, clean_string=True)

print('loading word vectors.')
if False:
	print('loading word vectors from files.')
	w2v = loadvecs('../pretrained/sbw_vectors.bin',vocab, binary=True)
	ft = loadvecs('../pretrained/fasttext_sbwc.vec',vocab,  binary=False)
	glv = loadvecs('../pretrained/glove_sbwc.vec',vocab, binary=False)
	cPickle.dump([w2v, ft, glv], open("mr.p", "wb"))
else:
	print('loading pre saved vectors.')
	x = cPickle.load(open("mr.p","rb"))
	w2v, ft, glv = x[0], x[1], x[2]


text, labels = extract_label(texts , 4)


##quitar palabras que no estan en el vocab en el text
n_nowords=0
for i , sentence in enumerate(text):
	sentence_aux = ""
	for word in sentence.split():
		if (word in w2v) and (word in ft) and (word in glv) :
			sentence_aux += " " + word + " " 
		else:
			n_nowords += 1
	text[i] = sentence_aux

print('remove ', n_nowords, ' words in not vocab')

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)


#select the greater ocurrence of len twwets
histo = np.zeros((35))
for s in sequences:
	histo[len(s)] += 1

imax = 0
temp = 0
for i, h in enumerate(histo):
	if h>temp:
		temp = h
		imax = i

MAX_SEQUENCE_LENGTH = 40#imax


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]



print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)+1 ) 
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
embedding_matrix2 = np.zeros((num_words, EMBEDDING_DIM))
embedding_matrix3 = np.zeros((num_words, EMBEDDING_DIM))



for word, i in word_index.items():
	if i >= MAX_NUM_WORDS:
		continue
	embedding_vector = w2v.get(word)
	if embedding_vector is not None:
	# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector
	
	embedding_vector = ft.get(word)
	if embedding_vector is not None:
	# words not found in embedding index will be all-zeros.
		embedding_matrix2[i] = embedding_vector

	
	embedding_vector = glv.get(word)
	if embedding_vector is not None:
	# words not found in embedding index will be all-zeros.
		embedding_matrix3[i] = embedding_vector

import random
random.seed(12345)


#define a model
inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embeddings_layer = Embedding(
				num_words, # due to mask_zero
				EMBEDDING_DIM,
				input_length=MAX_SEQUENCE_LENGTH,
				weights=[embedding_matrix3],
				trainable=False
			)(inp)

embeddings_layer_t = Embedding(
				num_words, # due to mask_zero
				EMBEDDING_DIM,
				input_length=MAX_SEQUENCE_LENGTH,
				weights=[embedding_matrix],
				trainable=True
			)(inp)

embeddings_layer_t2 = Embedding(
				num_words, # due to mask_zero
				EMBEDDING_DIM,
				input_length=MAX_SEQUENCE_LENGTH,
				weights=[embedding_matrix2],
				trainable=True
			)(inp)

#Convolution

filter_sizes = [2,2,2]
filter_numbers = [100, 100 , 100]
filter_pool_lengths = [2,2,2]

convolution_features_list = []
for filter_size,pool_length,num_filters in zip(filter_sizes, filter_pool_lengths, filter_numbers):
	conv_layer = Conv1D(nb_filter=num_filters, filter_length=filter_size, activation='relu')(embeddings_layer)
	pool_layer = GlobalMaxPooling1D()(conv_layer)
	#pool_layer = MaxPooling1D(pool_length=pool_length)(conv_layer)
	#flatten = Flatten()(pool_layer)
	convolution_features_list.append(pool_layer)

for filter_size,pool_length,num_filters in zip(filter_sizes, filter_pool_lengths, filter_numbers):
	conv_layer = Conv1D(nb_filter=num_filters, filter_length=filter_size, activation='relu')(embeddings_layer_t)
	pool_layer = GlobalMaxPooling1D()(conv_layer)#MaxPooling1D(pool_length=pool_length)(conv_layer)
	#flatten = Flatten()(pool_layer)
	convolution_features_list.append(pool_layer)

for filter_size,pool_length,num_filters in zip(filter_sizes, filter_pool_lengths, filter_numbers):
	conv_layer = Conv1D(nb_filter=num_filters, filter_length=filter_size, activation='relu')(embeddings_layer_t2)
	pool_layer = GlobalMaxPooling1D()(conv_layer)
	#pool_layer = MaxPooling1D(pool_length=pool_length)(conv_layer)
	#flatten = Flatten()(pool_layer)
	convolution_features_list.append(pool_layer)


out1 = Merge(mode='concat')(convolution_features_list) 
network = Model(input=inp, output=out1)

# Model	
model = Sequential()
model.add(network)

#Add dense layer to complete the model
model.add(Dense(16,init='uniform',activation='relu'))
#model.add(Dropout(0.2))
model.add( Dense(4, init='uniform', activation='softmax')  )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=6, batch_size=32)

#from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)