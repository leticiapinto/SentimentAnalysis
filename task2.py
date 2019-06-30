import xml.etree.ElementTree as etree
from  collections import defaultdict
import sys, re
import numpy as np
import _pickle as cPickle
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers.core import Dropout, Flatten
from keras.layers.merge import concatenate

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

#jaox imports
from gensim.models import KeyedVectors
import _pickle as cPickle
import sys, re
import xml.etree.ElementTree
from  collections import defaultdict




def clean_str(string, TREC=False):
	#string = re.sub(r"[^A-Za-z0-9(),!?\'\`\ó´´]", " ", string)
	string = re.sub(r"\|", " ", string) 
	string = re.sub(r"\'s", " \'s", string) 
	string = re.sub(r"\'ve", " \'ve", string) 
	string = re.sub(r"n\'t", " n\'t", string) 
	string = re.sub(r"\'re", " \'re", string) 
	string = re.sub(r"\'d", " \'d", string) 
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r"\.", " ", string)  
	string = re.sub(r",", " ", string) 
	string = re.sub(r"!", " ", string) 
	string = re.sub(r"¡", " ", string) 
	string = re.sub(r"\(", " ", string) 
	string = re.sub(r"\)", " ", string) 
	string = re.sub(r"\?", " ", string) 
	string = re.sub(r"\¿", " ", string) 
	string = re.sub(r"\:", " ", string) 
	string = re.sub(r"\"", " ", string) 
	string = re.sub(r"\'", " ", string) 
	string = re.sub(r"\-", " ", string) 
	string = re.sub(r"\_", " ", string)
	string = re.sub(r"\s{2,}", " ", string)    
	return string.strip() if TREC else string.strip().lower()

def load_samples(pathtrain, pathtest , cv=10, clean_string=True):
	e = etree.parse(pathtrain).getroot()
	revs = []
	vocab = defaultdict(float)


	for line in e.findall('tweet'):

		text = etree.tostring(line, encoding='utf-8', method='text').decode('utf-8')

		rev = []
		rev.append( text )
		
		if clean_string:
			orig_rev = clean_str(" ".join(rev))
		else:
			orig_rev = " ".join(rev).lower()

		

		words = set(orig_rev.split())
		for word in words:
			vocab[word] += 1


		aspects = []
		polaritys = []
		for sentiment in line.findall('sentiment'):
			asp = sentiment.get('aspect')
			if asp != None:
				aspects.append(asp.lower())
			pol = 0
			if sentiment.get('polarity') == 'N':
				pol = 0
			elif sentiment.get('polarity') == 'P':
				pol = 1
			else:
				pol = 2

			polaritys.append(pol)

		datum  = {"is_train": 1,
					"text": orig_rev,
					"num_words": len(orig_rev.split()),
					"split": np.random.randint(0,cv),
					"aspects": aspects,
					"polaritys": polaritys
				}

		revs.append(datum)

	e = etree.parse(pathtest).getroot()
	
	for line in e.findall('tweet'):

		text = etree.tostring(line, encoding='utf-8', method='text').decode('utf-8')

		rev = []
		rev.append( text )
		
		if clean_string:
			orig_rev = clean_str(" ".join(rev))
		else:
			orig_rev = " ".join(rev).lower()

		

		words = set(orig_rev.split())
		for word in words:
			vocab[word] += 1


		aspects = []
		polaritys = []
		for sentiment in line.findall('sentiment'):
			asp = sentiment.get('aspect')
			if asp != None:
				aspects.append(asp.lower())
			pol = 0
			if sentiment.get('polarity') == 'N':
				pol = 0
			elif sentiment.get('polarity') == 'P':
				pol = 1
			else:
				pol = 2

			polaritys.append(pol)

		datum  = {"is_train": 0,
					"text": orig_rev,
					"num_words": len(orig_rev.split()),
					"split": np.random.randint(0,cv),
					"aspects": aspects,
					"polaritys": polaritys
				}

		revs.append(datum)

	print('found ',len(revs), ' tweets and ', len(vocab), 'words')

	return revs , vocab

def extract_label(texts , numbers_categories):
	text = []
	labels = []
	aspects = []
	for doc in texts:
		sentence = doc['text']
		text.append(sentence)
		aspects.append(doc['aspects'])
		polaritys = np_utils.to_categorical(doc['polaritys'], numbers_categories)
		labels.append(polaritys)
	return text, labels , aspects
def format_data_aspect (x_train_text, x_train_aspect, y_train_text, word_index, embedding_matrix1, embedding_matrix2 , embedding_matrix3 , w2v, ft, glv, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH ):

	x_train1, y_train = [], []
	x_train2 = []
	x_train3 = []


	for i in range ( len( x_train_text ) ):
		for a, aspect in enumerate(x_train_aspect[i]):

			x_sentence_aspect1 = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM*2))
			x_sentence_aspect2 = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM*2))
			x_sentence_aspect3 = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM*2))

			avg1 = np.zeros((EMBEDDING_DIM))
			avg2 = np.zeros((EMBEDDING_DIM))
			avg3 = np.zeros((EMBEDDING_DIM))

			if aspect != None:
				words_aspect = aspect.split('-')
				if(len( words_aspect)>1 ):
					
					counter1 = 0
					counter2 = 0
					counter3 = 0
					for word_aspect in words_aspect:
						if w2v.get(word_aspect) is not None:
							counter1 += 1
							vector_aspect1 = embedding_matrix1 [ word_index[word_aspect] ]
							avg1 +=vector_aspect1
						if ft.get(word_aspect) is not None:
							counter2 += 1
							vector_aspect2 = embedding_matrix2 [ word_index[word_aspect] ]
							avg2 +=vector_aspect2
						if glv.get(word_aspect) is not None:
							counter3 += 1
							vector_aspect3 = embedding_matrix3 [ word_index[word_aspect] ]
							avg3 +=vector_aspect3	
								

					avg1 = avg1/max(counter1,1)
					avg2 = avg2/max(counter2,1)
					avg3 = avg3/max(counter3,1)

			#conc + avg
			temp = np.zeros((EMBEDDING_DIM*2))
			for j,index in enumerate(x_train_text[i]): #recorremos las palabras en la oracion
				
				temp1 = embedding_matrix1 [ index ]

				temp1 = np.concatenate((temp1,avg1), axis = 0)

				x_sentence_aspect1[j] = temp1

				temp2 = embedding_matrix2 [ index ]
				temp2 = np.concatenate((temp2,avg2), axis = 0)
				x_sentence_aspect2[j] = temp2

				temp3 = embedding_matrix3 [ index ]
				temp3 = np.concatenate((temp3,avg3), axis = 0)
				x_sentence_aspect3[j] = temp3

			x_train1.append(x_sentence_aspect1)
			x_train2.append(x_sentence_aspect2)
			x_train3.append(x_sentence_aspect3)
			y_train.append(y_train_text[i][a])

	return np.asarray(x_train1), np.asarray(x_train2), np.asarray(x_train3), np.asarray(y_train)


#vars global-------------------------------------------------------------
MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 20000
num_validation_samples = 1000 
#end---------------------------------------------------------------------


#MAIN ------------------------------------------------

revs , vocab = load_samples('../datasets/socialtv-train-tagged.xml','../datasets/socialtv-test-tagged.xml',3,True)


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

text, labels , aspects = extract_label(revs, 3)

##quitar palabras que no estan en el vocab en el text

for i , sentence in enumerate(text):
	sentence_aux = ""
	for word in sentence.split():
		if (word in w2v) or (word in ft) or (word in glv) :
			sentence_aux += " " + word + " " 
	text[i] = sentence_aux

# finally, vectorize the text samples into a 2D integer tensor

temp = ' '
for aspect in aspects:
	#print(aspect)
	temp += clean_str(' '.join(aspect))+' '

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
tokenizer.fit_on_texts(temp)

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

MAX_SEQUENCE_LENGTH = imax

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)

# split the data into a training set and a validation set
x_train_text = data[:-num_validation_samples]
x_train_aspect = aspects[:-num_validation_samples]
y_train_text = labels[:-num_validation_samples]


x_val_text = data[-num_validation_samples:]
x_val_aspect = aspects[-num_validation_samples:]
y_val_text = labels[-num_validation_samples:]


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



#convertir las entradas al espacio vectorial w2v:

x_train1, x_train2, x_train3, y_train = format_data_aspect (x_train_text, x_train_aspect, y_train_text, word_index, embedding_matrix, embedding_matrix2 , embedding_matrix3 , w2v, ft, glv, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH )
x_val1, x_val2, x_val3, y_val = format_data_aspect (x_val_text, x_val_aspect, y_val_text, word_index, embedding_matrix, embedding_matrix2 , embedding_matrix3 , w2v, ft, glv, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH )

print(x_train1.shape)

import random
random.seed(12345)

print('Training model.')


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed

input1 = Input(shape=( MAX_SEQUENCE_LENGTH , EMBEDDING_DIM*2), dtype='float32')
x1 = input1

input2 = Input(shape=( MAX_SEQUENCE_LENGTH , EMBEDDING_DIM*2), dtype='float32')
x2 = input2

input3 = Input(shape=( MAX_SEQUENCE_LENGTH , EMBEDDING_DIM*2), dtype='float32')
x3 = input3


merge =  concatenate([x1 ,x2, x3])

print(merge.shape)


x = Conv1D(128, 5, activation='relu', border_mode='same')(merge)
x = MaxPooling1D(4)(x)
x = Conv1D(128, 3, activation='relu', border_mode='same')(x)
x = Conv1D(128, 4, activation='relu', border_mode='same')(x)
x = Conv1D(128, 5, activation='relu', border_mode='same')(x)
x = GlobalMaxPooling1D()(x)
#x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)
model = Model(inputs=[input1, input2, input3], outputs=preds)

#rmsprop
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])



model.fit([x_train1,x_train2,x_train3], y_train, validation_data=([x_val1,x_val2,x_val3], y_val),epochs=10, batch_size=128 , shuffle = True )


loss, acc = model.evaluate([x_val1,x_val2,x_val3], y_val, verbose=0)
print('Train Accuracy: %f' % (acc*100))
