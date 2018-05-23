import json
import numpy as np
import pandas as pd
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


# GLOBAL VARIABLES

filename = 'data/training-categories.csv' #file from where we read the data
max_words=1000 # the maximum number of words in our vocabulary. It will also be the number of columns for the one hot encoding
tokenizer = Tokenizer(num_words=max_words) # Tokenizer object from keras
number_labels = 17

dictionary_filename = 'data/dictionary.json' # file to save our vocabulary
model_filename = 'data/model.json'  # file to save the model structure
weights_filename = 'data/model.h5'  # file to save the model weights


# Function to read the CSV of data
def read_data(file_name):
	df = pd.read_csv(file_name)
	tokens= [] # ordered list of customer tokens
	train_reasons_x = [] # ordered list with the verbatim comments for training
	train_y=[] # ordered list with the existing reasons for training as IDs (in our case from 0 to 18)
	train_str=[] # ordered list with the existing reasons fortraining as labels / labels

	for index, row in df.iterrows():
		train_reasons_x.append(row['no_reasons'])
		tokens.append(row['token'])
		train_y.append(row['categ_index'])
		train_str.append(row['categ_1'])

	return train_reasons_x, train_y, train_str, tokens # returning the ordered lists


train0_x, train0_y, train_str, tokens  = read_data(filename) # assigning the training variables
labels = [] # array with unique value of the labels
for label in train_str:
	if label not in labels:
		labels.append(label)
tokenizer.fit_on_texts(train0_x)
dictionary = tokenizer.word_index

#print("Train ):{}".format(train0_y))
#def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    
#    dic = [dictionary[word] for word in kpt.text_to_word_sequence(text)]
#    return dic

def convert_text_to_index_array(text):
	words = kpt.text_to_word_sequence(text)
	wordIndices = []
	for word in words:
		if word in dictionary:
			wordIndices.append(dictionary[word])
		else:
			print("'%s' not in training corpus; ignoring." %(word))
	return wordIndices


# function that takes a file and returns an array of tokens, a train_x and a train_y
def process_text(csv_filename):

	with open(dictionary_filename, 'w') as dictionary_file:
		json.dump(dictionary, dictionary_file)
	allWordIndices = []
	for text in train0_x:
		wordIndices = convert_text_to_index_array(text)
		allWordIndices.append(wordIndices)
	allWordIndices = np.asarray(allWordIndices)
	train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary') # one hot encoding with the number of rows = the number of comments and the number of columns = max_words + 1. The first column will always be with 0
	train_y = keras.utils.to_categorical(train0_y, 19) # one hot encoding for the labels: number of rows = number of comments and number of columns = number of labels + 1. The first column will always be 0
	#print(train_y.shape)
	#print(train0_x[-1])
	return train_x, train_y

# Creating the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='softmax'))

model.compile(loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy'])


train_x, train_y = process_text(filename) 

model.fit(train_x, train_y, batch_size=32, 
	epochs=14, verbose=1, 
	validation_split=0.1, 
	shuffle=True)


# Saving the model structure
model_json = model.to_json()
with open(model_filename, 'w') as json_file:
    json_file.write(model_json)

# Saving the model weights
model.save_weights(weights_filename)


# loading the model
# read in your saved model structure
json_file = open(model_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights(weights_filename)


# evaluating the model
eval_texts =["My order has not shipped and I placed my order 8 days ago PLUS I have tried to contact customer support through phone and email and no one has got back to me.",
"I signed up for the elite box and I have never received a box and I am not sure what the problem is my account says that I have one that will arrive however I have never received a box. I would like to know what the problem is.",
"I accidentally cancelled my order #304529605. I in fact do not want to cancel this order.", "Advertised new line - $19.99. And the plus size was double the price", "There was no sale on what i am looking to purchase", "Won’t let me complete order", "I purchased an item about two weeks ago and up to present I haven't received it.", "I thought these items were 3 for $30", "Now it's saying can't read undefined"]

#print(labels)
#print(labels.__len__())

for text in eval_texts:
	testArr = convert_text_to_index_array(text)
	input = tokenizer.sequences_to_matrix([testArr], mode='binary')
	pred = model.predict(input)
	print("---------------")
	print("Text:{}".format(text))
	#print("Label:{0} with confidence {1}% \n".format(labels[np.argmax(pred)-2], pred[0][np.argmax(pred)] * 100))
	for i in range(labels.__len__()):
		print("Label:{0} with confidence {1}% \n".format(labels[i], pred[0][i+2] * 100))


