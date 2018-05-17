import pandas as pd
import numpy as np

# reading the reasons from the CSV file
#df = pd.read_csv('data/training-categories.csv')


def read_data(file_name):
	df = pd.read_csv(file_name)
	tokens= []
	train_reasons_x = []
	train_y=[]
	train_str=[]

	for index, row in df.iterrows():
		train_reasons_x.append(row['no_reasons'])
		tokens.append(row['token'])
		train_y.append(row['categ_index'])
		train_str.append(row['categ_1'])

	return train_reasons_x, train_y, train_str, tokens

