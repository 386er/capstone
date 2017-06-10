from pandas import DataFrame, read_csv
from sklearn import preprocessing
import numpy as np


field_types = read_csv('./data/Field Names.csv', names=['field_name','field_type'])
field_names = field_types['field_name'][0:41]



def readDataKDD99():
	
	header = list(read_csv('./data/Field Names.csv', header=None)[0])
	data_train = read_csv('./data/KDD99Train.csv', names=header)
	data_test = read_csv('./data/KDD99Test.csv', names=header)

	return data_train, data_test


def replaceAttackTypes(dataset):
	dataset['attack_type'] = dataset['attack_type'].str.replace('.', '')

	return dataset


def dropAttackIndex(dataset):
	dataset = dataset.drop('attack_type_index', 1)

	return dataset


def mapAttackTypes(dataset):
	
	intrusions = read_csv('./data/Attack Types.csv', names=['attack_type','attack_group','dataset'])
	attack_tuples = list(zip(intrusions.attack_type, intrusions.attack_group))
	attack_mapping = {entry[0]:entry[1] for entry in attack_tuples}

	dataset = dataset.replace({'attack_type':attack_mapping})

	return dataset


def encodeFeatures(data_train, data_test):

	le = preprocessing.LabelEncoder()

	protocol_types = np.union1d(data_train.protocol_type.unique(), data_test.protocol_type.unique())
	services = np.union1d(data_train.service.unique(), data_test.service.unique()) 
	flags = np.union1d(data_train.flag.unique(), data_test.flag.unique()) 

	
	le.fit(protocol_types)
	data_train.protocol_type = le.transform(data_train.protocol_type)
	data_test.protocol_type = le.transform(data_test.protocol_type)

	le.fit(services)
	data_train.service = le.transform(data_train.service)
	data_test.service = le.transform(data_test.service)

	le.fit(flags)
	data_train.flag = le.transform(data_train.flag)
	data_test.flag = le.transform(data_test.flag)

	return data_train, data_test



def encodeLabels(data_train, data_test):

	attack_mapping = {'NORMAL':0,'PROBE':1,'DOS':2,'U2R':3,'R2L':4,}

	data_train = data_train.replace({'attack_type':attack_mapping})
	data_test  = data_test.replace({'attack_type':attack_mapping})


	return data_train, data_test



def binarizeLabels(labels):

	lb = preprocessing.LabelBinarizer()

	lb.fit([0,1,2,3,4])
	labels = lb.transform(labels['attack_type'])

	return labels



def scaleFeatures(dataset_train, dataset_test):

	scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

	for field in dataset_train.columns[:-1]:

		max_train = np.max(dataset_train[field])
		max_test = np.max(dataset_test[field])

		if max_train > max_test:
			min_max_tuple = (0,max_train)
		else:
			min_max_tuple = (0,max_test)


		scaler.fit(min_max_tuple)

		dataset_train[field] = scaler.transform(dataset_train[field])
		dataset_test[field] = scaler.transform(dataset_test[field])

	return dataset_train, dataset_test


def splitData(dataset):

	features = dataset.ix[:,:-1]
	labels = dataset.ix[:,-1:]

	return features, labels










