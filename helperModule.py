from pandas import DataFrame, read_csv
from sklearn import preprocessing
import numpy as np


field_types = read_csv('Field Names.csv', names=['field_name','field_type'])
field_names = field_types['field_name'][0:41]



def readData():
	
	header = list(read_csv('Field Names.csv', header=None)[0])
	data_train = read_csv('KDDTrain+.csv', names=header)
	data_test = read_csv('KDDTest+.csv', names=header)

	return data_train, data_test


def splitData(data_train, data_test):

	x_train = data_train.ix[:,:-2]
	y_train = data_train.ix[:,-2:-1]

	x_test = data_test.ix[:,:-2]
	y_test = data_test.ix[:,-2:-1]

	return x_train, y_train, x_test, y_test


def mapAttackTypes(y_train, y_test):
	
	intrusions = read_csv('Attack Types.csv', names=['attack_type','attack_group','dataset'])
	attack_tuples = list(zip(intrusions.attack_type, intrusions.attack_group))
	attack_mapping = {entry[0]:entry[1] for entry in attack_tuples}

	y_train = y_train.replace({'attack_type':attack_mapping})
	y_test = y_test.replace({'attack_type':attack_mapping})

	return y_train, y_test


def encodeFeatures(dataset):

	le = preprocessing.LabelEncoder()

	protocol_types = dataset.protocol_type.unique()
	services = dataset.service.unique()
	flags = dataset.flag.unique()
	
	le.fit(protocol_types)
	dataset.protocol_type = le.transform(dataset.protocol_type)

	le.fit(services)
	dataset.service = le.transform(dataset.service)

	le.fit(flags)
	dataset.flag = le.transform(dataset.flag)

	return dataset



def encodeLabels(dataset):

	le = preprocessing.LabelEncoder()

	attack_types = dataset['attack_type'].unique()

	le.fit(attack_types)
	dataset['attack_type'] = le.transform(dataset['attack_type'])

	return dataset


def binarizeLabels(dataset):

	lb = preprocessing.LabelBinarizer()

	lb.fit([0,1,2,3,4])
	dataset = lb.transform(dataset)

	return dataset



def getValueRange(dataset_train, dataset_test):

    value_range = {field:float(np.amax(dataset_train[field])) for field in field_names}
    value_range = {field:float(np.amax(dataset_test[field])) if np.amax(dataset_test[field]) > value_range[field] else value_range[field] for field in field_names}                                                                                                                                             	 
    value_range = {field:[value_range[field],0] for field in field_names}

    return value_range



def scaleFeatures(x_train, x_test):

	value_range = getValueRange(x_train, x_test)
	scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

	for field in field_names:

		scaler.fit(value_range[field])

		x_train[field] = scaler.transform(x_train[field])
		x_test[field] = scaler.transform(x_test[field])

	return x_train, x_test



'''
SVM with default parameters

Accuracy: 0.7428026438362241
F1 Score: 0.4714713550153423
Precision Score: 0.4874805614106393
Recall Score: 0.47799178865117165

-------------------------------

Standart DecisionTreeClassifier with default parameters

Accuracy: 0.7767821496695205
F1 Score: 0.5452761186107676
Precision Score: 0.7739885128253184
Recall Score: 0.5386814973134587

-------------------------------

XGBoost Classifier with default parameters

Accuracy: 0.7672448210087388
F1 Score: 0.48177712781306675
Precision Score: 0.48863402673427914
Recall Score: 0.4914928357729108



from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, predictions)
f1_score = metrics.f1_score(y_test, predictions, average='macro')
precision_score = metrics.precision_score(y_test, predictions, average='macro')
recall_score = metrics.recall_score(y_test, predictions, average='macro')

print('Accuracy: {}'.format(accuracy))
print('F1 Score: {}'.format(f1_score))
print('Precision Score: {}'.format(precision_score))
print('Recall Score: {}'.format(recall_score))

'''



