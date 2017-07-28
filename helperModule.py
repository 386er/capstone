from pandas import DataFrame, read_csv
from sklearn import preprocessing
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt


field_types = read_csv('./data/Field Names.csv', names=['field_name','field_type'])
field_names = field_types['field_name'][0:41]



def readDataKDD99():

	'''
	Reads in the data
	'''
	
	header = list(read_csv('./data/Field Names.csv', header=None)[0])
	data_train = read_csv('./data/KDD99Train.csv', names=header)
	data_test = read_csv('./data/KDD99Test.csv', names=header)

	return data_train, data_test


def replaceAttackTypes(dataset):

	'''
	Removes a . from the labels.
	'''

	dataset['attack_type'] = dataset['attack_type'].str.replace('.', '')

	return dataset


def dropAttackIndex(dataset):

	'''
	Removes the columns 'attack_type_index' from the dataset
	'''

	dataset = dataset.drop('attack_type_index', 1)

	return dataset


def mapAttackTypes(dataset):

	'''
	Takes in the dataset.
	Maps the different attack types to the attack type categories.
	Returns the dataset with mapped labels.
	'''
	
	intrusions = read_csv('./data/Attack Types.csv', names=['attack_type','attack_group','dataset'])
	attack_tuples = list(zip(intrusions.attack_type, intrusions.attack_group))
	attack_mapping = {entry[0]:entry[1] for entry in attack_tuples}

	dataset = dataset.replace({'attack_type':attack_mapping})

	return dataset


def encodeFeatures(data_train, data_test):

	'''
	Takes in the training and testing set.
	Encodes the three categorical features to integers.
	Returns the training and testing set with encoded categorical features.
	'''


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

	'''
	Takes in the training and testing set.
	Maps the connection types to integers.
	Returns the training and testing set with encoded labels.
	'''

	attack_mapping = {'NORMAL':0,'PROBE':1,'DOS':2,'U2R':3,'R2L':4,}

	data_train = data_train.replace({'attack_type':attack_mapping})
	data_test  = data_test.replace({'attack_type':attack_mapping})


	return data_train, data_test



def binarizeLabels(labels):

	'''
	Takes in a list of labels
	Returns a list of arrays of the size 5.
	'''

	lb = preprocessing.LabelBinarizer()

	lb.fit([0,1,2,3,4])
	labels = lb.transform(labels['attack_type'])

	return labels



def scaleFeatures(dataset_train, dataset_test):

	'''
	Takes in the training and testing dataset.
	Scales the features to values between 0 and 1
	Returns scaled training and testing sets.
	'''

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

	'''
	Takes in the dataset and splits the data into features and labels.
	Returns the features and labels.
	'''

	features = dataset.ix[:,:-1]
	labels = dataset.ix[:,-1:]

	return features, labels





def getBenchmarkResults():

	'''
	Returns a confusion table of the KDD99 Competition winning model.
	'''


	results_bench = [60262, 243, 78, 4, 6,511, 3471, 184, 0, 0, 5299, 1328, 223226, 0, 0, 68, 20, 0, 30, 10, 14527, 294, 0, 8, 1360]
	conf_bench = np.array(results_bench).reshape(5,5)

	return conf_bench



def computePerformanceMetrics(conf_benchmark, conf_model):


    '''
    Takes in the confusion table of the benchmark model and the suggested model. Based on these confusion tables, accuracy, TPR, FPR, Precsion and the F1-score are computed.
    Returns a dictionary of the computed metrics.
    '''
    
    metrics = {}
    
    accuracy_bench =np.sum([float(conf_benchmark[i][i])/np.sum(conf_benchmark) for i in range(5)])
    accuracy_model =np.sum([float(conf_model[i][i])/np.sum(conf_model) for i in range(5)])

    tpr_bench = [float(conf_benchmark[i][i])/(conf_benchmark.sum(axis=1)[i]) if conf_benchmark.sum(axis=1)[i] > 0 else float(conf_benchmark[i][i])/(conf_benchmark.sum(axis=0)[i] + 1)  for i,value in enumerate(conf_benchmark.sum(axis=1))]
    tpr_model = [float(conf_model[i][i])/(conf_model.sum(axis=1)[i]) if conf_model.sum(axis=1)[i] > 0 else float(conf_model[i][i])/(conf_model.sum(axis=0)[i] + 1)  for i,value in enumerate(conf_model.sum(axis=1))]
    
    fpr_bench = [float((conf_benchmark.sum(axis=1)[i] - conf_benchmark[i][i])) / conf_benchmark.sum(axis=1)[i] if conf_benchmark.sum(axis=1)[i] > 0 else float((conf_benchmark.sum(axis=1)[i] - conf_benchmark[i][i])) / (conf_benchmark.sum(axis=1)[i] + 1 ) for i,value in enumerate(conf_benchmark.sum(axis=1))]
    fpr_model = [float((conf_model.sum(axis=1)[i] - conf_model[i][i])) / conf_model.sum(axis=1)[i] if conf_model.sum(axis=1)[i] > 0 else float((conf_model.sum(axis=1)[i] - conf_model[i][i])) / (conf_model.sum(axis=1)[i] + 1 ) for i,value in enumerate(conf_model.sum(axis=1))]

    prec_bench = [float(conf_benchmark[i][i])/(conf_benchmark.sum(axis=0)[i]) if conf_benchmark.sum(axis=0)[i] > 0 else float(conf_benchmark[i][i])/(conf_benchmark.sum(axis=0)[i] + 1)  for i,value in enumerate(conf_benchmark.sum(axis=0))]
    prec_model = [float(conf_model[i][i])/(conf_model.sum(axis=0)[i]) if conf_model.sum(axis=0)[i] > 0 else float(conf_model[i][i])/(conf_model.sum(axis=0)[i] + 1)  for i,value in enumerate(conf_model.sum(axis=0))]
    
    f1_bench = [2*(prec_bench[i]*tpr_bench[i])/(prec_bench[i]+tpr_bench[i] + 0.00000001) for i in range(len(tpr_bench))]
    f1_model = [2*(prec_model[i]*tpr_model[i])/(prec_model[i]+tpr_model[i] + 0.00000001) for i in range(len(tpr_model))]

    
    metrics['accuracy_bench'] = accuracy_bench
    metrics['accuracy_model'] = accuracy_model

    metrics['tpr_bench'] = tpr_bench
    metrics['tpr_model'] = tpr_model
    
    metrics['fpr_bench'] = fpr_bench
    metrics['fpr_model'] = fpr_model
    
    metrics['prec_bench'] = prec_bench
    metrics['prec_model'] = prec_model
    
    metrics['f1_bench'] = f1_bench
    metrics['f1_model'] = f1_model
    
    return metrics




def plotMetrics(metrics):

	'''
	Takes in a dictionary of metrics for both the benchmark and the suggested model. 
	Creates a dashboard that plots the accuracy and all other evaluation metrics for the different attack types. 
	'''


	fig = plt.figure(figsize=(18,8))
	gs  = gridspec.GridSpec(2, 4, height_ratios=[0.3,1])
	ax1 = plt.subplot(gs[0, 0:2])
	axt = plt.text(1.3, 1.5, str(metrics['accuracy_bench'])[0:5], size=50, ha="center", va="center")
	axt = plt.text(1.8, 1.5, str(metrics['accuracy_model'])[0:5], size=50, ha="center", va="center")
	axt = plt.text(1.3, 1, 'Benchmark Accuracy', size=15, ha="center", va="center")
	axt = plt.text(1.8, 1, 'Model Accuracy', size=15, ha="center", va="center")
	ax2 = plt.subplot(gs[1, 0:1])
	ax3 = plt.subplot(gs[1, 1:2])
	ax4 = plt.subplot(gs[1, 2:3])
	ax5 = plt.subplot(gs[1, 3:4])

	x1 = [1,5,9,13,17]
	x2 = [2,6,10,14,18]


	ax1.barh([1], [metrics['accuracy_bench']], height=0.4, color='steelblue')
	ax1.barh([2], [metrics['accuracy_model']], height=0.4, color='darkorange')
	ax1.set_yticks([1,2])
	ax1.set_yticklabels(['Bench','Model'])
	ax1.set_xlabel('Accuracy', weight='bold')


	ax2.bar(x1, metrics['tpr_bench'], label='Benchmark')
	ax2.bar(x2, metrics['tpr_model'], label="Model")
	ax2.set_xlabel('TPR', weight='bold')
	ax2.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax2.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')
	ax2.legend()

	ax3.bar(x1, metrics['fpr_bench'], label='Benchmark')
	ax3.bar(x2, metrics['fpr_model'], label="Model")
	ax3.set_xlabel('FPR', weight='bold')
	ax3.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax3.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')

	ax4.bar(x1, metrics['prec_bench'], label='Benchmark')
	ax4.bar(x2, metrics['prec_model'], label="Model")
	ax4.set_xlabel('Precision', weight='bold')
	ax4.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax4.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')

	ax5.bar(x1, metrics['f1_bench'], label='Benchmark')
	ax5.bar(x2, metrics['f1_model'], label="Model")
	ax5.set_xlabel('F1-Score', weight='bold')
	ax5.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax5.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')

	plt.show()

	return fig



def getLabelCount(dataset):

	'''
	Takes in an array of binarized labels. 
	Returns the count of every connection type.
	'''


	normal = 0
	dos = 0
	probe = 0
	u2r = 0
	r2l = 0

	for entry in dataset:
		if entry[0] == 1:
			normal += 1
		if entry[1] == 1:
			dos += 1
		if entry[2] == 1:
			probe += 1
		if entry[3] == 1:
			u2r += 1
		if entry[4] == 1:
			r2l += 1

	return [normal, dos, probe, u2r, r2l]
    



def running_mean(l, N):

    '''
    Takes in an array of training or validation loss and an integer.
    Applies a running mean to the loss based on the chosen integer.

    Returns the transforemd array
    '''

    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result


def showDataDistribution(data):

	'''
	Plots the distribution of the first 18 features of the data, excluding categorial data.
	Returns the matplotlib figure object.
	'''

	features = data.columns

	fig = plt.figure(figsize=(12,13))
	gs  = gridspec.GridSpec(5, 3, height_ratios=[0.2,0.2,0.2,0.2,0.2],width_ratios=[0.33,0.33,0.33])

	ax1 = plt.subplot(gs[0, 0:1])
	ax2 = plt.subplot(gs[0, 1:2])
	ax3 = plt.subplot(gs[0, 2:3])
	
	ax4 = plt.subplot(gs[1, 0:1])
	ax5 = plt.subplot(gs[1, 1:2])
	ax6 = plt.subplot(gs[1, 2:3])

	ax7 = plt.subplot(gs[2, 0:1])
	ax8 = plt.subplot(gs[2, 1:2])
	ax9 = plt.subplot(gs[2, 2:3])
	
	ax10 = plt.subplot(gs[3, 0:1])
	ax11 = plt.subplot(gs[3, 1:2])
	ax12 = plt.subplot(gs[3, 2:3])

	ax13 = plt.subplot(gs[4, 0:1])
	ax14 = plt.subplot(gs[4, 1:2])
	ax15 = plt.subplot(gs[4, 2:3])


	ax1.plot(data[features[0]], label=features[0])
	ax2.plot(data[features[4]], label=features[4])
	ax3.plot(data[features[5]], label=features[5])
	ax4.plot(data[features[6]], label=features[6])

	ax5.plot(data[features[7]], label=features[7])
	ax6.plot(data[features[8]], label=features[8])
	ax7.plot(data[features[9]], label=features[9])
	ax8.plot(data[features[10]], label=features[10])

	ax9.plot(data[features[11]], label=features[11])
	ax10.plot(data[features[12]], label=features[12])
	ax11.plot(data[features[13]], label=features[13])
	ax12.plot(data[features[14]], label=features[14])

	ax13.plot(data[features[15]], label=features[15])
	ax14.plot(data[features[16]], label=features[16])
	ax15.plot(data[features[17]], label=features[17])



	plt.show()

	return fig




def plotMetricsWithoutAccuracy(metrics):

	'''
	Works like the plotMetrics method, but without showing the accuracy.
	Returns the matplotlib figure object.
	'''


	fig = plt.figure(figsize=(18,8))
	gs  = gridspec.GridSpec(1, 4, height_ratios=[1])

	ax2 = plt.subplot(gs[0, 0:1])
	ax3 = plt.subplot(gs[0, 1:2])
	ax4 = plt.subplot(gs[0, 2:3])
	ax5 = plt.subplot(gs[0, 3:4])

	x1 = [1,5,9,13,17]
	x2 = [2,6,10,14,18]

	ax2.bar(x1, metrics['tpr_bench'], label='Benchmark')
	ax2.bar(x2, metrics['tpr_model'], label="Model")
	ax2.set_xlabel('TPR', weight='bold')
	ax2.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax2.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')
	ax2.legend()

	ax3.bar(x1, metrics['fpr_bench'], label='Benchmark')
	ax3.bar(x2, metrics['fpr_model'], label="Model")
	ax3.set_xlabel('FPR', weight='bold')
	ax3.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax3.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')

	ax4.bar(x1, metrics['prec_bench'], label='Benchmark')
	ax4.bar(x2, metrics['prec_model'], label="Model")
	ax4.set_xlabel('Precision', weight='bold')
	ax4.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax4.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')

	ax5.bar(x1, metrics['f1_bench'], label='Benchmark')
	ax5.bar(x2, metrics['f1_model'], label="Model")
	ax5.set_xlabel('F1-Score', weight='bold')
	ax5.set_xticks([1.5,5.5,9.5,13.5,17.5])
	ax5.set_xticklabels(['NORMAL','PROBE','DOS','U2R','R2L'], size='smaller')

	plt.show()

	return fig





