from pandas import DataFrame, read_csv
from sklearn import preprocessing
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt


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


def getBenchmarkResults():

	results_bench = [60262, 243, 78, 4, 6,511, 3471, 184, 0, 0, 5299, 1328, 223226, 0, 0, 68, 20, 0, 30, 10, 14527, 294, 0, 8, 1360]
	conf_bench = np.array(results_bench).reshape(5,5)

	return conf_bench



def computePerformanceMetrics(conf_benchmark, conf_model):
    
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



def getLabelCount(dataset):

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
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N

    return result





