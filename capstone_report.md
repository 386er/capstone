# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
December 31st, 2050

## I. Definition

(approx. 1-2 pages)

### Project Overview

Since the rise of the Internet in the late 90s, an exponentially growing number of connected devices is shaping our world. While initially limited to personal computers and notebooks in mostly Western societies, smartphones have emerged in the late 2000s, bringing connectivity to formerly excluded societies in developing countries as well. This development is now increasingly accelerated by the rise of IoT, affecting both consumer and manufacturing markets.

With this ever growing number of connected devices and the technical infrastructure to maintain them, more and more traffic is transferred over networks, generating a confusing amount of log data that has to be monitored. Given this development, manual monitoring becomes increasingly infeasible to prevent cyber-attacks. Network Intrusion Detections Systems (NIDS) can help system administrators to detect network breaches, however setting up policies that are both flexible and effective against unforeseen attacks can be challenging.

While applicable to the detection of well-known and precisely defined attacks, traditional approaches of intrusion detection have proven to be ineffective in cases of novel intrusions. Such methods are mostly threshold-based and expressed in counts or distributions (See Garcia-Toedora, 2009).
Applying machine learning in the analysis of log files can help NIDS to learn more complex behavior. In particular it allows them to differentiate whole classes of traffic and hence enables them to detect irregularities and previously unseen attacks. With their ability to achieve high levels of abstractions, deep learning methods seem suitable for tackling this kind of problem.


### Problem Statement

The problem that will be addressed in this project is to create a model, which predicts whether a connection between a source and target IP, represented by a set of features, is an attempt to attack the source network or not. In case of an attack, the model has to predict what kind of attack a malicious connection represents.

An additional challenge which will be addressed in this project is that the type of attacks that the model will be trained on, differ from the type of attacks that its performance will tested on. This is a specific characteristic of the used dataset to make it more realistic. However this does not represent a major constraint, as there is widespread believe that most of the novel attacks can be derived from known attacks (see Tavallee, 2009).

Assuming that attacks produce a distinctive sequence of events, this capstone project seeks to model network traffic as a time series by applying a long short-term memory (LSTM) Recurrent neural network. Unlike feedforward neural networks, RNNs have cyclic connections making them powerful for modeling sequences. While standard RNNs are only able to model sequences that are up to 10 time steps long, LSTM based RNNs overcome this constraints and allow for much longer sequences to be considered.



### Metrics

The following metrics will be used to evaluate the performance of the applied model in comparison to the benchmark model:

- Accuracy: Defined as the proportion of true results (both true positives and true negatives) among the total number of datapoints examined.

- True Positive Rate: Defined as the ratio between the number of attacks correctly categorized as attacks and the total number of attacks.

- False Positive Rate: Defined as the ratio between the number of normal connections wrongly categorized as attacks and the total number of normal connections.

- Precision: Defined as the ratio of the number of true positives divided by the number of true positives and false positives.

- Recall: Defined as the ratio of number of true positives records divided by the number of true positives and false negatives.
Evaluation Metrics will be applied for the Global Test Dataset as well as for every type of connection (NORMAL; PROBE; DOS; U2R; R2L)

- F1-Score: Defined as the harmonic mean of precision and recall

The use of accuracy for this dataset is problematic insofar as there is a high imbalance between different types of attack which might be misleading. However for every generated model, the considered metrics will be complemented by the corresponding confusion matrix. Hence the global performance of the model will be considered, as well as the attack-type specific performance.



## II. Analysis

(approx. 2-4 pages)

### Data Exploration


The dataset used for the capstone project will be the KDD Cup 1999 Data, which was prepared for the Third International Knowledge Discovery and Data Mining Tools Competition. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between 'bad' connections, called intrusions or attacks, and 'good' normal connections.

The KDD dataset was created by collecting raw TCP dump data from a simulated U.S. airforce local-area network in the course of 9 weeks. During the data collection the network was deliberately attacked multiple times. Each datapoint represents a connection between a source and target IP, using a well defined protocol, during a well defined time frame. Each datapoint is made up of 41 features and is labeled either as a normal connections or as an attack. Some of the features are directly derived from the TCP/IP connections during a time interval, however the dataset includes also "higher-level" features that were derived from some of the basic features of the dataset.

The 'basic' set of features include inputs like duration of the connection (in seconds), the protocol type, the number of bytes from the source to the destination and vice versa.

The 'higher-level' features include inputs like the number of connections to the same host as the current connection in the past two seconds, number of connection that use the same service, number of failed login attempts.

A data point looks as follows:
XXX
Both test and training dataset contain 43 columns (41 input features, 1 label, 1 label index). Of the 41 input features, 38 features are continuous, 3 of them are nominal (protocol_type, service, flag).
As mentioned before the test and training set differ in terms of the attack types that occur in them. There are a total of 23 attack types in the training set and 38 types of attack in the test set.


Attacks that occur only in the training set are: [warezclient., spy.] 

Attacks that occur only in the test set are: [snmpgetattack., named., xlock., xsnoop., sendmail., saint., apache2., udpstorm., xterm., mscan., processtable., ps., httptunnel., worm., mailbomb., sqlattack., snmpguess.] 
 
Attacks that occur both sets: [normal., buffer_overflow., loadmodule., perl., neptune., smurf., guess_passwd., pod., teardrop., portsweep., ipsweep., land., ftp_write., back., imap., satan., phf., nmap., multihop., warezmaster., rootkit.] 



There are a total of 494021 data points in the training set and 311029 observations in the test set.  The distribution of attack types in the sets are as follows:


Type		Occurrence Train	Share Train		Occurrence Test		Share Test
Normal		97278				0.196			60593				0.194815
DOS			391458				0.792391		229853				0.739008
PROBE		4107				0.008313		4166				0.013394
U2R			52					0.000105		228					0.000733
R2L			1126				0.002279		16189				0.052050
Total		494021				1				311029				1


As it becomes obvious from this this table, that there is clear imbalance between different groups of attack types. The bulk of the connections in both training and testing set are DOS-type attacks. With normal connections making up for about another 20% of the dataset, only few connections are of type PROBE, U2R or R2L.  
In particular U2R and R2L attacks are extremely scarce in the training set, which sometimes either require only a single connection or are without any prominent sequential pattern (See Staudemeyer, 2015).
There is a revisited version of the dataset with an altered class distribution that has an increased share of U2R and R2L attacks to prevent learning algorithms to be biased towards the NORMAL and DOS connections (see Tavallee, 2009). However the author auf this report has decided to use the original dataset, as it ultimately represents a more realistic real-world environment.


### Exploratory Visualization




### Algorithms and Techniques



### Benchmark




