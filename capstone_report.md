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

The information content of the features becomes obvious in the correlation heat matrix. In particular the distinction between "basic features", which were derived from the packet data itself such as the duration, or the source and destination bytes and "higher features", which were engineered with additional domain knowledge. The first 20 features (exluding categorial variables, such as flag or service) show very little correlation among each other, meaning they contain a lot of information that potentially helps to classify different attack types. The remaining 20 features show very high levels of correlation, meaning they contain a l ot of redundant information.

Heatcorrelation matrix


Looking at the distribution of the first 15 (discrete) features, it becomes obvious that most data points are concentrated on certain values, making some features almost categorial, although they are considered having discrete distributions.




### Algorithms and Techniques


Given the type of problem for this project, applying a standard feedforward neural network presumably produces suboptimal results. Assuming that most attacks produce a specific signature, meaning a pattern that can be observed for multiple consecutive connections, the standard neural network would treat every single connection independently of its current position. Recurrent neural networks however maintain an internal state at each time step of the classification, which allows them to use information from previous connections for the processing of current connections. This stored "memory" should greatly increase their effectiveness in classifiying sequences of network connections. 

Besides general parameter that can be tuned for most models (such as the chosen optimizer, the number of training iterations, the batch size and the numbers of input variables considered in case of multivariate data), LSTM networks consist of cells, or also often referred to as blocks, that contain hidden units, which are a direct representation the learning capacity of the neural network.


Following parameters can be tuned for the model implementation:


1. Neural Network Architecture

 - Number of Cells (Default = 2)
 - Number of Hidden Nodes
 - Cell type (tensorflow offers a variety of cells, LSTM cell was used throughout.)

2. Training Parameter

 - Batch Size (how many time steps to include during a single training step; )
 - Optimizer Function (which function to optimize by mimizing error; used “Adam” throughout)
 - Epochs (how many times to run through the training process; kept mostly at 1 for time savings until later studies)




### Benchmark


The benchmark model for this capstone project will be the winning model used in the 1999 Data Mining and Knowledge Discovery competition (KDD Cup). The model applied was an ensemble of C5 decision trees. See (Pfahringer, 2000) for a more detailed overview of how this model was implemented.

This model resulted in the following confusion matrix (Taken from Elkan, 2000):

Actual →		0		1		2		3		4		%Correct
Predicted ↓						
0				60262	243		78		4		6		99.5%
1				511		3471	184		0		0		83.3%
2				5299	1328	223226	0		0		97.1%
3				68		20		0		30		10		13.2%
4				14527	294		0		8		1360	8.4%
%correct		74.6%	64.8%	99.9%	71.4%	98.8%	



## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

Since the dataset used in this project was initially prepared for data mining competition no effort had to be spend on cleaning it. First this can be justified by the fact that it contains no missing entries. Second, since this project deals with intrusion detection, the author of this report saw no need to remove potential outliers.

The data was then preprocessed in the following order:


1. The 38 different attack types are mapped to their corresponding attack group.
-----------
normal                      ==>   NORMAL
back,land ..                ==>   DOS
satan,ipsweep,nmap, ..      ==>   PROBE
ftp_write,guess_passwd, ..  ==>   R2L
rootkit,perl ..             ==>   U2R


2. 5 different types of connections are encoded into integers
-----------
NORMAL         ==>   0
DOS            ==>   1
PROBE          ==>   2
R2L            ==>   3
U2R            ==>   4


3. Categorial features (protocol_types, service, flags) are encoded into integers.
-----------
protocol_types:
['tcp' 'udp' 'icmp']                          ==> [0, 1, 2]
service:
['ftp_data', 'telnet', ... 'rje', 'harvest']  ==> [0, 1, .... 67, 68]
flags:
['SF', 'S0', ...  ,'S2', 'OTH']               ==> [ 0, 1 ... , 9, 10]


4. Scale features to values between 0 and 1
------------
For each feature, subtract the min value from the datapoint and divide it by the max value subtracted by the min value.


5. Binarize labels to vectors of the size 5
------------
0            ==>   [1,0,0,0,0]
1            ==>   [0,1,0,0,0]
2            ==>   [0,0,1,0,0]
3            ==>   [0,0,0,1,0]
4            ==>   [0,0,0,0,1]



### Implementation

The complete project was setup using jupyter notebooks. As a first step a new anaconda environment was created to ensure all employed software meets the Udacity project requirement. A number of basic libraries (numpy, pandas, scikit-learn, tensorflow) were then installed.  The first notebook was setup to preproccess the data as described in the previous section (transform_data.ipynb)

To ensure the features and labels were correctly formatted a basic feed-forward neural network was created (basic_nn.ipynb) and tested successfully. The output of the loss function of the train and validation data were then visualized with matplotlib for later hyperparameter tuning. In the next step, functions were provided to compute evaluation metrics (TPR/Recall, FPR, Precision, F1-Score), all based on the confusion matrix, which is computed every time the model is trained. Finally a dashboard was created with matplotlib that compares the model performance with the benchmark performance for each type of connection.

In the next step, the exisiting simple feed forward neural network architecture was modified to an RNN LSTM architecture. This included the reshaping of inputs as well as the network function itself (basic_rnn.ipynb). This notebook was then extended by a multi-layer LSTM archticture (mutlicell_rnn).

Finally a dropout wrapper was added to the lstm network (multicell_rnn_dropout).




### Refinement

%learning_rates = [1./10**i for i in range(1,7)]
%batch_size = [2**i for i in range(4,10)] # [16, 32, 64, 128, 256, 512]
%n_hidden = [2**i for i in range(4,10)]  [16, 32, 64, 128, 256, 512]
%keep_rates = [0.5 + 0.05 * i for i in range(10)]
%training_iters = [i for i in range(5000000, 25000000, 1000000)]
%training_iters = 12000000




### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_


The model parameters chosen for the project all seem in a reasonable range. Testing was performed with varius parameter combinations, and small changes in the parameter did not greatly affect the performance of the results. The author comes to the conclusion that the model can be trusted.


As previous authors have shown (Staudemeyer, Wei, aufzählen) LSTMs architectures are generally well suited for the classification of malicous connections based on the KDD99 dataset. Unfortunately with the given architecture the author was not able to beat benchmark accuracy and other evelaution matrics, however got very close (umformulieren).

All the chosen parameters (number of hidden units, number of layers, batch size, training iterations) are in a reasonable range, even though the learning rate is rather low. Since the testing set contains attack types that are not present in the training set, with an accuracy of 92.xx the model generalizes well, even though r2l and u2l metrics perform poor, however they perform in the benchmak model also poor (umschreiben).

Pertubations ?

- The results perform slightly worse than the benchmark model, but all chosen indicates are in a similar range. Hence the model can be trusted.






### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


The found results are slightly worse than the benchmark model. Discussion of paramters of every parameter.

The final solution is able to identify the majority of attacks, generatif low false positive rates for most of the malicous connections. 





### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:

- _Have you thoroughly summarized the entire process you used for this project?_  DONE
- _Were there any interesting aspects of the project?_ DONE
- _Were there any difficult aspects of the project?_ DONE
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_ DONE


The project started by searching for dataset and problem, that would both satisfy my interest for deep learning and security related topis. Once an appropriate dataset was found, a technique had to be determined that would provide meaningful results for the chosen problem. 

After both dataset and approach was chosen, a work environment was set up (installing packages, creating first notebook) and the data  preprocessed to have the right format to be feed to a neural network. To verify the format of the data, a basic feed-forward network was setup and the data was tested on it. Subsequently the basic neural network was altered and extended to the final RNN-LSTM architecture. 

After setting up the code for the model, the hyperparameters (number of layers, number of hidden layers, batch size, learning rate, number of training iterations) were chosen after numerous experiments via grid search. Hier noch zu Dropout schreiben.

The project was definitely interesting, given the freedom to choose from any dataset and use any appropriate algorithm for the chosen goal. Without a template provided that would guide through the project, exploring the problem and gradually building up knowledge and a code base was a great experience.

Since a lot of examples of LSTM implementations work with univariate sequences, like text or unlabeled time-series, it was definitely challenging to adjust to architecture of the model to make use of a multivariate dataset.

The expectation was to at least match the accuracy and the other evaluation metrics of the benchmark model with the chosen approach. That expectation unfortunately could not be live up to. With the current architecture it is clearly inferior to the benchmark model.



### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_


One approach that should be considered is to group the inputs into longer sequences, instead of feeding single connections to the model batchwise. A major obstacle related to this approach lies in the order of the connections. Since the connections are chronologically ordered, connections from different ip adresses are mixed together, making it really hard to filter connections that represent a sequence of attacks. Being able to filter these sequences would potentially incraease the performance of the model. However it remains unclear to the author how inputs and labels would have to be reshaped or newly created in order to be feed to an LSTM RNN. 

Another potential improvement in terms of computational efficieny lies in the reduction of the input features used in the project. As seen in the correlation matrix most of the higher level features are strongly correlated and thus provide little, if not any additional information for classifying different types of malicous connections.






























