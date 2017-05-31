# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
December 31st, 2050

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

Since the rise of the Internet in the late 90s, an exponentially growing number of connected devices is shaping our world. While initially limited to personal computers and notebooks in mostly Western societies, smartphones have emerged in the late 2000s, bringing connectivity to formerly excluded societies in developing countries as well. This development is now increasingly accelerated by the rise of IoT, affecting both consumer and manufacturing markets.

With this ever growing number of connected devices, more and more traffic is transferred over networks, generating a confusing amount of log data that has to be monitored. Given this development, manual monitoring becomes increasingly infeasible to prevent cyber-attacks. Network Intrusion Detections Systems (NIDS) can help system administrators to detect network breaches, however setting up policies that are both flexible and effective against unforeseen attacks can be challenging.

While applicable to the detection of well-known and precisely defined attacks, traditional approaches of intrusion detection have proven to be ineffective in cases of novel intrusions. Such methods are mostly threshold-based and expressed in counts or distributions (See Garcia-Toedora, 2009).
Applying machine learning in the analysis of log files can help NIDS to learn more complex behavior. In particular it allows them to differentiate whole classes of traffic and hence enables them to detect irregularities and previously unseen attacks. With their ability to achieve high levels of abstractions, deep learning methods seem suitable for tackling this kind of problem.


### Problem Statement

The problem that will be addressed in this project is to create a model, which predicts whether a connection between a source and target IP, represented by a set of features, is an attempt to attack the source network or not. In case of an attack, the model has to predict what kind of attack a malicious connection represents.

An additional challenge which will be addressed in this project is that the type of attacks that the model will be trained on, differ from the type of attacks that its performance will tested on. This is a specific characteristic of the used dataset to make it more realistic. However this does not represent a major constraint; as there is widespread believe that most of the novel attacks can be derived from the known attacks (see Tavallee, 2009).

Assuming that attacks produce a distinctive sequence of events, this capstone project seeks to model network traffic as a time series by applying a long short-term memory (LSTM) Recurrent neural network. Unlike feedforward neural networks, RNNs have cyclic connections making them powerful for modeling sequences. While standard RNNs are only able to model sequences that are up to 10 time steps long, LSTM based RNNs overcome this constraints and allow for much longer sequences to be considered.



### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


The measures of performance for this project will be the Mean Squared Error (MSE) and Root Mean Squared
Error (RMSE) calculated as the difference between predicted and actual values of the target stock ticker at daily
market close. The MSE and RMSE are standards for measuring error for machine learning predictions and will offer
consistency across all the studies developed, given that all the datasets will be normalized prior to model fitting.
Additionally, the visualizations from each study will be graphically analyzed for their degree of fit, ability incorporate
volatility or lag across time, and macro trends for diverging from the test data set. Lastly, the finalized study will be
measured for the delta between the performance of the benchmark model(s) and our finalized and tuned deep learning

Deep Learning Stock Value Predictor
model. For this last metric, comparison will be made to the SPY ETF, the initial linear regression model developed,
and the Lucena Research tool shown in the Machine Learning for Trading11 course. The graph displayed in that course
shows a 3.64x improvement over the S&P benchmark from 1/1/2009 to 6/18/



The following metrics will be used to evaluate the performance of the applied model in comparison to the benchmark model:
Accuracy: Defined as the proportion of true results (both true positives and true negatives) among the total number of datapoints examined.
True Positive Rate: Defined as the ratio between the number of attacks correctly categorized as attacks and the total number of attacks.
False Postitive Rate: Defined as the ratio between the number of normal connections wrongly categorized as attacks and the total number of normal connections.
Precision: Defined as the ratio of the number of true positives divided by the number of true positives and false positives.
Recall: Defined as the ratio of number of true positives records divided by the number of true positives and false negatives.
Evaluation Metrics will be applied for the Global Test Dataset as well as for every type of connection (NORMAL; PROBE; DOS; U2R; R2L)