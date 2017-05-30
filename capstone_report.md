# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
December 31st, 2050

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

Since the rise of the Internet in the late 90s, an exponentially growing number of connected devices is shaping our world. While initially limited to personal computers and notebooks in mostly Western societies, smartphones have emerged in the late 2000s, bringing connectivity to formerly excluded societys in developing countries as well. This development is now increasingly accelerated by the rise of IoT, affecting both consumer and manufacturing markets.
With this ever growing number of connected devices, more and more traffic is transferred over networks, generating a confusing amount of log data that has to be monitored. Given this development, manual monitoring becomes increasingly infeasable to prevent cyber attacks. Network Intrusion Detections Systems (NIDS) can help system administrators to detect network breaches, however setting up policies that are both flexible and effectice against unforseen attacks can be challenging.

While applicable to the detection of wellknown and precisely defined attacks, traditional approaches of intrusion detection have proven to be inneffective in cases of novel intrusions. Such methods are mostly threshold-based and expressed in counts or distributions (See Garcia-Toedora, 2009).

Applying machine learning in the analysis of log files, can help NIDS to learn more complex behaviour. In particular it allows them to differentiate whole classes of traffic and hence enables them to detect irregularities and previously unseen attacks.
With their ability to achieve high levels of abstractions, deep learning methods seem suitable for tackling this kind of problems.
This capstone project seeks to model network traffic as a time series by applying a long short-term memory (LSTM) Recurrent neural network.



### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_


The challenge of this project is to accurately predict the future closing value of a given stock across a given time period in the future. Because almost all stocks have a large range of historical daily closing price values, machine learning models for consideration should be capable of making predictions with time series data and should be trainable over
a long period of time. The subsequently predicted values should be backtested for their accuracy against a sizeable period of time as well. With those constraints over the stock data duration and model capabilities, the project will experiment with various types of machine learning and deep learning models as well as various architectures of assembling such models.
The key questions this investigation seeks to answer are:
• How well do deep learning models perform compared to machine learning models for stock price prediction?
• Within the category of deep learning models that work effectively with time series data, which model types and
architectures perform optimally?
• Can a relatively simple deep learning model predict well enough to de ne a trading strategy that outperforms
benchmarks?
Before beginning development, the originally hypothesized solution for predicting stock prices was a based on recent literature8 and was centrally composed of a Recurrent Neural Network built with Keras9. During development, the experimentation with model types diverged into alternate model types and more complex architectures, including those in the Long Short-Term Memory10 category.







### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_




The following metrics will be used to evaluate the performance of the applied model in comparison to the benchmark model:
Accuracy: Defined as the proportion of true results (both true positives and true negatives) among the total number of datapoints examined.
True Positive Rate: Defined as the ratio between the number of attacks correctly categorized as attacks and the total number of attacks.
False Postitive Rate: Defined as the ratio between the number of normal connections wrongly categorized as attacks and the total number of normal connections.
Precision: Defined as the ratio of the number of true positives divided by the number of true positives and false positives.
Recall: Defined as the ratio of number of true positives records divided by the number of true positives and false negatives.
Evaluation Metrics will be applied for the Global Test Dataset as well as for every type of connection (NORMAL; PROBE; DOS; U2R; R2L)