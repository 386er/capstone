# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
December 31st, 2050

## I. Definition
_(approx. 1-2 pages)_

### Project Overview


Since the rise of the Internet in the late 90s, an exponentially growing number of connected devices is shaping our world. While initially limited to personal computers and notebooks in mostly Western societies, smartphones have emerged in the late 2000s, bringing connectivity to formerly excluded societys in developing countries as well. This development is now increasingly accelerated by the rise of IoT, affecting both consumer and manufacturing markets.
With this ever growing number of connected devices,  more and more traffic is transferred over networks, generating a confusing amount of log data that has to be monitored. Given this development, manual monitoring becomes increasingly infeasable to prevent cyber attacks. Network Intrusion Detections Systems (NIDS) can help system administrators to detect network breaches, however setting up policies that are both flexible and effectice against unforseen attacks can be challenging.


Applying machine learning in the analysis of log file datasets can help to improve NIDS and thus strengthening the security posture of organizations.





### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_



Given a sequence of network connections between a source and target IP, each of which is represented by a total of 41 features, the problem is to predict whether a connection represents an attempt to attack the source network and to correctly predict the type of attack. The suggested solution is to model the network traffic as a time series by applying a long short-term memory (LSTM) recurrent neural network. Whether connections were correctly labeled can be clearly observed and by setting a random seed the problem can be reproduced.









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