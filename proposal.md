# Machine Learning Engineer Nanodegree
## Capstone Proposal
Tobias Burri   
April 7th, 2017
## Proposal

### Domain Background

With an ever growing number of connected devices, more and more data is transferred over networks, generating a confusing amount of data that has to be monitored. Given this development, manual monitoring becomes increasingly infeasable to prevent cyber attacks. Network Intrusion Detections Systems (NIDS) can help system administrators to detect network breaches, however setting up policies that are both flexible and effectice against unforseen attacks can be challenging. Applying machine learning in the analysis of large log file datasets can help to improve NIDS and thus strengthening the security posture of organizations.

Just like machine learning itself, acedemic research in the application of machine learning to cyber security has grown exponentially over the last years. A brief glance at the search results of Google Scholar for the terms "machine learning" and "cyber security" confirms this development.

Year | Number of results
------------ | -------------
2001 | 61
2006 | 309
2011 | 1260
2016 | 6010

However early research in this field dates back to the early nineties (see Debar, 1992). (Noch etwas beifügen!)

### Problem Statement

Given a set of the inputs from a labeled dataset of network connections, malicious connections shall be identified and correctly labeled. Whether connections were correctly labeled can be clearly observed and the problem can be reprocued.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The dataset used for the capstone project will be the NSL-KDD dataset, which is an improved* version of the KDD Cup 99 dataset. It was downloaded from https://github.com/defcom17/NSL_KDD
A major improvement of the NSL-KDD dataset is the removal of redundant records and some further rearrangement of the data. For an exact description of how the dataset was modified please see http://www.unb.ca/cic/research/datasets/nsl.html

The original KDD dataset was created by collecting raw TCP dump data from a simulated U.S. airforce local-area network in the course of 9 weeks. During the data collection the network was deliberately attacked multiple times. Each datapoint represents a connection between a source and target IP, using a well defined protocol, during a well defined time frame. The datapoints consist of 41 features (!!prüfen!!) and are labeled either as normal connections or as an attack.

Some of the features are directly derived from the TCP/IP connections during a time interval, however the dataset includes also "higher-level" features that were derived from some of the basic features of the dataset. 

- The 'basic' set of features include inputs like duration of the connection (in seconds), the protocol type, the number of bytes from the source to the destination and vice versa. 
- The 'higher-level' features include inputs like the number of connections to the same host as the current connecton in the past two seconds, number of connection that use the same service, number of failed login attempts.

There is a total of 38 types of attacks, grouped into 4 categories:

- DOS (denial of service) e.g. syn flood
- R2L (unauthorized access from a remote machine) e.g. guessing password
- U2R (unauthorized access to local superuser privileges)  e.g. various 'buffer overflow' attacks
- PROBING (surveillance and other probing) e.g. port scanning

Not all of the attack types that occur in the test set, occur in the training set. This is a specific characteristic of the dataset to make it more realistic. However it is believed, that most of the novel attacks can be derived from the known attacks (see Tavallee, 2009) All types of attacks will be labeled as one of the categories they are grouped in. 


### Solution Statement
_(approx. 1 paragraph)_

For the appropriate classification of malicious connections, a neural network will be calibrated and applied.

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_
In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.


### Evaluation Metrics

The following metrics will be used to evaluate the performance of both the benchmark model and the applied model:

- Accuracy: Defined as the proportion of true results (both true positives and true negatives) among the total number of datapoints examined.

- Precision: Defined as the ratio of the number of true positives divided by the number of true positives and false positives.

- Recall: Defined as the ratio of number of true positives records divided by the number of true positives and false negatives.

- F-Measure: Defined as the harmonic mean of precision and recall and represents a balance between them.


### Project Design
_(approx. 1 page)_
In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.


-----------
**REFERENCES**
1. Does the proposal you have written follow a well-organized structure similar to that of the project template?
1. Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
1. Would the intended audience of your project be able to understand your proposal?
1. Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
1. Are all the resources used for this project correctly cited and referenced?

[1] M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

[2] H. Debar; M. Becker; D. Siboni, "A Neural Network Component for an Intrusion Detection System", Proceeding SP '92 Proceedings of the 1992 IEEE Symposium on Security and Privacy, 1992

