# Machine Learning Engineer Nanodegree
## Capstone Proposal
Tobias Burri   
April 7th, 2017

## Proposal

### Domain Background
_(approx. 1-2 paragraphs)_

With an ever growing number of connected devices, more and more data is transferred over networks, generating a confusing amount of data that has to be monitored. Given this development, manual monitoring becomes increasingly infeasable to prevent cyber attacks. Network Intrusion Detections Systems (NIDS) can help system administrators to detect network breaches, however setting up policies that are both flexible and effectice against unforseen attacks can be challenging. Applying machine learning in the analysis of large log file datasets can help to improve NIDS and thus strengthening the security posture of organizations.

Just like machine learning itself, acedemic research in the application of machine learning to cyber security has grown exponentially in the last years. A brief glance at the search results of Google Scholar for the terms "machine learning cyber security" confirms this development. nochmal überarbeiten, paper nennen


A Neural Network Component for an Intrusion Detection System
Herve DEBAR; Monique BECKER; Didier SIBONI	
Published in: Proceeding SP '92 Proceedings of the 1992 IEEE Symposium on Security and Privacy
 





### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).





### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.


Dataset downloaded from: https://github.com/defcom17/NSL_KDD


The dataset used for the capstone project will be the NSL-KDD dataset, which is an improved* version of the KDD Cup 99 dataset. 


The latter was created by collecting raw TCP dump data from a simulated U.S. airforce local-area network in the course of 9 weeks. During the data collection the network was deliberately attacked multiple times. Each datapoint represent a connection between a source and target IP, using a well defined protocol, during a well defined time frame. The datapoints consist of 41 features and are labeled either as normal connections or as an attack.


Some of the features are directly derived from the TCP/IP connections during a time interval, however the dataset includes also "higher-level" features that were derived from some of the basic features of the dataset. The first, 'basic' set of features include inputs like duration of the connection (in seconds), the protocol type, the number of bytes from the source to the destination and vice versa. 

The latter, 'higher-level' features include inputs like the number of connections to the same host as the current connecton in the past two seconds, number of connection that use the same service, number of failed login attempts.



udpstorm,dos,test








These features include basic features derived directly
from a TCP/IP connection, traffic features accumulated
in a window interval, either time, e.g. two seconds or
number of connections, and content features extracted from
the application layer data of connections. Out of 41 features,
three are nominal, four are binary, and remaining 34 features
are continuous. The training data contains 23 traffic classes
that include 22 classes of attack and one normal class. The
test data contains 38 traffic classes that include 21 attacks
Figure 2: Various steps involved in our NIDS implementation
classes from the training data, 16 novel attacks, and one normal
class.







There is a total of 38 types of attacks, however they are grouped into 4 categories

- DOS (denial of service) e.g. syn flood;
- R2L (unauthorized access from a remote machine) e.g. guessing password;
- U2R (unauthorized access to local superuser privileges)  e.g., various 'buffer overflow' attacks;
- PROBING (surveillance and other probing) e.g., port scanning;

It is important to note that the test data is not from the same probability distribution as the training data, and it includes specific attack types not in the training data. 
The test data contains many attacks that were not injected during the training data collection phase to make the intrusion detection task realistic. It is believed that most of the novel
attacks can be derived from the known attacks. Thereafter, the training and test data were processed into the datasets
of five million and two million TCP/IP connection records, respectively. UMFORMULIEREN.











The KDD Cup dataset has been widely used as a benchmark
dataset for many years in the evaluation of NIDS. One
of the major drawback with the dataset is that it contains an
enormous amount of redundant records both in the training
and test data. It was observed that almost 78% and 75%
records are redundant in the training and test data, respectively
[20]. This redundancy makes the learning algorithms
biased towards the frequent attack records and leads to poor
classification results for the infrequent, but harmful records.
The training and test data were classified with the minimum
accuracy of 98% and 86% respectively using a very simple
machine learning algorithm. It made the comparison task
difficult for various IDSs based on different learning algorithms.
NSL-KDD was proposed to overcome the limitation
of KDD Cup dataset. The dataset is derived from the KDD
Cup dataset. It improved the previous dataset in two ways.
First, it eliminated all the redundant records from the training
and test data. Second, it partitioned all the records in
the KDD Cup dataset into various difficulty levels based on
the number of learning algorithms that can correctly classify
the records. After that, it selected the records by random
sampling of the distinct records from each difficulty level in a
fraction that is inversely proportional to their fraction in the
distinct records. These multi-steps processing of KDD Cup
dataset made the number of records in NSL-KDD dataset
reasonable for the training of any learning algorithm and
realistic as well.
















### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

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