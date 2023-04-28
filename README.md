# distributed-naive-bayes
Distributed Naive Bayes algorithm from scratch in PySpark using Kubernetes cluster on Google Cloud Platform

## Training and Evaluation:

1. Train and evaluate the algorithm in a distributed manner with n machines using Kubernetes on Google Cloud Platform.

2. Save the learned parameters.


## Deployment:
1. Implement a server that runs on a single machine on Google Cloud Platform and does the following:

   a. Loads the learned parameters.

   b. Provides an HTTP API that receives a text as input, processes it using the Naive Bayse algorithm and returns the sentiment score to the client as a response.
