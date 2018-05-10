# Classifying Amazon Reviews: Helpful or Not?

## Motivation

Have you ever been looking for an item on Amazon that wasn’t particular popular and noticed that none of the reviews are tagged helpful? This project builds a model to classify whether a review is “helpful” in order to prioritize “helpful” untagged reviews.

## Model

I used AWS EMR with Apache Spark to create combinations of feature pipelines that feed into various models to end up with a large number of models to pick the best from.

### Feature Pipeline:
- Featurize review text
- Choose reduction method
- Engineer new features
- Train models

### Top Pipeline + Model: TFIDF -> Ridge -> Logistic w/CV
- Class Balance: .535
- Accuracy: .797
- Precision: .767
- Recall: .811
- F1 Score: .78

## Technology

To be able to run all these different feature pipeline and model combinations on such a large dataset of reviews, I had to make use of Apache Spark and Amazon’s EMR Cluster. I stored all my data into an AWS S3 bucket and connected it to my AWS EMR cluster with the configuration shown below.
- Master Node - 8 Cores 15G Mem
- 14 Core Nodes - 8 Cores 15G Mem

