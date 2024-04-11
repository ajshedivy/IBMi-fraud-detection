# IBM i Fraud Detection Demo

# Overview

This demo contains an end-to-end AI/ML opps example that integrates IBM i and Openshift environments. In this demo, tabular transactions stored on a Db2 for i database (Db2) are used to train a fraud detection model using a reusable Kubeflow Pipeline. The model makes a "fraud" or no fraud" prediction for a given transaction. 

![alt text](docs/image-1.png)

### End-to-end workflow outline
The end-to-end workflow covers the following steps:
1. The training data is stored inside a IBMi db2 database.
2. By applying pattern 1 and using trino, this data is integrated into the end-to-end workflow.
3. By applying pattern 3, MLOps steps continue the end-to-end workflow:
    - Explore the data using EvidentlyAI
    - Experiment and train a fraud detection model using Jupyter Lab and TensorFlow
    - Monitor model training using TensorBoard and evaluate the model using TensorFlow
    - Export the model to the ONNX format, a portable format for exchanging AI models
4. By applying pattern 2, deploy the model using KServe and connect it to a fraud detection Flask application


## The Dataset 💳 

This example data set uses a collection of transactions consisting of Online, Swipe, and Chip transaction types. 

The data is designed to be used as sample data for exploring IBM data science and AI tools. Models trained from this data are not suitable for real world applications, but the dataset is a wonderful resource for education and training use cases.

The data can be downloaded [here](https://ibm.ent.box.com/v/tabformer-data/folder/130747715605)

# Installation Guide

The assumption is that the example will be deployed on:

    Red Hat OpenShift 4.12 running on IBM Power.
    Kubeflow 1.7 or newer
    IBM Power 9 or newer hardware

The container images in this example use a Python from RocketCE with is built with optimizations for IBM Power. These images only run on Power 9 or newer hardware.

The steps to deploy the application are outlined in this section.

## Configure Db2 for i database

First and foremost, we need to upload the transaction data to an IBM i system. This can be done using Data Transfer in ACS (Access Client Solutions). More detailed approach coming. 

## Configuring Trino and Db2 for i Connector

Detailed instructions for installing Trino on OpenShift: [here](https://community.ibm.com/community/user/powerdeveloper/blogs/natalie-jann/2022/11/07/simplify-data-access-using-trino-on-ibm-power)







