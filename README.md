---
title: DATA103_filipino_spam_detection
emoji: 🚀
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: spam classifier in the filipino context
license: mit
---

# **Scam/Spam SMS Checker and the Efficacy of the SIM Registration Act**

## **Problem Statement**

Spam text messages in the Philippines have increased in number in recent years, with more than 6 million reported SMS scams in 2024. In an article published by GMA News Online, Undersecretary Alexander Ramos, the executive director of the Cybercrime Investigation and Coordinating Center, shares that one of the widespread schemes of scammers has been the imitation of official brands.

The SIM Registration Act, implemented in 2022, was created to reduce such scams and cybercrimes. However, it seems that the issue is still prevalent. Through this project, the group aims to detect spam SMS using machine learning techniques, determine their frequency, and compare the same with data from before the implementation of the SIM Registration Act to determine whether this law has proved to be efficacious or not.

## **Dataset Description**

**Merged Filipino SMS Messages**

The merged dataset comprise of three different sms messages datasets available online within the filipino-context curated for the application of this project. We avoided the UCI SMS Repository as this does not provide messages being received specific by filipinos. 

## **How to use app**

Simply use one of the trained clasifier models to classify if the message in input prompt is spam or not. Alternatively, you may select a sample from our `test.csv` file for you to check how well the performance is for sms messages.

## **Model training**

The project will consider a train-val-test split for a cross-validation (cv) training with hyperparameter tuning considered per fold-run.

The group will consider <mark> four (4) traditional and explainable classifiers</mark> that are known to be used for spam detection. These are to be the two variants of `Naive-Bayes (NB)`, multinomial and complement (noted to handle class imbalances well), `Support Vector Machine`, and `RandomForest`.

The project utilized `MLflow` to track training and artificats (evaluation metrics) per general run when the model is called; We have put this all under a function.

## **Summary of best model configuration and model metrics**

The model training above already provides how the model metrics are extracted; All evaluation metrics and visualization are saved as artificats under `mlflow`. With this, presented below is the training summary done under initial run parameters for `preprocessor=tfidf` and `cv_folds=5` for all models considered of the study.

Models considered are the following: complement_NB (cNB), multinomial_NB (mNB), random_forest (rf), support_vector_machine (svm)

