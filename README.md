Locating and Filling Missing Words in Sentences
========================================================

## Introduction
This is a project, which considers locating and filling missing words in sentences based on language modeling. By investigating the statistical connection among words surrounding the candidate location and the candidate word, we manage to achieve a missing word location accuracy and a missing word filling accuracy as high as 52% and 32%, respectively.

## Table of Contents
* What it can do
* What it includes
* Contributors
* Additional information

## What it can do
Train the language model by a large training set including millions of complete sentences, then apply the model to predict missing word location and filling the missing word for each incomplete sentence. Note that there should only one missing word for each sentence, and it can neither be the first nor the last one.

## What it includes
* data\vocabulary-14126.txt: vocabulary listing 14216 high-frequency words and their corresponding frequencies
* data\train_v2.txt and test_v2.txt: too large to include, please download them from https://www.kaggle.com/c/billion-word-imputation/data, or you can use your own data and regenerate the high-frequency vocabulary.
* LocationAndFillingCrossValidation.py: model training and cross validation (based on only train_v2.txt)
* LocationAndFillingTestData.py: model training and produce results for testing data (train_v2.txt and test_v2.txt)
* Presentation Slides.pdf & Project Report.pdf: supporting documents

## Contributors
* Tianlong Song
* Zhe Wang

## Additional information
Please refer to the presentation slides and project report in this repository.
