# The Eye’s Tale: Understanding Empathy through Gaze Patterns

## Description

This repository contains research findings and analysis focused on the intriguing relationship between gaze dynamics and empathy. Leveraging extensive datasets from EyeTzip and Raw Data, we explore the correlation between gaze patterns and empathy levels. Our study employs cutting-edge machine learning models, with a spotlight on the Random Forest Regressor, to uncover the pivotal role of gaze metrics. In-depth insights are revealed through unsupervised learning techniques like K-Means clustering. The implications of this research extend to refining models, expanding datasets, and applying findings in clinical and social contexts, merging psychology and technology in the realm of empathy assessment.

# Jupyter Notebook Analysis

## Overview

This Jupyter Notebook contains a total of 141 cells, categorized as follows:

- Code Cells: 107
- Markdown Cells: 34

## Purpose

The notebook appears to be a comprehensive file that likely includes both code and explanations. Given the higher number of code cells, it's likely that the notebook is heavily focused on computational tasks.

## Quick Navigation

- Code cells primarily consist of Python code.
- Markdown cells likely contain explanations, notes, or other types of textual content.
  
## How to Use

1. Clone the repository where this notebook is stored.
2. Navigate to the notebook file and open it using Jupyter Notebook or Jupyter Lab.
3. Execute the cells in sequence or as needed, based on your requirements.

## Requirements

- Python environment (preferably 3.x)
- Jupyter Notebook or Jupyter Lab for running the notebook

## Note

For a more detailed understanding of each section, it is advisable to go through the notebook.


## Analysis Method

The analysis is primarily conducted using Python programming language. Various data manipulation and analysis libraries like Pandas, NumPy, and Matplotlib are used for pre-processing and visualization. The study employs statistical methods to understand the underlying patterns in the data.

### Machine Learning Part

The machine learning component of this research focuses on both supervised and unsupervised learning techniques:
We add featuring new metrics like 'Gaze Duration', 'Fixation Count', and 'Saccade Velocity'.
In our comprehensive analysis of gaze dynamics and empathy, we evaluated a total of seven machine-learning models.

A deep study was conducted using different algorithms using the machine learning and deep learning and all the results are summarized perfactly in the report. 

Type of ML and DL algorithms that are used in it.
1. Finding the correlation matrix to find the nuber of correlated features with the target variable.
2. Using the unsupervised learning to find the optimal number of cluster which can provide us more insights into empathy score.
3. Next we have used the neural network using the keras api on the top of tensorflow.
4. we have used to RRN_GRU to find the classification task with getting the resonable accuracy.

### Conclusion and Key Takeaways

1. There is a significant correlation between gaze patterns and empathy levels.
2. Random Forest Regressor proves to be an effective model for predicting empathy based on gaze metrics.
3. K-Means clustering reveals inherent groupings within the data, which can be useful for targeted interventions.
4. The findings have implications in both clinical and social contexts, merging psychology and technology.
