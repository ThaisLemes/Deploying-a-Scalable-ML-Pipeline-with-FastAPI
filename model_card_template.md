# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The goal of this model is to find out if an individual made over or less than 50 thousand a year based on workclass, education, marital-status,occupation,relationship,race,sex, and native-country.

Type of model: RainForestClassifier 

Dataset: UCI Census Income
https://archive.ics.uci.edu/dataset/20/census+income.

## Intended Use

The goal of this model is  to predict income category, if an individual made more or less than $50 thousand year based on categorical information from the U.S. Census dataset. It is intended for educational and exploratory purposes, such as demonstrating  ML pipelines with FastAPI.

## Training Data

This model was trained on the UCI Census database from 1994, which contains 48842 individual above to 16 and 99 years old. The dataset contains 15 features such as: age, workclass,fnlgt, education,education-num, marital-status, occupation,relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary. 
The training data used was 80%
label = Salary


## Evaluation Data

The model was evaluated on a 20% from the UCI Census Income dataset
Performance was measured using precision, recall and F1-Score.

## Metrics

Precision: 0.7222 | Recall: 0.6194 | F1: 0.6669

## Ethical Considerations

The model is intended for learning purposes, the model was trained on the UCI Census data, there are some sensitive data such as age, gender, and race. The data shuld not be used in decision-making contexts. 

## Caveats and Recommendations

This model is intended for educational used. 
The limitations of this data is that only apply to US population and the data is from 1994. 
To improve the model the recommendations would be evaluating fairness metrics, maybe hyperparamenter tuning and feature selection. 


