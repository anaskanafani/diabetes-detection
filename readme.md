# Diabetes Classification Project Documentation

  

## Introduction

This project aims to build a machine learning model to classify individuals as diabetic or non-diabetic based on various health indicators. The dataset used is the "Diabetes Binary Health Indicators BRFSS 2015"  from the CDC.

  

## Table of Contents

1.  [Import Libraries](#import-libraries)

2.  [Load Dataset](#load-dataset)

3.  [Data Exploration](#data-exploration)

4.  [Data Preprocessing](#data-preprocessing)

5.  [Model Building](#model-building)

6.  [Model Evaluation](#model-evaluation)

7.  [Findings and Learnings](#findings-and-learnings)

  

## Import Libraries

    python
    
    %matplotlib inline
    
    import numpy as np
    
    import pandas as pd
    
    import matplotlib.pyplot  as plt
    
    import seaborn as sns
    
    from ydata_profiling import ProfileReport
    
    from sklearn.model_selection  import train_test_split, GridSearchCV
    
    from sklearn.preprocessing  import StandardScaler, MinMaxScaler, RobustScaler
    
    from sklearn.linear_model  import LogisticRegression
    
    from sklearn.pipeline  import Pipeline
    
    from imblearn.over_sampling  import SMOTE
    
    from sklearn.ensemble  import RandomForestClassifier, GradientBoostingClassifier
    
    from sklearn.neighbors  import KNeighborsClassifier
    
    from sklearn.naive_bayes  import GaussianNB
    
    from sklearn.tree  import DecisionTreeClassifier
    
    from xgboost import XGBClassifier
    
    from catboost import CatBoostClassifier

  

## Load Dataset

`df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")`

  

## Data Exploration

### Profiling Report

    profile = ProfileReport(df,  title="Profiling Report")
    profile.to_file("analysis_report.html")

The profiling report generated provides a comprehensive overview of the dataset, including distributions, missing values, correlations,  and more.
  

### Basic Exploration

    print("First few rows of the dataset:")
    df.head()

    print("Columns in the dataset:")
    df.columns
    
    print("Statistical summary of the dataset:")
    df.describe().T
    
    print("Information about the dataset:")
    df.info()
    
    print("Number of missing values in each column:")
    df.isnull().sum()
    
    print("Number of duplicated rows in the dataset:")
    df.duplicated().sum()
    
    print("Number of unique values in each column:")
    df.nunique()
    
    print("Correlation matrix:")
    df.corr(numeric_only=True)

### Visual Exploration

    plt.figure(figsize  =  (16,10))
    sns.heatmap(df.corr(),  annot=True)
    plt.show()
    
![heatmap](https://github.com/user-attachments/assets/2f947c33-1030-47fe-a616-628c01549c2a)

    sns.countplot(x='Diabetes_binary',  data=df)
    plt.title("Class Distribution of Diabetes_binary")
    plt.show()

![output](https://github.com/user-attachments/assets/fc1d293d-583a-46c6-8739-b13a39c7c9bb)

    plt.figure(figsize=(12,  8))
    df.corr()['Diabetes_binary'].sort_values().plot(kind='bar')
    plt.title('Correlation with Diabetes_binary')
    plt.show()
    
![output](https://github.com/user-attachments/assets/948ee57c-fccb-4a2b-a4e4-9c026df020fd)

### Create a mask
    mask  = np.triu(np.ones_like(corr,  dtype=bool))

### Create a custom diverging palette

    cmap  = sns.diverging_palette(100,  7,  s=75,  l=40,  n=5,  center="light",  as_cmap=True)
    plt.figure(figsize=(15,  12))
    sns.heatmap(corr,  mask=mask,  center=0,  annot=True,  fmt='.2f',  square=True,  cmap=cmap)
    plt.show()
    
![output](https://github.com/user-attachments/assets/56db6445-8511-4ac3-b42c-ac3b5aa2cc4e)

  

## Data Preprocessing

### Handling Missing Values and Duplicates

The dataset does not contain any missing values but has duplicated rows which were handled accordingly.

### Splitting Data

    X = df.drop(columns='Diabetes_binary')
    
    y  = df['Diabetes_binary']
    
    X_train, X_test, y_train,  y_test  = train_test_split(X, y,  test_size=0.2,  random_state=42)

  

### Scaling Data
Various scalers were used to handle the data:

 - StandardScaler
 - MinMaxScaler
 - RobustScaler

### Handling Imbalanced Data

    smote = SMOTE(random_state=42)
    X_train_res,  y_train_res  = smote.fit_resample(X_train, y_train)

  

## Model Building
Various models were built using the following classifiers:

 - Logistic Regression
 - RandomForestClassifier
 - GradientBoostingClassifier
 - KNeighborsClassifier
 - GaussianNB
 - DecisionTreeClassifier
 - XGBClassifier
 - CatBoostClassifier

### Example Pipeline with Logistic Regression

    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
    ])
    
    param_grid  =  {
    'classifier__C':  [0.1,  1,  10],
    'classifier__penalty':  ['l2']
    }
    
    grid_search  = GridSearchCV(pipeline, param_grid,  cv=5)
    grid_search.fit(X_train_res, y_train_res)

## Model Evaluation
Models were evaluated using metrics such as:

 - Accuracy
 - Precision
 - Recall
 - F1 Score

### Example Evaluation Code

    from sklearn.metrics  import classification_report, confusion_matrix
    y_pred  = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

## Findings and Learnings
1. **Data Quality:** The dataset had a significant number of duplicated rows which needed to be removed.
2. **Feature Importance:** Certain features like BMI, HighBP,  and Age showed higher correlation with diabetes.
3. **Class Imbalance:** The target variable was imbalanced, necessitating the use of techniques like SMOTE to handle it.
4. **Model Performance:** Ensemble models like Random Forest and Gradient Boosting performed better compared to simpler models like Logistic Regression and Naive Bayes.
5. **Hyperparameter Tuning:** GridSearchCV was effective in tuning the hyperparameters and improving model performance.

  

## Conclusion
The project successfully classified individuals as diabetic or non-diabetic using various machine learning models. Ensemble methods proved to be the most effective,  and handling class imbalance was crucial for improving model performance.
