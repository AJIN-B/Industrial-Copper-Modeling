
# Industrial Copper Modeling

## Overview

This project aims to develop two machine learning models for the copper industry to 
address the challenges of predicting selling price and lead classification.
**ML Regression model** which predicts continuous variable of **Selling_Price**.
**ML Classification model** which predicts **Status: WON or LOST**.
Created a **streamlit page** where you can insert each column value and you will get 
the predicted value of **Selling_Price  or Status(Won/Lost)**.


## Dataset Overview

The copper industry deals with less complex data related to sales and pricing.
However, this data may suffer from issues such as **skewness** and noisy data, 
which can affect the accuracy of manual predictions. Dealing with these challenges 
manually can be time-consuming and may not result in optimal pricing decisions. 
By utilizing advanced techniques such as **data normalization, feature scaling, outlier detection** and 
leveraging algorithms that are robust to skewed and noisy data.

### Approach 

- **Data Understanding:** Identify the types of variables (continuous, categorical) and their distributions. 
- **Data Preprocessing:** 
    - Handle missing values with mean/median/mode.
    - Treat Outliers using IQR or Isolation Forest from sklearn library.
    - Identify Skewness in the dataset and treat skewness with appropriate data transformations,
    - Encode categorical variables  as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable.

- **EDA:** visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot, violinplot.
- **Feature Engineering:** Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. 

### Model Building and Evaluation:
- Split the dataset into training and testing/validation sets. 
- Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. 
- Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.
- Interpret the model results and assess its performance based on the defined problem statement.

#### To run this app

`python -m streamlit run app.py`  **or**  `streamlit run app.py`


## Installation

### Basic Requirements:

- __[Python 3.10](https://docs.python.org/3/)__
- __[sklearn](https://pypi.org/project/scikit-learn/1.2.2/)__ 
- __[Pandas](https://pandas.pydata.org/docs/)__
- __[Streamlit](https://docs.streamlit.io/)__
- __[Numpy](https://numpy.org/doc/)__ 
- __[matplotlib](https://pypi.org/project/matplotlib/)__ 


### To install the basic Requirements

`pip install - r requirements.txt`

### Contact

- Name : Ajin B

- GITHUB : https://github.com/AJIN-B

- LINKEDIN: https://www.linkedin.com/in/ajin-b-0851191b0/

- Mail : ajinleo9940@gmail.com
