# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# PROGRAM
```

# Filter Methods

import pandas as pd
import numpy as np
import seaborn as sns

wine = load_wine()
x = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

x

y

x.isnull().sum()

col_list=list(x.columns)
print(col_list)

x['magnesium'].max()

x['magnesium'].min()

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
sScaler=StandardScaler()
mmScaler=MinMaxScaler()
maScaler=MaxAbsScaler()
rScaler=RobustScaler()

x['magnesium']

Sscaled_mag=sScaler.fit_transform(x[['magnesium']])
Sscaled_mag

MMscaled_mag=mmScaler.fit_transform(x[['magnesium']])
MMscaled_mag

MAscaled_mag=maScaler.fit_transform(x[['magnesium']])
MAscaled_mag

Rscaled_mag=rScaler.fit_transform(x[['magnesium']])
Rscaled_mag

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_classif 

x

selector1=SelectKBest(score_func=chi2,k=5)
x_new=selector1.fit_transform(x,y)

selected=selector1.get_support(indices=True)
selected_features=x.columns[selected]
print(selected_features)

selector2=SelectKBest(score_func=f_regression,k=5)
x_new=selector2.fit_transform(x,y)

selected=selector2.get_support(indices=True)
selected_features=x.columns[selected]
print(selected_features)

selector3=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector3.fit_transform(x,y)

selected=selector3.get_support(indices=True)
selected_features=x.columns[selected]
print(selected_features)


# Wrapper Method

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000, solver='saga')

sfs_forward = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward')
sfs_forward.fit(X_train_scaled, y_train)

sfs_backward = SequentialFeatureSelector(model, n_features_to_select=5, direction='backward')
sfs_backward.fit(X_train_scaled, y_train)

forward_features = [wine.feature_names[i] for i in sfs_forward.get_support(indices=True)]
backward_features = [wine.feature_names[i] for i in sfs_backward.get_support(indices=True)]

print("Forward Selected Features:", forward_features)
print("Backward Selected Features:", backward_features)


# Embedded Method

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_feature=x.columns[sfm.get_support()]
print(selected_feature)
```
# OUTPUT:
# Dataset (wine dataset)

<img width="1371" height="483" alt="image" src="https://github.com/user-attachments/assets/0eac6431-9a5c-417f-af67-013b31688185" />

# FEATURE SCALING

# Standard Scaler
<img width="518" height="476" alt="image" src="https://github.com/user-attachments/assets/b4377809-8780-48d9-a030-a7a43d7a4c8e" />

# MinMaxScaler

<img width="552" height="475" alt="image" src="https://github.com/user-attachments/assets/08d51869-c1fa-4f50-aea8-9b046351d215" />

# MaxAbsScaler

<img width="538" height="463" alt="image" src="https://github.com/user-attachments/assets/ed974783-be45-4432-b35f-005305759118" />

# Robust Scaler

<img width="517" height="473" alt="image" src="https://github.com/user-attachments/assets/72123bc4-47d7-4c78-a158-959ae607023d" />

# FEATURE SELECTION 

# 1) Filter Methods
# Chi Square Method (Selected Features)

<img width="731" height="77" alt="image" src="https://github.com/user-attachments/assets/381cf364-fe90-4086-9bd9-6fe325054683" />

# Fisher Score Method (Selected Features)

<img width="756" height="80" alt="image" src="https://github.com/user-attachments/assets/77d6a826-55d8-4b60-b11d-466416362d4a" />

# Information Gain (Selected Features)

<img width="510" height="88" alt="image" src="https://github.com/user-attachments/assets/7ef09ff0-85bd-4015-88fe-f03c628cbeb1" />

# 2) Wrapper Methods (Forward Selection and Backward Elimination Methods)

<img width="916" height="58" alt="image" src="https://github.com/user-attachments/assets/968b46cc-a6c4-44c9-a213-af32b30ac2c7" />

# 3) Embedded Methods (Selected Features)

<img width="587" height="76" alt="image" src="https://github.com/user-attachments/assets/486bb3c8-6218-4b63-ab82-742c8baf3d2c" />

# RESULT:
Therefore, various feature scaling methods such as standard scaler, minmax scaler, maxabs scaler, robust scaler is used for the magnesium column in the wine dataset since it ranges from _ to _. It requires proper scaling of the field to improve model performance. Various feature selection methods are also used. They fall under three categories.
1) Filter method which uses statistical models to select the best features (Chi Square, Fisher Score and Information Gain)
2) Wrapper method applies machine learning algorithm iteratively to different subsets of features and identifies the best. There are two kinds: Foward Selection and Backward Elimination.
3) Embedded Methods
