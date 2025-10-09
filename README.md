# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

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

# CODING AND OUTPUT:
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

# 2) Wrapper Methods

# 3) Embedded Methods


# RESULT:
       # INCLUDE YOUR RESULT HERE
