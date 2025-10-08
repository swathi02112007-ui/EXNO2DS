# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
~~~
# -*- coding: utf-8 -*-
"""EXNO2DS.py
Converted from Google Colab to VS Code Compatible Script
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# READ CSV FILE
# Replace with your CSV file path
dt = pd.read_csv("C:/path/to/your/titanic.csv")  # 👈 Change this path
print("DATA PREVIEW:")
print(dt.head())

# %%
# DISPLAY INFORMATION ABOUT DATASET
print("\nDATASET INFO:")
print(dt.info())

# %%
# DISPLAY NO. OF ROWS AND COLUMNS
print("\nNumber of Rows and Columns:")
print(dt.shape)

# %%
# SET PASSENGER ID AS INDEX COLUMN
if 'PassengerId' in dt.columns:
    dt.set_index('PassengerId', inplace=True)
print("\nDataset after setting PassengerId as index:")
print(dt.head())

# %%
# DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS:")
print(dt.describe())

# =======================================================
# **CATEGORICAL DATA ANALYSIS**
# =======================================================

# %%
# USE VALUE_COUNTS FUNCTION AND PERFORM CATEGORICAL ANALYSIS
print("\nValue Counts for 'Sex' column:")
if 'Sex' in dt.columns:
    print(dt['Sex'].value_counts())

print("\nValue Counts for 'Embarked' column (if available):")
if 'Embarked' in dt.columns:
    print(dt['Embarked'].value_counts())

# =======================================================
# **UNIVARIATE ANALYSIS**
# =======================================================

# %%
# USE COUNTPLOT AND PERFORM UNIVARIATE ANALYSIS FOR "SURVIVED" COLUMN
if 'Survived' in dt.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Survived', data=dt)
    plt.title("Univariate Analysis - Survived Count")
    plt.show()

# %%
# IDENTIFY UNIQUE VALUES IN "PASSENGER CLASS" COLUMN
if 'Pclass' in dt.columns:
    print("\nUnique values in Pclass column:")
    print(dt['Pclass'].unique())

# %%
# RENAMING COLUMN
if 'Sex' in dt.columns:
    dt.rename(columns={'Sex': 'Gender'}, inplace=True)
print("\nDataset after renaming 'Sex' to 'Gender':")
print(dt.head())

# =======================================================
# **BIVARIATE ANALYSIS**
# =======================================================

# %%
# USE CATPLOT METHOD FOR BIVARIATE ANALYSIS
if {'Pclass', 'Survived'} <= set(dt.columns):
    sns.catplot(x='Pclass', hue='Survived', kind='count', data=dt, height=5, aspect=1.2)
    plt.title("Bivariate Analysis - Survival by Passenger Class")
    plt.show()

# %%
# USE COUNTPLOT WITH ANNOTATIONS
if {'Pclass', 'Survived'} <= set(dt.columns):
    fig, ax1 = plt.subplots(figsize=(8,5))
    graph = sns.countplot(x='Pclass', hue='Survived', data=dt)
    plt.title("Bivariate CountPlot - Survival vs Pclass")
    graph.set_xticklabels(graph.get_xticklabels())
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2, height + 1, height, ha="center")
    plt.show()

# %%
# USE BOXPLOT METHOD TO ANALYZE AGE AND SURVIVED COLUMN
if {'Age', 'Survived'} <= set(dt.columns):
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Survived', y='Age', data=dt)
    plt.title("Bivariate Analysis - Age vs Survived")
    plt.show()

# =======================================================
# **MULTIVARIATE ANALYSIS**
# =======================================================

# %%
# USE BOXPLOT METHOD AND ANALYZE THREE COLUMNS (PCLASS, AGE, GENDER)
if {'Pclass', 'Age', 'Gender'} <= set(dt.columns):
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Pclass', y='Age', hue='Gender', data=dt)
    plt.title("Multivariate Analysis - Age vs Pclass and Gender")
    plt.show()

# %%
# USE CATPLOT METHOD AND ANALYZE THREE COLUMNS (PCLASS, SURVIVED, GENDER)
if {'Pclass', 'Survived', 'Gender'} <= set(dt.columns):
    sns.catplot(x='Pclass', hue='Gender', col='Survived', kind='count', data=dt, height=5, aspect=1)
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Multivariate Analysis - Pclass, Survived, Gender")
    plt.show()

# %%
# IMPLEMENT HEATMAP AND PAIRPLOT FOR THE DATASET
plt.figure(figsize=(10,6))
sns.heatmap(dt.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(dt, hue='Survived', diag_kind='kde')
plt.show()

~~~
        

# RESULT
DATA PREVIEW:
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

DATASET INFO:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Gender       891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None

Number of Rows and Columns:
(891, 12)

Dataset after setting PassengerId as index:
            Survived  Pclass  ... Cabin Embarked
PassengerId                 ...                
1                 0       3  ...   NaN        S
2                 1       1  ...   C85        C
3                 1       3  ...   NaN        S
4                 1       1  ...  C123        S
5                 0       3  ...   NaN        S

DESCRIPTIVE STATISTICS:
          Survived     Pclass         Age       SibSp       Parch        Fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

Value Counts for 'Sex' column:
male      577
female    314
Name: Sex, dtype: int64

Value Counts for 'Embarked' column (if available):
S    644
C    168
Q     77
Name: Embarked, dtype: int64

Unique values in Pclass column:
[3 1 2]

Dataset after renaming 'Sex' to 'Gender':
            Survived  Pclass  ... Cabin Embarked
PassengerId                 ...                
1                 0       3  ...   NaN        S
2                 1       1  ...   C85        C
3                 1       3  ...   NaN        S
4                 1       1  ...  C123        S
5                 0       3  ...   NaN        S

        