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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
df=pd.read_csv("titanic_dataset.csv")
df

![alt text](b1.png)

df.info()

![alt text](b2.png)

df.shape

![alt text](b3.png)

df.set_index("PassengerId",inplace=True)
df.describe()

![alt text](b4.png)

df.shape

![alt text](b5.png)

Categorical data analysis

df.nunique()

![alt text](b6.png)

df["Survived"].value_counts()

![alt text](b7.png)

per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per

![alt text](b8.png)

sns.countplot(data=df,x="Survived")

![alt text](b9.png)

df

![alt text](b10.png)

df.Pclass.unique()

![alt text](b11.png)

df.rename(columns={'Sex':'Gender'},inplace=True)
df

![alt text](b12.png)

Bivariate Analysis

sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)

![alt text](b13.png)

sns.catplot(x="Survived",hue="Gender",data=df,kind="count")

![alt text](b14.png)

df.boxplot(column="Age",by="Survived")

![alt text](b15.png)

sns.scatterplot(x=df["Age"],y=df["Fare"])

![alt text](b16.png)

sns.jointplot(x="Age",y="Fare",data=df)

![alt text](b17.png)

Multivariate Analysis

fig, ax1 = plt.subplots(figsize=(8,5))
plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)

![alt text](b18.png)

sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")

![alt text](b19.png)

Co-relation

corr=df.corr()
sns.heatmap(corr,annot=True)

![alt text](b20.png)

sns.pairplot(df)

![alt text](b21.png)

# RESULT

Thus, the programs are executed and verified successfully.