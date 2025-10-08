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
