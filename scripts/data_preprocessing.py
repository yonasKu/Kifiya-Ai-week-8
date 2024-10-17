import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


       
    
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to display the overview of the dataset
# def data_overview(df):
#     print("Data Overview:")
#     print(df.info())
#     print("\nFirst few rows of the dataset:")
#     print(df.head())
def data_overview(df):
    print("Data Overview:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

def handle_missing_values(self, strategy="drop", fill_value=None, columns=None):
    """
    Handle missing values by either dropping or imputing them.
    
    Parameters:
    - strategy (str): "drop" to remove rows/columns with missing values, "impute" to fill missing values.
    - fill_value (optional): If strategy is "impute", this is the value used to fill missing data. Can be "mean", "median", or a custom value.
    - columns (optional): List of columns to apply this to. If None, applies to all columns.
    
    Returns:
    - df (pd.DataFrame): DataFrame with missing values handled.
    """
    if columns is None:
        columns = self.df.columns
    
    if strategy == "drop":
        self.df = self.df.dropna(subset=columns)
    elif strategy == "impute":
        for col in columns:
            if fill_value == "mean":
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif fill_value == "median":
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(fill_value, inplace=True)
    return self.df

def remove_duplicates(self):
    """
    Remove duplicate rows from the DataFrame.
    
    Returns:
    - df (pd.DataFrame): DataFrame with duplicates removed.
    """
    self.df = self.df.drop_duplicates()
    return self.df

def correct_data_types(self, conversions):
    """
    Correct data types of specific columns.
    
    Parameters:
    - conversions (dict): Dictionary where keys are column names and values are the desired data types (e.g., {'age': 'int', 'purchase_time': 'datetime64'}).
    
    Returns:
    - df (pd.DataFrame): DataFrame with corrected data types.
    """
    for column, dtype in conversions.items():
        self.df[column] = self.df[column].astype(dtype)
    return self.df

def univariate_analysis(self, column):
    """
    Perform univariate analysis by plotting histograms and boxplots.
    
    Parameters:
    - column (str): Column to analyze.
    """
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(self.df[column], kde=True)
    plt.title(f'Histogram of {column}')
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=self.df[column])
    plt.title(f'Boxplot of {column}')
    
    plt.show()

def bivariate_analysis(self, column1, column2, kind="scatter"):
    """
    Perform bivariate analysis by plotting scatter plots or other kinds of plots.
    
    Parameters:
    - column1 (str): First column for analysis.
    - column2 (str): Second column for analysis.
    - kind (str): Type of plot. Options: 'scatter', 'bar', 'box', etc.
    """
    plt.figure(figsize=(10, 6))
    
    if kind == "scatter":
        sns.scatterplot(x=self.df[column1], y=self.df[column2])
    elif kind == "box":
        sns.boxplot(x=self.df[column1], y=self.df[column2])
    elif kind == "bar":
        sns.barplot(x=self.df[column1], y=self.df[column2])
    elif kind == "heatmap":
        corr = self.df[[column1, column2]].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
    
    plt.title(f'{kind.capitalize()} plot of {column1} vs {column2}')
    plt.show()

# Example of using this in a Jupyter notebook:
# from data_preprocessing import DataPreprocessing
# df = pd.read_csv('Fraud_Data.csv')
# dp = DataPreprocessing(df)
# dp.handle_missing_values(strategy="impute", fill_value="mean")
# dp.remove_duplicates()
# dp.correct_data_types({'purchase_time': 'datetime64', 'age': 'int'})
# dp.univariate_analysis('purchase_value')
# dp.bivariate_analysis('purchase_value', 'age', kind='scatter')
