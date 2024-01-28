#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Vaani Rawat
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
import csv
#%% Q1
# a)
# Open and Read the Dataset
with open('SPXVIX.csv', 'r') as file:
    csv_reader = csv.reader(file)
    # Iterate over each row in the csv file
    for row in csv_reader:
        print(row)
        
"""
['2016-08-03', '2163.79', '12.86']
['2016-08-04', '2164.25', '12.42']
['2016-08-05', '2182.87', '11.39']
...
"""
# Initialize lists to store data
date = []
spx = []
vix = []

with open('SPXVIX.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        # Check if the row has any missing data
        if '' in row or len(row) < 3:
            continue

        # Append data to respective lists
        date.append(row[0])          # Date is in the first column
        spx.append(float(row[1]))    # SPX is in the second column
        vix.append(float(row[2]))    # VIX is in the third column

# b)
# Create a dataframe from the lists
SPXVIX_df = pd.DataFrame({'SPX': spx, 'VIX': vix}, index=date)
SPXVIX_df.index.name = 'Date'

# c)
print(SPXVIX_df.head().to_string())
#Docstring with top 5 resulting rows of dataframe
""" 
                SPX    VIX
Date                      
2010-01-04  1132.99  20.04
2010-01-05  1136.52  19.35
2010-01-06  1137.14  19.16
2010-01-07  1141.69  19.06
2010-01-08  1144.98  18.13
"""
#%% Q2
# a)
# Read the CSV file into a DataFrame
df = pd.read_csv('SPXVIX.csv', index_col='Date')  

# Display the top 8 rows of the DataFrame
print(df.head(8).to_string)
"""
                SPX    VIX
Date                      
2010-01-01      NaN    NaN
2010-01-04  1132.99  20.04
2010-01-05  1136.52  19.35
2010-01-06  1137.14  19.16
2010-01-07  1141.69  19.06
2010-01-08  1144.98  18.13
2010-01-11  1146.98  17.55
2010-01-12  1136.22  18.25
"""

# b)
# Remove rows with NaN values
df = df.dropna()
    
# Print dataframe head to get top 5 resulting rows
print(df.head().to_string)

#Docstring with top 5 resulting rows of dataframe
""" 
                SPX    VIX
Date                      
2010-01-04  1132.99  20.04
2010-01-05  1136.52  19.35
2010-01-06  1137.14  19.16
2010-01-07  1141.69  19.06
2010-01-08  1144.98  18.13
"""
# c)
# Display row index labels
print("Row Index Labels:", df.index)

# Display column index labels
print("Column Index Labels:", df.columns)

# d)
# Convert the index to datetime
df.index = pd.to_datetime(df.index)

# Filter the data for the dates between 2017-05-03 and 2018-06-29
start_date = '2017-05-03'
end_date = '2018-06-29'
df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot S&P500 prices
ax1.plot(df_filtered.index, df_filtered['SPX'], color='b', label='S&P500')
ax1.set_xlabel('Date')
ax1.set_ylabel('S&P500 Price', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for the VIX prices
ax2 = ax1.twinx()
ax2.plot(df_filtered.index, df_filtered['VIX'], color='r', label='VIX')
ax2.set_ylabel('VIX Price', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding title and legends
plt.title('S&P500 and VIX Prices (2017-05-03 to 2018-06-29)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.show()

# e)
"""
Based on the plot displaying the S&P500 and VIX prices from 
May 3, 2017 to June 29, 2018,we observe that the S&P500 and 
VIX tend to exhibit an inverse relationship. 

This means that when the the VIX increases, S&P500 tends 
to decrease and vice versa. 

The VIX is often used as a market sentiment indicator. 
A rising VIX suggests increasing fear or uncertainty 
in the market, which typically corresponds with a decline 
in the S&P500. A falling VIX suggests increasing confidence 
or stability, often accompanying a rise in the S&P500. 

However, the relationship is not always perfectly inverse. 
There are periods in the plot where both move in the 
same direction or where one remains relatively flat while 
the other fluctuates.
"""

# f) 
# Calculate simple returns for SPX and VIX
df['retSPX'] = df['SPX'].pct_change()
df['retVIX'] = df['VIX'].pct_change()

print(df.head())

"""
Since the simple returns is calculated using the formula 
R(t) = P(t) - P(t-1) / P(t-1), we use the pct_change() 
method in pandas as it computes the percentage change 
compared to the previous row. 

For the first row, since there's no "previous row" to 
compare with, the result is undefined, hence the first 
element of the two new columns is NaN.
"""
df = df.dropna()

# g)
# Calculate the number of observations (rows) and variables (columns)
rr = df.shape[0]  # Number of rows
cc = df.shape[1]  # Number of columns

# Count the number of positive return days for S&P500
positive_spx_days = df[df['retSPX'] > 0].shape[0]

# Calculate the proportion and convert it to a percentage
proportion_positive_spx = (positive_spx_days / rr) * 100

# Round the result to 1 decimal place and print
print("Percentage of S&P up days: {:.1f}%".format(proportion_positive_spx))

"""
Percentage of S&P up days: 54.7%
"""

# h)
# Select the return columns
returns_df = df[['retSPX', 'retVIX']]

# Create the scatter matrix plot
pd.plotting.scatter_matrix(returns_df, alpha=0.2, figsize=(10, 10), diagonal='hist')

# Show the plot
plt.show()

# i)
# Calculate the correlation matrix
correlation_matrix = df[['retSPX', 'retVIX']].corr()
print(correlation_matrix)
"""
          retSPX    retVIX
retSPX  1.000000 -0.789413
retVIX -0.789413  1.000000
"""

# Extract the correlation between the returns of S&P and VIX
correlation_spx_vix = correlation_matrix.loc['retSPX', 'retVIX']
print("The correlation between SPX and VIX is:", correlation_spx_vix)

"""
The correlation between SPX and VIX is: -0.7894127848767359
"""

# j)
"""
There is a strong negative correlation of approximately
 -0.789 between the returns of the S&P500 and the VIX
suggesting a sgnificant inverse relationship between 
the two. When the S&P500 increases in value, the VIX tends 
to decrease, and vice versa. 

Based on this inverse relationship, if the S&P500 were 
to drop in price, one would expect the VIX to rise. 
However, the relationship is not always perfectly inverse 
and while the negative correlation is strong the effect on 
VIX might also depend on other factors like economic indicators, 
geopolitical events, or policy changes.
"""

#%% Q3 intro analysis code
snp   = pd.read_csv("snp.csv",   index_col = 0, parse_dates = True) 
crude = pd.read_csv("crude.csv", index_col = 0, parse_dates = True) 
gold  = pd.read_csv("gold.csv",  index_col = 0, parse_dates = True) 
sse   = pd.read_csv("sse.csv",   index_col = 0, parse_dates = True) 

crude.dropna(inplace = True)
gold.dropna(inplace = True)

# merge dataframes by date
temp = gold.merge(crude, left_index = True, right_index = True, how = 'inner')
temp2 = temp.merge(sse, left_index = True, right_index = True, how = 'inner')
df = temp2.merge(snp, left_index = True, right_index = True, how = 'inner')

df.isnull().values.sum() # 0 (no missing values)

# Align SSE time with S&P (i.e. this will shift SSE up in column by 1 day)
df['sse'] = df['sse'].shift(-1)
df.dropna(inplace = True)
 
print(df.columns)  
# ['gold', 'gold vol', 'cl', 'cl vol', 'sse', 'sse vol', 'snp', 'snp vol']

# Historical Gold traded volume
plt.figure(figsize = (10, 8))
plt.plot(df.iloc[:, 1], 'xg-')
plt.show()

# Covid-19 Gold traded volume spike!!!
plt.figure(figsize = (10, 8))
plt.plot(df.iloc[-100:-1, 1], 'xr-')
plt.show()

# Let's plot the price movement of S&P with Crude:    
df[['snp', 'cl']].plot(figsize = (10, 8), secondary_y = 'cl') 


# DATA PREPROCESSING ----------------------------------------------------------

# a)
# Calculate log returns for each asset
df['rGold'] = np.log(df['gold'] / df['gold'].shift(1))
df['rCl'] = np.log(df['cl'] / df['cl'].shift(1))
df['rSse'] = np.log(df['sse'] / df['sse'].shift(1))
df['rSnp'] = np.log(df['snp'] / df['snp'].shift(1))

# Remove any rows with missing values
df.dropna(inplace=True)

# b)
# Create a 1-time step lag column for S&P500 returns data
df['rSnp_lag1'] = df['rSnp'].shift(1)

# c)
# Create a 1-time step ahead column for the S&P500
df['DayAheadSnp'] = df['snp'].shift(-1)

# Drop any rows with missing values
df.dropna(inplace=True)

# d)
# Shift the 'rSnp' series to align each row with the next day's return, 
# Then apply function to classify each value as 1 if it's positive or -1 if not
y = df['rSnp'].shift(-1).apply(lambda x: 1 if x > 0 else -1)

# e)
# List of columns to keep
columns_to_keep = ['gold vol', 'cl vol', 'sse vol', 'snp vol', 'rGold', 'rCl', 'rSse', 'rSnp', 'rSnp_lag1']

# Keep only the specified columns in df
df = df[columns_to_keep]

# f) 
# Using MaxAbsScaler to scale features to values between -1 to 1
scaler = preprocessing.MaxAbsScaler()

# Fit the scaler to the data and transform it
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# g)
# Define the feature columns and the target variable
feature_columns = ['rSnp', 'rSnp_lag1', 'snp vol', 'rCl', 'rGold']
X = df[feature_columns]

# Split the dataset into training and testing sets (2/3 training, 1/3 testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
#%% Running SVM algorithms
# Fit the SVM with a polynomial kernel
svm_poly = SVC(kernel='poly')
svm_poly.fit(X_train, y_train)

# Fit the SVM with an RBF kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# Classification accuracy on the training dataset for both models
accuracy_train_poly = svm_poly.score(X_train, y_train)
accuracy_train_rbf = svm_rbf.score(X_train, y_train)

# Classification accuracy on the testing dataset for both models
accuracy_test_poly = svm_poly.score(X_test, y_test)
accuracy_test_rbf = svm_rbf.score(X_test, y_test)

print("Training Accuracy (Polynomial Kernel):", accuracy_train_poly)
print("Training Accuracy (RBF Kernel):", accuracy_train_rbf)
print("Testing Accuracy (Polynomial Kernel):", accuracy_test_poly)
print("Training Accuracy (RBF Kernel):", accuracy_test_rbf)

"""
Training Accuracy (Polynomial Kernel): 0.5482200647249191
Training Accuracy (RBF Kernel): 0.5482200647249191
Testing Accuracy (Polynomial Kernel): 0.5705045278137129
Testing Accuracy (RBF Kernel): 0.5705045278137129
"""
