import pandas as pd 

# Load - data
df_train = pd.read_csv('./data/cs-training.csv')
df_train

# drop - the first column
df_train = df_train.iloc[:, 1:]
df_train

# explore - SeriousDlqin2yrs
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='SeriousDlqin2yrs', data=df_train)
print("percentage {}".format(df_train['SeriousDlqin2yrs'].sum()/len(df_train)))

# Check - missing data
null_num = df_train.isnull().sum()
pd.DataFrame({'Column': null_num.index, 'Value':null_num.values, '%':null_num.values/len(df_train)})

df_train['RevolvingUtilizationOfUnsecuredLines'].describe()
sns.distplot(df_train['RevolvingUtilizationOfUnsecuredLines'])

# Fill - missing data
df_train.info()  
df_train = df_train.fillna(df_train.median())
df_train.isnull().sum()

# Binning

import math
# Bin - age [-math.inf, 25, 40, 50, 60, 70, math.inf]
age_bins = [-math.inf, 25, 40, 50, 60, 70, math.inf]
df_train['age'].value_counts()
df_train['bin_age'] = pd.cut(df_train['age'], bins=age_bins)
df_train['bin_age'].value_counts()
df_train[['age', 'bin_age']]
df_train['bin_age'].value_counts()

# Bin - NumberOfDependents [-math.inf,2,4,6,8,10,math.inf]
dependent_bin =[-math.inf,2,4,6,8,10,math.inf]
df_train['bin_NumberOfDependents'] = pd.cut(df_train['NumberOfDependents'], bins=dependent_bin)
df_train[['NumberOfDependents', 'bin_NumberOfDependents']]
df_train['bin_NumberOfDependents'].value_counts()

