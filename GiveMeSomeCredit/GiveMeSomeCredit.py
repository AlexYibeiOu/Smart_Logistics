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
pd.DataFrame({'Column': null_num.index, 'Value':null_num.values, \
    '%':null_num.values/len(df_train)})

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
df_train['bin_NumberOfDependents'] = \
    pd.cut(df_train['NumberOfDependents'], bins=dependent_bin)
df_train[['NumberOfDependents', 'bin_NumberOfDependents']]
df_train['bin_NumberOfDependents'].value_counts()

# Bin - 
# NumberOfTime30-59DaysPastDueNotWorse，
# NumberOfTime60-89DaysPastDueNotWorse，
# NumberOfTimes90DaysLate
# [-math.inf,1,2,3,4,5,6,7,8,9,math.inf]
dpd_bins = [-math.inf,1,2,3,4,5,6,7,8,9,math.inf]
df_train['bin_NumberOfTime30-59DaysPastDueNotWorse'] = \
    pd.cut(df_train['NumberOfTime30-59DaysPastDueNotWorse'], bins=dpd_bins)
df_train['bin_NumberOfTime60-89DaysPastDueNotWorse'] = \
    pd.cut(df_train['NumberOfTime60-89DaysPastDueNotWorse'], bins=dpd_bins)
df_train['bin_NumberOfTimes90DaysLate'] = \
    pd.cut(df_train['NumberOfTimes90DaysLate'], bins=dpd_bins)
df_train['bin_NumberOfTime30-59DaysPastDueNotWorse'].value_counts()
df_train['bin_NumberOfTime60-89DaysPastDueNotWorse'].value_counts()
df_train['bin_NumberOfTimes90DaysLate'].value_counts()

# Bin - the rest 5 columns
# RevolvingUtilizationOfUnsecuredLines, 
# DebtRatio, 
# MonthlyIncome, 
# NumberOfOpenCreditLinesAndLoans, 
# NumberRealEstateLoansOrLines 
# 5 sections
df_train['bin_RevolvingUtilizationOfUnsecuredLines'] = pd.qcut(df_train['RevolvingUtilizationOfUnsecuredLines'], q=5, duplicates='drop')
df_train['bin_DebtRatio'] = pd.qcut(df_train['DebtRatio'], q=5, duplicates='drop')
df_train['bin_MonthlyIncome'] = pd.qcut(df_train['MonthlyIncome'], q=5, duplicates='drop')
df_train['bin_NumberOfOpenCreditLinesAndLoans'] = pd.qcut(df_train['NumberOfOpenCreditLinesAndLoans'], q=5, duplicates='drop')
df_train['bin_NumberRealEstateLoansOrLines'] = pd.qcut(df_train['NumberRealEstateLoansOrLines'], q=5, duplicates='drop')
df_train[['bin_RevolvingUtilizationOfUnsecuredLines', \
    'bin_DebtRatio','bin_MonthlyIncome', \
    'bin_NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']]
df_train['bin_NumberRealEstateLoansOrLines'].value_counts()
df_train['NumberRealEstateLoansOrLines'].value_counts()

# Show - all bin columns
bin_cols = [c for c in df_train.columns.values if c.startswith('bin')]
bin_cols
