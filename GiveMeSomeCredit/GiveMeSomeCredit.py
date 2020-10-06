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
#df_train.median()
#df_train['MonthlyIncome'].isnull().sum()

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
dependent_bins =[-math.inf,2,4,6,8,10,math.inf]
df_train['bin_NumberOfDependents'] = pd.cut(df_train['NumberOfDependents'], bins=dependent_bins)
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


df_train['bin_NumberOfDependents'].value_counts()
df_train['bin_MonthlyIncome'].value_counts()

import numpy as np

# IV
def cal_IV(df, feature, target):
    lst = []
    cols = ['Variable', 'Value', 'All', 'Bad']
    for i in range(df[feature].nunique()):   # nunique = number of unique
        val = list(df[feature].unique())[i] 
        temp1 = df[df[feature]==val].count()[feature]  # total
        temp2 = df[(df[feature]==val) & (df[target]==1)].count()[feature] # number of target=1
        #print(feature, val, temp1, temp2)
        lst.append([feature, val, temp1, temp2])
    data = pd.DataFrame(lst, columns=cols)
    data = data[data['Bad'] > 0]
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']  # this value leads to bad
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum() 
    data['Distribution Good'] = \
        (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum()) 
    data['WOE'] = np.log(data['Distribution Bad'] / data['Distribution Good'])
    data['IV'] = \
        (data['Distribution Bad'] - data['Distribution Good']) * data['WOE']

    data = data.sort_values(by=['Variable', 'Value'],ascending=True)
    #print(data)
    return data['IV'].sum()

for col in bin_cols:
    print(col)
    print(cal_IV(df_train, col, 'SeriousDlqin2yrs'))


for col in bin_cols:
    #print(col)
    #print(cal_IV(df_train, col, 'SeriousDlqin2yrs'))
    temp_IV = cal_IV(df_train, col, 'SeriousDlqin2yrs')
    print (col, temp_IV)
