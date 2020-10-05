import pandas as pd 

# Load data
df_train = pd.read_csv('./data/cs-training.csv')
df_train

# drop the first column
df_train = df_train.iloc[:, 1:]
df_train

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='SeriousDlqin2yrs', data=df_train)
print("percentage {}".format(df_train['SeriousDlqin2yrs'].sum()/len(df_train)))

null_num = df_train.isnull().sum()
pd.DataFrame({'Column': null_num.index, 'Value':null_num.values, '%':null_num.values/len(df_train)})