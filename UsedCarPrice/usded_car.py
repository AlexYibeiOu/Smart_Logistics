# https://tianchi.aliyun.com/competition/entrance/231784/introduction

'''
Field	    Description
SaleID	    交易ID，唯一编码
name	    汽车交易名称，已脱敏
regDate	    汽车注册日期，例如20160101，2016年01月01日
model	    车型编码，已脱敏
brand	    汽车品牌，已脱敏
bodyType	车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：
            5，商务车：6，搅拌车：7
fuelType	燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：
            5，电动：6
gearbox	    变速箱：手动：0，自动：1
power	    发动机功率：范围 [ 0, 600 ]
kilometer	汽车已行驶公里，单位万km
notRepairedDamage	汽车有尚未修复的损坏：是：0，否：1
regionCode	地区编码，已脱敏
seller	    销售方：个体：0，非个体：1
offerType	报价类型：提供：0，请求：1
creatDate	汽车上线时间，即开始售卖时间
price	    二手车交易价格（预测目标）
v系列特征	  匿名特征，包含v0-14在内15个匿名特征
'''

import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load data
data_train = pd.read_csv('used_car_train_20200313.csv', sep =' ')
data_test = pd.read_csv('used_car_testB_20200421.csv', sep =' ')

# Check - dtypes
#data_train.dtypes

#data_train
#data_test
#data_train.columns.tolist()
'''
['SaleID',
 'name',
 'regDate',
 'model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'power',
 'kilometer',
 'notRepairedDamage',
 'regionCode',
 'seller',
 'offerType',
 'creatDate',
 'price',
 'v_0',
 'v_1',
 'v_2',
 'v_3',
 'v_4',
 'v_5',
 'v_6',
 'v_7',
 'v_8',
 'v_9',
 'v_10',
 'v_11',
 'v_12',
 'v_13',
 'v_14']
 '''

# Check na
#data_train.isnull().sum()
#data_test.isnull().sum()
# model                   1
# bodyType             4506
# fuelType             8680
# gearbox              5981

# Fillna - model 
# ( model == na ) => 0
#data_train[data_train['model'].isnull()==True]
#data_train['model'].value_counts()
data_train['model'][38424]=0
#data_train.isnull().sum()

# Cleaning
data_train.replace(to_replace = '-', value = np.nan, inplace=True)
data_test.replace(to_replace = '-', value = np.nan, inplace=True)

# Fill - notRepairedDamage
#data_train['notRepairedDamage'].fillna(1.2, inplace=True)
#data_test['notRepairedDamage'].fillna(1.2, inplace=True)
#data_train['notRepairedDamage'] = pd.DataFrame(data_train['notRepairedDamage'], dtype=np.float)

data_train.fillna(data_train.median(), inplace=True)
data_test.fillna(data_test.median(), inplace=True)

# Revised - power
# ( power > 600 ) => 600
# ( power == 0 ) => median
#data_train['power'].plot()
#data_train['power'].describe()
#data_train['power'].value_counts()
#data_train['price'][data_train['power']<40].describe()
#data_train['power'][data_train['power']>600].describe()
#data_test['power'][data_test['power']<40].describe()
#data_test['power'][data_test['power']>600].describe()

# mean power = 127
#temp = data_train
#temp['power'][temp['power']>600] = 0
#temp = temp['power'][temp['power']>0]
#temp.mean()

data_train['power'][data_train['power']==0] = 127
data_test['power'][data_test['power']==0] = 127
data_train['power'][data_train['power']>600] = 600
data_test['power'][data_test['power']>600] = 600
#data_train['power'].describe()
#data_test['power'].describe()

# Drop - SaleID => replace by removing tag
# data_train.drop(['SaleID'], axis=1, inplace=True)
# data_test.drop(['SaleID'], axis=1, inplace=True)

# Drop - name => replace by removing tag
# data_train.drop(['name'], axis=1, inplace=True)
# data_test.drop(['name'], axis=1, inplace=True)

# Drop - seller => replace by removing tag
# data_train['seller'].value_counts()
# data_test['seller'].value_counts()
# data_train.drop(['seller'], axis=1, inplace=True)
# data_test.drop(['seller'], axis=1, inplace=True)

# Drop - offerType => replace by removing tag
# data_train['offerType'].value_counts()
# data_test['offerType'].value_counts()
# data_train.drop(columns=['offerType'], inplace=True)
# data_test.drop(columns=['offerType'], inplace=True)

# Create - Veh_Age = ( creatDate - regDate ) / 10000
data_train['vehAge'] = ( data_train['creatDate'] - data_train['regDate'] ) / 10000
data_test['vehAge'] = ( data_test['creatDate'] - data_test['regDate'] ) / 10000

# Check - vehAge
# data_train['vehAge'][data_train['vehAge']>26].count()
# 0
# data_test['vehAge'][data_test['vehAge']>26].count()
# 0

# Check - kilometer
# data_train['kilometer'].describe()
# 0.5 ~ 15
# data_test['kilometer'].describe()
# 0.5 ~ 15

# Drop - creatDate => replace by removing tag
# data_train.drop(columns=['creatDate'], inplace=True)
# data_test.drop(columns=['creatDate'], inplace=True)

# Drop - regDate => replace by removing tag
# data_train.drop(columns=['regDate'], inplace=True)
# data_test.drop(columns=['regDate'], inplace=True)

# Drop - v_1 => replace by removing tag
#plt.figure(figsize=(20,10))
#sns.heatmap(data_train.corr(), cmap='coolwarm')
#corr = data_train.corr()
#corr.to_csv('./corr.csv', index=False)
# v_1 = v_6
# data_train.drop(columns=['v_1'], inplace=True)
#data_test.drop(columns=['v_1'], inplace=True)


# Check - regionCode
#data_train['regionCode'].value_counts()

# Check - distributioin
'''
plt.scatter(data_train.index,data_train['v_2'])

plt.scatter(data_train.index,data_train['v_3'])

plt.scatter(data_train.index,data_train['v_4'])

plt.scatter(data_train.index,data_train['v_5'])

plt.scatter(data_train.index,data_train['v_6'])

plt.scatter(data_train.index,data_train['v_7'])

plt.scatter(data_train.index,data_train['v_8'])

plt.scatter(data_train.index,data_train['v_9'])

plt.scatter(data_train.index,data_train['v_10'])

plt.scatter(data_train.index,data_train['v_11'])

plt.scatter(data_train.index,data_train['v_12'])

plt.scatter(data_train.index,data_train['v_13'])

plt.scatter(data_train.index,data_train['v_14'])
'''


# Drop - index = 38424
# data_train['v_14'].max()
# data_train[data_train['v_14']==8.658417876941384]
# index = 38424
data_train.drop(index=38424, inplace=True)
# Check
#plt.scatter(data_train['v_2'],data_train['v_5'])
#plt.scatter(data_test['v_2'],data_test['v_5'])
#plt.scatter(data_train['v_2'],data_train['v_7'])
#plt.scatter(data_test['v_2'],data_test['v_7'])
#data_train['v_5'].value_counts()
#data_test['v_5'].value_counts()


# Check
#plt.scatter(data_train['price'],data_train['power'])
#plt.scatter(data_train['vehAge'],data_train['price'])
#plt.scatter(data_train['price'],data_train['notRepairedDamage'])
#data_train['notRepairedDamage'].value_counts()
#data_test['notRepairedDamage'].value_counts()


# Finish Data Cleaning
# --------------------------
# Start Modeling

# Drop tags - 'price','SaleID','name','seller','offerType','creatDate','regDate','v_1'
tags = data_train.columns.tolist()
tags.remove('price')
tags.remove('SaleID')
tags.remove('name')
tags.remove('seller')
tags.remove('offerType')
tags.remove('creatDate')
tags.remove('regDate')
tags.remove('v_1')
tags
#data_train[tags]



# Encode - no good
#data_train['model']
#data_train['model'].value_counts()
# lbe = LabelEncoder()
# data_train['model'] = lbe.fit_transform(data_train['model'])
# data_test['model'] = lbe.transform(data_test['model'])
#data_train['model']
#data_train['model'].value_counts()

'''
data_train['brand']
data_train['brand'].value_counts()
lbe = LabelEncoder()
data_train['brand'] = lbe.fit_transform(data_train['brand'])
data_test['brand'] = lbe.transform(data_test['brand'])
'''

#data_train['bodyType']
#data_train['bodyType'].value_counts()
# lbe = LabelEncoder()
# data_train['bodyType'] = lbe.fit_transform(data_train['bodyType'])
# data_test['bodyType'] = lbe.transform(data_test['bodyType'])
#data_train['bodyType']
#data_train['bodyType'].value_counts()

'''
unseen = [1738, 1781, 2040, 2080, 2580, 2696, 2776, 3284, 3462, 3584, 3928, \
3935, 3990, 4095, 4636, 4709, 4737, 5041, 5426, 5464, 5524, 5676, 5914, \
5968, 6402, 6451, 6504, 6524, 6607, 6664, 6687, 6712, 6771, 6850, 6873, \
995, 6999, 7016, 7071, 7073, 7082, 7160, 7262, 7308, 7337, 7348, 7362, \
7363, 7394, 7406, 7504, 7530, 7554, 7565, 7587, 7600, 7607, 7650, 7663, \
7670, 7684, 7729, 7782, 7783, 7807, 7811, 7831, 7837, 7839, 7844, 7853, \
7869, 7870, 7878, 7880, 7882, 7891, 7895, 7898, 7907, 7921, 7928, 7951, \
7962, 7980, 7983, 7995, 7996, 7998, 8004, 8008, 8010, 8011, 8016, 8019, \
8021, 8027, 8032, 8035, 8066, 8075, 8079, 8085, 8095, 8098]

data_test['regionCode']  = data_test['regionCode'].map(lambda x : 9999 if data_test['regionCode'] in unseen else x )

data_train['regionCode'] = lbe.fit_transform(data_train['regionCode'])
data_test['regionCode'] = lbe.transform(data_test['regionCode'])


'''

# Check - Corr
#plt.figure(figsize=(20,10))
#corr = data_train[tags].corr()
#sns.heatmap(corr, cmap='coolwarm')

# Normalization
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data_train[tags].values)
x_train = min_max_scaler.transform(data_train[tags].values)
x_test = min_max_scaler.transform(data_test[tags].values)

# Split dataset
y = data_train['price'].values
train_x, test_x, train_y, test_y = train_test_split(x_train, y, test_size=0.05)

# Build model
model =  keras.Sequential([
    keras.layers.Dense(250, activation='relu', input_shape=[len(tags)]),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(250, activation='relu'), 
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1)
])
# Define loss function
model.compile(loss='mean_absolute_error', optimizer='Adam')

# Train
model.fit(train_x, train_y, batch_size=256, epochs=100)

# Validation
print('Train dataset MAE:', mean_absolute_error(train_y, model.predict(train_x)))
print('Test dataset MAE:', mean_absolute_error(test_y, model.predict(test_x)))


# Test Model
predict_y = model.predict(x_test)

# Output result: SaleID, price
result = pd.DataFrame()

result['SaleID'] = data_test.SaleID
result['price'] = predict_y

plt.scatter(result['SaleId'],result['price'])
result.to_csv('./ans_nn_09.csv', index=False)


# fail (base 10) -
# epochs=80 -> 100
# Fillna - notRepairedDamage, fillna = 0.8 & change dtype from object to float
# Train dataset MAE: 431.69294715479634
# Test dataset MAE: 464.6514124265035


# fail (base 10) -73 (should fillna = 0.8)
# Fillna - notRepairedDamage, fillna = 1.2
# Train dataset MAE: 470.66255708721536
# Test dataset MAE: 515.3054725673676

# fail (base 10) -99 (fill with wrong number)
# Fillna - notRepairedDamage, fillna = '2'
# Train dataset MAE: 524.4819312514492
# Test dataset MAE: 547.6832448112488

# fail (base 10) -48
# epochs =100 -> 80
# Drop brand
# Train dataset MAE: 458.89400421152186
# Test dataset MAE: 496.9910494635264

# fail (base 10) same over fitting
# v_13 ^ 2
# Train dataset MAE: 437.947246250813
# Test dataset MAE: 451.95853670857747

# fail (base 10) -78 over fitting
# Drop v_13 
# Train dataset MAE: 472.13538650957776
# Test dataset MAE: 530.8448741765341


# Version 10 (base 09) +5
# Drop regionCode
# Train dataset MAE: 431.3961721832854
# Test dataset MAE: 448.2378973698934

# fail (base 09) same 
# Drop v_7


# fail (base 09) -56 over fitting
# epochs 10  batch size 1024
# epochs 10  batch size 512
# epcchs 40  batch size 256
# epcchs 20  batch size 128
# epcchs 15  batch size 64
# epcchs 5  batch size 32
# Train dataset MAE: 476.89471053804766
# Test dataset MAE: 509.980633203125

# fail (base 09) -42
# batch_size=512->256->64, epochs=100->50 
# Train dataset MAE: 484.2685384866615
# Test dataset MAE: 495.2958327618917


# fail (base 09) -42
# DenseNet 3+2+2 => 3+2+2+1
# Train dataset MAE: 456.57875003406724
# Test dataset MAE: 495.33780178578695

# fail -29
# batch_size=512->256->128
# Train dataset MAE: 447.8423846878453
# Test dataset MAE: 482.5797127087911

# Versioin 09 good +32
# batch_size=512->256, epochs=100, test_size = 0.2->0.1->0.05 
# Train dataset MAE: 427.13026016995406
# Test dataset MAE: 453.7730372197469
# score 462

# Version 08 ok +15
# DenseNet 3 + 2 + 2, test_size 0.2->0.1
# model.fit(train_x, train_y, batch_size=4096 ->512, epochs=80->40)
# Train dataset MAE: 476.32180486311637
# Test dataset MAE: 485.8805928206126

# good
# batch_size=4096 -> 8192 -> 2048 -> 1024

# fail 07-2 same +2
# batch_size=4096 -> 8192 -> 2048
# Train dataset MAE: 484.0189002021387
# Test dataset MAE: 498.4448374432882

# fali -30
# batch_size=4096 -> 8192
# Train dataset MAE: 534.0685215491687
# Test dataset MAE: 530.037574518903

# fail no imporve, over fitting -14
# DenseNet 3 + 2 + 1, epochs=80->90
# Train dataset MAE: 491.29107517998625
# Test dataset MAE: 514.0841620043436

# fail -45
# power =0 => 127 => 0
# Train dataset MAE: 541.6818959873042
# Test dataset MAE: 545.1462017818451

# Version 07 good +35
# test_size 0.2->0.1
# Train dataset MAE: 501.51260987465054
# Test dataset MAE: 500.71453985646565
# score 504


# Version 06 good +33
# DenseNet 3 + 2
# Train dataset MAE: 518.7910735452233
# Test dataset MAE: 535.1797369550387

# Version fail -33
# Encode  model, bodyType
# Train dataset MAE: 592.1297171025892
# Test dataset MAE: 598.6645795150915

# Version 04 +32
# Drop - v_1
# Train dataset MAE: 570.5627961249878
# Test dataset MAE: 565.5876860450148
# score = 568

# Version 03 small improve +11
# power =0 => 127
# Train dataset MAE: 587.1443784633636
# Test dataset MAE: 597.5741834983507

# Version 02 
# Baseline
# drop(SaleID,name,seller,offerType,creatDate,regDate)
# Create(vehAge)
# power <40 => 127
# power >600 = 600
# DenseNet
# model.fit(train_x, train_y, batch_size=4096, epochs=80)
# Train dataset MAE: 592.2529963332062
# Test dataset MAE: 608.951632693721

# Version 01
# Baseline 
# DenseNet
# model.fit(train_x, train_y, batch_size=4096, epochs=80)
# Train dataset MAE: 551.8627396380623
# Test dataset MAE: 558.9530082368453















''''

#  price to be revised
data_train['price'].boxplot()
data_train[data_train['price']==11]
plt.scatter(data_train['power'], data_train['price'])
plt.show()




# - bodyType
#data_train['bodyType'].isnull().sum()
#data_train['bodyType'].describe()
#data_train['bodyType'].value_counts()
#data_test['bodyType'].isnull().sum()
#data_test['bodyType'].describe()
#data_test['bodyType'].value_counts()

temp = data_train[data_train['bodyType'].isnull()==False]
temp['bodyType'].isnull().sum()

temp1 = temp['bodyType']
temp1['v_01'] = temp['v_0']
t
plt.figure(figsize=(20,10))
sns.heatmap(temp1.corr(), annot=True, cmap='coolwarm')


data_train['fuelType'].describe()

data_train['Type'].describe()



'''


