import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load - data
dataset = pd.read_csv('SupplyChain.csv', encoding='unicode_escape')




# Check - Missing value
dataset.isnull().sum()
# Customer Zipcode
# Order Zipcode 
# Product Description 

# Create - Customer Full Name
dataset[['Customer Fname', 'Customer Lname']] 
dataset['Customer Full Name'] = dataset['Customer Fname'] + dataset['Customer Lname']
dataset['Customer Full Name']

# Fill - Zipcode
dataset['Customer Zipcode'].value_counts() 
dataset['Customer Zipcode'].isnull().sum()
dataset['Customer Zipcode'] = dataset['Customer Zipcode'].fillna(0)

# Check - Correlation
data = dataset
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
# Order Item Total = Sales per customer
# Order Item Id = Order Id
# Product Card Id = Order Item Cardprod Id
# Order Customer I d = Customer Id

# EDA
# Explore - Sales by Market
data['Market'].value_counts()
market = data.groupby('Market')
data['Sales per customer']
market['Sales per customer'].sum()
market['Sales per customer'].sum().sort_values(ascending=False).plot.bar(figsize=(12,6),title='Sales in different markets')

# Explore - Sales by Region
region = data.groupby('Order Region')
region['Sales per customer'].sum().sort_values(ascending=False).plot.bar(figsize=(12,6),title='Sales in different region')

# Explore - Sales by Category name
cat = data.groupby('Category Name')
cat['Sales per customer'].sum().sort_values(ascending=False).plot.bar(figsize=(12,6),title='Total Sales in different category')
cat['Sales per customer'].mean().sort_values(ascending=False).plot.bar(figsize=(12,6),title='Mean in different category')

# Explore - order date (DateOrders) => year, month, week_day,....
data['order date (DateOrders)'].value_counts()
temp = pd.DatetimeIndex(data['order date (DateOrders)'])
data['order_year'] = temp.year
data['order_month'] = temp.month
data['order_week_day'] = temp.weekday
data['order_hour'] = temp.hour
data['order_month_year'] = temp.to_period('M')
data[['order_year','order_month','order_week_day','order_hour','order_month_year']] 

# Explore - Sales by time ( year, month, day, hour)
plt.figure(figsize = (10,12))
plt.subplot(2,2,1)
df_year = data.groupby('order_week_day')
df_year['Sales'].mean().plot(figsize=(12,12), title='Avg sales by year')

plt.subplot(2,2,2)
df_day = data.groupby('order_year')
df_day['Sales'].mean().plot(figsize=(12,12), title='Avg sales by week day')

plt.subplot(2,2,3)
df_hour = data.groupby('order_hour')
df_hour['Sales'].mean().plot(figsize=(12,12), title='Avg sales by hour')

plt.subplot(2,2,4)
df_month = data.groupby('order_month')
df_month['Sales'].mean().plot(figsize=(12,12), title='Avg sales by month')
plt.show()

# Explore - Product Price / Sales per customer
data.plot( x ='Product Price', y= 'Sales per customer')
plt.title('Relation btw Product Price / Sales per customer')
plt.xlabel('Product Price')
plt.ylabel('Sales per customer')
plt.show()

# RFM
data['TotalPrice'] = data['Order Item Quantity'] * data['Order Item Total']
data[['TotalPrice', 'Order Item Quantity', 'Order Item Total']]
data['Order Item Quantity'].value_counts()

# Change Format - order date (DateOrders) to_datetime
data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])
data['order date (DateOrders)']

# check last order time
data['order date (DateOrders)'].max()

# set now = 2018-2-1
import datetime
present  = datetime.datetime(2018, 2, 1)

# Calculate RFM of each user
# groupby - Order Customer Id
# rencency: order data (DateOrder) = x.max is the last order day
# frequency: Order Id = len('Order Id')
# monetary: TotalPrice  = sum('TotalPrice')
customer_seg = data.groupby('Order Customer Id').agg({\
    'order date (DateOrders)': lambda x: (present-x.max()).days, \
    'Order Id': lambda x: len(x), \
    'TotalPrice': lambda x: x.sum()})
customer_seg

customer_seg.rename(columns={'order date (DateOrders)':'R_Value', \
                             'Order Id':'F_Value', \
                             'TotalPrice':'M_Value'}, inplace=True)
customer_seg


# separate into 4 quarters
quantiles = customer_seg.quantile([0.25, 0.5, 0.75])
quantiles
quantiles.shape
quantiles.columns

# R_Value : less is better  
def R_Score(a, b, c):
    if a <= c[b][0.25]:
        return 4
    elif a <= c[b][0.50]:
        return 3
    elif a <= c[b][0.75]:
        return 2
    else:
        return 1

# F_Value & M_Value 
def FM_Score(a, b, c):
    if a <= c[b][0.25]:
        return 1
    elif a <= c[b][0.50]:
        return 2
    elif a <= c[b][0.75]:
        return 3
    else:
        return 4

customer_seg['R_Score'] = customer_seg['R_Value'].apply(R_Score, args=('R_Value', quantiles))
customer_seg['F_Score'] = customer_seg['F_Value'].apply(FM_Score, args=('F_Value', quantiles))
customer_seg['M_Score'] = customer_seg['M_Value'].apply(FM_Score, args=('M_Value', quantiles))
customer_seg

def RFM_User(df):
    if df['M_Score'] > 2 and df['F_Score'] > 2 and df['R_Score'] > 2:
        return 'important_value_customer'
    if df['M_Score'] > 2 and df['F_Score'] <= 2 and df['R_Score'] > 2:
        return 'important_develop_customer'
    if df['M_Score'] > 2 and df['F_Score'] > 2 and df['R_Score'] <= 2:    
        return 'important_retain_customer'
    if df['M_Score'] > 2 and df['F_Score'] <= 2 and df['R_Score'] <= 2:
        return 'important_detain_customer'
    
    if df['M_Score'] <= 2 and df['F_Score'] > 2 and df['R_Score'] > 2:
        return 'normal_value_customer'
    if df['M_Score'] <= 2 and df['F_Score'] <= 2 and df['R_Score'] > 2:
        return 'normal_develop_customer'
    if df['M_Score'] <= 2 and df['F_Score'] > 2 and df['R_Score'] <= 2:    
        return 'normal_retain_customer'
    if df['M_Score'] <= 2 and df['F_Score'] <= 2 and df['R_Score'] <= 2:
        return 'normal_detain_customer'

customer_seg['Customer_Segmentation'] = customer_seg.apply(RFM_User, axis=1)
customer_seg