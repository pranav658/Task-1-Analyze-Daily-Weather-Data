#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# # Load the data

# In[2]:


df=pd.read_csv('weather.csv')


# # Data Exploration 

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.describe(include='O')


# Date-Time Breaking down

# In[8]:


df[["Date-Time","TZ"]]=df["Formatted Date"].str.split("+",expand=True)
df=df.drop(columns="Formatted Date")
df.head()


# In[9]:


columns_order=["Date-Time","TZ","Summary","Precip Type","Temperature (C)","Apparent Temperature (C)",
                "Humidity","Wind Speed (km/h)","Wind Bearing (degrees)","Visibility (km)","Loud Cover",
                "Pressure (millibars)", "Daily Summary"]
df=df.reindex(columns=columns_order)
df=df.drop(columns="TZ")
df.head()


# In[10]:


df["Date-Time"]=pd.to_datetime(df["Date-Time"])
df.info()


# In[11]:


df.head()


# In[12]:


df["Year"]=pd.DatetimeIndex(df["Date-Time"]).year
df["Month"]=df["Date-Time"].dt.month_name()
df["day"]=df["Date-Time"].dt.day
df.head()


# # Data Analysis

# Wind speed

# In[13]:


df['Wind Speed (km/h)'].describe()


# In[14]:


avg_wind_speed=pd.DataFrame(df.groupby("Year")["Wind Speed (km/h)"].mean())


# In[15]:


avg_wind_speed


# # Data Visualization

# In[16]:


df['Summary'].value_counts().plot(kind='bar', cmap='rainbow')
plt.title('SUMMARY OF THE WEATHER')
plt.show()


# In[17]:


df['Precip Type'].value_counts().plot(kind='bar', cmap='rainbow')
plt.title('SUMMARY OF THE Precip Type')
plt.show()


# Summary vs Precip Type

# In[19]:


sns.scatterplot(y=df['Summary'], x=df['Precip Type'], color='blue')
plt.title('SUMMARY VS Precip Type')
plt.show()


# the possibility of rain exists in all summary weather while the precipitation to be snowing shows some difference.

# # precipitation type vs other variables

# relation of Precip Type with all other numerical variables.

# In[29]:


df_new_num=df.drop(['Summary','Date-Time','Year','Month','day' ], axis=1)
df_new_num=df_new_num.drop(['Precip Type','Daily Summary'],axis=1)
df_new_num=df_new_num.drop(['Loud Cover'],axis=1)
df_new_num


# In[31]:


plt.figure(figsize=(20,10))
re=1
for i in df_new_num.columns:
    plt.subplot(2,4,re)
    sns.barplot(x=df['Precip Type'], y=df_new_num[i],palette='Blues')
    re+=1
    plt.title(i)
    
plt.show()


# # summary vs other variables

# In[35]:


plt.figure(figsize=(10,50))
re=1
for i in df_new_num.columns:
    plt.subplot(7,1,re)
    sns.barplot(y=df['Summary'], x=df_new_num[i],palette='GnBu')
    re+=1
    plt.title(i)
    plt.xticks(rotation=90)
    
plt.show()
plt.tight_layout()


# In[55]:


fig,ax=plt.subplots(figsize=(10,8))
sns.lineplot(x=avg_wind_speed.index,y=avg_wind_speed["Wind Speed (km/h)"])
plt.title("Average wind speed over the years")


# In[56]:


month_avg_wind_Speed=pd.DataFrame(df.groupby("Month")["Wind Speed (km/h)"].mean())
order=["January","February","March","April","May","June","July","August","September",
            "October","November","December"]
monthly_wind_speed=month_avg_wind_Speed.reindex(index=order)
monthly_wind_speed


# In[57]:


fig,ax=plt.subplots(figsize=(10,8))
sns.lineplot(x=monthly_wind_speed.index,y=monthly_wind_speed["Wind Speed (km/h)"])
plt.title("Monthly Average wind speed over the years")


# Weather Condition

# In[58]:


df['Summary'].value_counts()


# In[59]:


weather_cond=pd.DataFrame(df.groupby("Year")["Summary"].describe(include="O").top)

weather_cond.rename(columns={"top":"most frequent weather"})


# In[60]:


m_weather_cond=pd.DataFrame(df.groupby("Month")["Summary"].describe(include="O").top)
order=["January","February","March","April","May","June","July","August","September",
            "October","November","December"]
m_weather_cond.rename(columns={"top":"most frequent weather"})
monthly=m_weather_cond.reindex(index=order)
monthly


# visibility

# In[61]:


month_avg_visibility=pd.DataFrame(df.groupby("Month")["Visibility (km)"].mean())
order=["January","February","March","April","May","June","July","August","September",
            "October","November","December"]
monthly_visibility=month_avg_visibility.reindex(index=order)
monthly_visibility


# In[62]:


fig,ax=plt.subplots(figsize=(15,8))
sns.lineplot(x=monthly_visibility.index,y=monthly_visibility["Visibility (km)"])
plt.title("Monthly visibility over the years")


# Precipitation: rainy and snowy months

# In[63]:


percip=pd.DataFrame(df.groupby("Month")["Precip Type"].describe(include="O").top)
order=["January","February","March","April","May","June","July","August","September",
            "October","November","December"]
m_p=percip.rename(columns={"top":"Precip Type"})
monthly_percip=m_p.reindex(index=order)
monthly_percip


# Temperature
# 

# In[64]:


fig,ax=plt.subplots(figsize=(10,8))
plt.hist(df["Temperature (C)"],bins=10,rwidth=0.9)
plt.xlabel("Temperature (C)")
plt.ylabel("freq")


# In[65]:


year_avg_temp=pd.DataFrame(df.groupby("Year")["Temperature (C)"].mean())
year_avg_temp


# In[124]:


min_temp=pd.DataFrame(df.groupby("Month")["Temperature (C)"].min())
df = pd.merge(df, min_temp, left_on='Month', right_index=True, how='left', suffixes=('', '_avg'))
df.head()


# In[125]:


max_temp=pd.DataFrame(df.groupby("Month")["Temperature (C)"].max())
df = pd.merge(df, max_temp, left_on='Month', right_index=True, how='left', suffixes=('', '_avg'))
df.rename(columns={'Temperature (C)_avg': 'monthly_avg_min_temp'}, inplace=True)
df.head()


# In[128]:


df.rename(columns={'monthly_avg_min_temp': 'min_temp'}, inplace=True)
df.rename(columns={'monthly_avg_max_temp': 'max_temp'}, inplace=True)
df.head()


# In[129]:


plt.figure(figsize=(12,5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show()


# In[119]:


df['Precip Type'].value_counts()


# In[120]:


df['Precip Type'] = df['Precip Type'].astype(str)


# In[121]:


df['Rainfall'] = df['Precip Type'].apply(lambda x: 1 if x.lower() == 'rain' else 0)


# In[130]:


df.head()


# Rainfall Prediction
# 
# Prepare the data for prediction

# In[132]:


X = df[['min_temp', 'max_temp']]
y = df['Rainfall']


# # Split the data into training and testing sets
# 

# In[133]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train a linear regression model
# 

# In[134]:


model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions and calculate the Mean Squared Error

# In[135]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')


# # Conclusions and Insights 

# In[139]:


highest_rainfall_month = max_temp.idxmax()
lowest_rainfall_month = max_temp.idxmin()
print(f'Highest rainfall month: {highest_rainfall_month}, Lowest rainfall month: {lowest_rainfall_month}')


# In[ ]:




