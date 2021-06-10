#!/usr/bin/env python
# coding: utf-8

# # follow up  template after that write the code
# 
# 
# 
# 
# 

# ## 1. Let's Now Import library of pandas and numpy
# 
# 
# 

# In[40]:


import pandas as pd
import numpy as np


# ## 2. Load The Data into a Pandas Frame , hint  use **pd.read_csv**

# In[41]:


df = pd.read_csv('kc_house_data.csv')


# ## 3. define x is the features of dataframe and y is label , hint use **df.iloc**

# In[42]:


x = df.drop('price',axis=1)
y = df.price


# ### 4.Drop (id,date) column from x(independent variables ),  hint use **drop()** function

# In[43]:


x = x.drop(['id','date'],axis=1)


# ## 5. Splitting the dataset into the Training set and Test set and put **random_state=2**
# 

# In[44]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2 , random_state=2)


# ## 6. Fitting Linear Regression to the dataset and count regression.score

# In[46]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test, y_test)


# In[ ]:




