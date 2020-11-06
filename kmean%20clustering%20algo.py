#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("project.csv")
df.head()


# In[5]:


plt.scatter(df["variable1"],df["variable2"])


# In[6]:


km =KMeans(n_clusters=3)
km


# In[7]:


y_predicted = km.fit_predict(df[['variable1','variable2']])
y_predicted


# In[8]:


df['cluster'] = y_predicted
df.head()


# In[11]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.variable1,df1['variable2'],color='green')
plt.scatter(df2.variable1,df2['variable2'],color='red')
plt.scatter(df3.variable1,df3['variable2'],color='black')

plt.xlabel('variable1')
plt.ylabel('variable2')
plt.legend()


# In[12]:


scaler = MinMaxScaler()
scaler.fit(df[['variable2']])
df['variable2']=scaler.transform(df['variable2'])

scaler.fit(df.variable1)
df.variable1 = scaler.transform(df.variable1)
df


# In[13]:


km= KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['variable1','variable2']])
y_predicted


# In[14]:


df['cluster'] = y_predicted
df.drop('cluster',axis='colums',inplace=True)
df


# In[15]:


km.cluster_centers_


# In[16]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.variable1,df1['variable2'],color='green')
plt.scatter(df2.variable1,df2['variable2'],color='red')
plt.scatter(df3.variable1,df3['variable2'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.legend()


# In[18]:


k_rng = range(1,7)
sse =[]
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['variable1','variable2']])
    sse.append(km.inertia_)


# In[19]:


sse


# In[20]:


plt.xlabel('k')
plt.ylabel('sum of squares error')
plt.plot(k_rng,sse)


# In[ ]:




