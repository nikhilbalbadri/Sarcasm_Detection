#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[2]:


#Upload Twitter Data
twitter_data = pd.read_csv(r'D:/Project_Sarcasm_Detection/Twitter_Data.csv')


# In[3]:


twitter_data.head()


# In[4]:


twitter_data.isnull().sum()


# In[5]:


twitter_data = twitter_data.replace(np.nan, '', regex=True)
twitter_data.isnull().sum()


# In[6]:


# train test split
train, test = train_test_split(twitter_data, test_size = 0.3, stratify = twitter_data['Sarcasm'], random_state=21)


# In[7]:


print(train.shape)
print(test.shape)


# In[8]:


pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=1000,
                                                      stop_words= ENGLISH_STOP_WORDS)),
                            ('model', LogisticRegression())])


# In[9]:


pipeline.fit(train.Tweet4, train.Sarcasm)


# In[10]:


# import joblib
from joblib import dump

# dump the pipeline model
dump(pipeline, filename="text_classification.joblib")


# In[ ]:




