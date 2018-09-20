
# coding: utf-8

# In[1]:


import nltk
import csv
import pandas as pd
import datetime
import numpy as np
import re


# In[3]:


with codecs.open('C:\\Users\\2010y\\Anaconda3\\Clean2.csv', "r", encoding='UTF-8', errors='ignore') as csvfile:
    df=pd.read_csv(csvfile)


# In[4]:


comment=df['comment body']


# In[5]:


comment=list(df['comment body'])


# In[6]:


date=list(df['comment_created_at'])


# In[7]:


regex=re.compile('2016-03.*')
march2016=[m.group(0) for l in date for m in [regex.search(l)] if m]


# In[36]:


regex=re.compile('2016-04.*')
april2016=[m.group(0) for l in date for m in [regex.search(l)] if m]


# In[37]:


common=[]
for i, v in enumerate(date):
    for j in march2016:
        if v==j:
            common.append(i)


# In[38]:


common1=[]
for i, v in enumerate(date):
    for j in april2016:
        if v==j:
            common1.append(i)


# In[39]:


comment_april2016=[]
for i in common1:
    comment_april2016.append(comment[i])


# In[40]:


str_april2016=str(comment_april2016)


# In[43]:


str_april2016=str_april2016.replace("'","\n")
str_april2016=str_april2016.replace("[","")
str_april2016=str_april2016.replace("]","")
str_april2016=str_april2016.replace(",","")


# In[44]:


file=open('C:\\Users\\2010y\\Anaconda3\\GravitySpy\\2016_04.txt', 'w')
file.write(str_april2016)
file.close()


# In[19]:


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim


# In[20]:


from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma=wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma  


# In[21]:


from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


# In[22]:


def prepare_text_for_lda(text):
    tokens=nltk.sent_tokenize(text)
    tokens=[token for token in tokens if len(token)>4]
    tokens=[get_lemma(token) for token in tokens]
    return tokens


# In[48]:


import random
text_data=[]
with codecs.open('C:\\Users\\2010y\\Anaconda3\\GravitySpy\\2016_04.txt', "r", encoding='UTF-8', errors='ignore') as state:
    for line in state:
        tokens=prepare_text_for_lda(line)
        if random.random() > .99:
            print(tokens)
            text_data.append(tokens)


# In[46]:


from gensim import corpora
Lda=gensim.models.ldamodel.LdaModel
dictionary = corpora.Dictionary(text_data)
doc_term_matrix=[dictionary.doc2bow(doc) for doc in text_data]
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=10))

