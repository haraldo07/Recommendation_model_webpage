# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 22:23:39 2023

@author: User
"""

import pandas as pd
import numpy as np
import re
df=pd.read_csv('Anime_data.csv')
df.drop(columns=['Producer','Studio','ScoredBy'],inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

tfidf_mat = tfidf.fit_transform(df['Genre'])
tfidf_mat
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_mat,tfidf_mat)
cosine_sim
data = pd.Series(df['Genre'],index = df.index)
data = pd.DataFrame(data)

class ItemRecommender:
    def __init__(self):
        self.data = data
        self.cosine_sim = cosine_sim
        
    def recommendation(self, keyword):
        index = self.data[self.data['Genre'].str.contains(keyword, flags=re.IGNORECASE, regex=True)].index[0]
        sim_score = list(enumerate(self.cosine_sim[index]))    
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)

        sim_score = sim_score[1:8]
        final_index = [i[0] for i in sim_score]
        return final_index
    
    def predict(self,ram):
        idx = self.recommendation(ram)
        b=pd.DataFrame()
        b['Title']=df['Title'].iloc[idx]
        b['No. Of Episodes']=df['Episodes'].astype('int64').iloc[idx]
        b.reset_index(drop=True,inplace=True)
        #b.set_index('Title',inplace=True)#.to_dict()
        return b
        #return b['Episodes']
    
rec=ItemRecommender()

import pickle
pickle.dump(rec,open('model.pkl','wb'))
            