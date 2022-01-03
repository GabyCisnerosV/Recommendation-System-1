# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:38:46 2021

@author: gaby
"""
# #
text=["London Paris London","Paris Paris London"]
#Representation of text as vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

#Visualization. Each row is is each string from the text
print(cv.get_feature_names()) #feature names
print(count_matrix.toarray()) #count for each one

