import pandas as pd
import numpy as np
import re
import nltk

nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


train = pd.read_csv("dataset/Train.csv")

# Associate Category names with numerical index and save it in new column CategoryId
target_category = train['category'].unique()

train['categoryId'] = train['category'].factorize()[0]

# Create a new pandas dataframe "category", which only has unique Categories, also sorting this list in order of CategoryId values
category = train[['category', 'categoryId']].drop_duplicates().sort_values('categoryId')

def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)
train['content'] = train['content'].apply(remove_tags)


def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews
train['content'] = train['content'].apply(special_char)


def convert_lower(text):
    return text.lower()
train['content'] = train['content'].apply(convert_lower)


x = train['content']
y = train['categoryId']

from sklearn.feature_extraction.text import CountVectorizer
x = np.array(train.iloc[:,0].values)
y = np.array(train.categoryId.values)
cv = CountVectorizer(max_features = 5000)
x = cv.fit_transform(train.content).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle = True)

splits = {
    "x_train": x_train, 
    "x_test": x_test, 
    "y_train": y_train, 
    "y_test": y_test
}

import pickle

with open('data_splits.pkl', 'wb') as file:
    pickle.dump(splits, file)




