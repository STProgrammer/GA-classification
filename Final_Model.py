# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 02:59:27 2020

@author: Zohainus
"""

import pandas as pd
import nltk
import string 
import re
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from nltk.corpus import stopwords
import unicodedata
#from gensim.parsing.preprocessing import STOPWORDS
from string import punctuation
from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,classification_report,accuracy_score
from nltk.stem.snowball import FrenchStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_union,make_pipeline
from sklearn.feature_selection import SelectFromModel,VarianceThreshold,  SelectPercentile,SelectKBest, f_classif,chi2
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import Imputer
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import csv
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
#Read Dataset
#To see full tweet


pd.set_option("display.max_colwidth", 100)
dataset = pd.read_csv("C:\\Users\\xxx\\Desktop\\engtweets1.csv",encoding="latin-1", names = ["label","Tweets"]).astype(str)
#new_data = pd.read_csv("F:\\april2019.csv",encoding="latin-1", names = ["label","Tweets"]).astype(str)
def preprocess_text(text):
         # Lowercase
        text = text.lower()
        
        words_seperated_by_space = text.split(" ")
        words_seperated_by_space = [k.replace("\\xa0", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\xc2", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\n", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\r", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\xc8", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x9b", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x99", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\xc4", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x83", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\"99%er\"", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x99", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("2x80", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("cxf3", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("2x80", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("u0111", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("cixe2x80m", " ") for k in words_seperated_by_space]

        words_seperated_by_space = [k.replace("don't", "dont") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("won't", "wont") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("can't", "cant") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("i\'m", "i am") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("ain't", "is not") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'ll", "will") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'t", "not") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'ve", "have") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("'s", "is") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'re", "are") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'d", "would") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\ \\ ", " ") for k in words_seperated_by_space]

        words_seperated_by_space = [re.sub(" u ", "you", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("[!]+", "!", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[?]+', "?", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[.]+', ".", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[\\\]+', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[\']+', ".", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(haha)+", "haha", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(l(ol)+)", "lol ", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(bw(a)+h)", "bwah", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(bwa(h)+)", "bwah", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(bwa(h)+)", "bwah", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("<[a-z]*>", "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("[*****]+", "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(xe2x80x(9|a6))", "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('_', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('-', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('<>', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('g(rrr)+', "grr", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('u(mmm)+|u(mm)+', "umm", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('@[A-Za-z0-9]+',' ',str(k))for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('https?://[A-Za-z0-9./]+',' ',str(k))for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('#[A-Za-z0-9]+',' ',str(k))for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('\W', ' ', str(k))for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('\s+', ' ', str(k))for k in words_seperated_by_space]
        text = ' '.join(words_seperated_by_space)
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', '', text)
        # Remove twitter handlers, hashtags symbols and URLs
        text = re.sub(r'@[\w_-]+', ' ', text)
        text = re.sub('https?://[^ ]+', ' ', text)
        text = re.sub('#', '', text)
        text = re.sub('rt', '', text)
        # Expand contractions
        text = re.sub(r"i'm", " i am ", text)
        text = re.sub(r" im ", " i am ", text)
        text = re.sub(r"\: p", "", text)
        text = re.sub(r" ive ", " i have ", text)
        text = re.sub(r" he's ", " he is ", text)
        text = re.sub(r" she's ", " she is ", text)
        text = re.sub(r" that's ", " that is ", text)
        text = re.sub(r" what's ", " what is ", text)
        text = re.sub(r" where's ", " where is ", text)
        text = re.sub(r" haven't ", " have not ", text)
        text = re.sub(r" ur ", " you are ", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r" won't ", " will not ", text)
        text = re.sub(r" wouldn't ", " would not ", text)
        text = re.sub(r" can't ", " cannot ", text)
        text = re.sub(r" cannot ", " cannot ", text)
        text = re.sub(r" don't ", " do not ", text)
        text = re.sub(r" didn't ", " did not ", text)
        text = re.sub(r" doesn't ", " does not ", text)
        text = re.sub(r" isn't ", " is not ", text)
        text = re.sub(r" it's ", " it is ", text)
        text = re.sub(r" who's ", " who is ", text)
        text = re.sub(r" there's ", " there is ", text)
        text = re.sub(r" weren't ", " were not ", text)
        text = re.sub(r" okay ", " o", text)
        text = re.sub(r" you're ", " you are ", text)
        text = re.sub(r" c'mon ", " come on ", text)
        text = re.sub(r"in'", "ing", text)
        text = re.sub(r"\'s", " s", text)
        # Remove ponctuation and special chars except ! and ?
        text = re.sub('[^a-zA-Z?!\s]', ' ', text)
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        sentence = []
        for word in text.split(' '):
            sentence.append(lemmatizer.lemmatize(word))
        # Rebuild sentences
        text = ' '.join(sentence)
        # Remove stopwords
        stopWords = set(stopwords.words('english'))
        sentence = []
        for word in text.split(' '):
            if word not in stopWords:
                sentence.append(word)
        # Rebuild sentences
        text = ' '.join(sentence)
        return text
# caling function
dataset['Tweets_clean'] = dataset['Tweets'].apply(preprocess_text)
dataset['Tweets_clean'].values.reshape(1,-1)
print (f'G = {len(dataset[dataset["label"]=="G"])}')
print (f'NG = {len(dataset[dataset["label"]=="NG"])}')
#new_data['Tweets_clean'] = new_data['Tweets'].apply(preprocess_text)
#print(new_data['Tweets_clean'])
# TFidfVectorization
tfidf_vect = TfidfVectorizer()
X = tfidf_vect.fit_transform(dataset['Tweets_clean'])
print(X.shape)
#X1 = tfidf_vect.transform(new_data['Tweets_clean']).toarray()
#print(X1.shape)
y = dataset['label']
print(dataset.groupby(['label']).size())
skf = StratifiedKFold(n_splits=10, random_state=18, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Comparison of Classifiers
models = {
            'LinearSVC':LinearSVC(),
            'SVC':SVC(C = 10, gamma= 0.4, kernel = 'sigmoid') ,
            #'SVC':SVC(C = 10, gamma= 0.2, kernel = 'linear') ,
            #'SVC':SVC(C = 10, gamma= 0.4, kernel = 'poly') ,
            'LogisticRegression':LogisticRegression(penalty='l2', dual=False, tol=0.0001, C= 0.5, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, verbose=5, warm_start=False, n_jobs=-1),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'RandomForestClassifier':RandomForestClassifier(),
            'GradientBoostClassifier':GradientBoostingClassifier(),
            'KNN':KNeighborsClassifier(),
            'NB':GaussianNB(),
            'AdaBoost':AdaBoostClassifier(),
            'BaggingClassifier':BaggingClassifier(),
            'XGBClassifier':XGBClassifier(),
            'LightGBM':LGBMClassifier()
            }


f = open('FianlReportClassifier.txt','w')
for name, model in models.items():
    clf = model
    clf.fit(X_train.toarray(), y_train)
    y_pred = clf.predict(X_test.toarray())
    #y_pred = clf.predict(X1)
    #print(y_pred)
    
        
    print('Precision score of ' + name , precision_score(y_test,y_pred, pos_label="G"))
    print('Recall score of ' + name , recall_score(y_test,y_pred, pos_label="G"))
    print('F_measure of ' + name , f1_score(y_test,y_pred, pos_label="G"))
    print('Accuracy score of ' + name , accuracy_score(y_test,y_pred))


    print('Accuracy score of ' + name , accuracy_score(y_test,y_pred))
    f.writelines('%s,%s\n,%s\n'%('Accuracy score of '+ name , accuracy_score(y_test, y_pred),classification_report(y_test, y_pred)))
    print('Accuracy score of '+ name , accuracy_score(y_test, y_pred),'\n',classification_report(y_test, y_pred,  average="binary", pos_label="G"))
    print(confusion_matrix(y_test, y_pred))

# it is the code when i use another dataset to train the model. ok    
'''
question1 = y_pred  #question 1 data
question2 = new_data['Tweets_clean'] #question 2 data
df = pd.DataFrame(columns=["label", "Tweets"])
df["label"] = question1
df["Tweets"] = question2
df.to_csv("C:\\Users\\Zohainus\\Desktop\\newdata\\april2019.csv",index=False)
new_data_op = pd.read_csv("C:\\Users\\Zohainus\\Desktop\\newdata\\april2019.csv",encoding="latin-1", names = ["label","Tweets"],delimiter=',').astype(str)
#How many labels in dataset
print (f'G = {len(new_data_op[new_data_op["label"]=="G"])}')
print (f'NG = {len(new_data_op[new_data_op["label"]=="NG"])}')



'''

  #pred = model.predict(X_test)
   
'''
    # Construct the Confusion Matrix
    label = ['G', 'NG']
    cm = confusion_matrix(y_test, pred, label)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + label)
    ax.set_yticklabels([''] + label)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title()
    plt.show()
    '''
'''
# Confusion Matrix of LinearSvc
#import modules
import warnings
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#ignore warnings
warnings.filterwarnings('ignore')

pred = clf.predict(X_test)

#Construct the Confusion Matrix
label = ['G', 'NG']
cm = confusion_matrix(y_test, pred, label)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + label)
ax.set_yticklabels([''] + label)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title()
plt.show()

'''






'''

'''
#Exploring the dataset
# Sahpe of data
print (f"Input data has {len(dataset)} rows, {len(dataset.columns)}columns")
#How many labels in dataset
print (f'G = {len(new_data[new_data["label"]=="G"])}')
print (f'NG = {len(new_data[new_data["label"]=="NG"])}')
# Missing values in any row, ignore those
print (f"Number of missing labels = {dataset ['label'].isnull().sum() }")
print (f"Number of missing labels = {dataset ['Tweets'].isnull().sum() }")

'''
'''
# Confusion Matrix of LinearSvc
#import modules
import warnings
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline  

#ignore warnings
warnings.filterwarnings('ignore')
model = LinearSVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

#Construct the Confusion Matrix
label = ['G', 'NG']
cm = confusion_matrix(y_test, pred, label)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
ax.set_xticklabels([''] + label)
ax.set_yticklabels([''] + label)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title()
plt.show()

'''

# Hyper  parameter Tunning
from sklearn.model_selection import GridSearchCV 
# defining parameter range 

param_grid = [
  {'C': [10,100,100,1000,10000,100000,1000000],
  'kernel': ['sigmoid', 'linear'],
  'gamma': [0.4, 0.5]}
 ]

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 2) 

# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 

# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 


grid_predictions = grid.predict(X_test) 

# print classification report 
print(classification_report(y_test, grid_predictions)) 
print(accuracy_score(y_test,grid_predictions))


