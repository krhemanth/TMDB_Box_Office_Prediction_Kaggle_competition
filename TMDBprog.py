# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:48:15 2019

@author: krhem
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast

Train = pd.read_csv("train.csv")
final_Test = pd.read_csv("test.csv")

# Feature Engineering
def GetCSVFromListOfDict(keyNameToFetch,column,columnName):
    column=column.copy()
    column=column.fillna('[{}]')
    columnList=[]
    for index,row in column.iteritems():
        columnStr=''
        listofDict=ast.literal_eval(row)
        for dic in listofDict:

            if(keyNameToFetch in dic.keys()):
                columnStr=columnStr+';'+str(dic[keyNameToFetch]) 
        columnStr=columnStr.strip(';') # trim leading ;
        columnList.append(columnStr)

    tempDF=pd.DataFrame(columnList,columns=[columnName])
    return tempDF[columnName] 


Train['belongs_to_collection']=GetCSVFromListOfDict('name',Train.belongs_to_collection,'belongs_to_collection')
Train['genres']=GetCSVFromListOfDict('name',Train.genres,'genres')
Train['production_companies']=GetCSVFromListOfDict('name',Train.production_companies,'production_companies')
Train['production_countries']=GetCSVFromListOfDict('name',Train.production_countries,'production_countries')
Train['spoken_languages']=GetCSVFromListOfDict('iso_639_1',Train.spoken_languages,'spoken_languages')
Train['Keywords']=GetCSVFromListOfDict('name',Train.Keywords,'Keywords')
Train['Crew_Dept']=GetCSVFromListOfDict('department',Train.crew,'crew')
Train['Crew_Job']=GetCSVFromListOfDict('job',Train.crew,'crew')
Train['Crew_Name']=GetCSVFromListOfDict('name',Train.crew,'crew')
Train['Crew_Gender']=GetCSVFromListOfDict('gender',Train.crew,'crew')


final_Test['belongs_to_collection']=GetCSVFromListOfDict('name',final_Test.belongs_to_collection,'belongs_to_collection')
final_Test['genres']=GetCSVFromListOfDict('name',final_Test.genres,'genres')
final_Test['production_companies']=GetCSVFromListOfDict('name',final_Test.production_companies,'production_companies')
final_Test['production_countries']=GetCSVFromListOfDict('name',final_Test.production_countries,'production_countries')
final_Test['spoken_languages']=GetCSVFromListOfDict('iso_639_1',final_Test.spoken_languages,'spoken_languages')
final_Test['Keywords']=GetCSVFromListOfDict('name',final_Test.Keywords,'Keywords')
final_Test['Crew_Dept']=GetCSVFromListOfDict('department',final_Test.crew,'crew')
final_Test['Crew_Job']=GetCSVFromListOfDict('job',final_Test.crew,'crew')
final_Test['Crew_Name']=GetCSVFromListOfDict('name',final_Test.crew,'crew')
final_Test['Crew_Gender']=GetCSVFromListOfDict('gender',final_Test.crew,'crew')


# =============================================================================
# print(len(Train.belongs_to_collection))
# Train.belongs_to_collection.value_counts()
# C = Train.loc[Train['belongs_to_collection'] == 'Singam Collection']
# =============================================================================

# Checking missing value in belongs_to_collection
Train['belongs_to_collection_ISMISSING']=(Train.belongs_to_collection.str.strip()=='').astype(int)
final_Test['belongs_to_collection_ISMISSING']=(final_Test.belongs_to_collection.str.strip()=='').astype(int)
cor1 =Train[['belongs_to_collection_ISMISSING','revenue']].corr()
Train.drop(columns=['belongs_to_collection'],inplace=True)
final_Test.drop(columns=['belongs_to_collection'],inplace=True)

# For genres column
print(len(Train.genres))
print(Train.genres.isna().sum())
Train['genres']=Train.genres.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns
Train['genres']=Train.genres.str.replace(';',' ')
# Converting to 0: Absent and 1:Present
from sklearn.feature_extraction.text import CountVectorizer

vectFeatures = CountVectorizer(max_features=10)
vectFeatures.fit(Train['genres'])

featuresTrainSplit=vectFeatures.transform(Train['genres'])
featuresUnseenTestSplit=vectFeatures.transform(final_Test['genres'])
featuresTrain=pd.DataFrame(featuresTrainSplit.toarray(),columns=vectFeatures.get_feature_names())
featuresfinal_Test=pd.DataFrame(featuresUnseenTestSplit.toarray(),columns=vectFeatures.get_feature_names())
featuresTrain.columns='genres_'+featuresTrain.columns
featuresfinal_Test.columns='genres_'+featuresfinal_Test.columns
Train=pd.concat([Train,featuresTrain],axis=1)
final_Test=pd.concat([final_Test,featuresfinal_Test],axis=1)

# Drop genres column
Train.drop(columns=['genres'],inplace=True)
final_Test.drop(columns=['genres'],inplace=True)

#Production Countries
print(len(Train.production_countries))
Train.production_countries.value_counts().head(20)

Train['production_countries']=Train.production_countries.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns
Train['production_countries']=Train.production_countries.str.replace(';',' ')
final_Test['production_countries']=final_Test.production_countries.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns
final_Test['production_countries']=final_Test.production_countries.str.replace(';',' ')

# Creating Column films produced in America
Train['IsProductionFromUSA']=(Train['production_countries']=='united_states_of_america').astype(int)
final_Test['IsProductionFromUSA']=(final_Test['production_countries']=='united_states_of_america').astype(int)
Train.drop(columns=['production_countries'],inplace=True)
final_Test.drop(columns=['production_countries'],inplace=True)

#Original language and Spoken Language
Train['IsEnglishLanguage']=(
                    (Train['spoken_languages'].str.contains('en'))
                    & 
                    (Train['original_language']=='en')).astype(int)



final_Test['IsEnglishLanguage']=(
                    (final_Test['spoken_languages'].str.contains('en'))
                    &
                    (final_Test['original_language']=='en')).astype(int)
Train[['IsEnglishLanguage','revenue']].corr()

Train.drop(columns=['spoken_languages','original_language'],inplace=True)
final_Test.drop(columns=['spoken_languages','original_language'],inplace=True)

#Keywords
Train['Keywords']=Train.Keywords.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns
Train['Keywords']=Train.Keywords.str.replace(';',' ')
Train['Keywords']=Train['Keywords'].str.lower()


final_Test['Keywords']=final_Test.Keywords.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns
final_Test['Keywords']=final_Test.Keywords.str.replace(';',' ')
final_Test['Keywords']=final_Test['Keywords'].str.lower()

#create columns for Keywords
from sklearn.feature_extraction.text import CountVectorizer

vectFeatures = CountVectorizer(max_features=20)
vectFeatures.fit(Train['Keywords'].str.lower())

featuresTrainSplit=vectFeatures.transform(Train['Keywords'])
featuresUnseenTestSplit=vectFeatures.transform(final_Test['Keywords'])



featuresTrain=pd.DataFrame(featuresTrainSplit.toarray(),columns=vectFeatures.get_feature_names())
featuresfinal_Test=pd.DataFrame(featuresUnseenTestSplit.toarray(),columns=vectFeatures.get_feature_names())


featuresTrain.columns='Keywords'+featuresTrain.columns
featuresfinal_Test.columns='Keywords'+featuresfinal_Test.columns
Train=pd.concat([Train,featuresTrain],axis=1)
final_Test=pd.concat([final_Test,featuresfinal_Test],axis=1)

Train.drop(columns=['Keywords'],inplace=True)
final_Test.drop(columns=['Keywords'],inplace=True)

#Homepage
Train['IsHomePageAvailable']=(Train.homepage.isna()==False).astype(int)
final_Test['IsHomePageAvailable']=(final_Test.homepage.isna()==False).astype(int)

Train[['IsHomePageAvailable','revenue']].corr()

#Train Date
dateSplit=Train.release_date.str.extract('([0-9]+)/([0-9]+)/([0-9]+)')
dateSplit.columns=['ReleaseMonth','ReleaseDate','ReleaseYear']


dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']='19'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']
dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']='20'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']

Train.drop(columns=['release_date'],inplace=True)
Train=pd.concat([Train,dateSplit.astype(int)],axis=1)

# Test Date
print(final_Test.release_date.mode())
final_Test['release_date'].fillna('9/9/11',inplace=True)

dateSplit1=final_Test.release_date.str.extract('([0-9]+)/([0-9]+)/([0-9]+)')
dateSplit1.columns=['ReleaseMonth','ReleaseDate','ReleaseYear']


dateSplit1.loc[dateSplit1.ReleaseYear.astype(int)>20,'ReleaseYear']='19'+dateSplit1.loc[dateSplit1.ReleaseYear.astype(int)>20,'ReleaseYear']
dateSplit1.loc[dateSplit1.ReleaseYear.astype(int)<=20,'ReleaseYear']='20'+dateSplit1.loc[dateSplit1.ReleaseYear.astype(int)<=20,'ReleaseYear']


final_Test.drop(columns=['release_date'],inplace=True)
final_Test=pd.concat([final_Test,dateSplit1.astype(int)],axis=1)

# Log Scaling
Train['revenue']=np.log1p(Train.revenue)

Train['budget']=np.log1p(Train.budget)
final_Test['budget']=np.log1p(final_Test.budget)


Train['popularity']=np.log1p(Train.popularity)
final_Test['popularity']=np.log1p(final_Test.popularity)

#Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
TrainNum=Train.select_dtypes(include=numerics)
final_TestNum=final_Test.select_dtypes(include=numerics)
TrainNum.drop(columns=['id'],inplace=True)
final_TestNum.drop(columns=['id'],inplace=True)

TrainNum=TrainNum.fillna(TrainNum.median())
final_TestNum=final_TestNum.fillna(TrainNum.median())

#Train Test split
from sklearn import model_selection # for splitting into train and test
import sklearn
# Split-out validation dataset
X = TrainNum.drop(columns=['revenue'])
Y = TrainNum['revenue']

validation_size = 0.2
seed = 100
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Import XG Boost
import xgboost
model_XG = xgboost.XGBRegressor() 
model_XG.fit(X_train, Y_train)

# make predictions for test data

trainResult_XG = model_XG.predict(X_train)
testResult_XG = model_XG.predict(X_test)
final_TestResult_XG=model_XG.predict(final_TestNum)


########## TRAIN DATA RESULT ##########

print('---------- TRAIN DATA RESULT ----------')
# The mean squared error
print("Mean squared error: %.5f"%np.sqrt( mean_squared_error(Y_train, trainResult_XG)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % r2_score(Y_train, trainResult_XG))

########## TEST DATA RESULT ##########

print('---------- TEST DATA RESULT ----------')
# The mean squared error
print("Mean squared error: %.5f"% np.sqrt(mean_squared_error(Y_test, testResult_XG)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % r2_score(Y_test, testResult_XG))

#############Submission##################

final_TestResult_XG=np.expm1(final_TestResult_XG)
submission=pd.DataFrame([final_Test.id,final_TestResult_XG]).T

submission.columns=['id','revenue']

submission.id=submission.id.astype(int)

submission.to_csv('submission.csv',index=False)