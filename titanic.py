#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore') #ignore warnings

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = train.append(test)

test.info()
data.info()

############## Observe Data #############
sns.countplot(data['Sex'], hue = data['Survived']) #hue:以色調分類
display(data[['Sex','Survived']].groupby(['Sex'],as_index = False).mean().round(3))
#observation(1):女性存活率大
sns.countplot(data['Pclass'], hue = data['Survived'])
display(data[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().round(3))
#observation(2):等級越高，存活率越大
plt.figure(figsize=(50,12))
sns.barplot(y=data['Survived'],x=data['Age'])
#observation(3):小孩或老年存活率較大，尤其小孩更明顯
sns.countplot(data['Ticket'].str[0], hue = data['Survived'])
#obsercation(4):有某些ticket的開頭字母或數字出現率比較高

############## Feature Engineering #############
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0) #把性別轉成0或1
data['Title1'] = data['Name'].str.split(",",expand=True)[1]
data['Title1'] = data['Title1'].str.split(".",expand=True)[0] #把稱謂擷取出來
data['Title2'] = data['Title1'].replace(regex={'Ms':'Miss',
                                               'Mme':'Mrs',
                                               'Mlle':'Miss',
                                               'Dona':'Mrs',
                                               'Dr':'Mr',
                                               'Major':'Mr',
                                               'Lady':'Mrs',
                                               'the Countess':'Mrs',
                                               'Jonkheer':'Mr',
                                               'Col':'Mr',
                                               'Rev':'Mr',
                                               'Capt':'Mr',
                                               'Sir':'Mr',
                                               'Don':'Mr'}) #把所有稱謂濃縮成四種稱謂
data['Embarked'] = data['Embarked'].fillna('S') #embarked的缺失值用最多人登陸的S港填補
data['Fare']=data['Fare'].fillna(data['Fare'].mean()) #fare用平均數填補
data['Cabin_Letter'] = data['Cabin'].apply(lambda x: str(x)[0])
train = data[:len(train)]
test = data[len(train):]

for i in [train] :
     i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)  
     dt = train.groupby(['Title2'])['Age']
     i['Age'] = dt.transform(lambda x: x.fillna(x.mean())) #age的空值用該稱謂的年齡平均值填補
     i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                       np.where((i['SibSp']+i['Parch']) <= 3,'Small', 'Big'))
     #把親屬統一成一個欄位，並改以solo,small,big表示
     del i['SibSp']
     del i['Parch']#刪除原本的兩個欄位
     
     i['Ticket_Letter'] = i['Ticket'].apply(lambda x: str(x)[0]) #取ticket的第一個字母或數字做為代表
     i['Ticket_Letter'] = i['Ticket_Letter'].apply(lambda x: str(x))
     i['Ticket_Letter'] = np.where((i['Ticket_Letter']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Letter'],
                            np.where((i['Ticket_Letter']).isin(['W', '4', '7', '6', 'L', '5','8','9']),'Low_ticket', 'Other_ticket'))
        #根據觀察結果，把較不常出現的字母或數字以'low＿ticket'表示
for i in [test] :
     i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)  
     dt = test.groupby(['Title2'])['Age']
     i['Age'] = dt.transform(lambda x: x.fillna(x.mean())) #age的空值用該稱謂的年齡平均值填補
     
     i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                       np.where((i['SibSp']+i['Parch']) <= 3,'Small', 'Big'))
     #把親屬統一成一個欄位，並改以solo,small,big表示
     del i['SibSp']
     del i['Parch']#刪除原本的兩個欄位
     
     i['Ticket_Letter'] = i['Ticket'].apply(lambda x: str(x)[0]) #取ticket的第一個字母或數字做為代表
     i['Ticket_Letter'] = i['Ticket_Letter'].apply(lambda x: str(x))
     i['Ticket_Letter'] = np.where((i['Ticket_Letter']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Letter'],
                            np.where((i['Ticket_Letter']).isin(['W', '4', '7', '6', 'L', '5','8','9']),'Low_ticket', 'Other_ticket'))
        #根據觀察結果，把較不常出現的字母或數字以'low＿ticket'表示
    
#類別資料轉數值資料
train['Embarked'] = train['Embarked'].astype('category').cat.codes
train['Pclass'] = train['Pclass'].astype('category').cat.codes
train['Title2'] = train['Title2'].astype('category').cat.codes
train['Fam_Size'] = train['Fam_Size'].astype('category').cat.codes
train['Ticket_Letter'] = train['Ticket_Letter'].astype('category').cat.codes
train['Cabin_Letter'] = train['Cabin_Letter'].astype('category').cat.codes


test['Embarked'] = test['Embarked'].astype('category').cat.codes
test['Pclass'] = test['Pclass'].astype('category').cat.codes
test['Title2'] = test['Title2'].astype('category').cat.codes
test['Fam_Size'] = test['Fam_Size'].astype('category').cat.codes
test['Ticket_Letter'] = test['Ticket_Letter'].astype('category').cat.codes
test['Cabin_Letter'] = test['Cabin_Letter'].astype('category').cat.codes


predictors = ['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Fam_Size', 'Title2','Ticket_Letter','Cabin_Letter']
 
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(train[predictors], train["Survived"])
print("%.4f" % rf.oob_score_) #交叉驗證

pred = rf.predict(test[predictors])

#################### Save Result ########################
submission = pd.DataFrame({
                            "PassengerId": test["PassengerId"],
                            "Survived": pred
                         })
submission.to_csv('submission.csv', index=False)

