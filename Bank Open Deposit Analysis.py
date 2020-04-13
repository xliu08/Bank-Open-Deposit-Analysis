#!/usr/bin/env python
# coding: utf-8

# ### Xiaoting(Theresa) Liu 
# ### Bank Open Deposit Analysis
# 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('bank.csv')
term_deposits = df.copy()
# Have a grasp of how our data looks.
df.head()


# In[5]:


types = df.dtypes
print(types)
sumNullRws = df.isnull().sum()
sumNullRws


# No missing values, good to go. 

# In[6]:


df.describe()


# #### EDA for numerical variables

# In[7]:


#Scatter Plot
g = sns.pairplot(df, hue="deposit", palette="husl")
#this provides an overall corrlationship between numerical variable.


# In[111]:


#Heatmap
#scale deposit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
df['deposit'] =  LabelEncoder().fit_transform(df['deposit'])

corr_matrix = df.corr()

mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

f, ax = plt.subplots(figsize=(20, 18)) 
heatmap = sns.heatmap(corr_matrix, 
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'PuBu',
                
                      vmin = -1, 
                      vmax = 1,
                      annot = True,
                      annot_kws = {"size": 12})
#add the column names as labels
plt.title('Correlation Matrix', fontsize =20)
ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns, rotation = 45)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


# In[235]:


#Duration analysis
#recalled that duration indicates last contact in duration in seconds.
#we can see what is the impact when clients are contacted above or below duration average. 

lst = [df]
df["duration_types"]  = np.nan
avg_duration = df['duration'].mean()

for col in lst:
    col.loc[col["duration"] < avg_duration, "duration_types"] = "Below_average"
    col.loc[col["duration"] > avg_duration, "duration_types"] = "Above_average"

pct_termsub = pd.crosstab(df['duration_types'],
                           df['deposit']).apply(lambda r:round(r/r.sum(),2)*100, axis=1)

ax = pct_termsub.plot(kind="bar",stacked = True,colormap="Set2")
ax.legend(["no","yes"],title='Subscription')
plt.title("The Impact of Duration \n in Term Deposit Subscription", fontsize=18)
plt.xticks(rotation=0)
plt.xlabel("Duration Types", fontsize=18)
plt.ylabel("Percentage (%)", fontsize=18)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            '{:.0f} %'.format(height), 
            horizontalalignment='center', 
            verticalalignment='center')

plt.show()

    


# #### EDA for Categorical Variables 

# In[44]:


df_cat = df.select_dtypes(include = 'object').copy()
df_cat.head(2)


# In[33]:


#Job Analysis 
job_type = pd.crosstab(index=df['job'], columns=df['deposit'])
job_type


# In[109]:


# visulization for job types
job_type.plot(kind='bar', figsize = (10, 8),stacked = True,colormap="Set3")
plt.legend(loc = 'upper right')
plt.xticks(rotation=45)
plt.xlabel("Job Types")
plt.ylabel("Counts")
plt.title("Relationship between Job Types and Term Deposit Subscription ", fontsize=20)
plt.show()


# In[90]:


#Martial, Education and Default Analysis 
plt.figure(figsize=[14,4])
plt.subplot(1,3,1)
sns.countplot(x='marital', hue='deposit', data=df_cat,palette="Set3")
plt.subplot(1,3,2)
sns.countplot(x='education', hue='deposit', data=df_cat,palette="Set3")
plt.subplot(1,3,3)
sns.countplot(x='default', hue='deposit', data=df_cat,palette="Set3")
plt.show()


# In[107]:


#Housing, loan, and contract Analysis 
plt.figure(figsize=[14,4])
plt.subplot(1,3,1)
sns.countplot(x='housing', hue='deposit', data=df_cat,palette="Set3")
plt.subplot(1,3,2)
sns.countplot(x='loan', hue='deposit', data=df_cat,palette="Set3")
plt.subplot(1,3,3)
sns.countplot(x='contact', hue='deposit', data=df_cat,palette="Set3")
plt.show()


# In[305]:


#month and poutcome  Analysis 
plt.figure(figsize=[14,4])
plt.subplot(1,2,1)
sns.countplot(x='month', hue='deposit',order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], 
              data=df_cat,palette="Set3")
plt.subplot(1,2,2)
sns.countplot(x='poutcome', hue='deposit', data=df_cat,palette="Set3")
plt.show()


# #### Predictive Analysis  

# In[8]:


#Pre-processing dataset
#transform categorical attributes
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
 
MultiColumnLabelEncoder(columns = ['job','marital','education','default','housing','loan',
                                   'contact','month','poutcome','deposit']).fit_transform(df).head(5)


# In[9]:


le = MultiColumnLabelEncoder()
df_le = le.fit_transform(df)
df_le.head()


# In[58]:


#Data Standardization 
from sklearn import preprocessing

X1 = df_le.drop('deposit',axis=1)
y = df_le.deposit
#standardize the data attributes
standardized_X = preprocessing.scale(X1)


# In[11]:


#Split data to test and train set
from sklearn.model_selection import train_test_split

y = df_le.deposit
X = standardized_X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[1]:


#Select the best model with the higest accuracy score. 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# prepare models
models = []
models.append(('Logistic Regression',LogisticRegression()))
models.append(('Nearest Neighbors', KNeighborsClassifier()))
models.append(('Linear SV', SVC()))
models.append(('Gradient Boosting Classifier',GradientBoostingClassifier() ))
models.append(('Decision Tree', tree.DecisionTreeClassifier()))
models.append(('Random Forest', RandomForestClassifier()))

         
#evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X=X_train, y=y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[363]:


#Train model with Gradient Boosting Classifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, 
                                        max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
    


# Learning rate of 0.5 gives us the best performance (0.851) on the validation set and good performance on the training set.

# In[29]:


from sklearn.metrics import classification_report

#calculate performance measures for Gradient Boosting Classifier model  
gb_clf = GradientBoostingClassifier(learning_rate=0.5)

gb_clf.fit(X_train, y_train)

y_pred = gb_clf.predict(X_test)

print(classification_report(y_test, y_pred))


# In[386]:


#Cross Validation 
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title("Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
ax.set_xticklabels(['Predicted Refused Deposit', 'Predicted Accepted Deposit'])
ax.set_yticklabels(['Actual Refused Deposits', 'Actual Accepted Deposits'], fontsize=16, rotation=360)
ax.set_yticks(np.arange(conf_matrix.shape[0]) + 0.5, minor=False)
plt.xlabel('Predicted label')
plt.show()


# In[46]:


#Precisions and Recall Tradeoff 
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

y_scores = cross_val_predict(gb_clf, X_train, y_train, cv=3, method="decision_function")

precisions, recalls, threshold = precision_recall_curve(y_train, y_scores)

def precision_recall_curve(precisions, recalls, thresholds):
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(thresholds, precisions[:-1], "r--", label="Precisions")
    plt.plot(thresholds, recalls[:-1], "#424242", label="Recalls")
    plt.title("Precision and Recall \n Tradeoff", fontsize=18)
    plt.ylabel("Level of Precision and Recall", fontsize=16)
    plt.xlabel("Thresholds", fontsize=16)
    plt.legend(loc="best", fontsize=14)
    plt.xlim([-2, 4.7])
    plt.ylim([0, 1])
    plt.axvline(x=0.18, linewidth=3, color="#0B3861")
    plt.annotate('Best Precision and \n Recall Balance \n is at 0.18 \n threshold ', xy=(0.18, 0.82), xytext=(55, -40),
             textcoords="offset points",
            arrowprops=dict(facecolor='black', shrink=0.03),
                fontsize=12, 
                color='k')
    
precision_recall_curve(precisions, recalls, threshold)
plt.show()


# In[37]:


#ROC Curve (Receiver Operating Characteristic)
from sklearn.metrics import roc_curve

gb_fpr, gb_tpr, thresold = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
plot_roc_curve(gb_fpr,gb_tpr)


# In[412]:


#calculate roc score
from sklearn.metrics import roc_auc_score

gb_score = roc_auc_score(y_train, y_scores)
print('Gradient Boost Classifier Score:',gb_score)


# In[23]:


#XGBooster
from xgboost import XGBClassifier

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

xgb_score = xgb_clf.score(X_train, y_train)
print("XGBoost Accuracy Score is", xgb_score)


# XGBoost accuracy score (0.86) is higher than Gradient Boosting Classifier(0.84) in training set. 

# In[26]:


#Make Predictions with XGBoost Model

#make predictions for test data

y_pred_xgb = xgb_clf.predict(X_test)
predictions = [round(value) for value in y_pred_xgb]

# evaluate predictions
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[38]:


#plot ROC
y_prob_gb = gb_clf.predict_proba(X_test)
y_score_gb = y_prob_gb[:,1]
fpr_gb,tpr_gb, threshold_gb = roc_curve(y_test, y_score_gb)
auc= accuracy_score(y_test, y_pred_gb)
plt.plot(fpr_gb,tpr_gb,label='Gradient Boosting Classifier,auc  = %0.2f' % auc)

y_prob_xgb = xgb_clf.predict_proba(X_test)
y_score_xgb = y_prob_xgb[:,1]
fpr_xgb,tpr_xgb, threshold_xgb = roc_curve(y_test, y_score_xgb)
auc= accuracy_score(y_test, y_pred_xgb)
plt.plot(fpr_xgb,tpr_xgb,label='XGBoosting Classifier,auc  = %0.2f' % auc)

# ROC curve plotting
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize = 16)
plt.legend(loc="lower right")

plt.show()


# In[61]:


from xgboost import plot_importance
import matplotlib.pyplot as plt

model = XGBClassifier(learning_rate = 0.5, n_estimators=300, max_depth=5)
model.fit(X_train, y_train)

# plot feature importance
plot_importance(model)
plt.show()


# In[86]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
plt.style.use('seaborn-white')

# Convert the columns into categorical variables
df2 = df
df2['job'] = df2['job'].astype('category').cat.codes
df2['marital'] = df2['marital'].astype('category').cat.codes
df2['education'] = df2['education'].astype('category').cat.codes
df2['contact'] = df2['contact'].astype('category').cat.codes
df2['poutcome'] = df2['poutcome'].astype('category').cat.codes
df2['month'] = df2['month'].astype('category').cat.codes
df2['default'] = df2['default'].astype('category').cat.codes
df2['loan'] = df2['loan'].astype('category').cat.codes
df2['housing'] = df2['housing'].astype('category').cat.codes

target_name = 'deposit'
X = df2.drop('deposit', axis=1)

label=df2[target_name]

X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.2, random_state=42, stratify=label)

model = XGBClassifier(learning_rate = 0.5, n_estimators=300, max_depth=5)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = df2.drop('deposit', axis=1).columns
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

def feature_importance_graph(indices, importances, feature_names):
    plt.figure(figsize=(12,6))
    plt.title("Determining Feature importances \n with XGBoost", fontsize=18)
    plt.barh(range(len(indices)), importances[indices], color='#ffd700',  align="center")
    plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal',fontsize=14)
    plt.ylim([-1, len(indices)])

feature_importance_graph(indices, importances, feature_names)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




