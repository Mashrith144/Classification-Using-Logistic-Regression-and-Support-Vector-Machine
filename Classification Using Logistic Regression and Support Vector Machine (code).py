#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Exploratory Data Analysis (EDA)

df = pd.read_csv("C://Users//rmrco//Desktop//diabetes.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.sample(10)


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


# Data Cleaning

df.shape


# In[11]:


df = df.drop_duplicates()


# In[12]:


df.shape


# In[13]:


# Count of null values, checking missing values and display the number of null values

df.isnull().sum()


# In[14]:


df.columns


# In[15]:


print('No. of zero values in Glucose  ',df[df['Glucose']==0].shape[0])


# In[16]:


print('No. of zero values in BloodPressure  ',df[df['BloodPressure']==0].shape[0])


# In[17]:


print('No. of zero values in SkinThickness  ',df[df['SkinThickness']==0].shape[0])


# In[18]:


print('No. of zero values in Insulin  ',df[df['Insulin']==0].shape[0])


# In[19]:


print('No. of zero values in BMI  ',df[df['BMI']==0].shape[0])


# In[20]:


df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
print('No. of zero values in Glucose  ',df[df['Glucose']==0].shape[0])

df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
print('No. of zero values in BloodPressure  ',df[df['BloodPressure']==0].shape[0])

df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
print('No. of zero values in SkinThickness  ',df[df['SkinThickness']==0].shape[0])

df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
print('No. of zero values in Insulin  ',df[df['Insulin']==0].shape[0])

df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
print('No. of zero values in BMI  ',df[df['BMI']==0].shape[0])


# In[21]:


df.describe()


# In[22]:


# Data Visualisation 
# Count Plot

f,ax=plt.subplots(1,2,figsize=(10,5))
df['Outcome'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%',ax = ax[0],shadow = True)
ax[0].set_title('Outcome')
ax[0].set_ylabel(' ')
sns.countplot('Outcome',data = df,ax = ax[1])
ax[1].set_title('Outcome')
N,P = df['Outcome'].value_counts()
print('Negative (0): ',N)
print('Positive (1): ',P)
plt.grid()
plt.show()


# In[23]:


# Histograms

df.hist(bins = 10, figsize = (20,20))
plt.show


# In[24]:


# Scatter plot 

scatter_matrix(df, figsize = (20,20))


# In[25]:


# Pairplot

sns.pairplot(data = df, hue = 'Outcome')
plt.show()


# In[26]:


# Correlation Analysis 
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot = True, cmap = "RdYlGn")


# In[27]:


# Splitting the data frame into X and Y

target_name = 'Outcome'
y = df[target_name]
X = df.drop(target_name, axis = 1)


# In[28]:


X.head()


# In[29]:


y.head()


# In[30]:


# Standard Scaler (Feature Scaling)

scaler = StandardScaler()
scaler.fit(X)
SSX = scaler.transform(X)


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(SSX, y, test_size = 0.2, random_state = 7)


# In[32]:


df.shape


# In[33]:


X_train.shape, y_train.shape


# In[34]:


X_test.shape, y_test.shape


# In[35]:


# Logistic Regression Classifier

lr = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')
lr.fit(X_train, y_train)


# In[36]:


# Support Vector Machine Classifier

sv = SVC()
sv.fit(X_train, y_train)


# In[37]:


# Predictions - LR

lr_pred = lr.predict(X_test)


# In[38]:


X_test.shape


# In[39]:


lr_pred.shape


# In[40]:


# Predictions -SVM

sv_pred = sv.predict(X_test)


# In[41]:


sv_pred.shape


# In[42]:


# Models Evaluation (LR and SVM)
# LR

print("Train Accuracy of Logistic Regression",lr.score(X_train,y_train)*100)
print("Accuracy (Test) score of Logistic Regression", lr.score(X_test, y_test)*100)
print("Accuracy (Test) score of Logistic Regression", accuracy_score(y_test, lr_pred)*100)


# In[43]:


# SVM

print("Train Accuracy of SMM", sv.score(X_train,y_train)*100)
print("Accuracy (Test) score of SVM", sv. score(X_test, y_test)*100)
print("Accuracy score of SVM", accuracy_score(y_test,sv_pred)*100)


# In[44]:


# Logistic Regression - Confusion Matrix, Classification Report, ROC AUC Score,  
# Confusion Matrix 

cm = confusion_matrix(y_test, lr_pred)
cm


# In[45]:


sns.heatmap(confusion_matrix(y_test, lr_pred),annot = True, fmt = "d")


# In[46]:


TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]

TN, FP, FN, TP


# In[47]:


cm = confusion_matrix(y_test, lr_pred)
print('TN - True Negative {}'.format (cm[0,0]))
print('FP - False Positive {}'. format (cm[0,1]))
print('FN - False Negative {}'. format (cm[1,0]))
print('TP - True Positive {}'.format (cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0], cm[1,1]]),np.sum(cm))*100))
print('Misclassification Rate: {}'.format (np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[48]:


# Classification Report

print('Classification Report of Logistic Regression: \n',classification_report(y_test, lr_pred, digits = 4))


# In[49]:


# ROC AUC Score

auc = round(roc_auc_score(y_test, lr_pred)*100,2)
print("ROC AUC SCORE of Logistic Regression is", auc)


# In[50]:


fpr, tpr, thresholds = roc_curve(y_test, lr_pred)
plt.plot(fpr, tpr, color = 'orange', label = 'ROC')
plt.plot([0, 1], [0, 1], color = 'darkblue', linestyle = '--',label ='ROC curve (area - %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of Logistic Regression')
plt.legend()
plt.grid()
plt.show()


# In[51]:


# Support Vector Machine - Confusion Matrix, Classification Report, ROC AUC Scores
# Confusion Matrix

cm = confusion_matrix(y_test, sv_pred)
cm


# In[52]:


sns.heatmap(confusion_matrix(y_test, sv_pred),annot = True, fmt = "d")


# In[53]:


cm = confusion_matrix(y_test, sv_pred)
print('TN - True Negative {}'.format (cm[0,0]))
print('FP - False Positive {}'. format (cm[0,1]))
print('FN - False Negative {}'. format (cm[1,0]))
print('TP - True Positive {}'.format (cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0], cm[1,1]]),np.sum(cm))*100))
print('Misclassification Rate: {}'.format (np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[54]:


# Classification Report 

print('Classification Report of SVM: \n',classification_report(y_test, sv_pred, digits = 4))


# In[55]:


# ROC AUC Scores

auc = round(roc_auc_score(y_test, sv_pred)*100,2)
print("ROC AUC SCORE of SVM is", auc)


# In[56]:


fpr, tpr, thresholds = roc_curve(y_test, sv_pred)
plt.plot(fpr, tpr, color = 'orange', label = 'ROC')
plt.plot([0, 1], [0, 1], color = 'darkblue', linestyle = '--',label ='ROC curve (area - %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of SVM')
plt.legend()
plt.grid()
plt.show()

