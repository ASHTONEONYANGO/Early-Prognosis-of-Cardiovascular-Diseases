#!/usr/bin/env python
# coding: utf-8

#    # Heart Disease Prediction
# The early prognosis of Cardiovascular Diseases can aid in making decisions on lifestyle changes in high risk patients, and in turn go a long way into reducing complications and saving lives.
# 
# This project intends to pinpoint the most relevant/risk factors of heart diseases as well as predict the overall risk using logistic regression and other allied ML technique.

# In[67]:


# importing libraries
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preparation
# ### Source:
# The dataset is publicly available on the kaggle website, and was collected from hospitals from an ongoing study on the residents of the town of Framingham. 
# The intention is to replicate the analysis from this data fo high risk patients in regions across the sub-Sahara.
# 
# - https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset/data
# 
# The classification goal is to predict whether a patient has a 10-year risk of future coronary heart disease (CHD). The dataset provides the patients' information. It includes over 4,000 records and 15 attributes. 

# In[75]:


heart_df = pd.read_csv("C:\Users\Ashe\Python Projects\Cardiovascular Diseases Prognosis\Framingham.csv")
#heart_df = pd.read_csv(r"C:\\Users\\Ashe\\Python Projects\\Cardiovascular Diseases Prognosis\\Framingham.csv")
heart_df.head()


# In[68]:


heart_df = pd.read_csv(r"C:/Users/Ashe/Python Projects/Cardiovascular Diseases Prognosis/Framingham.csv")
# dropping the attribute education - not much sense as a risk factor.
heart_df.drop(['education'],axis=1,inplace=True)
heart_df.head()


# ### Variables:
# 
# Each attribute is a potential risk factor. There are both demographic, behaviourial and medical risk factors.
# 
# - Demographic
# 
#  - sex: male or female;(Nominal)
# 
#  - age: age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# 
# 
# - Behavioural
# 
#  - currentSmoker: whether or not the patient is a current smoker (Nominal)
# 
#  - cigsPerDay: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarrettes, even half a cigarette.)
# 
# 
# - Medical Risk Factors
#    
# - Medical(history)
# 
#  - BPMeds: whether or not the patient was on blood pressure medication (Nominal)
# 
#  - prevalentStroke: whether or not the patient had previously had a stroke (Nominal)
# 
#  - prevalentHyp: whether or not the patient was hypertensive (Nominal)
# 
#  - diabetes: whether or not the patient had diabetes (Nominal)
# 
# 
# - Medical(current):
# 
#  - totChol: total cholesterol level (Continuous)
# 
#  - sysBP: systolic blood pressure (Continuous)
# 
#  - diaBP: diastolic blood pressure (Continuous)
# 
#  - BMI: Body Mass Index (Continuous)
# 
#  - heartRate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
# 
#  - glucose: glucose level (Continuous)
# 
# 
# - Predict variable (desired target):
# 
#  - 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)

# In[17]:


heart_df.rename(columns={'male':'Sex_male'},inplace=True)


# ### Missing values

# In[21]:


heart_df.isnull().sum()


# In[24]:


count = 0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missng values is ', count)
print()
print('since it is only ', round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# In[27]:


heart_df.dropna(axis=0,inplace=True)


# ## Exploratory Analysis

# In[36]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
    fig.tight_layout()
    plt.show()
draw_histograms(heart_df,heart_df.columns,6,3)


# In[38]:


heart_df.TenYearCHD.value_counts()


# In[40]:


sn.countplot(x='TenYearCHD',data=heart_df)


# - There are 3177 patients with no heart diseases and 572 patients with risk of heart disease.

# In[44]:


sn.pairplot(data=heart_df)


# In[47]:


heart_df.describe()


# ## Logistic Regression
# 
# Logistic regression, as a type of regression analysis in statistics, has been emloyed to predict the outcome of categorical dependent variable from a set of predictor or independet variables. As is always the case in logistic regression, the dependent variable is binary (Nominal).
# 
# Logistic regression has been used for prediction as well as calculating the probability of success; ten year risk of coronary heart disease (TenYearCHD).

# In[49]:


from statsmodels.tools import add_constant as add_constant
heart_df_constant = add_constant(heart_df)
heart_df.head()


# In[51]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=heart_df_constant.columns[:-1]
model=sm.Logit(heart_df.TenYearCHD,heart_df_constant[cols])
result=model.fit()
result.summary()


# The Logit regression results above show ssome of the attributes with P value higher than the preferred alpha (5%) and thereby showing low statistically significant relationship with the probability of heart disease. 
# 
# Backward elimination approach is used here to remove those attributes with highest P value one at a time followed by running the regression repeatedly until all attributes have P values less than 0.05.

# ### Feature Selection: Backward elimination (P-value approach)

# In[60]:


def back_feature_elem(data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly
    eliminating feature with the highest P-value above alpha (5%) one at a time
    and returns the regression summary with all P-values below alpha """
    
    while len(col_list)>0:
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)
            
result=back_feature_elem(heart_df_constant, heart_df.TenYearCHD,cols)


# In[62]:


result.summary()


# #### Logisstic regression equation
# 
#           P = e(β0+β1X1) / 1 + e(β0+β1X1)
#  
# When all features plugged in:
# 
# logit(p)=log(p/(1−p))=β0+β1∗Sexmale+β2∗age+β3∗cigsPerDay+β4∗totChol+β5∗sysBP+β6∗glucose

# ### Interpreting results: Odds Ratio, Confidence Intervals and Pvalues

# In[64]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio', 'pvalue']
print((conf))


# #### To Do next
# 
# - splitting data into train and test split
# - model evaluation 
#  - model accuracy
#  - confusion matrix
#  - model evaluation (statistics)
#  
#    - predicted probabilities of 0 (no CHD: No) and 1 (CHD: Yes) for the test data with a default classification threshold of 0.5
#   
# - lowering the threshold
# - ROC curve
# - Area under the curve AUC
# - conclusions

# In[ ]:




