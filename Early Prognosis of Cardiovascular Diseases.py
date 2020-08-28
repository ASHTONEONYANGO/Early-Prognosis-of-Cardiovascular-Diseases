#!/usr/bin/env python
# coding: utf-8

#  # Heart Disease Prediction
# The early prognosis of Cardiovascular Diseases can aid in making decisions on lifestyle changes in high risk patients, and in turn go a long way into reducing complications and saving lives.
# 
# This project intends to pinpoint and analyse the most relevant/risk factors of heart diseases as well as predict the overall risk using logistic regression and other allied ML technique.

# In[1]:


# importing libraries
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')


# ### Data Preparation
# **Source:**
# The dataset is publicly available on the kaggle website, and was collected from hospitals from an ongoing study on the residents of the town of Framingham. 
# The intention is to replicate the analysis from this data for high risk patients in regions across the sub-Sahara.
# 
# - https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset/data
# 
# The classification goal is to predict whether a patient has a 10-year risk of future coronary heart disease (CHD). The dataset provides the patients' information. It includes over 4,000 records and 15 attributes. 

# In[2]:


heart_df = pd.read_csv('Framingham.csv')
# dropping the attribute education - not much sense as a risk factor. 
# heart_df.drop(['education'],axis=1,inplace=True)
heart_df.head()


# **Renaming of columns**
# - Some columns which are in complex medical terms are renamed for easy understanding by medical and non-medical persons alike.
# eg. 'systolic_Blood_Pressure' is renamed to simply 'Blood_Pressure'

# In[3]:


# renaming of columns for ease of understanding
heart_df.rename(columns={'male':'Gender'},inplace=True)
heart_df.rename(columns={'BPMeds':'Blood_Pressure_Medication'},inplace=True)
heart_df.rename(columns={'prevalentStroke':'stroke'},inplace=True)
heart_df.rename(columns={'prevalentHyp':'hypertension'},inplace=True)
heart_df.rename(columns={'totChol':'cholesterol'},inplace=True)
heart_df.rename(columns={'sysBP':'systolic_Blood_Pressure'},inplace=True)
heart_df.rename(columns={'diaBP':'diastolic_Blood_Pressure'},inplace=True)
#heart_df.rename(columns={'male':'Sex_male'},inplace=True)

heart_df.head()


# ### Variables:
# 
# Each attribute is a potential risk factor. There are both demographic, behaviourial and medical risk factors.
# 
# - Demographic
# 
#  - Gender: male - 1 or female - 0;(Nominal)
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
#  - Blood_Pressure_Medication: whether or not the patient was on blood pressure medication (Nominal)
# 
#  - stroke: whether or not the patient had previously had a stroke (Nominal)
# 
#  - Hypertension: whether or not the patient was hypertensive (Nominal)
# 
#  - diabetes: whether or not the patient had diabetes (Nominal)
# 
# 
# - Medical(current):
# 
#  - cholesterol: total cholesterol level (Continuous)
# 
#  - systolic_Blood_Pressure: systolic blood pressure (Continuous)
# 
#  - diastolic_Blood_Pressure: diastolic blood pressure (Continuous)
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
#  - 10 year risk of coronary heart disease TenYearCHD (binary: “1”, means “Yes”, “0” means “No”)

# In[4]:


heart_df


# ### Missing Values
# - The biggest deal breakers in in data is "missing data".

# In[5]:


count = 0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print()
print('since it is only ', round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# The code above parses the data to determine the total number of null entries row-wise, and their percentage.

# Getting the sum of all null entries for every column in the data .

# In[6]:


# identifying missing values
heart_df.isnull().sum()


# Visualising the missing data.

# In[7]:


# visualization off the missing values
sb.heatmap(heart_df.isnull(),cbar=False)


# **Handling missing data**
# #### Missing Data Imputation
# - Imputing the missing data with mean.
# 
# This replaces all the null entries with the mean of the particular column

# In[8]:


heart_df.education.fillna(heart_df.education.mean(),inplace=True)
heart_df.cigsPerDay.fillna(heart_df.cigsPerDay.mean(),inplace=True)
heart_df.Blood_Pressure_Medication.fillna(heart_df.Blood_Pressure_Medication.mean(),inplace=True)
heart_df.cholesterol.fillna(heart_df.cholesterol.mean(),inplace=True)
heart_df.BMI.fillna(heart_df.BMI.mean(),inplace=True)
heart_df.heartRate.fillna(heart_df.heartRate.mean(),inplace=True)
heart_df.glucose.fillna(heart_df.glucose.mean(),inplace=True)


# In[9]:



heart_df.isnull().sum()


# - The visualization below indicates no presence of null entries in the data, all of them having been replaced with the mean values.
# 
# This is in contrast to the earlier visualisation of the missing data before imputation.

# In[10]:


sb.heatmap(heart_df.isnull(),cbar=False)


# ### Correlation Analysis
# 
#  - Checking the correlations between columns by visualizing the correlation matrix as a heatmap ie. correlation matrix plots

# In[11]:


import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7,7 
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Checking how one set of data corresponds to another set of data (columns) in the dataset. 

# In[12]:


# correlation analysis
heart_df.corr()


# #### Analysing the relationships between the independent variables and target variable. 

# In[13]:


# matrix plot as correlation of all variables
plt.figure(figsize = (20, 8))
sns.heatmap(heart_df.corr(method='spearman',min_periods=1), annot = True)
plt.show()

# there's a larger correlation magnitude towards the regions with stronger colors,aside from the diagonal


# - heatmap showing correlation between variables using different shades of color to indicate the discrepancies.
# - the lighter shades indicate stronger correlation while the darker shades show weaker correlation or no correlation at all. 

# In[14]:


plt.figure(figsize = (20, 8))
sns.heatmap(heart_df.corr(method='spearman'),cmap='coolwarm',annot=True)


# - heatmap showing the correlations between variables using their correlation values.

# In[15]:


def heatmap(x, y, size):
    fig, ax = plt.subplots()

    plt.figure(figsize = (10, 6))
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
data = heart_df
columns = ['Gender', 'age', 'education', 'currentSmoker','cigsPerDay', 'Blood_Pressure_Medication', 'stroke','hypertension','diabetes','cholesterol','systolic_Blood_Pressure','diastolic_Blood_Pressure','BMI','heartRate','glucose','TenYearCHD'] 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# From all the above plots:
# - the lighter regions/color represent more positive and larger magnitude of correlation between the attributes, whereas the darker shades represent more negative and lower magnitude correlation.
# - As such, systolic and diastolic blood pressure are attributed more to the prevalence of stroke and diabetes (they have a higher correlation)
# - Likewise, cholesterol, glucose levels and the BMI are attributed to hypertension, diabetes and stroke.
# - conversely, Gender, age and education have no significant correlation with any other attributes thus are no major medical risk factors for any health condition.
# 
# From these visualisations, it becomes very clear how systolic_Blood_Pressure, diastolic_Blood_Pressure, and hypertension play a big role in determining the occurence of TenYearCHD - coronary heart disease.

# ### **Exploratory Visualisations**
# 
# **Histograms**
# 
# Graphical representations of the distribution of variables in a dataset. 
# 
# They show the underlying frequency or probability distribution of continuous numerical variables.

# In[16]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,40))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkGreen')
    fig.tight_layout()
    plt.show()
draw_histograms(heart_df,heart_df.columns,8,2)


# - In a majority of the contuinuous numerical variables distribution, the histograms are skewed to the right. 
# - There are definitely more females than males in the study conducted in Farmingham.
# 
# cholesterol distribution: the majority of the individuals have cholesterol levels between 200 - 250.
# 
# cigsPerDay: most most of the patients smoke less than 10 cigarettes per day.
# 
# age distribution: the highest number of patients are aged between 40 - 45 years.
# 
# glucose distribution: the most patients have glucose levels approximately 75.
# 
# BMI distribution: uniform skewness - most patients have a BMI of around 25.
# 
# heatRate: a majority of the patients have a heartRate of about 75.
# 
# Blood_Prerssure_Medication distribution: about 90% of all patients are not on blood pressure medication.
# 
# stroke distribution: almost all patients 99% are not prevalent to stroke.

# #### Scatter plot visualisations

# In[17]:


plt.scatter(heart_df.index,heart_df.cholesterol)
plt.xlabel('Index')
plt.ylabel('cholesterol level')
plt.title('Cholesterol scatter plot')


# - The univariate scatter plot shows the observations of different cholesterol levels corresponding to the index/observation number stored in the index of the DataFrame.

# In[18]:


fig = sns.scatterplot(heart_df.index,heart_df.cholesterol,hue=heart_df.Gender)
fig.set(xlabel='index')
fig.set(title='Cholesterol scatter distribution in Gender groupings')


# - seaborn scatter plot showing the observations of varying cholesterol levels corresponding to the index. 
# - The different hues and color mapping help differentiate the cholesterol data values according to the Gender type as the categorical variable.

# In[27]:


fig = sns.scatterplot(heart_df.index,heart_df.glucose,hue=heart_df.TenYearCHD)
fig.set(xlabel='index')
fig.set(title='Glucose scatter distribution in TenYearCHD groupings')


# - from the above scatter plot, there are definitely more females with glucose levels ranging 100 - 150.

# In[28]:


fig = sns.scatterplot(heart_df.index,heart_df.BMI,hue=heart_df.TenYearCHD)
fig.set(xlabel='index')
fig.set(title='BMI scatter distribution in TenYearCHD groupings')


# - seaborn scatter plot showing the observations of varying BMI values corresponding to the index.
# - The different hues and color mapping help differentiate the BMI data values according to the TenYearCHD as the categorical variable.

# #### Swarm plot visualisation

# In[25]:


# age vs TenYearCHD
plt.figure(figsize=(10,10))
sns.swarmplot(x='TenYearCHD', y='age', data=heart_df)


# In[30]:


# BMI vs TenYearCHD
plt.figure(figsize=(10,10))
sns.swarmplot(x='BMI', y='age', data=heart_df,hue='Gender')


# #### Density plot visualisations

# In[31]:


heart_df.head(2)


# In[32]:


sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.kdeplot(heart_df.cholesterol,shade=True)
fig.set(title='Cholesterol level density plot',xlabel='cholesterol',ylabel='density')


# - cholesterol levels have a high density between 200 - 250

# In[33]:


sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.kdeplot(heart_df.glucose,shade=True)
fig.set(title='Glucose density plot',xlabel='glucose',ylabel='density')


# - glucose level is very dense between 50 - 100

# In[34]:


sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.kdeplot(heart_df.BMI,shade=True)
fig.set(title='BMI density plot',xlabel='BMI',ylabel='density')


# - BMI is very dense at an approximate value of 25

# In[35]:


sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.kdeplot(heart_df.heartRate,shade=True)
fig.set(title='heartRate density plot',xlabel='heartRate',ylabel='density')


# - there is high density of heartRate values between 60 - 80

# In[36]:


sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.kdeplot(heart_df.age,shade=True)
fig.set(title='age density plot',xlabel='age',ylabel='density')


# - it is more dense for ages 35 - 60

# #### Seaborn distribution plots

# In[37]:


# seaborn distribution plot
sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.distplot(heart_df.age,bins=10,rug=True)
fig.set(title='age distribution plot')


# - age is more distributed between 40 - 55 years of age

# In[38]:


sns.set(rc=({'figure.figsize':(6,6)}))
fig = sns.distplot(heart_df.cholesterol,kde=False,bins=30,rug=True)
fig.set(title='cholesterol distribution plot')


# - very high distribution of cholesterol levels between 200 - 300

# In[39]:


sns.set(rc=({'figure.figsize':(7,7)}))
fig = sns.distplot(heart_df.systolic_Blood_Pressure,kde=False,bins=20)
fig.set(title='systolic_Blood_Pressure distribution plot')


# - systolic_Blood_Pressure between around 110 - 130 is highly distributed

# In[40]:


sns.distplot(heart_df.BMI,kde=False,bins=10)


# In[41]:


sns.distplot(heart_df.heartRate,kde=False,bins=30,rug=True)


# In[42]:


sns.distplot(heart_df.glucose,kde=False,bins=40)


# In[43]:


fig = sns.barplot(x='diabetes',y='TenYearCHD',data=heart_df)
fig.set(title='Barplot for Diabetes and TenYearCHD')


# In[44]:


sns.barplot(x='stroke',y='TenYearCHD',data=heart_df)


# In[45]:


# male and female having coronary disease or not
sns.countplot(x=heart_df['Gender'], hue=heart_df['TenYearCHD'])


# Here from the above countplot, we see that most data are females
# 
# There are more females having no risk than males having no risk
# 
# There are slightly more males having risk than females having risk

# In[46]:


sns.set(rc=({'figure.figsize':(12,8)}))
fig = sns.boxplot(data=heart_df,palette='coolwarm',orient='h')
fig.set(title='Quartile distribution of patient data')


# In[47]:


sns.set(rc=({'figure.figsize':(8,10)}))
fig = sns.lmplot(x='glucose',y='TenYearCHD',data=heart_df,hue='Gender',palette='coolwarm',markers=['o','v'],scatter_kws={'s':100})
fig.set(title='Glucose against TenYearCHD according to Gender')


# #### Using faceting variables 'glucose' and 'cholesterol' to visualise multiple higher-dimensional relationships between them.

# In[48]:


sns.lmplot(x='cholesterol',y='glucose',data=heart_df,col='TenYearCHD')


# In[49]:


sns.lmplot(x='cholesterol',y='glucose',data=heart_df,col='TenYearCHD',hue='currentSmoker',aspect=1.5)


# - alot of the patients have glucose levels between 50 - 150 and cholesterol levels lower than 400.
# - of those at risk of contracting coronary heart disease, a majority are current smokers.

# ### Multicollinearity Visualisations
# - Using variable inflation factors (VIF) to determine the strength of the correlation between independent variables.

# In[50]:


# detecting multicollinearity using VIF

# importing a library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(x):
    # calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = x.columns
    vif['VIF'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
    
    return(vif)


# The function calc_vif() calculates the VIF of individual variables to determine their correlation.
# - VIF starts at 1 and has no upper limit
# - VIF = 1, no  correlation between the independent variable and the other variable
# - VIF exceeding 5 or 10 indicates high multicollinearity between this independent variable and the others

# In[51]:


x = heart_df.iloc[:,:-1]
calc_vif(x)


# - more than half of the independent predictor variables have moderate multicollinearity.
# - 'age', 'cholesterol', 'BMI', 'heartRate', 'glucose', 'systolic_Blood_Pressure' and ''diastolic_Blood_Pressure' have high VIF values; they can be predicted by other independent variables in the dataset. 
# - Although there is no upper limit for the VIF value, multicollinearity between systolic_Blood_Pressure and other variables, and diastolic_Blood_Pressure and other variables is very high.

# I preferred VIF to correlation matrix and scatter plots, because it can show the correlation of a variable with other a group of other variables, and not just the the bivariate relationship between independent variables.

# In[ ]:





# ### Combining the correlated variables of interest into one and dropping the others.
# - Combining 'systolic_Blood_Pressure' and 'diastolic_Blood_Pressure' into one variable 'Blood_Pressure' to reduce the multicollinearity.

# In[52]:


df2 = heart_df.copy()
n = heart_df
df2['Blood_Pressure'] = heart_df.apply(lambda n: (n.systolic_Blood_Pressure + n.diastolic_Blood_Pressure)*.5,axis=1)
x = df2.drop(['systolic_Blood_Pressure','diastolic_Blood_Pressure','TenYearCHD'],axis=1)
x['TenYearCHD'] = heart_df.TenYearCHD

calc_vif(x)


# The above actions are to reduce the multicollinearity between the independent variables.
# 
# This is necessary because high correlation between predictor variables would obscure the real relationship between the predictor and response variables.

# In[53]:


# new DataFrame
x


# ### **Normalization pre-modelling**
#  - transformation by feature scaling 0 - 1

# In[54]:


# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(x)

# transform data
scaled_df = norm.transform(x)
scaled_df = pd.DataFrame(scaled_df,columns=['Gender','age','education','currentSmoker','cigsPerDay','Blood_Pressure_Medication','stroke','hypertension','diabetes','cholesterol','BMI','heartRate','glucose','Blood_Pressure','TenYearCHD'])

scaled_df


# In[55]:


fig, ax1 = plt.subplots(figsize=(20,5))
ax1.set_title('Before scaling')
sns.kdeplot(x.age,ax=ax1)
sns.kdeplot(x.cigsPerDay,ax=ax1)
sns.kdeplot(x.heartRate,ax=ax1)
sns.kdeplot(x.BMI,ax=ax1)


fig, ax2 = plt.subplots(figsize=(20,5))
ax2.set_title('After scaling')
sns.kdeplot(scaled_df.age,ax=ax2)
sns.kdeplot(scaled_df.cigsPerDay,ax=ax2)
sns.kdeplot(scaled_df.heartRate,ax=ax2)
sns.kdeplot(scaled_df.BMI,ax=ax2)


# In[56]:


fig, ax1 = plt.subplots(figsize=(20,5))
ax1.set_title('Before scaling')
sns.kdeplot(x.cholesterol,ax=ax1)
sns.kdeplot(x.glucose,ax=ax1)
sns.kdeplot(x.Blood_Pressure,ax=ax1)


fig, ax2 = plt.subplots(figsize=(20,5))
ax2.set_title('After scaling')
sns.kdeplot(scaled_df.cholesterol,ax=ax2)
sns.kdeplot(scaled_df.glucose,ax=ax2)
sns.kdeplot(scaled_df.Blood_Pressure,ax=ax2)


# In[57]:



# data standardization with sklearn

'''
from sklearn.preprocessing import StandardScaler

# copy of datasets
x_stand = x.copy()

# numerical features
num_cols = ['Gender','age','education','currentSmoker','cigsPerDay','Blood_Pressure_Medication','stroke','hypertension','diabetes','cholesterol','BMI','heartRate','glucose','TenYearCHD','Blood_Pressure',]

# apply standardization on numerical features
for i in num_cols:
    # fit on training data column
    scale = StandardScaler().fit(x_stand[[i]])
    
    # transform the training data column
    x_stand[i] = scale.transform(x_stand[[i]])

x_stand
'''


# ### Splitting training and testing data for modelling

# In[58]:


import sklearn
from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split

X = scaled_df.iloc[:,:-1].values # feature matrix
# Y = scaled_df.iloc[:14].values
Y = scaled_df.iloc[:,-1].values # dependet target variable

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3,random_state=5)


# In[59]:


X_train.shape


# In[60]:


Y_train.shape


# ###**Modelling using Logistic Regression**
# 
# Logistic regression, as a type of regression analysis in statistics, has been emloyed to predict the outcome of categorical dependent variable from a set of predictor or independet variables. As is always the case in logistic regression, the dependent variable is binary (Nominal).
# 
# Logistic regression has been used for prediction as well as calculating the probability of success; ten year risk of coronary heart disease (TenYearCHD).

# In[61]:


from statsmodels.tools import add_constant as add_constant
df_constant = add_constant(scaled_df)
df_constant.head()


# In[62]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq,scaled_df)
cols = df_constant.columns[:-1]
model = sm.Logit(scaled_df.TenYearCHD,df_constant[cols])
result = model.fit()


# In[63]:


result.summary()


# The results above show some of the attributes with P value higher than the preferred alpha(5%) and thereby showing low statistically significant relationship with the probability of heart disease. 
# 
# Backward elimination approach is used here to remove those attributes with highest Pvalue one at a time follwed by running the regression repeatedly until all attributes have P Values less than 0.05.

# **Feature Selection**
# 
# Backward elimination (P-value approach)

# In[64]:


'''
def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(heart_df_constant,heart_df.TenYearCHD,cols)
'''


# In[65]:


# result.summary()


# #### Interpreting results: 
# Odds Ratio, Confidence Intervals and Pvalues

# In[66]:


p = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = p

pv = round(result.pvalues,3)
conf['pvalue'] = pv
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio', 'pvalue']
print((conf))


#   - This fitted model shows that, holding all other features constant, the odds of getting diagnosed with heart disease for males (Gender = 1)over that of females (Gender = 0) is exp(0.4774) = 1.611866.

# ### Model Evaluation
# 
# **Model accuracy**

# In[67]:


from sklearn.linear_model import LogisticRegression

# creating an instance of a logistic regression model
logreg = LogisticRegression()

# fiiting the model onto the train data
logreg.fit(X_train,Y_train)


# In[68]:


# making predictions of the test data
y_pred = logreg.predict(X_test)
y_pred


# In[69]:


sklearn.metrics.accuracy_score(Y_test,y_pred)


#  - 
# Accuracy of the model is 0.845125786163522.

# **Confusion matrix**

# In[70]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,y_pred)
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='YlGnBu')


# The confusion matrix shows 1063+4 = 1067 correct predictions, and 193+12 = 205 incorrect predictions/Type II errors (False Negatives). 

# In[71]:


TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]

sensitivity = TP/float(TP+FN)
specificity = TN/float(TN+FP)


# **Model Evaluation - Statistics**

# In[72]:


print('The accuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',

'The Misclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n\n',

'Specificity or True Negative Rate = TN/(TN+FP)',TN/float(TN+FP),'\n\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n\n',

'Negative Predictive value = TN/(TN+FN) = ',TN/float(TN+FN),'\n\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n\n',

'Negative Likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity )


# The logistic regression model is highly specific that it is sensitive. The negative values are predicted more accurately that the positives.
# 

# *Predicted probabilities of 0 (No Coronary Heart Disease) and 1 (Coronary Heart Disease: Yes) for the test data with a default classification threshold of 0.5*

# In[75]:


y_pred_prob = logreg.predict_log_proba(X_test)[:,:]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob,columns=['No Heart Disease (0)','Heart Disease (1)'])

y_pred_prob_df.head()


# **Lowering the threshold**
# 
# For the reason that the model is prediicting Heart disease, too many type II errors is not advisable. 
# 
# A False Negative (ignoring the probability of disease when there actually is one) is more danfgerous than a False Positive in this case.
# 
# To that effect, in order to increase the sensitivity, the threshold is lowered. 

# In[76]:


from sklearn.preprocessing import binarize

for i in range(1,5):
  cm2 = 0
  y_pred_prob_yes = logreg.predict_log_proba(X_test)
  
  y_pred2 = binarize(y_pred_prob_yes,i/10)[:,1]
  cm2 = confusion_matrix(Y_test,y_pred2)

  print('With ',i/10,' threshold the Confusion Matrix is ','\n',cm2,'\n',
        'with ',cm2[0,0]+cm2[1,1],' correct predictions and ',cm2[1,0],
        ' Type II errors (False Negatives)','\n\n',
        'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),
        'Specitifity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n' )
  


# ### Model performance
# **ROC Curce**
# 
# The ROC curve helps in deciding the best threshold value. it is generated by plotting TP rate(y-axis) against FP rate(x-axis).
# - It will always end at (1,1) while the threshold point being at 0.
# - It helps in visuslising the performance of a classification model, and showing the model efficiency by by detecting True Positives(recall).
# - Based on the **assumption** of a lower classiffication threshold, the logistic rregression model classifies more items as positive. 
# - The threshold based evaluation metrics (precision recall curve) tells the optimal threshold to select.
# 
# **Area under Curve(AUC)**
# 
# - gives the rate of successful classification bythe  logistic regression  model.
# 
# The higher the erea, the greater the disparity between true and false positives, and the stronger the model is in classifying members of the dataset. 
# 
# Closer to 1 -> better
# 
# **If high threshold**,
# - High specificity
# - Low sensitivity
# 
# **If low threshold**,
# - Low specificity
# - High sensitivity
# 

# In[77]:


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(Y_test,y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[78]:


sklearn.metrics.roc_auc_score(Y_test,y_pred_prob_yes[:,1])


# An area of 0.5 corresponds to a model that performs no better than random classification, and a good classifier stays as far away from that as possible.
# 
# An area of 1 is ideal. The closer the AUC to 1 the better.
# 
# This model has an AUC of 0.7397947287814022, approaching 1. As such, the model is stronger and has a good performance. 
# 

# In[ ]:




