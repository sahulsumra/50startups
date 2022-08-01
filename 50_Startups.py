

# Libraries 
import pandas as pd         # For data manipulation  
import numpy as np          # For mathematical calculation
import matplotlib.pyplot as plt    #For visualization
import seaborn as sns       # For advance visualization
from sklearn.metrics import mean_squared_error
import math


#Loading the dataset
df = pd.read_csv(r'C:\Users\ASUS\Desktop\sahul\50_Startups.csv')


#Information
df.info()


# Descriptive statistics 
df.describe()


# Checking for the missing values 
df.isnull().sum()


# Checking for the duplicates if any
df.duplicated().sum()


#Column name 
df.columns 


# Renaming column name 
df = df.rename(columns= {'R&D Spend': 'RD', 'Administration': 'Admin', 'Marketing Spend': 'MS'})

# Checking variance 
df.var() == 0


#Checking for the outlier if any
sns.boxplot(df.RD)

sns.boxplot(df.Admin)          

sns.boxplot(df.MS)         

sns.boxplot(df.Profit)    # Outlier Present




#Removing the outlier 
from feature_engine.outliers import Winsorizer as win

winsor = win(capping_method='iqr',  tail='left', fold=1.5, variables=['Profit'])

df['Profit'] = winsor.fit_transform(df[['Profit']])

sns.boxplot(df.Profit)       #Outlier removed 






############     Exploratory data analysis      ######################


# First moment business decision 
df.mean()

df.median()


# Second moment business decision
df.var()

df.std()


# Third moment business decision
df.skew()


# Fourth moment business decision
df.kurt()




######     Univariate and bivariate analysis       ##########


#ploting histogram 
sns.histplot(data= df, x = df.RD, kde = True)

sns.histplot(data= df, x= df.Admin, kde = True)

sns.histplot(data= df, x= df.MS, kde = True)

sns.histplot(data= df, x= df.Profit, kde = True)


#ploting scatter plot
sns.scatterplot( x= df.State , y= df.Profit)

sns.scatterplot(x= df.MS, y= df.Profit, hue = df.State)


# Jointplot
import seaborn as sns
sns.jointplot(x= df.RD, y= df.Profit)




#Checking for the normal distribution of the data 
# Q-Q Plot
from scipy import stats
import pylab

stats.probplot(df.RD, dist = "norm", plot = pylab)     #for RD

stats.probplot(df.Admin, dist = "norm", plot = pylab)     #for Admin

stats.probplot(df.MS, dist = "norm", plot = pylab)     #for MS

stats.probplot(df.Profit, dist = "norm", plot = pylab)     #for Profit



# Scatter plot between the variables along with histograms
sns.pairplot(df.iloc[:, :])



#Converting States in dummies 
df1 = pd.get_dummies(data= df, columns= ['State'], drop_first=True)


#Column name
df1.columns

#Renaming columns
df1 = df1.rename(columns= { 'State_Florida': 'Florida', 'State_New York': 'NewYork'})



# Ploting the correlation heatmap 
plt.figure(figsize=(10,8))
sns.heatmap(df1.corr(),annot=True)




#Spliting data
from sklearn.model_selection import train_test_split

X= df1.drop(['Profit'], axis=1)
Y = df1['Profit']

x_train, x_test, y_train,  y_test = train_test_split(X, Y, test_size = 0.3) 


# Fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 


# predicting the test set results
y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)


#Finding error test 
mse_test = mean_squared_error(y_test, y_pred)
test_rmse = math.sqrt(mse_test)
print(test_rmse)


#Finding error on train
mse_train = mean_squared_error(y_train, y_pred_train)
train_rmse = math.sqrt(mse_train)
print(train_rmse)

#R2 of test
from sklearn.metrics import r2_score
r2_model_test = r2_score(y_test, y_pred)
r2_model_test


#R2 of train
r2_model_train = r2_score(y_train, y_pred_train)
r2_model_train



###############################################################################

# Create a pickle file using serialization
import pickle
pickle_out = open('startup.pkl', 'wb')
pickle.dump(regressor, pickle_out)
pickle_out.close()












