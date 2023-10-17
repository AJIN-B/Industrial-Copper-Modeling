# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


#%%

df = pd.read_excel('Copper_Set.xlsx')


df['thickness'].fillna(value=df['thickness'].mean(), inplace=True)
df['material_ref'].fillna('unknown', inplace=True)

# dropping the material reference column 
#df.drop(columns=['material_ref'],inplace=True)

# deleting the remaining null values as they are less than 1% of data which can be neglected
df = df.dropna()

#%%

# dealing with data in wrong format 
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce', downcast='signed')
df['country'] = pd.to_numeric(df['country'], errors='coerce', downcast='signed')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce', downcast='signed')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0') 

#%% Regression

Rdf = df.copy() 

a = Rdf['selling_price'] <= 0
print(a.sum())
Rdf.loc[a, 'selling_price'] = np.nan

a = Rdf['quantity tons'] <= 0
print(a.sum())
Rdf.loc[a, 'quantity tons'] = np.nan

a = Rdf['thickness'] < 0
print(a.sum())

# deleting the remaining null values as they are less than 1% of data which can be neglected
Rdf = Rdf.dropna()

Rdf['selling_price'] = np.log(Rdf['selling_price'])
Rdf['quantity tons'] = np.log(Rdf['quantity tons'])
Rdf['thickness'] = np.log(Rdf['thickness'])

#use ordinal encoder to convert categorical data into numerical data.
from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
Rdf[['status','item type']] = OE.fit_transform(Rdf[['status','item type']])

#split data into X, y
X = Rdf[['quantity tons','status','item type','application','thickness','width','country','customer','product_ref']]
y = Rdf['selling_price']


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform(X_test)



#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


tree = DecisionTreeRegressor()
#tree.fit(X_train, Y_train)

# hyperparameters
param_grid = {'max_depth': [10, 20, 30],
              'min_samples_split': [10,15,20],
              'min_samples_leaf': [4,6,8,10],
              'max_features': ['auto', 'sqrt', 'log2']}
# gridsearchcv
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


print("Training accuracy of the model is {:.2f}".format(best_model.score(X_train, Y_train)))
print("Testing accuracy of the model is {:.2f}".format(best_model.score(X_test, Y_test)))


training_data_predction = best_model.predict(X_train)
r_train_score1 = metrics.r2_score(Y_train, training_data_predction)
print("R squared score of Training Set : ", r_train_score1)

test_data_predction = best_model.predict(X_test)
r2_test_dt = metrics.r2_score(Y_test, test_data_predction)
print("R squared score of Test Set : ", r2_test_dt)

mae_test_dt = metrics.mean_absolute_error(Y_test, test_data_predction)
print("Mean Absolute Error score of Test Set : ", mae_test_dt)

mse_test_dt = metrics.mean_squared_error(Y_test, test_data_predction)
print("Mean Squared Error score of Test Set : ", mse_test_dt)


#%%

# Removing the status other than won and lost
dfc = df[df['status'].isin(('Won','Lost')) ]
dfc.status.replace('Won', 1 ,inplace= True)
dfc.status.replace('Lost', 0 ,inplace= True)

#use ordinal encoder to convert categorical data into numerical data.
from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
dfc['item type'] = OE.fit_transform(dfc[['item type']])


#%%

columns = ['quantity tons','thickness','width'] 


plt.figure()
sns.boxplot(data=dfc[columns], orient='h')  
plt.xlabel('Values')

#%% 
# Treat Outliers using IQR or Isolation Forest from sklearn library
# Find InterQuantile range 
up_low_col = dict()
for i in columns:
    q1 = dfc[i].quantile(0.25)
    q3 = dfc[i].quantile(0.75)
    iqr = q3-q1
    upperL = q3+1.5*iqr
    lowerL = q1-1.5*iqr
    dfc[i] = np.where(dfc[i]>upperL,upperL,np.where(dfc[i]<lowerL,lowerL,dfc[i]))
    up_low_col.update({i:[upperL,lowerL]})


#%%

plt.figure()
sns.boxplot(data=dfc[columns], orient='h')  
plt.xlabel('Values')

#%%

#split data into X, y
dfc = dfc.dropna()
X = dfc[['quantity tons','item type','application','thickness','width','country','selling_price','customer','product_ref']] 
Y = dfc[['status']]


# splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test ,y_train ,y_test = train_test_split(X,Y,test_size=0.2,random_state=100)
print("Train shape ",X_train.shape)
print("Test shape ",X_test.shape)

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler()
X_train = X_std.fit_transform(X_train)
X_test = X_std.transform(X_test)


#%%

#import classifier algorithms.
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score, f1_score
from sklearn.metrics import plot_confusion_matrix,classification_report, roc_curve, auc


dtc = DecisionTreeClassifier(max_depth = 5, random_state = 1)
dtc.fit(X_train, y_train)
train_score = dtc.score(X_train, y_train)
test_score = dtc.score(X_test, y_test)
print("Training Score : ",train_score)
print("Testing Score : ",test_score)

print(classification_report(y_test, dtc.predict(X_test)))


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
print("Training Score : ",knn.score(X_train, y_train))
print("Testing Score : ",knn.score(X_test, y_test))
print(classification_report(y_test, knn.predict(X_test)))


gbc = GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,random_state = 28)
gbc.fit(X_train, y_train)
print("Training Score : ",gbc.score(X_train, y_train))
print("Testing Score : ",gbc.score(X_test, y_test))
print(classification_report(y_test, gbc.predict(X_test)))


rfc = RandomForestClassifier(n_estimators = 20, max_depth =6,random_state = 35)
rfc.fit(X_train, y_train)
print("Training Score : ",rfc.score(X_train, y_train))
print("Testing Score : ",rfc.score(X_test, y_test))
print(classification_report(y_test, rfc.predict(X_test)))



LR = LogisticRegression()
LR.fit(X_train,y_train)
print("Training Score : ",LR.score(X_train, y_train))
print("Testing Score : ",LR.score(X_test, y_test))
print(classification_report(y_test, LR.predict(X_test)))


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)
print("Training Score : ",clf.score(X_train, y_train))
print("Testing Score : ",clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))



