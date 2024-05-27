#!/usr/bin/env python
# coding: utf-8

# Import packages

# In[64]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from boruta import BorutaPy
from scipy.stats import zscore
import re
import glob
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pickle


# In[4]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Load & Join Data Sources

# In[6]:


acs = pd.read_csv('acs5yr2010_cookv2.csv')


# In[7]:


mapping = pd.read_excel('il17trf.xlsx',sheet_name='il17trf')


# In[8]:


energy = pd.read_csv('energy_chicago.csv')


# In[9]:


acs['Geography'] = acs.Geography.str.rsplit('US',expand=True)[1]


# In[10]:


acs = pd.merge(acs, mapping[['GEOID10','AREALANDPT']]
, how='left', left_on= acs['Geography'], right_on= mapping['GEOID10'].astype('str'))


# In[11]:


acs['GEOID10'].isna().sum()


# In[12]:


energy['GEO_ID_TRACT'] = energy['CENSUS BLOCK'].astype('str').str[:11]


# In[13]:


#e_tot = energy.groupby(['COMMUNITY AREA NAME','GEO_ID_TRACT'])[['TOTAL KWH','TOTAL THERMS']].sum()


# In[14]:


kwh_columns = ['KWH JANUARY 2010', 'KWH FEBRUARY 2010',
       'KWH MARCH 2010', 'KWH APRIL 2010', 'KWH MAY 2010', 'KWH JUNE 2010',
       'KWH JULY 2010', 'KWH AUGUST 2010', 'KWH SEPTEMBER 2010',
       'KWH OCTOBER 2010', 'KWH NOVEMBER 2010', 'KWH DECEMBER 2010']
therm_columns = ['THERM JANUARY 2010', 'THERM FEBRUARY 2010', 'THERM MARCH 2010',
       'TERM APRIL 2010', 'THERM MAY 2010', 'THERM JUNE 2010',
       'THERM JULY 2010', 'THERM AUGUST 2010', 'THERM SEPTEMBER 2010',
       'THERM OCTOBER 2010', 'THERM NOVEMBER 2010', 'THERM DECEMBER 2010']
melted_df_kwh = pd.melt(energy, id_vars=energy.columns.difference(kwh_columns), value_vars=kwh_columns, var_name='Month', value_name='KWH')

# Melt the DataFrame for THERM consumption
melted_df_therm = pd.melt(energy, id_vars=energy.columns.difference(therm_columns), value_vars=therm_columns, var_name='Month', value_name='THERM')

# Concatenate the melted DataFrames
energy = pd.concat([melted_df_kwh, melted_df_therm['THERM']], axis=1)

# Reset index
energy.reset_index(drop=True, inplace=True)


# In[15]:


energy['Month'] = energy['Month'].str.split().str[1]


# In[16]:


energy.head()


# In[17]:


energy = energy[(energy['KWH']!=0) & (energy['THERM']!=0)]


# In[18]:


energy['sqft_h_unit_kwh'] = energy['KWH TOTAL SQFT'] / energy['TOTAL UNITS']
energy['sqft_h_unit_therm'] = energy['THERMS TOTAL SQFT'] / energy['TOTAL UNITS']


# In[19]:


drop_cols = ['KWH MEAN 2010',
       'KWH STANDARD DEVIATION 2010', 'KWH MINIMUM 2010',
       'KWH 1ST QUARTILE 2010', 'KWH 2ND QUARTILE 2010',
       'KWH 3RD QUARTILE 2010', 'KWH MAXIMUM 2010',
       'KWH SQFT STANDARD DEVIATION 2010', 'KWH SQFT MINIMUM 2010',
       'KWH SQFT 1ST QUARTILE 2010', 'KWH SQFT 2ND QUARTILE 2010',
       'KWH SQFT 3RD QUARTILE 2010', 'KWH SQFT MAXIMUM 2010',
       'THERM MEAN 2010', 'THERM STANDARD DEVIATION 2010',
       'THERM MINIMUM 2010', 'THERM 1ST QUARTILE 2010',
       'THERM 2ND QUARTILE 2010', 'THERM 3RD QUARTILE 2010',
       'THERM MAXIMUM 2010', 'THERMS SQFT STANDARD DEVIATION 2010',
       'THERMS SQFT MINIMUM 2010', 'THERMS SQFT 1ST QUARTILE 2010',
       'THERMS SQFT 2ND QUARTILE 2010', 'THERMS SQFT 3RD QUARTILE 2010',
       'THERMS SQFT MAXIMUM 2010', 'TOTAL CONSUMPTION',
       'AVERAGE_CONSUMPTION_PER_UNIT','TOTAL UNITS','OCCUPIED UNITS PERCENTAGE',
       'RENTER-OCCUPIED HOUSING UNITS', 'RENTER-OCCUPIED HOUSING PERCENTAGE','AVERAGE_CONSUMPTION_PER_UNIT',
        'ELECTRICITY ACCOUNTS', 'ZERO KWH ACCOUNTS','GAS ACCOUNTS','TOTAL THERMS','KWH SQFT MEAN 2010','THERMS SQFT MEAN 2010', 'TOTAL KWH']


# In[20]:


energy.drop(columns=drop_cols, inplace=True)


# In[21]:


energy.head()


# In[22]:


acs_keep_columns = ['Estimate!!SEX AND AGE!!Median age (years)',
                    'Percent!!SEX AND AGE!!21 years and over',
                    'Estimate!!HOUSEHOLDS BY TYPE!!Average household size',
                    'Percent!!HOUSING OCCUPANCY!!Occupied housing units',
                    'Percent!!HOUSE HEATING FUEL!!Utility gas',
                    'Percent!!HOUSE HEATING FUEL!!Bottled, tank, or LP gas',
                    'Percent!!HOUSE HEATING FUEL!!Electricity',
                    'Percent!!HOUSE HEATING FUEL!!Fuel oil, kerosene, etc.',
                    'Percent!!COMMUTING TO WORK!!Workers 16 years and over',
                    'Percent!!COMMUTING TO WORK!!Car, truck, or van -- drove alone',
                    'Percent!!COMMUTING TO WORK!!Car, truck, or van -- carpooled',
                    'Percent!!COMMUTING TO WORK!!Public transportation (excluding taxicab)',
                    'Percent!!COMMUTING TO WORK!!Walked',
                    'Percent!!COMMUTING TO WORK!!Other means',
                    'Percent!!COMMUTING TO WORK!!Worked at home',
                    'Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)',
                    'Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median earnings for workers (dollars)',
                    'Geography'
                   ]


# In[23]:


acs = acs[acs_keep_columns]


# In[24]:


df = acs.merge(energy, how='right', right_on='GEO_ID_TRACT', left_on='Geography')


# Save/Load Intermediate Data

# In[26]:


df.to_csv('acs_energy2.csv')


# In[27]:


#df = pd.read_csv('acs_energy2.csv')


# Clean Data

# Create Features

# In[30]:


df['kwh_per_unit'] = df['KWH']/df['OCCUPIED UNITS']
df['therm_per_unit'] = df['THERM']/df['OCCUPIED UNITS']


# In[ ]:





# In[31]:


df.shape


# Remove Outliers

# In[33]:


def remove_outliers(df_subset):
    z_scores_kwh_sqft = zscore(df_subset['kwh_per_unit'].dropna())
    z_scores_therm_sqft = zscore(df_subset['therm_per_unit'].dropna())
    filtered_df = df_subset[(np.abs(z_scores_kwh_sqft) < 3) &
                            (np.abs(z_scores_therm_sqft) < 3)]
    return filtered_df

# Initialize an empty DataFrame for the final concatenation
df_filtered = pd.DataFrame()

# Process each group individually to avoid excessive memory usage
grouped = df.groupby(['Month', 'BUILDING TYPE', 'BUILDING_SUBTYPE'])
for name, group in grouped:
    print(f"Processing group: {name}")
    filtered_group = remove_outliers(group)
    df_filtered = pd.concat([df_filtered, filtered_group], ignore_index=True)

print("All groups processed and concatenated successfully.")

# Verify the shape of the final DataFrame
print(df_filtered.shape)


# In[34]:


df_filtered.columns


# In[35]:


df_filtered.drop(columns='OCCUPIED HOUSING UNITS', inplace=True)


# In[36]:


drop_cols = ['TERM APRIL 2010',
       'THERM AUGUST 2010', 'THERM DECEMBER 2010', 'THERM FEBRUARY 2010',
       'THERM JANUARY 2010', 'THERM JULY 2010', 'THERM JUNE 2010',
       'THERM MARCH 2010', 'THERM MAY 2010', 'THERM NOVEMBER 2010',
       'THERM OCTOBER 2010', 'THERM SEPTEMBER 2010','GEO_ID_TRACT']


# In[37]:


df_filtered.drop(columns=drop_cols, inplace=True)


# In[38]:


id_cols = [
    'Geography',                                                                                                                                                                                                                              
    'CENSUS BLOCK', 
    'COMMUNITY AREA NAME'
    ]


# In[39]:


columns_to_convert = ['Geography','CENSUS BLOCK','COMMUNITY AREA NAME']
df_filtered[columns_to_convert] = df_filtered[columns_to_convert].astype(str)


# In[40]:


id_df = df_filtered[id_cols]


# In[76]:


df_filtered.columns


# In[92]:


X = df_filtered.drop(columns=['KWH', 'THERM', 'Geography','CENSUS BLOCK','Estimate!!HOUSEHOLDS BY TYPE!!Average household size',
                              'Percent!!HOUSING OCCUPANCY!!Occupied housing units',
                              'Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median earnings for workers (dollars)',
                              'THERMS TOTAL SQFT','KWH TOTAL SQFT','Estimate!!SEX AND AGE!!Median age (years)',
                              'Percent!!COMMUTING TO WORK!!Workers 16 years and over','sqft_h_unit_kwh', 'sqft_h_unit_therm',
                             'kwh_per_unit','therm_per_unit'])
y_kwh = df_filtered['kwh_per_unit']
y_therm = df_filtered['therm_per_unit']

# Split the data into training and testing sets
X_train, X_test, y_kwh_train, y_kwh_test = train_test_split(X, y_kwh, test_size=0.2, random_state=666)
X_train, X_test, y_therm_train, y_therm_test = train_test_split(X, y_therm, test_size=0.2, random_state=666)


# In[94]:


categorical_columns = X_train.select_dtypes(include=['category', 'object']).columns.tolist()


# In[96]:


# Initialize Target Encoders
target_encoder_kwh = TargetEncoder(target_type='continuous',random_state=666)
target_encoder_therm = TargetEncoder(target_type='continuous', random_state=666)

# Fit and transform the categorical columns for KWH
X_train_encoded_kwh = X_train.copy()
X_train_encoded_kwh[categorical_columns] = target_encoder_kwh.fit_transform(
    X_train[categorical_columns], y_kwh_train
)

# Fit and transform the categorical columns for THERM
X_train_encoded_therm = X_train.copy()
X_train_encoded_therm[categorical_columns] = target_encoder_therm.fit_transform(
    X_train[categorical_columns], y_therm_train
)

# Apply the same transformation to the test sets
X_test_encoded_kwh = X_test.copy()
X_test_encoded_kwh[categorical_columns] = target_encoder_kwh.transform(
    X_test[categorical_columns]
)

X_test_encoded_therm = X_test.copy()
X_test_encoded_therm[categorical_columns] = target_encoder_therm.transform(
    X_test[categorical_columns]
)


# ## Feature Selection (Boruta)

# In[98]:


# Initialize XGBoost regressor
xgb_kwh = XGBRegressor(tree_method = "hist", device = "cuda")
xgb_therm = XGBRegressor(tree_method = "hist", device = "cuda")
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Initialize Boruta
boruta_kwh = BorutaPy(estimator=xgb_kwh, n_estimators='auto', verbose=2, random_state=666, perc=90)
boruta_therm = BorutaPy(estimator=xgb_therm, n_estimators='auto', verbose=2, random_state=666, perc=90)

# Fit Boruta for KWH
boruta_kwh.fit(X_train_encoded_kwh.values, y_kwh_train.values)
feature_names = X_train_encoded_kwh.columns
feature_ranks = list(zip(feature_names, boruta_kwh.ranking_, boruta_kwh.support_, boruta_kwh.support_weak_ )) 
boruta_kwh_results = pd.DataFrame(feature_ranks, columns=['feature_name','rank', 'confirmed','tentative'])
boruta_kwh_results.to_csv('boruta_kwh_results.csv')


# Fit Boruta for THERM
boruta_therm.fit(X_train_encoded_therm.values, y_therm_train.values)
feature_names = X_train_encoded_therm.columns
feature_ranks = list(zip(feature_names, boruta_therm.ranking_, boruta_therm.support_, boruta_therm.support_weak_ )) 
boruta_therm_results = pd.DataFrame(feature_ranks, columns=['feature_name','rank', 'confirmed','tentative'])
boruta_therm_results.to_csv('boruta_therm_results.csv')



# In[100]:


# Transform the training sets to select the important features
X_train_selected_kwh = X_train_encoded_kwh.iloc[:, boruta_kwh.support_]
X_train_selected_therm = X_train_encoded_therm.iloc[:, boruta_therm.support_]

# Transform the test sets to select the important features
X_test_selected_kwh = X_test_encoded_kwh.iloc[:, boruta_kwh.support_]
X_test_selected_therm = X_test_encoded_therm.iloc[:, boruta_therm.support_]


# In[101]:


X_train_selected_therm.columns


# In[107]:


X_train_selected_kwh.to_csv('X_train_selected_kwh.csv')
X_train_selected_therm.to_csv('X_train_selected_therm.csv')


# In[109]:


xgb_gpu_model = XGBRegressor(tree_method = "hist", device = "cuda")

# Hyperparameter grid for XGBoost
param_grid = {

    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0] 
}

# Initialize GridSearchCV for KWH
grid_search_kwh = GridSearchCV(estimator=xgb_gpu_model, param_grid=param_grid, 
                               cv=5, scoring='neg_mean_squared_error',
                               verbose= 2, n_jobs=1)
grid_search_kwh.fit(X_train_selected_kwh, y_kwh_train)

# Initialize GridSearchCV for THERM
grid_search_therm = GridSearchCV(estimator=xgb_gpu_model, param_grid=param_grid, 
                                 cv=5, scoring='neg_mean_squared_error', 
                                 verbose=2, n_jobs=1)
grid_search_therm.fit(X_train_selected_therm, y_therm_train)

# Best models
best_model_kwh = grid_search_kwh.best_estimator_
best_model_therm = grid_search_therm.best_estimator_

# Evaluate the models on the test sets
y_kwh_pred = best_model_kwh.predict(X_test_selected_kwh)
y_therm_pred = best_model_therm.predict(X_test_selected_therm)

print("KWH Model Test Score:", best_model_kwh.score(X_test_selected_kwh, y_kwh_test))
print("THERM Model Test Score:", best_model_therm.score(X_test_selected_therm, y_therm_test))


# In[111]:


# Save the best model for KWH
filename_kwh = 'xgboost_model_kwh.pkl'
with open(filename_kwh, 'wb') as file:
    pickle.dump(best_model_kwh, file)

# Save the best model for THERM
filename_therm = 'xgboost_model_therm.pkl'
with open(filename_therm, 'wb') as file:
    pickle.dump(best_model_therm, file)


# In[113]:


kwh_cv_results_df = pd.DataFrame(grid_search_kwh.cv_results_)


# In[115]:


therm_cv_results_df = pd.DataFrame(grid_search_therm.cv_results_)


# In[61]:


print(kwh_cv_results_df.head())


# In[ ]:




