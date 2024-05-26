#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from census import Census
import us
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


# API endpoint for testing connection
url = "https://api.census.gov/data/2020/acs/acs5/profile"

# Make a GET request to the API endpoint
response = requests.get(url)

# Check the response status code
if response.status_code == 200:
    print("Connection successful! API is accessible.")
else:
    print(f"Connection failed. Status code: {response.status_code}")


# In[3]:


c = Census("c38bb496a1b22bfbd25f27e39804ccc3f9b473de",year=2010)
groups = ['DP05','DP02','DP04','DP03']
cook_data =[]
for group in groups:

  c_data = c.acs5dp.state_county_tract(
    ('NAME', f'group({group})'), us.states.IL.fips, '031', c.ALL, timeout=30
  )
  cook_data.append(c_data)
flat_data = {}
for data in cook_data:
  for tract_data in data:
    tract_identifer = tract_data['NAME']
    if tract_identifer not in flat_data:
      flat_data[tract_identifer] = tract_data
    else:
      flat_data[tract_identifer].update(tract_data)

flat_data_list = list(flat_data.values())
cook_df = pd.DataFrame(flat_data_list)


# In[4]:


#cook_df.to_csv('acs5yr2010.csv')
cook_df = pd.read_csv('acs5yr2010.csv')


# In[7]:


urls = [
    "https://api.census.gov/data/2010/acs/acs5/profile/groups/DP02/",
    "https://api.census.gov/data/2010/acs/acs5/profile/groups/DP03/",
    "https://api.census.gov/data/2010/acs/acs5/profile/groups/DP04/",
    "https://api.census.gov/data/2010/acs/acs5/profile/groups/DP05/"
]
variable_labels = {}

for url in urls:

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        variables = data['variables']
        variable_labels.update({variable: info['label'] for variable, info in variables.items()})
    else:
        print("Failed to retrieve data for URL:", url)


# In[8]:


cook_df = cook_df.rename(columns=variable_labels)


# In[61]:


#column_names = cook_df_clean.columns.tolist()
#df_column_names = pd.DataFrame(column_names, columns=['Column Names'])
#df_column_names.to_csv('columns2010.csv')


# In[14]:


column_names = pd.read_csv('columns2010v3.csv')
keep = column_names[column_names['include'].isin([1,2])]


# In[16]:


cook_df_clean = cook_df[keep.iloc[:,2]]
missing = cook_df_clean.isna().sum()
missing[missing != 0].count()
cook_df_clean.head(10).describe().T


# In[17]:


#Some variables appear more than once in the dataset because it is combined from multiple queries pulling preset groups of variables using the census.gov API which appears to have overlap.

# 1. Identify duplicate column names
duplicate_columns = cook_df_clean.columns[cook_df_clean.columns.duplicated()]

# 2. Decide which duplicate columns to keep
# For example, if you want to keep the first occurrence of each duplicate column:
columns_to_keep = ~cook_df_clean.columns.duplicated(keep='first')

# 3. Drop the duplicate columns you don't want to keep
cook_df_clean = cook_df_clean.loc[:, columns_to_keep]


# In[18]:


numeric_df = cook_df_clean.select_dtypes(include=['number'])

# Filter DataFrame for columns containing at least one negative value
negative_columns = numeric_df.lt(0).sum()

# Filter out columns with 0 negative values
negative_columns = negative_columns[negative_columns > 0]

# Print the columns with at least one negative value and the number of negative values in those columns
print("Columns with at least one negative value and the number of negative values:")
print(negative_columns)


# In[22]:


#In this dataset, whenever a data point is unavailable or could not be estimated, they are coded as a negative value. In order to prevent issues with calculations, they will be replaced with NaNs here.
numeric_df = numeric_df.mask(numeric_df < 0, np.nan)
cook_df_clean[numeric_df.columns] = numeric_df


# In[25]:


null_columns = cook_df_clean.columns[cook_df_clean.isnull().all()]


# In[27]:


cook_df_clean.drop(columns=null_columns, inplace=True)


# In[29]:


cook_df_clean.to_csv('acs5yr2010_cookv2.csv')


# In[23]:


missing = cook_df_clean.isna().sum()
missing[missing != 0].count()


# In[124]:


sns.histplot(cook_df_clean['Estimate!!Total housing units'])


# In[125]:


sns.histplot(cook_df_clean['Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)'])


# In[196]:


sns.scatterplot(cook_df_clean, x='Estimate!!Total housing units',y='Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)') 


# In[126]:


dfc = cook_df_clean
substring = 'HEATING'
mask = dfc.columns.str.contains(substring)
subset_dfc = dfc.loc[:, mask]
sums = subset_dfc.sum()
sums_df = pd.DataFrame({'Column': sums.index, 'Sum': sums.values})
# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(data=sums_df, x='Sum', y='Column')
plt.xlabel('# of Houses')
plt.title('Houses By Energy Used For Heating')
plt.show()


# In[127]:


substring = 'SEX AND AGE'
substring2 = 'over'
substring3 = '85 years and over'
mask = dfc.columns.str.contains(substring) & dfc.columns.str.contains(substring2) & ~dfc.columns.str.contains(substring3)
columns_to_keep = ~mask
dfc = dfc.loc[:, columns_to_keep]
dfc.to_csv('acs5yr2010_cook.csv')


# In[128]:


dfc.columns = dfc.columns.str.replace('Estimate!!', '')


# In[129]:


substring = 'SEX AND AGE'
substring2 = ' years'
mask = dfc.columns.str.contains(substring) & dfc.columns.str.contains(substring2)
subset_dfc = dfc.loc[:, mask]
sums = subset_dfc.sum()
sums_df = pd.DataFrame({'Column': sums.index, 'Sum': sums.values})
# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(data=sums_df, x='Sum', y='Column')
plt.xlabel('Population')
plt.title('Population by Age Group')
plt.show()


# In[172]:


import matplotlib.ticker as ticker

substring = 'INCOME'
substring2 = 'to'
substring3 = 'Less'
substring4 = 'more'
# Use & for an AND condition to ensure both substrings are present in column names
mask = dfc.columns.str.contains(substring) & (dfc.columns.str.contains(substring2) | dfc.columns.str.contains(substring3) | dfc.columns.str.contains(substring4))
subset_dfc = dfc.loc[:, mask]
sums = subset_dfc.sum()
sums_df = pd.DataFrame({'Column': sums.index, 'Sum': sums.values})


# In[173]:


pd.set_option('display.max_colwidth', None)
print(sums_df.iloc[:,0])


# In[174]:


order = ['INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$10,000 to $14,999',
            'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$15,000 to $24,999',
            'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$25,000 to $34,999',
            'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$35,000 to $49,999',
            'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$50,000 to $74,999',
            'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$75,000 to $99,999',
            'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$100,000 to $149,999',
           'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$150,000 to $199,999',
           'INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!$200,000 or more'
          ]


# In[176]:


sums_df_sorted = sums_df.set_index('Column').reindex(order).reset_index()

# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
ax = sns.barplot(data=sums_df_sorted, x='Sum', y='Column')
ax.set_xlabel('Income Group')
plt.title('Population by Income Group')
plt.show()


# In[144]:


#Generate Summary Stats for Groups of Related Variables and Export to Excel

column_names = dfc.columns.to_list()
var_group = [name.split('!!', 1)[0] for name in column_names]
var_group = list(set(var_group))
with pd.ExcelWriter('summary_stats.xlsx') as writer:
    for prefix in var_group:
        # Subset the DataFrame for columns with the current prefix
        subset_df = dfc[[col for col in dfc.columns if col.startswith(prefix)]]
        numeric_subset_df = subset_df.select_dtypes(include='number')
        # Generate summary statistics using describe()
        if not numeric_subset_df.empty:
            # Generate summary statistics for the numeric subset DataFrame using describe()
            summary_stats = numeric_subset_df.describe()
            
            # Export the summary statistics to a separate Excel worksheet
            summary_stats.to_excel(writer, sheet_name=prefix)
        else:
            print(f"No numeric columns found for prefix '{prefix}'. Skipping export.")


# In[180]:


substring = 'Built'
mask = dfc.columns.str.contains(substring)
subset_dfc = dfc.loc[:, mask]
sums = subset_dfc.sum()
sums_df = pd.DataFrame({'Column': sums.index, 'Sum': sums.values})
# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(data=sums_df, x='Sum', y='Column')
plt.xlabel('# Of Homes')
plt.title('# Of Homes By Year Built')
plt.show()


# In[186]:


substring = 'UNITS'
sub2 = 'Total'
mask = dfc.columns.str.contains(substring) & ~dfc.columns.str.contains(sub2)
subset_dfc = dfc.loc[:, mask]
sums = subset_dfc.sum()
sums_df = pd.DataFrame({'Column': sums.index, 'Sum': sums.values})
# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(data=sums_df, x='Sum', y='Column')
plt.xlabel('Residential Buildings')
plt.title('Residential Buildings By # of Units')
plt.show()


# In[189]:


substring = 'ROOMS'
sub2 = 'Median'
mask = dfc.columns.str.startswith(substring) & ~dfc.columns.str.contains(sub2)
subset_dfc = dfc.loc[:, mask]
sums = subset_dfc.sum()
sums_df = pd.DataFrame({'Column': sums.index, 'Sum': sums.values})
# Create the bar plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(data=sums_df, x='Sum', y='Column')
plt.xlabel('Residential Buildings')
plt.title('Residential Buildings By # of Units')
plt.show()


# In[190]:


sns.scatterplot(dfc,x='HOUSEHOLDS BY TYPE!!Average household size',
y='INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)')


# In[195]:


dfc_cols = pd.Series(dfc.columns.to_list())
dfc_cols.to_csv('dfc_cols.csv')


# In[ ]:




