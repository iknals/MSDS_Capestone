#!/usr/bin/env python
# coding: utf-8

# In[481]:


from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[461]:


app = Flask(__name__)
model_kwh = joblib.load('xgboost_model_kwh.pkl')  # Load your pre-trained model 
model_thermal = joblib.load('xgboost_model_therm.pkl') 


# In[181]:


community_mapping = {
    1: 'Albany Park',
    2: 'Archer Heights',
    3: 'Armour Square',
    4: 'Ashburn',
    5: 'Auburn Gresham',
    6: 'Austin',
    7: 'Avalon Park',
    8: 'Avondale',
    9: 'Belmont Cragin',
    10: 'Beverly',
    11: 'Bridgeport',
    12: 'Brighton Park',
    13: 'Burnside',
    14: 'Calumet Heights',
    15: 'Chatham',
    16: 'Chicago Lawn',
    17: 'Clearing',
    18: 'Douglas',
    19: 'Dunning',
    20: 'East Garfield Park',
    21: 'East Side',
    22: 'Edgewater',
    23: 'Edison Park',
    24: 'Englewood',
    25: 'Forest Glen',
    26: 'Fuller Park',
    27: 'Gage Park',
    28: 'Garfield Ridge',
    29: 'Grand Boulevard',
    30: 'Greater Grand Crossing',
    31: 'Hegewisch',
    32: 'Hermosa',
    33: 'Humboldt Park',
    34: 'Hyde Park',
    35: 'Irving Park',
    36: 'Jefferson Park',
    37: 'Kenwood',
    38: 'Lakeview',
    39: 'Lincoln Park',
    40: 'Lincoln Square',
    41: 'Logan Square',
    42: 'Loop',
    43: 'Lower West Side',
    44: 'McKinley Park',
    45: 'Montclare',
    46: 'Morgan Park',
    47: 'Mount Greenwood',
    48: 'Near North Side',
    49: 'Near South Side',
    50: 'Near West Side',
    51: 'New City',
    52: 'North Center',
    53: 'North Lawndale',
    54: 'North Park',
    55: 'Norwood Park',
    56: "O'Hare",
    57: 'Oakland',
    58: 'Portage Park',
    59: 'Pullman',
    60: 'Riverdale',
    61: 'Rogers Park',
    62: 'Roseland',
    63: 'South Chicago',
    64: 'South Deering',
    65: 'South Lawndale',
    66: 'South Shore',
    67: 'Uptown',
    68: 'Washington Heights',
    69: 'Washington Park',
    70: 'West Elsdon',
    71: 'West Englewood',
    72: 'West Garfield Park',
    73: 'West Lawn',
    74: 'West Pullman',
    75: 'West Ridge',
    76: 'West Town',
    77: 'Woodlawn'
}


# In[183]:


kwh_mapping = {'BUILDING TYPE': {'Commercial': 1880.2849761703008,
  'Industrial': 237.61571206982185,
  'Residential': 363.9098267522914},
 'BUILDING_SUBTYPE': {'Commercial': 3004.046996909193,
  'Industrial': 237.61571206982185,
  'Multi 7+': 198.64255391839546,
  'Multi < 7': 245.4300033572473,
  'Municipal': 2163.000065332185,
  'Single Family': 452.5104404869979},
 'COMMUNITY AREA NAME': {'Albany Park': 351.41022766909595,
  'Archer Heights': 613.9566246722954,
  'Armour Square': 643.6542719204457,
  'Ashburn': 826.5836639783819,
  'Auburn Gresham': 432.83070345512834,
  'Austin': 514.2085719535593,
  'Avalon Park': 569.700970477468,
  'Avondale': 385.6926900366552,
  'Belmont Cragin': 496.568455783016,
  'Beverly': 1110.7326818375145,
  'Bridgeport': 738.939782706508,
  'Brighton Park': 369.8115695694734,
  'Burnside': 822.4036111491894,
  'Calumet Heights': 691.2185695026313,
  'Chatham': 466.983336891234,
  'Chicago Lawn': 532.8778568957854,
  'Clearing': 604.139294248808,
  'Douglas': 366.5523959557818,
  'Dunning': 620.3046125327398,
  'East Garfield Park': 803.3500004020442,
  'East Side': 398.09269506658677,
  'Edgewater': 887.0948851362203,
  'Edison Park': 739.4664747168446,
  'Englewood': 314.73429592349424,
  'Forest Glen': 900.7776010305236,
  'Fuller Park': 1035.3238703504408,
  'Gage Park': 519.7107721684657,
  'Garfield Ridge': 1008.7242649282632,
  'Grand Boulevard': 358.09779662487506,
  'Greater Grand Crossing': 308.1756645561831,
  'Hegewisch': 470.8642332959669,
  'Hermosa': 343.27557088906974,
  'Humboldt Park': 363.15320964148265,
  'Hyde Park': 301.5561185501022,
  'Irving Park': 473.7038955722906,
  'Jefferson Park': 776.0428693905407,
  'Kenwood': 305.5567381716549,
  'Lakeview': 527.4855501970222,
  'Lincoln Park': 556.9509364289636,
  'Lincoln Square': 368.40836843933073,
  'Logan Square': 352.2306606375553,
  'Loop': 4563.6877388904095,
  'Lower West Side': 332.91950799859836,
  'McKinley Park': 340.39438296599087,
  'Montclare': 744.4133135301994,
  'Morgan Park': 687.9459191192176,
  'Mount Greenwood': 940.2711127080916,
  'Near North Side': 1043.5321659210567,
  'Near South Side': 3538.0479997610073,
  'Near West Side': 967.1321471856656,
  'New City': 1160.6251177739864,
  'North Center': 475.9632610818988,
  'North Lawndale': 410.6712300031871,
  'North Park': 755.7597470854946,
  'Norwood Park': 739.1459799590606,
  "O'Hare": 318.72019163460516,
  'Oakland': 280.7633451316832,
  'Portage Park': 517.3696819558481,
  'Pullman': 544.3942726002325,
  'Riverdale': 1011.337290817395,
  'Rogers Park': 366.7940936580375,
  'Roseland': 658.9466984736803,
  'South Chicago': 391.3746590342554,
  'South Deering': 454.3738235157358,
  'South Lawndale': 486.1343910357671,
  'South Shore': 274.76729295940584,
  'Uptown': 584.0416876668387,
  'Washington Heights': 581.2642414670269,
  'Washington Park': 234.59342449828597,
  'West Elsdon': 590.5806897019821,
  'West Englewood': 546.9839015157382,
  'West Garfield Park': 338.48446687752397,
  'West Lawn': 685.7803045011353,
  'West Pullman': 532.651389550477,
  'West Ridge': 538.5819603057247,
  'West Town': 559.4338763529975,
  'Woodlawn': 232.80944618935794},
 'Month': {'APRIL': 442.94497004768886,
  'AUGUST': 690.0432986245012,
  'DECEMBER': 678.825753553143,
  'FEBRUARY': 432.66890519943803,
  'JANUARY': 450.16166926587596,
  'JULY': 809.5403914570477,
  'JUNE': 729.9822560558424,
  'MARCH': 440.6758769703953,
  'MAY': 555.2567620376575,
  'NOVEMBER': 604.0164317221936,
  'OCTOBER': 496.1177138833193,
  'SEPTEMBER': 524.6481932946102}}

therm_mapping = {'BUILDING TYPE': {'Commercial': 121.9682509217725,
  'Industrial': 31.79622079387195,
  'Residential': 59.214339130109074},
 'BUILDING_SUBTYPE': {'Commercial': 182.52878596048302,
  'Industrial': 31.79622079387195,
  'Multi 7+': 34.77375187566835,
  'Multi < 7': 46.50150804426434,
  'Municipal': 183.36391355975675,
  'Single Family': 67.67411187998921},
 'COMMUNITY AREA NAME': {'Albany Park': 55.71118403240077,
  'Archer Heights': 73.83837934614228,
  'Armour Square': 131.28107236436455,
  'Ashburn': 99.63272625500271,
  'Auburn Gresham': 75.37135465661329,
  'Austin': 69.69648699296754,
  'Avalon Park': 91.76569644961997,
  'Avondale': 45.933762963853425,
  'Belmont Cragin': 60.88038081275178,
  'Beverly': 128.3854877810497,
  'Bridgeport': 53.232907486712925,
  'Brighton Park': 53.04762860178468,
  'Burnside': 120.58891555019979,
  'Calumet Heights': 109.4678243174597,
  'Chatham': 76.76847292767806,
  'Chicago Lawn': 67.88002760910182,
  'Clearing': 70.62913749176315,
  'Douglas': 56.00559598955545,
  'Dunning': 82.17948941276967,
  'East Garfield Park': 81.69745161155149,
  'East Side': 66.59248520218476,
  'Edgewater': 61.004621063159455,
  'Edison Park': 94.17495316556484,
  'Englewood': 62.28859422073265,
  'Forest Glen': 109.29808990944626,
  'Fuller Park': 64.08137718636554,
  'Gage Park': 68.52341660246907,
  'Garfield Ridge': 89.08014426186058,
  'Grand Boulevard': 48.236945267294224,
  'Greater Grand Crossing': 54.77530328848674,
  'Hegewisch': 64.68692603744805,
  'Hermosa': 52.48454906320213,
  'Humboldt Park': 58.14386526043422,
  'Hyde Park': 57.29178287473828,
  'Irving Park': 56.953896888800976,
  'Jefferson Park': 78.55653126614908,
  'Kenwood': 51.35951786399324,
  'Lakeview': 41.85518392388877,
  'Lincoln Park': 48.27586686888549,
  'Lincoln Square': 47.36471642767394,
  'Logan Square': 44.62412975463093,
  'Loop': 222.18816735793519,
  'Lower West Side': 58.090882141779886,
  'McKinley Park': 50.47863636540414,
  'Montclare': 74.12805102062865,
  'Morgan Park': 95.019399607572,
  'Mount Greenwood': 97.8120318358267,
  'Near North Side': 58.63995175956005,
  'Near South Side': 156.61100214185365,
  'Near West Side': 97.75701918356698,
  'New City': 82.3175681725108,
  'North Center': 50.94998149958223,
  'North Lawndale': 65.68183781046955,
  'North Park': 91.48694291420286,
  'Norwood Park': 86.27434120278947,
  "O'Hare": 51.408572129182105,
  'Oakland': 43.98360563912709,
  'Portage Park': 63.61139379980858,
  'Pullman': 81.42273986954997,
  'Riverdale': 74.22466639296854,
  'Rogers Park': 44.61672064970327,
  'Roseland': 87.5982013858567,
  'South Chicago': 64.65853291463043,
  'South Deering': 69.15991755466068,
  'South Lawndale': 54.13473264380084,
  'South Shore': 56.09444129263113,
  'Uptown': 46.518281431725484,
  'Washington Heights': 96.2224718085185,
  'Washington Park': 48.6756284447776,
  'West Elsdon': 74.77412742170817,
  'West Englewood': 76.51569447137001,
  'West Garfield Park': 63.76757656583714,
  'West Lawn': 86.44427094519426,
  'West Pullman': 79.88043862829876,
  'West Ridge': 68.52980878213997,
  'West Town': 45.481786376432275,
  'Woodlawn': 44.963291939901495},
 'Month': {'APRIL': 63.05848825515348,
  'AUGUST': 16.26118174350093,
  'DECEMBER': 131.25340453457054,
  'FEBRUARY': 144.68200634806772,
  'JANUARY': 168.79152735752822,
  'JULY': 18.05900400080653,
  'JUNE': 22.84275813627468,
  'MARCH': 119.7636807910163,
  'MAY': 38.114160998917676,
  'NOVEMBER': 52.1942531670539,
  'OCTOBER': 24.24073672702877,
  'SEPTEMBER': 16.946604916677085}}


# In[217]:


commute_mapping = {
    1: 'Percent!!COMMUTING TO WORK!!Car, truck, or van -- drove alone',
    2: 'Percent!!COMMUTING TO WORK!!Car, truck, or van -- carpooled',
    3: 'Percent!!COMMUTING TO WORK!!Public transportation (excluding taxicab)',
    4: 'Percent!!COMMUTING TO WORK!!Walked',
    5: 'Percent!!COMMUTING TO WORK!!Worked at home'
}

housing_type_mapping = {
    1: 'Single Family Home',
    2: 'Multi 7+',
    3: 'Multi < 7'
}

heating_mapping = {
    1: 'Percent!!HOUSE HEATING FUEL!!Electricity',
    2: 'Percent!!HOUSE HEATING FUEL!!Utility gas',
    3: 'Percent!!HOUSE HEATING FUEL!!Bottled, tank, or LP gas',
    4: 'Percent!!HOUSE HEATING FUEL!!Fuel oil, kerosene, etc.'
}


# In[441]:


kwh_cols = ['Percent!!SEX AND AGE!!21 years and over',
       'Percent!!HOUSE HEATING FUEL!!Utility gas',
       'Percent!!HOUSE HEATING FUEL!!Bottled, tank, or LP gas',
       'Percent!!HOUSE HEATING FUEL!!Electricity',
       'Percent!!HOUSE HEATING FUEL!!Fuel oil, kerosene, etc.',
       'Percent!!COMMUTING TO WORK!!Car, truck, or van -- drove alone',
       'Percent!!COMMUTING TO WORK!!Car, truck, or van -- carpooled',
       'Percent!!COMMUTING TO WORK!!Public transportation (excluding taxicab)',
       'Percent!!COMMUTING TO WORK!!Walked',
       'Percent!!COMMUTING TO WORK!!Other means',
       'Percent!!COMMUTING TO WORK!!Worked at home',
       'Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)',
       'AVERAGE BUILDING AGE', 'AVERAGE HOUSESIZE', 'AVERAGE STORIES',
       'BUILDING_SUBTYPE', 'COMMUNITY AREA NAME', 'KWH TOTAL SQFT',
       'THERMS TOTAL SQFT', 'Month']

therm_cols = ['Percent!!SEX AND AGE!!21 years and over',
       'Percent!!HOUSE HEATING FUEL!!Utility gas',
       'Percent!!HOUSE HEATING FUEL!!Bottled, tank, or LP gas',
       'Percent!!HOUSE HEATING FUEL!!Electricity',
       'Percent!!COMMUTING TO WORK!!Car, truck, or van -- drove alone',
       'Percent!!COMMUTING TO WORK!!Car, truck, or van -- carpooled',
       'Percent!!COMMUTING TO WORK!!Public transportation (excluding taxicab)',
       'Percent!!COMMUTING TO WORK!!Walked',
       'Percent!!COMMUTING TO WORK!!Other means',
       'Percent!!COMMUTING TO WORK!!Worked at home',
       'Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)',
       'AVERAGE BUILDING AGE', 'AVERAGE HOUSESIZE', 'AVERAGE STORIES',
       'COMMUNITY AREA NAME', 'KWH TOTAL SQFT', 'THERMS TOTAL SQFT', 'Month']


# In[10]:


@app.route('/')
def index():
    # Shared features between KWH and Thermal
    # Shared features between KWH and Thermal
    shared_features = ['Household Size','Adults In Household','Commute Type','Annual Household Income', 'Housing Type', 'Home Square Footage', 'Building Age (Years)', 'Number Of Stories', 'Heating', 'Community']
    # Unique features for KWH and Thermal
    unique_features_kwh = []  # Add any unique features for KWH here
    unique_features_thermal = []  # Add any unique features for Thermal here
    # Combine shared and unique features for KWH and Thermal
    features_kwh = shared_features + unique_features_kwh
    features_thermal = shared_features + unique_features_thermal
    # Remove duplicates and pass to the template
    all_features = list(dict.fromkeys(features_kwh + features_thermal))
    return render_template('index.html', features=all_features)


# In[443]:


#Data For Testing
data = [
    {
        'Household Size': 4,
        'Adults In Household': 2,
        'Commute Type': 1,
        'Annual Household Income': 75000,
        'Housing Type': 1,  # Assume 1 corresponds to 'Single Family Home'
        'Home Square Footage': 1500,
        'Building Age (Years)': 20,
        'Number Of Stories': 2,
        'Heating': 2,  # Assume 1 corresponds to 'Electricity'
        'Community': 1  # Assume 1 corresponds to 'Albany Park'
    }
]
shared_features = ['Household Size','Adults In Household','Commute Type','Annual Household Income', 'Housing Type', 'Home Square Footage', 'Building Age (Years)', 'Number Of Stories', 'Heating', 'Community']
df = pd.DataFrame(data, columns=shared_features)
df['Percent!!SEX AND AGE!!21 years and over'] = df['Adults In Household']/df['Household Size']*100
df['Housing Type'] = df['Housing Type'].map(housing_type_mapping)
df['Community'] = df['Community'].map(community_mapping)


# In[445]:


combined_cols = list(set(kwh_cols + therm_cols))
df2 = pd.DataFrame(columns=combined_cols)


# In[449]:


column_mapping = {
    'Percent!!SEX AND AGE!!21 years and over': 'Percent!!SEX AND AGE!!21 years and over',
    'Annual Household Income': 'Estimate!!INCOME AND BENEFITS (IN 2010 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)',
    'Building Age (Years)': 'AVERAGE BUILDING AGE',
    'Housing Type': 'BUILDING_SUBTYPE',
    'Number Of Stories': 'AVERAGE STORIES',
    'Community': 'COMMUNITY AREA NAME',
    'Home Square Footage':'KWH TOTAL SQFT',
    'Household Size': 'AVERAGE HOUSESIZE'
}
for col_df, col_df2 in column_mapping.items():
    df2[col_df2] = df[col_df]
    
df2['THERMS TOTAL SQFT']= df2['KWH TOTAL SQFT']
commute_type = df['Commute Type'][0]
column_name = commute_mapping.get(commute_type)
df2[column_name] = 100.0
heat_type = df['Heating'][0]
column_name = heating_mapping.get(heat_type)
df2[column_name] = 100.0

columns_to_fill = df2.columns.difference(['Month'])
df2[columns_to_fill] = df2[columns_to_fill].fillna(value=0)
df2 = pd.concat([df2] * 12, ignore_index=True)

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df2['Month'] = months
df2['Month'] = df2['Month'].str.upper()


# In[453]:


df_kwh = df2[kwh_cols]
df_therm = df2[therm_cols]


# In[457]:


cols_to_map = ['BUILDING_SUBTYPE','COMMUNITY AREA NAME','Month']
for col in cols_to_map:
    if col in df_kwh.columns:
        df_kwh[col] = df_kwh[col].map(kwh_mapping[col])
    if col in df_therm.columns:
        df_therm[col] = df_therm[col].map(therm_mapping[col])


# In[463]:


prediction_kwh = model_kwh.predict(df_kwh)


# In[471]:


prediction_therm = model_thermal.predict(df_therm)


# In[477]:


pred_df = pd.DataFrame({
    'Month': months,
    'KWH': prediction_kwh,
    'THERM': prediction_therm
})
pred_df.loc[pred_df['THERM'] < 0, 'THERM'] = 0


# In[513]:


pred_df['Month'] = pd.Categorical(pred_df['Month'], categories=months, ordered=True)

# Sort DataFrame by 'Month' column
pred_df_sorted = pred_df.sort_values(by='Month')

# Plot lines
plt.figure(figsize=(12, 6))
sns.lineplot(data=pred_df_sorted, x='Month', y='KWH', label='KWH')
sns.lineplot(data=pred_df_sorted, x='Month', y='THERM', label='THERM')
plt.xticks(rotation=45)
plt.xlabel('')
plt.ylabel('Units of Energy')
plt.title('Predicted Household Energy Consumption By Month')
plt.legend()

# Remove x-axis ticks and labels
plt.tick_params(axis='x', bottom=False)
plt.xticks([])  # Remove x-axis tick labels

# Create a table with rounded values
table_data = pred_df_sorted[['KWH', 'THERM']].round(2).values.T
row_labels = ['KWH', 'THERM']
col_labels = pred_df_sorted['Month'].unique()
table = plt.table(cellText=table_data,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='bottom')

# Adjust table layout
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.show()


# In[ ]:


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    numeric_data = {}
    for key in data.keys():
        try:
            numeric_data[key] = float(data[key])
        except ValueError:
            return jsonify({'error': f'Invalid input for {key}, please enter numeric values.'})

    df = pd.DataFrame([numeric_data])
    # Predict KWH and Thermal
    prediction_kwh = model_kwh.predict(df_kwh)
    prediction_therm = model_thermal.predict(df_therm)

    # Return predictions as JSON
    return jsonify({'prediction_kwh': prediction_kwh, 'prediction_thermal': prediction_thermal})

if __name__ == '__main__':
    app.run(debug=True)

