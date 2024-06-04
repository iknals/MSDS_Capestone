# Predictive Analysis of Urban Energy Consumption

In today's rapidly urbanizing world, efficient energy management is critical for sustainable development and economic growth. Urban areas, which house the majority of the global population, account for over 70% of global energy consumption and significantly contribute to greenhouse gas emissions. As urban populations grow, the demand for energy rises, putting pressure on existing infrastructure and underscoring the need for sustainable energy solutions.

## Project Overview

Our project, "Predictive Analysis of Urban Energy Consumption," aims to address these challenges by leveraging data science techniques to understand energy consumption patterns in urban centers better. By analyzing historical data and developing predictive models, we provide actionable insights to optimize energy use, reduce costs, and mitigate environmental impacts. 

### Objectives

The primary objectives of our project are as follows:

1. **Analyze Historical Data:** Explore historical urban energy consumption data to identify trends and patterns.
2. **Identify Key Factors:** Determine the factors influencing energy use in urban areas.
3. **Develop Predictive Models:** Build accurate models to forecast energy consumption.
4. **User-Friendly Web Application:** Create a user-friendly web application for local Chicago residents to forecast their energy consumption over a 12-month period.

### Datasets

We utilize the following datasets for our analysis:

1. **American Community Survey Data (acs_energy.csv):**
   - Provides extensive demographic, socioeconomic, and housing data.
   - Data available down to census tract level, which may lack the granularity needed for block-level analysis.
   - Misalignment in data granularity and frequency can lead to imprecise modeling of energy demand based on demographic factors.

2. **City of Chicago Energy Usage 2010 (columns2010v3.csv):**
   - Includes building-level data on electricity and natural gas usage across Chicago for 2010.
   - Data is outdated (more than a decade old), which may not reflect current energy consumption patterns or the impact of recent efficiency improvements.
   - Projected trends may not be accurate for current or future decision-making without newer data.

3. **Additional Files:**
   - `final.csv`: The final prediction table output used to generate the line plot showing the forecasted energy consumption over a 12-month period.
   - `Input Fields.xlsx`: Contains the input fields needed for the application inputs.
   - `xgboost_model_therm.pkl` and `xgboost_model_kwh.pkl`: Saved XGBoost model files used for making the respective predictions.
   - `web_app.py`: Run this Flask application to launch the web app.
   - Folders:
     - `templates`: Contains the HTML code for the web application.
     - `static`: Static files for the web app.
       
4. **Contributors:**
   - Sofia Cornali
   - Akira Noda (GitHub username: @Akimon85)
   - Ikna Shillingford (GitHub username: @iknals)
   - Braden Wellman
   
