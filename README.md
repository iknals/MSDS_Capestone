us tract level, which may lack the granularity needed for block-level analysis.
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
   
