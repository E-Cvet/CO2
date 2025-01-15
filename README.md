# CO2_project

### Data is received in the form of an excel file 
https://docs.google.com/spreadsheets/d/1e3UHTczADp20aTlmeOdv1TCw_J5Vt0FnRzHBGLIg4Iw/edit?gid=487576916#gid=487576916

### Data preprocessing needed 
- sorting and creating proper datetime indexes based on available data
- expanding range of data
- data cleaning - invalid dates/nan values/missing values
- initial analysis and plots

### Sources for additional data - https://gml.noaa.gov/ccgg/trends/graph.html
All data is in situ samples from Mauna Loa and/or the suitable coordinates

- Sulfur hexafluoride (SF6) data from hourly in situ samples analyzed on a gas chromatograph located at Mauna Loa (MLO), Hawaii (19.539 N, 155.578 W, elevation: 3397m https://gml.noaa.gov/aftp/data/hats/sf6/insituGCs/CATS/daily/mlo_SF6_Day.dat

- Atmospheric Methane (CH4) Dry Air Mole Fractions from quasi-continuous measurements at Mauna Loa, Hawaii https://gml.noaa.gov/aftp/data/trace_gases/ch4/in-situ/surface/txt/ch4_mlo_surface-insitu_1_ccgg_DailyData.txt

- Nitrous Oxide (N2O) data from hourly in situ samples analyzed on a gas chromatograph located at Mauna Loa (MLO), Hawaii (19.539 N, 155.578 W, elevation: 3397 m) https://gml.noaa.gov/aftp/data/hats/n2o/insituGCs/CATS/daily/mlo_N2O_Day.dat

- Historical meteorological variables (temperature, humidity, wind speed, pressure) https://open-meteo.com/en/docs/historical-weather-api

### Data preparations for models 
- merging based on datetime 
- current, lagged and rolling correlation analysis and tests (Kendall, Pearson, Granger..)
- stationarity, ACF and PACF tests

## Models
### Model 1 - Only original data is used
### Model 2 - Short-Medium term model - predicting 1 month ahead (multiple features)
### Model 3 - Medium-Long term model - predicting up to 5 years ahead (multiple features)
### Model 4 - Long term model - predicting 10+ years ahead (multiple features)
