# preprocessing
# The quantity of CO2 is determined and described by the chemical term “mole fraction”, defined as the number of carbon dioxide molecules in a given number of molecules of air, after removal of water vapor. For example, 413 parts per million of CO2 (abbreviated as ppm) means that in every million molecules of (dry) air there are on average 413 CO2 molecules.

# %% [markdown]
# # Importing libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from helpers import coerce_into_full_datetime, add_missing_one_year_rows, plot_column, add_missing_dates, plot_rolling_correlations 
from helpers import interpret_p_value, plot_lagged_correlations
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# %% [markdown]
# # Importing, checks and formatting of station data

# %%
df_station = pd.read_csv('data_project - Sheet1.csv')

# %%
df_station

# %%
invalid_10years = (df_station['10 years ago'] == -999.99).sum()
invalid_10years

# %% [markdown]
# ## Adjusting station data

# %% [markdown]
# **Some of the years are in an invalid format, so i will be using the information from the other columns**

# %% [markdown]
# i am using the date from the 'decimal' column along with the 'day' and 'month' columns in order to adjust the datetime index accordingly

# %%
df_station['year'] = df_station['year'].astype(str).str.strip()
mask = df_station['year'] == 'year'
df_station.loc[mask, 'year'] = df_station.loc[mask, 'decimal'].fillna(0).apply(lambda x: int(float(x)))

# %%
df_station.dtypes

# %% [markdown]
# ensuring that all columns are in a proper datetime format, including a datetime index

# %%
df_station.drop(columns = ['decimal'], inplace = True)
df_station = coerce_into_full_datetime(df_station)
df_station

# %%
invalid_average = (df_station['average'] == -999.99).sum()
invalid_1year = (df_station['1 year ago'] == -999.99).sum()
invalid_10years = (df_station['10 years ago'] == -999.99).sum()

print(invalid_average)
print(invalid_1year)
print(invalid_10years)

# %% [markdown]
# **using the data from the '1 year ago' and '10 years ago' to include and adjust missing datetimes, thus giving us a richer dataframe**

# %% [markdown]
# i replaced all invalid data with the values from '10 years ago'

# %%
df_station = add_missing_dates(df_station) # using the function to create new rows using the '10 year ago' column

# %%
df_station = add_missing_one_year_rows(df_station) # using the function to create new rows using the '1 year ago' column

# %%
df_station.drop(df_station[df_station['average'] == -999.99].index, inplace=True)

# %%
df_station

# %%
invalid_average = (df_station['average'] == -999.99).sum()
print(invalid_average)

# %% [markdown]
# # Importing, preprocessing and checking importance of - open meteo weather data

# %%
df_history = pd.read_csv(r'open-meteo-19.44N155.62E0m - Sheet1.csv', skiprows=3)
df_history.index = pd.to_datetime(df_history.index).strftime('%Y-%m-%d %H:%M:%S')

# %%
df_history.reset_index(inplace=True)
df_history.set_index('time', inplace=True)
df_history.drop(columns = 'index', inplace = True)
df_history.index = pd.to_datetime(df_history.index)
df_history = df_history.resample('D').mean()

# %%
df_history

# %%
common_dates = df_history.index.intersection(df_station.index)
df_history = df_history.loc[common_dates]

# %%
df_history

# %%


# %% [markdown]
# # Evaluating correlations between weather data and CO2 levels from station data

# %%
df_station_column = df_station[['average']]
df_CO2_meteo = df_history.join(df_station_column, how='inner')

# %%
df_CO2_meteo

# %%
# Compute Kendall's Tau correlation matrix
kendall_corr = df_CO2_meteo.corr(method='kendall')

plt.figure(figsize=(12, 6))
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f")
div_palette = sns.color_palette("RdBu", 12)
plt.rcParams.update({'font.size': 13})
plt.title("Kendall Correlation Heatmap")
plt.show()

# %% [markdown]
# **Surprisingly and unfortunately, there seems to be little to no correlation between the weather data and the average levels of CO2 at this particular location.
# I will experiment only with the temperature, relative humidity, evapotranspiration and soil temperature for now.**
# 
# 

# %%
df_CO2_meteo.drop(['surface_pressure (hPa)'],axis=1,inplace = True)

# %%
df_CO2_meteo.rename(columns={'average': 'average_CO2', 'temperature_2m (°C)' : 'temperature', 
                             'relative_humidity_2m (%)':'humidity', 'dew_point_2m (°C)' : 'dew_point',
                             'precipitation (mm)' : 'precipitation', 'pressure_msl (hPa)' : 'pressure',
                             'et0_fao_evapotranspiration (mm)' : 'evapotranspiration', 
                             'wind_speed_10m (m/s)' : 'wind_speed', 'soil_temperature_0_to_7cm (°C)' : 'soil_temperature'}, inplace=True)

# %%
df_CO2_meteo.columns

# %%
df_CO2_meteo.to_csv('df_CO2_meteo.csv', index=True)

# %% [markdown]
# # N2O Importation and analysis

# %%
df_N2O = pd.read_csv(r'mlo_N2O_Day.csv', skiprows = 1)

# %%
df_N2O

# %%
new_column_names = ['year', 'month', 'day', 'median_N2O', 'std._dev.N2O', 'samplesN20']
df_N2O.columns = new_column_names
df_N2O

# %%
df_N2O = coerce_into_full_datetime(df_N2O)
df_N2O

# %%
df_N2O.interpolate(method='time', inplace=True)
df_N2O.fillna(method='ffill', inplace=True)
df_N2O.fillna(method='bfill', inplace=True)


# %%
df_N2O

# %% [markdown]
# # Methane importation and analysis

# %%
df_CH4 = pd.read_csv(r'mlo_CH4_Day.csv')

# %%
new_column_names = [
    "site_code", "year", "month", "day", "hour", "minute", "second",
    "datetime", "time_decimal", "midpoint_time", "value_CH4", "value_std_dev_CH4",
    "nvalue_CH4", "latitude", "longitude", "altitude", "elevation", "intake_height", "qcflag"
]

df_CH4.columns = new_column_names

df_CH4.drop(
    columns=["site_code", "year", "month", "day", "hour", "minute", "second", "time_decimal",
             "midpoint_time", "latitude", "longitude", "altitude", "elevation", "intake_height", "qcflag"],
    inplace=True
)

df_CH4["datetime"] = pd.to_datetime(df_CH4["datetime"]).dt.date
df_CH4.set_index("datetime", inplace=True)

df_CH4


# %%
invalid_CH4 = (df_CH4['value_CH4'] == -999.99).sum()
invalid_CH4

# %%
df_CH4 = df_CH4.loc[df_CH4["value_CH4"] != -999.99]


# %% [markdown]
# # SF6 importation and checks

# %%
df_SF6 = pd.read_csv(r'mlo_SF6_Day.csv', skiprows = 1)

# %%
new_column_names = ['year', 'month', 'day', 'median_SF6', 'std.dev_SF6', 'samples']

df_SF6.columns = new_column_names

# %%
df_SF6['year'] = df_SF6['year'].astype(str).str.strip()
mask = df_SF6['year'] == 'year'

df_SF6 = coerce_into_full_datetime(df_SF6)
df_SF6

# %%
df_SF6.interpolate(method='time', inplace=True)
df_SF6.fillna(method='ffill', inplace=True)
df_SF6.fillna(method='bfill', inplace=True)

# %% [markdown]
# # Evaluating importance of each feature

# %%
df_CO2_meteo.rename_axis('datetime', inplace=True)
df_CO2_meteo.columns

# %%
df_N2O.columns

# %%
df_CH4.columns

# %%
df_SF6.columns

# %% [markdown]
# # Merging and visualising all data

# %%
plot_column(df_CO2_meteo, 'average_CO2', 'red')
plot_column(df_N2O, 'median_N2O', 'blue')
plot_column(df_CH4, 'value_CH4', 'green')
plot_column(df_SF6, 'median_SF6', 'magenta')
plt.tight_layout

# %%
dfs = [df.copy() for df in [df_CO2_meteo, df_N2O, df_CH4, df_SF6]]
for i in range(len(dfs)):
    dfs[i].index = pd.to_datetime(dfs[i].index)  # Converting index to proper datetime64[ns]

start_date = df_CO2_meteo.index.min()  # Get the earliest date from df_CO2_meteo

dfs = [df[df.index >= start_date] for df in dfs]

df_combined_outer = pd.concat(
    [dfs[0][['temperature', 'humidity', 'dew_point', 'precipitation', 'pressure',
             'evapotranspiration', 'wind_speed', 'soil_temperature', 'average_CO2']],
     dfs[1][["median_N2O"]],
     dfs[2][["value_CH4"]],
     dfs[3][["median_SF6"]]],
    axis=1, join="outer")

# %%
dfs = [df.copy() for df in [df_CO2_meteo, df_N2O, df_CH4, df_SF6]]
for i in range(len(dfs)):
    dfs[i].index = pd.to_datetime(dfs[i].index)  # Converting index to proper datetime64[ns]

start_date = df_CO2_meteo.index.min()  # Get the earliest date from df_CO2_meteo

dfs = [df[df.index >= start_date] for df in dfs]

df_combined_inner = pd.concat(
    [dfs[0][['temperature', 'humidity', 'dew_point', 'precipitation', 'pressure',
             'evapotranspiration', 'wind_speed', 'soil_temperature', 'average_CO2']],
     dfs[1][["median_N2O"]],
     dfs[2][["value_CH4"]],
     dfs[3][["median_SF6"]]],
    axis=1, join="inner")


# %%
df_combined_outer.columns

# %%
columns_to_fill = ['temperature', 'humidity', 'dew_point', 'precipitation', 'pressure',
                   'evapotranspiration', 'wind_speed', 'soil_temperature', 'average_CO2']

df_combined_outer[columns_to_fill] = df_combined_outer[columns_to_fill].interpolate(method='time')
df_combined_outer[columns_to_fill] = df_combined_outer[columns_to_fill].fillna(method='ffill') # Forward fill remaining missing values
df_combined_outer[columns_to_fill] = df_combined_outer[columns_to_fill].fillna(method='bfill') # Backward fill remaining missing values


# %%
columns_to_plot = ['temperature', 'humidity', 'dew_point', 'precipitation', 'pressure',
                   'evapotranspiration', 'wind_speed', 'soil_temperature', 'average_CO2',
                   'median_N2O', 'value_CH4', 'median_SF6']

plt.figure(figsize=(17, 26))

for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(13, 1, i)
    plt.plot(df_combined_outer.index, df_combined_outer[col], label=col, color='b')  # Plot each column
    plt.xlabel("Datetime")
    plt.ylabel("levels")
    plt.title(f"{col} Over Time")
    plt.legend()
    plt.grid(True)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# %%
# Compute Kendall's Tau correlation matrix
kendall_corr = df_combined_outer.corr(method='kendall')

plt.figure(figsize=(12, 7))
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Kendall Correlation Heatmap")
plt.show()

# %%
# Compute Pearson's Tau correlation matrix
pearson_corr = df_combined_outer.corr(method='pearson')

plt.figure(figsize=(10, 7))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

# %%
df_combined_outer.to_csv("df_combined_outer.csv", index=True)
df_combined_inner.to_csv("df_combined_inner.csv", index=True)

# %%
plot_rolling_correlations(df_combined_inner)

# %%
# Step 1: Check for stationarity (Augmented Dickey-Fuller Test)
def check_stationarity(series, variable_name):
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value

# Run stationarity tests for all variables
stationarity_results = {col: check_stationarity(df_combined_inner[col], col) for col in df_combined_inner.columns}

# Step 2: Determine the optimal lag length (using a maximum of 12 lags)
max_lag = 12
granger_results = {}

for col in df_combined_inner.columns:
    if col != "average_CO2":
        test_result = grangercausalitytests(df_combined_inner[['average_CO2', col]].dropna(), max_lag, verbose=False)
        granger_results[col] = {lag: test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}

# Display stationarity and Granger causality test results
stationarity_results, granger_results


# %%
# Convert Granger causality test results into a readable DataFrame
granger_df = pd.DataFrame.from_dict(
    {var: [granger_results[var][lag] for lag in range(1, max_lag + 1)] for var in granger_results.keys()},
    orient='index',
    columns=[f'Lag {i}' for i in range(1, max_lag + 1)]
)

granger_df["Interpretation"] = granger_df.apply(lambda row: interpret_p_value(row.values), axis=1)
granger_df

# %% [markdown]
# Granger Causality Test Results
# 1. Stationarity Check (Augmented Dickey-Fuller Test)
# CO₂ (average_C2O) is non-stationary (p-value = 0.86) → This means CO₂ has a trend and may need differencing to make it stationary before modeling.
# Temperature (temperature_2m (°C)) and Humidity (relative_humidity_2m (%)) are stationary (p-value ≈ 0).
# N₂O, CH₄, and SF₆ are non-stationary (p-values > 0.67).
# 
# 2. Granger Causality (p-values across 1-12 lags)
# Lower p-values (< 0.05) indicate strong causality.
# The smaller the p-value, the more significant the predictive relationship.
# Variable	Significant at Multiple Lags?	Interpretation
# Temperature (temperature_2m (°C))	✅ Strong causality (p < 1e-10) across all lags	Predicts future CO₂ trends
# Relative Humidity (relative_humidity_2m (%))	✅ Significant up to lag 12 (p < 0.01)	Influences CO₂ levels, but weaker than temperature
# N₂O (median_N2O)	✅ Significant causality at longer lags (p < 1e-8 at lag 12)	Has a delayed effect on CO₂
# CH₄ (value_CH4)	✅ Very strong causality (p < 1e-20 at multiple lags)	CH₄ changes predict CO₂ variations
# SF₆ (median_SF6)	✅ Moderate causality at higher lags (p < 1e-10 at lag 12)	SF₆ shows long-term predictive power
# Key Takeaways
# Temperature is the strongest predictor of CO₂
# 
# The p-values are extremely low, suggesting that past temperature values contain significant information about future CO₂ levels.
# CH₄ also has a very strong causal effect on CO₂
# 
# This makes sense since methane (CH₄) and CO₂ are both greenhouse gases affected by similar processes.
# Humidity, N₂O, and SF₆ also predict CO₂, but to a lesser extent
# 
# Their effects are more delayed, as seen in longer lags (9-12 months).
# CO₂ itself is non-stationary
# 
# If we want to build predictive models, we might need to difference CO₂ to remove trends.

# %% [markdown]
# Why Did Pearson & Kendall Show Weak Correlation, But Granger Shows Strong Causality?
# The difference comes from how these methods analyze relationships:
# 
# 1. Pearson/Kendall Correlation (Static, Instantaneous Relationship)
# These methods only measure direct relationships between variables at the same time.
# Pearson Correlation checks linear relationships at one point in time.
# Kendall Correlation looks at rank-based (monotonic) relationships but still without considering time delays.
# Since CO₂, temperature, and humidity might have delayed effects on each other, Pearson/Kendall may fail to detect a strong relationship.
# 
# 2. Granger Causality (Temporal Dependency)
# This test considers past values of temperature, humidity, and other variables to see if they help predict future CO₂.
# Many climate and atmospheric processes do not act immediately—they take weeks or months to show an impact.
# Example: Higher temperatures today might increase plant respiration or ocean CO₂ release over the next few weeks or months.
# That’s why even if Pearson/Kendall showed weak relationships, Granger Causality detects delayed effects that standard correlation ignores.
# Real-World Example of Delayed Causality in Climate Data
# Temperature & CO₂: When temperatures rise, it may take weeks to months before we see a significant change in CO₂ levels due to ocean-atmosphere exchange.
# Humidity & CO₂: Humidity affects cloud cover, precipitation, and soil moisture, which influence carbon absorption and release but not immediately.
# Methane (CH₄) & CO₂: CH₄ breaks down into CO₂ over time, meaning its effects on CO₂ might appear after several months.
# 

# %%


# %% [markdown]
# ### 1 year in the past - analysis

# %%
plot_lagged_correlations(df_combined_inner, 'average_CO2', 365)

# %% [markdown]
# Key Observations:
# 🌡 Temperature (temperature_2m (°C)) shows a periodic pattern
# 
# The correlation peaks at around 90, 180, and 360 days.
# This suggests a seasonal effect, where temperature changes predict CO₂ levels months later.
# Possible explanation: Seasonal cycles of vegetation, ocean uptake, or industrial activity.
# 💧 Humidity (relative_humidity_2m (%)) has a weak but noticeable lag effect
# 
# Correlation is slightly negative, meaning higher humidity may be linked to lower CO₂ later.
# This could be due to increased plant growth (photosynthesis) reducing CO₂.
# 🛑 N₂O (median_N2O) shows some delayed correlation
# 
# N₂O is related to industrial activity and fossil fuel combustion.
# If N₂O increases, CO₂ might follow due to shared emission sources.
# *🔥 Methane (value_CH4) has a strong positive correlation with CO₂ over time
# 
# CH₄ and CO₂ both contribute to greenhouse effects.
# CH₄ breaks down into CO₂ over time, explaining why higher CH₄ leads to increased CO₂ later.
# 🌎 SF₆ (median_SF6) shows the strongest overall correlation
# 
# SF₆ is a long-lived greenhouse gas, and its correlation with CO₂ is almost constant.
# This suggests shared sources or long-term emission trends.
# 🔬 Interpretation
# Temperature and CH₄ are the strongest predictors of CO₂ over time.
# The delayed effects (~90 to 360 days) explain why Pearson/Kendall correlation missed these relationships.
# There is a clear seasonal component, especially for temperature and humidity.
# Industrial gases (N₂O, SF₆) show long-term trends in relation to CO₂.
# 

# %% [markdown]
# ### 5 year in the past -  analysis

# %%
plot_lagged_correlations(df_combined_inner, 'average_CO2', 1825)

# %% [markdown]
# Key Findings:
# 📉 Temperature (temperature_2m (°C)) Shows Strong Multi-Year Cycles
# 
# Clear periodic pattern every ~365 days → Suggests annual climate cycles affecting CO₂.
# Peaks every ~1 year, aligning with seasonal and yearly CO₂ fluctuations.
# 💧 Humidity (relative_humidity_2m (%)) Shows Opposite Cycles
# 
# Correlation oscillates inversely to temperature.
# Suggests that higher humidity is associated with lower future CO₂ levels.
# Likely linked to vegetation absorption and precipitation cycles.
# 🔥 CH₄ (value_CH4) and SF₆ (median_SF6) Show Strong Long-Term Correlations
# 
# CH₄ and SF₆ maintain high correlation for multiple years.
# This suggests a sustained relationship, possibly due to shared emission sources or slow chemical breakdown.
# 📈 N₂O (median_N2O) Shows Long-Term Influence
# 
# Maintains a high correlation for several years.
# Indicates that industrial emissions trends impact CO₂ over the long run.
# 🌍 What This Means for Long-Term Forecasting
# Temperature and CH₄ are the strongest long-term predictors of CO₂.
# The yearly cycles suggest that seasonality must be considered in CO₂ forecasting models.
# Industrial pollutants (N₂O, SF₆) have long-term correlations, meaning they could be used for policy-driven forecasts.

# %% [markdown]
# ### 10 year back analysis

# %%
plot_lagged_correlations(df_combined_inner, 'average_CO2', 3650)

# %% [markdown]
# 🔍 Very Long-Term Lagged Correlation Analysis (10 Years)
# This plot extends the lagged correlation window to 10 years (3650 days) to examine even longer-term dependencies between CO₂ and its predictors.
# 
# Key Findings:
# 🌡 Temperature (temperature_2m (°C)) Shows a Strong Multi-Year Cycle
# 
# Clear oscillations approximately every year (365 days).
# This suggests strong seasonal and multi-year trends in CO₂.
# The correlation maintains a repeating wave-like pattern, possibly due to global temperature cycles and ocean-atmosphere interactions.
# 💧 Humidity (relative_humidity_2m (%)) Maintains an Opposite Cycle
# 
# Shows an inverse correlation to temperature.
# Indicates that higher humidity leads to lower CO₂ after several months/years (likely due to enhanced vegetation growth, precipitation, and CO₂ absorption).
# 🔥 CH₄ (value_CH4) and SF₆ (median_SF6) Maintain Strong Long-Term Correlations
# 
# CH₄ and SF₆ consistently correlate with CO₂ across multiple years.
# Suggests shared emission sources or slow accumulation effects over time.
# 📈 N₂O (median_N2O) Shows a Gradual Decrease in Correlation Over Time
# 
# The influence of N₂O on CO₂ appears strongest in the first few years but then declines.
# This may reflect industrial emissions policies affecting long-term N₂O and CO₂ trends.
# 📉 After ~8-10 Years, Correlations Become More Unstable
# 
# Around 2500-3500 days (7-10 years), the correlations become noisier.
# This could be due to external climate variability, policy changes, or model limitations in capturing such long-term effects.
# 🌍 Implications for Long-Term CO₂ Forecasting
# Temperature, CH₄, and SF₆ remain the best long-term predictors of CO₂.
# Annual cycles are clearly visible, meaning that any CO₂ forecasting model should incorporate seasonality.
# Humidity has a delayed inverse effect, possibly due to its influence on carbon sinks.
# Industrial pollutants (N₂O) show shorter-term influence, making them more useful for mid-term (1-5 year) forecasting.
# 

# %%
df_combined_inner.drop(columns = ['precipitation', 'wind_speed', 'dew_point', 'pressure'])
df_combined_outer.drop(columns = ['precipitation', 'wind_speed', 'dew_point', 'pressure'])
df_full_CO2 = df_combined_outer[['average_CO2']]


# %%
df_combined_inner.to_csv('df_combined_inner.csv', index=True)
df_combined_outer.to_csv('df_combined_outer.csv', index=True)
df_full_CO2.to_csv('df_full_CO2.csv', index=True)

# %% [markdown]
# 


