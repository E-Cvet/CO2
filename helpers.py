
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_column(df, column_name, colour):
  '''simple plot of column over TIME'''

  plt.figure(figsize=(17, 6))
  plt.plot(df.index, df[column_name], label=column_name, color=colour)
  plt.grid(
    color="gray",      
    linestyle="--",   
    linewidth=0.5,   
    alpha=0.7)
  plt.title(f'{column_name} Over Time')
  plt.xlabel('Date')
  plt.ylabel(column_name)
  plt.legend()
  plt.show()


def add_missing_one_year_rows(df):
  ''' # function for adding rows from the column '1 year ago' where the calculated
        dates do not coencide with the existing rows - gives us more accurate data in terms of trends and movement'''

  df = df.sort_index()
  new_rows = []

  for current_date, row in df.iterrows():
      one_year_ago_date = current_date - pd.DateOffset(years=1)

      if one_year_ago_date not in df.index:
        new_row = {
                'average': row['1 year ago'],
                '1 year ago': None,  # This would be blank for the new row
                '10 years ago': None,  # This would be blank for the new row
            }
        new_rows.append((one_year_ago_date, new_row))

  new_rows_df = pd.DataFrame([r[1] for r in new_rows], index=[r[0] for r in new_rows])
  df = pd.concat([df, new_rows_df]).sort_index()

  return df


def coerce_into_full_datetime(df):
  ''' coercing datetime indexed column from separate non numeric columns '''

  df['year'] = pd.to_numeric(df['year'], errors='coerce')
  df['month'] = pd.to_numeric(df['month'], errors='coerce')
  df['day'] = pd.to_numeric(df['day'], errors='coerce')
  df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')

  df.set_index('datetime', inplace=True)
  df.drop(columns = ['year', 'month', 'day'], inplace = True)

  return df


def add_missing_dates(df):
  ''' inserting new rows with data from '10 years ago' column' to give us a richer timeframe'''
  df = df.sort_index()

  placeholder_mask = (df['10 years ago'] == -999.99)

  shifted_average = df['average'].copy()
  shifted_average.index = shifted_average.index - pd.DateOffset(years=10)

  aligned_shifted_average = shifted_average.reindex(df.index)
  df.loc[placeholder_mask, '10 years ago'] = aligned_shifted_average[placeholder_mask]
  df.loc[placeholder_mask, ['average', '10 years ago']]

  return df


def plot_rolling_correlations(df):
    ''' plotting rolling corelations between the CO2 column and other columns'''

    rolling_corr_30 = df.rolling(window=30).corr(df["average_CO2"])
    rolling_corr_365 = df.rolling(window=365).corr(df["average_CO2"])

    # Plot individual rolling correlations for both window sizes
    for col in df.columns:
        if col != "average_CO2":
            plt.figure(figsize=(16, 5))
            plt.plot(rolling_corr_30.index, rolling_corr_30[col], label=f"30-day Corr with {col}", color="lightblue", alpha=0.7)
            plt.plot(rolling_corr_365.index, rolling_corr_365[col], label=f"365-day Corr with {col}", color="darkred", linestyle="dashed", alpha=0.99)  
            plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            plt.title(f"Rolling Correlation (30-day & 365-day) between CO2 and {col}")
            plt.xlabel("Date")
            plt.ylabel("Correlation")
            plt.legend()
            plt.show()
    

def interpret_p_value(p_values):
    ''' Add an interpretation column based on p-values'''

    if all(p < 0.01 for p in p_values):  # Strong causality across all lags
        return "Strong causality (p < 0.01)"
    elif any(p < 0.05 for p in p_values):  # Moderate causality at some lags
        return "Moderate to low causality (p < 0.05 at some lags)"
    else:
        return "No significant causality"


def plot_lagged_correlations(df, target_column, max_lag_days, significance_threshold=0.05):
    """Plots lagged correlations between the target column and other columns in the DataFrame."""
    
    lagged_correlation_results = pd.DataFrame(index=range(1, max_lag_days + 1))
    for col in df.columns:
        if col != target_column:
            correlations = [
                df[target_column].corr(df[col].shift(lag)) for lag in range(1, max_lag_days + 1)
            ]
            lagged_correlation_results[col] = correlations

    plt.figure(figsize=(16, 6))
    for col in lagged_correlation_results.columns:
        plt.plot(lagged_correlation_results.index, lagged_correlation_results[col], label=col)

    plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(y=significance_threshold, color="red", linestyle="dashed", label=f"Significance Threshold (±{significance_threshold})")
    plt.axhline(y=-significance_threshold, color="red", linestyle="dashed")
    plt.title(f"Lagged Correlation Analysis (Up to {max_lag_days} Days): {target_column} vs Predictors")
    plt.xlabel("Lag (Days)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.show()


def plot_increase_decades(df):
    '''function for plotting the increase of levels through the decades'''
    df_viz = df.reset_index()  # Reset the index if needed
    df_viz.rename(columns={'index': 'datetime'}, inplace=True)

    df_viz['datetime'] = pd.to_datetime(df_viz['datetime'], errors='coerce')
    df_viz['decade'] = (df_viz['datetime'].dt.year // 10) * 10

    # Group by decade and calculate the mean CO2 levels for each decade
    decade_avg_CO2 = df_viz.groupby('decade')['average_CO2'].mean().reset_index()

    plt.figure(figsize=(16, 6))
    colors = ['#A8D5BA', '#C1E0C5', '#DDE8B9', '#F4D9A6', '#F8B88B', '#F29A8E']
    bars = plt.bar(decade_avg_CO2['decade'], decade_avg_CO2['average_CO2'], color=colors, width=8, edgecolor='black')

    plt.title('Layered Level Chart of CO₂ Levels by Decade', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('CO₂ Levels (ppm)', fontsize=12)
    plt.xticks(decade_avg_CO2['decade'], labels=[f"{int(year)}s" for year in decade_avg_CO2['decade']])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_labels = [f"{int(year)}s" for year in decade_avg_CO2['decade']]
    plt.legend(bars, legend_labels, title='Decades', loc='lower right')

    plt.tight_layout()
    plt.show()




