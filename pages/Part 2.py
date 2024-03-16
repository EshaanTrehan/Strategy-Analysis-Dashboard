import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import altair as alt
import yfinance as yf
from st_aggrid import AgGrid
from itertools import permutations

# Colour decleartion
colors = ['#1f77b4', '#ff7f0e']

# Function Definitions for Analysis

def get_t_bill_rates():
    # Tickers for Treasury Bills on Yahoo Finance
    tickers = {
        '5 Year US Treasury Note yeild': '^FVX',            # 5 year treasury note Yeild
        '10 Year US Treasury Note Intreast rate': '^TNX',   # 10 year note Intreast rate
        '30 Year uS Treasury Bond yield': '^TYX'            # 30 year treasury bond yield
    }
    rates = {'Duration Type': [], 'Rate': []}
    
    for duration, ticker in tickers.items():
        tbill = yf.Ticker(ticker)
        hist = tbill.history(period="1d")
        last_close = hist['Close'].iloc[-1]
        rates['Duration Type'].append(duration)
        rates['Rate'].append(last_close)
    
    return pd.DataFrame(rates)

def max_drawdown(return_series):
    cumulative_returns = (1 + return_series).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def sortino_ratio(return_series):
    mean_return = return_series.mean()
    negative_volatility = return_series[return_series < 0].std()
    return mean_return / negative_volatility

def var_cvar(return_series, confidence_level):
    var = return_series.quantile(1 - confidence_level)
    cvar = return_series[return_series <= var].mean()
    return var, cvar

def calculate_metrics(filtered_df, risk_free_rate, confidence_level, selected_combination):

    calculated_metrics = {
        'Mean Return': filtered_df[selected_combination].mean(),
        'Standard Deviation': filtered_df[selected_combination].std(),
        'Sharpe Ratio': (filtered_df[selected_combination].mean() - risk_free_rate) / filtered_df[selected_combination].std(),
        'Cumulative Returns': (1 + filtered_df[selected_combination]).cumprod().iloc[-1] - 1,
        'Annualized Return': np.power((1 + filtered_df[selected_combination].mean()), 252) - 1,
        'Annualized Volatility': filtered_df[selected_combination].std() * np.sqrt(252),
        'Maximum Drawdown': max_drawdown(filtered_df[selected_combination]),
        'Sortino Ratio': sortino_ratio(filtered_df[selected_combination]),
        'VaR': var_cvar(filtered_df[selected_combination], confidence_level)[0],
        'CVaR': var_cvar(filtered_df[selected_combination], confidence_level)[1]
    }
    
    return calculated_metrics

def plot_daily_returns_echarts(data_frame, selected_combination):
    # Define the series for the selected combination
    series_data = data_frame[selected_combination].tolist()
    series_option = {
        "name": selected_combination,
        "type": 'line',
        "data": series_data,
        "smooth": True,
        "symbol": 'none',  # Hide symbols on the line
        "lineStyle": {"width": 2},  # Slightly thicker line for visibility
        "itemStyle": {
            "color": colors[0]  # Set the color for the selected series
        }
    }
    
    # Options for the chart
    options = {
        "tooltip": {"trigger": 'axis'},
        "legend": {
            "data": [selected_combination]  # Only the selected combination in the legend
        },
        "xAxis": {
            "type": 'category',
            "data": data_frame['Date'].dt.strftime('%Y-%m-%d').tolist()
        },
        "yAxis": {
            "type": 'value'
        },
        "series": [series_option]  # Only one series option
    }
    
    # Display the chart with only the selected combination
    st_echarts(options=options, height="600px")

def plot_metric_altair(metric_name, metric_value):
    # Prepare data for Altair
    source = pd.DataFrame({
        'Metric': [metric_name],  # Enclose the single metric name in a list
        'Value': [metric_value]   # Enclose the single metric value in a list
    })

    # Altair Bar Chart
    chart = alt.Chart(source).mark_bar().encode(
        x='Metric:N',
        y='Value:Q',
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=[metric_name],  # Enclose the single metric name in a list
            range=[colors[0]]      # Enclose the single color in a list
        ))
    ).properties(title={"text": metric_name, "fontSize": 20})

    st.altair_chart(chart, use_container_width=True)


# Streamlit App
st.title('Abbey Capital Graduate Program Task Part 2')

# Load data and pre-process
data_path = 'Graduate Programme 2024 - Exercise Data.xlsx'

# Renaming first col in Sheet 2 to Date
sheet2_df = pd.read_excel(data_path, sheet_name='Sheet2', index_col=0)
sheet2_df.index = pd.to_datetime(sheet2_df.index)
sheet2_df.reset_index(inplace=True)
sheet2_df.rename(columns={'index': 'Date'}, inplace=True)
sheet2_df['Date'] = pd.to_datetime(sheet2_df['Date'], dayfirst=True)

# Renaming first col in Sheet 3 to Date
sheet3_df = pd.read_excel(data_path, sheet_name='Sheet3', index_col=0)
sheet3_df.index = pd.to_datetime(sheet3_df.index)
sheet3_df.reset_index(inplace=True)
sheet3_df.rename(columns={'index': 'Date'}, inplace=True)
sheet3_df['Date'] = pd.to_datetime(sheet3_df['Date'], dayfirst=True)

# Merging Data frames
merged_df = pd.merge(sheet2_df, sheet3_df, on='Date')
merged_df = merged_df.drop(columns=['Unnamed: 2'])

# All permutations
column_permutations_60_40 = list(permutations(merged_df.columns[2:], 2))
column_permutations_60_40 = [perm for perm in column_permutations_60_40 if perm[0] != perm[1]]

# Creating combinations
def create_combinations_60_40(merged_df, permutations):
    combination_names_60_40 = []
    for perm in permutations:
        col60, col40 = perm
        # 60% of first column, 40% of second column
        merged_df[f'60% {col60} - 40% {col40}'] = 0.6 * merged_df[col60] + 0.4 * merged_df[col40]
        combination_names_60_40.append([f'60% {col60} - 40% {col40}'])

    return combination_names_60_40

def create_combinations_50_30_20(merged_df):
    col50 = merged_df.columns[2]  # Fixed 50% column
    col30 = merged_df.columns[3]  # Fixed 30% column
    variable_cols = merged_df.columns[4:7]  # Columns 5, 6, 7

    combination_names_50_30_20 = []
    for col20 in variable_cols:
        # 50% of the third column, 30% of the fourth, 20% of the variable column
        merged_df[f'50% {col50} - 30% {col30} - 20% {col20}'] = (
            0.5 * merged_df[col50] + 0.3 * merged_df[col30] + 0.2 * merged_df[col20]
        )
        combination_names_50_30_20.append(f'50% {col50} - 30% {col30} - 20% {col20}')

    return combination_names_50_30_20


# Apply the function to create combinations
combination_names_60_40 = create_combinations_60_40(merged_df, column_permutations_60_40)
combination_names_50_30_20 = create_combinations_50_30_20(merged_df)

# Combine the combination names lists
flat_combination_names_60_40 = [item for sublist in combination_names_60_40 for item in sublist]

# Combine and ensure unique combination names if needed
combination_names = list(set(flat_combination_names_60_40 + combination_names_50_30_20))

# Date range slider
min_date = pd.to_datetime(merged_df['Date']).min().to_pydatetime()
max_date = pd.to_datetime(merged_df['Date']).max().to_pydatetime()
start_date, end_date = st.slider('Select the Date Range', min_value=min_date, max_value=max_date, value=(min_date, max_date))
filtered_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]

# Slider for confidence level
confidence_level = st.slider('Select the Confidence Level for VaR and CVaR', 0.01, 0.99, 0.95)

# Slider for risk-free rate (as a percentage, e.g., 2 for 2%)
risk_free_rate = st.slider('Select the Risk-Free Rate (%)', 0.0, 10.0, 4.44) / 100

# Cureent T-Bill rates to help identify risk free rate
st.write('Current T-Bill Rates')
t_bill_rates_df = get_t_bill_rates()
st.table(t_bill_rates_df)

# Creating a dropdown for users to select the combination
selected_combination = st.selectbox('Select a combination to analyze:', options=combination_names)

# Recalculate metrics with user input
calculated_metrics = calculate_metrics(filtered_df, risk_free_rate, confidence_level, selected_combination)

# Plotting Daily Returns
plot_daily_returns_echarts(filtered_df, selected_combination)

# Display metrics and graphs
for metric_name, metric_value in calculated_metrics.items():
    plot_metric_altair(metric_name, metric_value)

    # Display the metric value as a number
    col1 = st.columns([1])[0]  # Create a single column
    col1.metric(label=f"{metric_name}", value=f"{metric_value:.4f}" if isinstance(metric_value, float) else f"{metric_value}")

# Display User parameters
user_params = pd.DataFrame({
    'Parameter': ['Risk-Free Rate', 'Confidence Level', 'Start Date', 'End Date'],
    'Value': [f"{risk_free_rate*100:.2f}%", f"{confidence_level*100:.2f}%", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
})
st.write('User Selected Parameters')
st.table(user_params)

# Create a DataFrame from the metrics dictionary
all_metrics = pd.DataFrame(list(calculated_metrics.items()), columns=['Metric', selected_combination])
st.write('Metrics Comparison')
AgGrid(all_metrics)


# Risk Range Explanation Table
risk_ranges = {
    'Metric': ['Annualized Volatility', 'Daily VaR', 'Daily CVaR', 'Maximum Drawdown'],
    'Low': ['< 5%', '< 0.5%', 'Less than double the VaR value', '< 10%'],
    'Moderate': ['5% - 25%', '0.5% - 3%', 'Less than double the VaR value', '10% - 30%'],
    'High': ['> 25%', '> 3%', 'More than double the VaR value', '> 30%']
}
risk_range_table = pd.DataFrame(risk_ranges)
st.write('Risk Range Definitions')
st.table(risk_range_table)

# Risk/Volatility Profile Table
# Define thresholds for low, moderate, and high risk/volatility (these are placeholders)
def risk_profile(metric, value, var):
    if metric == 'Annualized Volatility':
        if value < 0.05:  # Less than 5%
            return 'Low'
        elif value <= 0.25:  # 5% to 25%
            return 'Moderate'
        else:  # Greater than 25%
            return 'High'
    elif metric == 'VaR':
        if value > -0.005:  # Less than 0.5%
            return 'Low'
        elif value >= -0.03:  # 0.5% to 3%
            return 'Moderate'
        else:  # Greater than 3%
            return 'High'
    elif metric == 'CVaR':
        if value > 2 * var:
            return 'Low'
        elif value >= 2 * var:
            return 'Moderate'
        else:  # Significantly more than double the VaR value
            return 'High'
    elif metric == 'Maximum Drawdown':
        if value > -0.10:  # Less than 10% drawdown
            return 'Low'
        elif value > -0.30:  # 10% to 30% drawdown
            return 'Moderate'
        else:  
            return 'High'
    return 'Undefined'

# Display the risk profile table with the name of the selected option
risk_profile_table = pd.DataFrame({
    'Metric Volatility': ['Annualized Volatility', 'VaR', 'CVaR', 'Maximum Drawdown'],
    selected_combination: [risk_profile(metric, calculated_metrics[metric], calculated_metrics['VaR']) for metric in ['Annualized Volatility', 'VaR', 'CVaR', 'Maximum Drawdown']]
})
st.write('Risk/Volatility Profile')
st.table(risk_profile_table)

def export_metrics_to_excel(data_frame, combinations, risk_free_rate, confidence_level, file_path='all_combinations_metrics.xlsx'):
    all_metrics_df = pd.DataFrame()

    for combo in combinations:
        metrics = calculate_metrics(data_frame, risk_free_rate, confidence_level, combo)
        combo_str = ' - '.join(combo) if isinstance(combo, (list, tuple)) else str(combo)
        metrics_df = pd.DataFrame(metrics, index=[combo_str])  # Use combo_str as the index
        all_metrics_df = pd.concat([all_metrics_df, metrics_df])  # Do not ignore the index

    # After concatenating, reset the index to turn the strategy names into a column. 
    # This step will name the index column automatically.
    all_metrics_df.reset_index(inplace=True)
    all_metrics_df.rename(columns={'index': 'Portfolio Split'}, inplace=True)

    all_metrics_df.to_excel(file_path, index=False)
    print(f"Metrics for all combinations have been saved to {file_path}.")

export_metrics_to_excel(filtered_df, combination_names, risk_free_rate, confidence_level, 'all_combinations_metrics.xlsx')

# Load data from an Excel file
file_path = 'all_combinations_metrics.xlsx'  # Adjust this to the path of your Excel file
df = pd.read_excel(file_path)

# Normalize metrics
# For metrics where higher is better, normalize directly
# For metrics where lower is better, invert the normalization
metrics_to_normalize_directly = ['Mean Return', 'Sharpe Ratio', 'Cumulative Returns', 'Annualized Return', 'Sortino Ratio']
metrics_to_normalize_inversely = ['Standard Deviation', 'Annualized Volatility', 'Maximum Drawdown', 'VaR', 'CVaR']

for metric in metrics_to_normalize_directly:
    df[f'{metric} Norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

for metric in metrics_to_normalize_inversely:
    df[f'{metric} Norm'] = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Calculate composite score with equal weights for simplicity
normalized_columns = [f'{metric} Norm' for metric in (metrics_to_normalize_directly + metrics_to_normalize_inversely)]
df['Composite Score'] = df[normalized_columns].mean(axis=1)

# Find the best strategy based on the highest composite score
# Assuming 'Strategy Name' is the column with the actual names of the strategies
composite_scores_df = df[['Portfolio Split', 'Composite Score']].sort_values(by='Composite Score', ascending=False)

# For displaying the best strategy
best_strategy_row = composite_scores_df.iloc[0]  # Get the first row of the sorted DataFrame
best_strategy_name = best_strategy_row['Portfolio Split']  # Assuming the names are stored in 'Strategy Name'

# Now use AgGrid with the DataFrame
st.write('Composite Scores for each Portfolio Split:')
AgGrid(composite_scores_df)

# Top 5
top_five_strategies = composite_scores_df.head(5)

# Create a bar chart
chart = alt.Chart(top_five_strategies, height=400).mark_bar().encode(
    x=alt.X('Portfolio Split', sort='-y', axis=alt.Axis(labelAngle=-45)), 
    y='Composite Score',
    color='Portfolio Split'  
).properties(
    title='Top 5 Portfolio Splits by Composite Score'
)

# Display the chart
st.altair_chart(chart, use_container_width=True)

st.write('Summary')
st.info(f"\nThe recommended Portfolio Split is: {best_strategy_name}")