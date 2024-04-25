import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import altair as alt
import yfinance as yf
from st_aggrid import AgGrid
from scipy.stats import gmean

# Colour decleartion
colors = ['#1f77b4', '#ff7f0e']

# Function Definitions for Analysis

def get_t_bill_rates():
    # Tickers for Treasury Bills from Yahoo Finance
    tickers = {
        # '5 Year US Treasury Note yeild': '^FVX',            # 5 year treasury note Yeild
        # '10 Year US Treasury Note Intreast rate': '^TNX',   # 10 year note Intreast rate
        # '30 Year uS Treasury Bond yield': '^TYX'            # 30 year treasury bond yield
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

def sortino_ratio(return_series,risk_free_rate):
    mean_return = return_series.mean()
    negative_volatility = return_series[return_series < 0].std()
    return (mean_return - risk_free_rate) / negative_volatility

def var_cvar(return_series, confidence_level):
    var = return_series.quantile(1 - confidence_level)
    cvar = return_series[return_series <= var].mean()
    return var, cvar

def plot_daily_returns_echarts(data_frame):
    options = {
        "tooltip": {"trigger": 'axis'},
        "legend": {"data": ['Series 1', 'Series 2']},
        "xAxis": {"type": 'category', "data": data_frame['Date'].dt.strftime('%Y-%m-%d').tolist()},
        "yAxis": {"type": 'value'},
        "series": [
            {"name": 'Series 1', "type": 'line', "data": data_frame['RoR Series 1'].tolist(), "color":colors[0]},
            {"name": 'Series 2', "type": 'line', "data": data_frame['RoR Series 2'].tolist(), "color":colors[1]}
        ]
    }
    
    st_echarts(options=options, height="400px")

def calculate_metrics(sheet1_df, risk_free_rate, confidence_level):

    series1_metrics = {
        'Geometric Mean Return': gmean(sheet1_df['RoR Series 1'] + 1) - 1,
        'Standard Deviation': sheet1_df['RoR Series 1'].std(),
        'Sharpe Ratio': (sheet1_df['RoR Series 1'].mean() - risk_free_rate) / sheet1_df['RoR Series 1'].std(),
        'Cumulative Returns': (1 + sheet1_df['RoR Series 1']).cumprod().iloc[-1] - 1,
        'Annualized Return': np.power(gmean(sheet1_df['RoR Series 1'] + 1), 252) - 1,
        'Annualized Volatility': sheet1_df['RoR Series 1'].std() * np.sqrt(252),
        'Maximum Drawdown': max_drawdown(sheet1_df['RoR Series 1']),
        'Sortino Ratio': sortino_ratio(sheet1_df['RoR Series 1'], risk_free_rate),
        'VaR': var_cvar(sheet1_df['RoR Series 1'], confidence_level)[0],
        'CVaR': var_cvar(sheet1_df['RoR Series 1'], confidence_level)[1]
    }
    series2_metrics = {
        'Geometric Mean Return': gmean(sheet1_df['RoR Series 2'] + 1) - 1,
        'Standard Deviation': sheet1_df['RoR Series 2'].std(),
        'Sharpe Ratio': (sheet1_df['RoR Series 2'].mean() - risk_free_rate) / sheet1_df['RoR Series 2'].std(),
        'Cumulative Returns': (1 + sheet1_df['RoR Series 2']).cumprod().iloc[-1] - 1,
        'Annualized Return': np.power(gmean(sheet1_df['RoR Series 2'] + 1), 252) - 1,
        'Annualized Volatility': sheet1_df['RoR Series 2'].std() * np.sqrt(252),
        'Maximum Drawdown': max_drawdown(sheet1_df['RoR Series 2']),
        'Sortino Ratio': sortino_ratio(sheet1_df['RoR Series 2'], risk_free_rate),
        'VaR': var_cvar(sheet1_df['RoR Series 2'], confidence_level)[0],
        'CVaR': var_cvar(sheet1_df['RoR Series 2'], confidence_level)[1]
    }
    
    return series1_metrics, series2_metrics

def plot_metric_altair(metric_name, series1_value, series2_value):
    source = pd.DataFrame({
        'Series': ['Series 1', 'Series 2'],
        metric_name: [series1_value, series2_value]
    })

    chart = alt.Chart(source).mark_bar().encode(
        x='Series:N',
        y=f'{metric_name}:Q',
        color=alt.Color('Series:N', scale=alt.Scale(
            domain=['Series 1', 'Series 2'],
            range=[colors[0], colors[1]] 
        ))
    ).properties(title={"text": metric_name, "fontSize": 40})
    
    st.altair_chart(chart, use_container_width=True)

def decimal_to_percentage_display(value):
    return f"{value:.2f}%"

def decimal_to_percentage_graph(value):
    return value * 100

# Streamlit App
st.title('Task 1')

# Load data and pre-process
data_path = 'Data.xlsx'
sheet1_df = pd.read_excel(data_path, sheet_name='Sheet1', index_col=0)
sheet1_df.index = pd.to_datetime(sheet1_df.index)
sheet1_df.reset_index(inplace=True)
sheet1_df.rename(columns={'index': 'Date'}, inplace=True)

# Date range slider
min_date = pd.to_datetime(sheet1_df['Date']).min().to_pydatetime()
max_date = pd.to_datetime(sheet1_df['Date']).max().to_pydatetime()
start_date, end_date = st.slider('Select the Date Range', min_value=min_date, max_value=max_date, value=(min_date, max_date))
filtered_df = sheet1_df[(sheet1_df['Date'] >= start_date) & (sheet1_df['Date'] <= end_date)]

# Slider for confidence level
confidence_level = st.slider('Select the Confidence Level for VaR and CVaR', 0.01, 0.99, 0.95)

# Slider for risk-free rate 
risk_free_rate = st.slider('Select the Risk-Free Rate (%)', 0.0, 10.0, 4.44) / 100

# Cureent T-Bill rates to help identify risk free rate
st.write('Current T-Bill Rates')
t_bill_rates_df = get_t_bill_rates()
st.table(t_bill_rates_df)

# Calculate metrics with user input
series1_metrics, series2_metrics = calculate_metrics(filtered_df, risk_free_rate, confidence_level)

# Plotting Daily Returns
plot_daily_returns_echarts(filtered_df)

# Display metrics and graphs
metrics_to_display_as_percentage = [
    'Geometric Mean Return', 'Standard Deviation', 'Cumulative Returns', 'Annualized Return', 'Annualized Volatility', 'Maximum Drawdown', 'VaR', 'CVaR'
]

for metric_name in series1_metrics.keys():
    series1_value = series1_metrics[metric_name]
    series2_value = series2_metrics[metric_name]

    # Check if this metric should be displayed as a percentage
    if metric_name in metrics_to_display_as_percentage:
        series1_value = decimal_to_percentage_graph(series1_value)
        series2_value = decimal_to_percentage_graph(series2_value)
    else:
        # Format as a decimal with four places if it's not a percentage
        series1_value = f"{series1_value:.4f}" if isinstance(series1_value, float) else series1_value
        series2_value = f"{series2_value:.4f}" if isinstance(series2_value, float) else series2_value

    plot_metric_altair(metric_name, series1_value, series2_value)

    if metric_name in metrics_to_display_as_percentage:
        series1_value = decimal_to_percentage_display(series1_value)
        series2_value = decimal_to_percentage_display(series2_value)

    col1, col2 = st.columns(2)
    col1.metric(label=f"Series 1 {metric_name}", value=series1_value)
    col2.metric(label=f"Series 2 {metric_name}", value=series2_value)


# Display User parameters
user_params = pd.DataFrame({
    'Parameter': ['Risk-Free Rate', 'Confidence Level', 'Start Date', 'End Date'],
    'Value': [f"{risk_free_rate*100:.2f}%", f"{confidence_level*100:.2f}%", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
})
st.write('User Selected Parameters')
st.table(user_params)


# Table Displaying all metrics
formatted_metrics = []
series1_values = []
series2_values = []

# Loop through the metrics, formatting as needed
for metric_name in series1_metrics.keys():
    formatted_metrics.append(metric_name)  

    if metric_name in metrics_to_display_as_percentage:
        series1_value = decimal_to_percentage_display(series1_metrics[metric_name] * 100)  
    else:
        series1_value = f"{series1_metrics[metric_name]:.4f}" if isinstance(series1_metrics[metric_name], float) else series1_metrics[metric_name]
    series1_values.append(series1_value)
    
    if metric_name in metrics_to_display_as_percentage:
        series2_value = decimal_to_percentage_display(series2_metrics[metric_name] * 100) 
    else:
        series2_value = f"{series2_metrics[metric_name]:.4f}" if isinstance(series2_metrics[metric_name], float) else series2_metrics[metric_name]
    series2_values.append(series2_value)

all_metrics = pd.DataFrame({
    'Metric': formatted_metrics,
    'Series 1 Value': series1_values,
    'Series 2 Value': series2_values
})
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
        else:  
            return 'High'
    elif metric == 'Maximum Drawdown':
        if value > -0.10:  # Less than 10% drawdown
            return 'Low'
        elif value > -0.30:  # 10% to 30% drawdown
            return 'Moderate'
        else:  
            return 'High'
    return 'Undefined'

risk_profile_table = pd.DataFrame({
    'Metric Volatility': ['Annualized Volatility', 'VaR', 'CVaR', 'Maximum Drawdown'],
    'Series 1 Profile': [risk_profile(metric, series1_metrics[metric], series1_metrics['VaR']) for metric in ['Annualized Volatility', 'VaR', 'CVaR', 'Maximum Drawdown']],
    'Series 2 Profile': [risk_profile(metric, series2_metrics[metric], series2_metrics['VaR']) for metric in ['Annualized Volatility', 'VaR', 'CVaR', 'Maximum Drawdown']]
})
st.write('Risk/Volatility Profile')
st.table(risk_profile_table)

# Basic Strategy
def get_strategy_preference(series1_val, series2_val, metric_name):
    if metric_name in ['Geometric Mean Return', 'Sharpe Ratio', 'Cumulative Returns', 'Annualized Return', 'Sortino Ratio']:
        return 'Series 1' if series1_val > series2_val else 'Series 2'
    else:
        return 'Series 1' if series1_val < series2_val else 'Series 2'

# Create a DataFrame to display which strategy is better for each metric
strategy_preference = pd.DataFrame({
    'Metric': series1_metrics.keys(),
    'Preferred Strategy': [get_strategy_preference(series1_metrics[metric_name], series2_metrics[metric_name], metric_name) for metric_name in series1_metrics.keys()]
})
st.write('Preferred Investment Strategy')
AgGrid(strategy_preference)

# Summary/Conclusion based on number of metrics in favor
series1_favored = strategy_preference['Preferred Strategy'].tolist().count('Series 1')
series2_favored = strategy_preference['Preferred Strategy'].tolist().count('Series 2')

if series1_favored > series2_favored:
    summary = "Series 1 is the preferred investment strategy based on the majority of metrics."
elif series1_favored < series2_favored:
    summary = "Series 2 is the preferred investment strategy based on the majority of metrics."
else:
    summary = "Both Series 1 and Series 2 are equally favored based on the metrics."

st.write('Summary/Conclusion')
st.info(summary)