# Strategy Analysis Dashboard

The **Strategy Analysis Dashboard** is a web-based application designed to provide a deep dive into the risk and return characteristics of various investment strategies. Utilizing Python and Streamlit, this dashboard visualizes data and analytical metrics to compare performance across different financial instruments and strategies.

## 🚀 Features

- **Dynamic Data Visualizations**: Leverage sophisticated charting and grid capabilities to explore data dynamically.
- **Financial Data Analysis**: Analyze financial data using built-in Python libraries to perform complex calculations and simulations.
- **Interactive User Interface**: Users can interact with data visualizations and modify parameters to see different analytics.

## 📁 File Structure

- `Data.xlsx` - Contains the raw data used for analyses.
- `Homepage.py` - Main script for the dashboard's homepage.
- `all_combinations_metrics.xlsx` - Additional metrics data.
- `requirements.txt` - Specifies the Python libraries needed.
- `pages/`
  - `Part 1.py` - Contains detailed analysis for one part of the dashboard.
  - `Part 2.py` - Contains continuation or additional analyses.

## 🔧 Setup & Execution

1. Ensure you have Python installed on your system.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit application:
   ```bash
   streamlit run Homepage.py
   
## 🧪 Testing

- Launch the dashboard to ensure all pages load without errors.
- Interact with the visualizations to test their responsiveness.
- Upload different datasets if applicable to test data handling and visualization capabilities.

## 🧠 Technical Details

- **Client-Side Technology**: HTML, CSS (leveraged through Streamlit).
- **Server-Side Technology**: Python with Streamlit.
- **Key Python Libraries**:
  - **Streamlit**: For creating and managing the web app.
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  - **Altair**: For declarative visualization in Python.
  - **yfinance**: For downloading historical market data from Yahoo Finance.
  - **SciPy**: For advanced mathematical operations.

## 🌟 Support

For any technical issues or contributions, please open an issue on the project's GitHub repository page or contact the project maintainer.
