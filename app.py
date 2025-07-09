# Import necessary libraries
import streamlit as st  # For creating the web app interface
import yfinance as yf   # To fetch stock market data
import pandas as pd     # For data manipulation
import plotly.graph_objects as go  # For interactive plots
import plotly.express as px        # For simpler interactive plots
from datetime import date  # To handle date input
from statsmodels.tsa.seasonal import seasonal_decompose  # For time series decomposition
import statsmodels.api as sm  # For SARIMAX model
from statsmodels.tsa.stattools import adfuller  # For stationarity test (optional)
from pmdarima import auto_arima # For automatic ARIMA model selection
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




# App title and description
app_name = "Stock Market Forecasting"
st.title(app_name)
# Description block with colored background using HTML
st.markdown(
    """
    <div style="background-color: red; padding: 10px; border-radius: 5px;">
        <h3 style="color: darkblue; text-align: center;">
            This app is created to forecast the stock market price of the selected company.
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)
# Add an image below the description

st.markdown(
    """
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzt48SFZoTyECjNrBs_4sS2hjVv6M9AY3x5A&s" 
    style="width: 700px; height: 300px; padding-top: 20px;">
    """,
    unsafe_allow_html=True
)

# Sidebar input section
st.sidebar.subheader("Select the parameters below")

# Date input for start and end
start_date = st.sidebar.date_input("Start date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", date(2021, 12, 31))

# Company stock ticker selection
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox("Select the company", ticker_list)

# Download stock data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

#  Rename the columns of the DataFrame (make sure the number matches the actual columns)
data.columns = ['Date' 'Open','High' ,'Low', 'Close', 'Volume']

# Reset index to make Data a column
data.reset_index(inplace=True)

# Show date range and dataset
st.write('Date from',start_date,'to',end_date)
st.write(data)

# Title for visualization section
st.header("Data Visualization")
st.write("Note: select your specific date range on the sidebar or zoom the plot and select the specific columns")

# Check if data is available and has required columns
# Ensure necessary columns exist
if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
    required_columns = ['Date', 'Close']
    if all(col in data.columns for col in required_columns):
        fig = px.line(data_frame=data, x='Date', y='Close', title=f'Closing Price of the {ticker} Stock', width=1000, height=600)
        st.plotly_chart(fig)
    else:
        st.error("The dataset does not contain 'Date' and 'Close' columns.")
else:
    st.error("The data is empty or invalid.")
    

# Let user select a column for forecasting
column= st.selectbox("select the column to be used for forcasting",data.columns[1:])
data=data[['Date',column]]
st.write("Selected Data")
st.write(data)

from pmdarima import auto_arima

# Automatically determine p, d, q
auto_model = auto_arima(data[column], seasonal=False, trace=True)

# Extract parameters
p = auto_model.order[0]
d = auto_model.order[1]
q = auto_model.order[2]





#ADF test check stationary
st.header('Is data is Stationary?')
st.write('**Note: if p-value is less than 0.05  than data is stationary')
st.write(adfuller(data[column])[1]<0.05)

# Decompose the selected column into trend, seasonality, and residuals
st.header("Decomposition of data")
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

# Plot the decomposed components using Plotly
trend_fig = px.line(x=data['Date'], y=decomposition.trend, title='Trend', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue')
st.plotly_chart(trend_fig)
seasonality_fig = px.line( x=data['Date'], y=decomposition.seasonal, title='Seasonality',  width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green')
st.plotly_chart(seasonality_fig)
residual_fig = px.line(x=data['Date'], y=decomposition.resid, title='Residuals', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='red', line_dash='dot')
st.plotly_chart(residual_fig)

# Define SARIMAX model parameters
st.success(f"Auto ARIMA selected: p={p}, d={d}, q={q}")

st.markdown(f"""
### 📊 Automatically Detected ARIMA Parameters

- **p = {p}** → Autoregressive term  number of past values used to predict the future
- **d = {d}** → number of times data is differenced to remove trend  
- **q = {q}** → Moving average term  number of past forecast errors included in the prediction
""")
# Seasonal order for SARIMAX model
seasonal_order = st.number_input("Select the the value of seasional order:", 0,24,12)








# Fit the SARIMAX model on selected column
model = sm.tsa.statespace.SARIMAX(data[column], 
                                  order=(p, d, q), 
                                  seasonal_order=(p, d, q, seasonal_order))
model = model.fit()

# Title for forecasting section
st.markdown(f"""
<p style="color: red; font-weight: bold; font-size: 40px; text-align: center;">
    📈 Forecasting the Data of <strong>{ticker}</strong> Stock <br>
    🔍 Column Name: <strong>{column}</strong>
</p>
""", unsafe_allow_html=True)



# Forecast future days input
forecast_period = st.number_input("Select the number of days to forecast:", 1, 365, 10)

# Predict future values using model
predictions = model.get_prediction(start=len(data), end=len(data) +forecast_period)
predictions = predictions.predicted_mean

# Create date index for predictions
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')

# Convert predictions to DataFrame
predictions = pd.DataFrame(predictions)
predictions.insert(0, "Date", predictions.index, True)
predictions.reset_index(drop=True,inplace=True)

# Show predictions and original data
st.write("Predictions", predictions)
st.write("Actual Data", data)
st.write("__________")

# Create an empty figure
fig = go.Figure()
# Add the actual data trace
fig.add_trace(go.Scatter(x=data['Date'],  y=data[column],mode='lines',name='Actual',line=dict(color='blue')))

# Add the predicted data trace
fig.add_trace(go.Scatter(x=predictions['Date'],  y=predictions["predicted_mean"],mode='lines',name='Predicted',line=dict(color='red')))

# Customize layout
fig.update_layout(title='Actual VS Predicted',xaxis_title='Date',yaxis_title='Price',width=1000,height=400,)

# Optional: show separated plots
st.plotly_chart(fig)
show_plot=False 
if st.button("Show  sepearted Plot"): 
    if not show_plot:  
        st.write(px.line (x=data['Date'],y=data[column],title='Actual ',labels={'x': 'Date', 'y': 'price'},width=1200,height=400))
        st.write(px.line (x=predictions['Date'],y=predictions['predicted_mean'],title='Predicted ',labels={'x': 'Date', 'y': 'price'},width=1200,height=400))       
        show_plot=True
    else: 
        show_plot=False
    hide_plot=False 
    if st.button("hide seperate plot"): 
        if not hide_plot:
            hide_plot=True 
    else: 
        hide_plot=False
st.write("________")
# Footer
st.markdown(
    """
    <p style="color:green; font-weight:bold; font-size:30px; text-align:center;">
        Thank you for using this app! 😊<br>
        Share it with your friends! 🤝
    </p>
    """,
    unsafe_allow_html=True
)
# Author Info
st.write("### About the Author:")
st.markdown(
    """
    <p style="color:blue; font-weight:bold; font-size:50px;">Muhammad Shoaib</p>
    """,
    unsafe_allow_html=True
)


    



