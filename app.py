import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go 
import plotly.express as px
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

# App title and description
app_name = "Stock Market Forecasting"
st.title(app_name)

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

st.markdown(
    """
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRzt48SFZoTyECjNrBs_4sS2hjVv6M9AY3x5A&s" 
    style="width: 700px; height: 300px; padding-top: 20px;">
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.subheader("Select the parameters below")
start_date = st.sidebar.date_input("Start date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", date(2021, 12, 31))

ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox("Select the company", ticker_list)

data = yf.download(ticker, start=start_date, end=end_date)
#data.columns = [None] * len(data.columns)
data.columns = ['Date' 'Open','High' ,'Low', 'Close', 'Volume']

data.reset_index(inplace=True)
st.write('Date from',start_date,'to',end_date)
st.write(data)
st.header("Data Visualization")
st.write("Note: select your specific date range on the sidebar or zoom the plot and select the specific columns")


if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
    required_columns = ['Date', 'Close']
    if all(col in data.columns for col in required_columns):
        fig = px.line(data_frame=data, x='Date', y='Close', title='Closing Price of the Stock', width=1000, height=600)
        st.plotly_chart(fig)
    else:
        st.error("The dataset does not contain 'Date' and 'Close' columns.")
else:
    st.error("The data is empty or invalid.")

column= st.selectbox("select the column to be used for forcasting",data.columns[1:])
data=data[['Date',column]]
st.write("Selected Data")
st.write(data)
#ADF test check stationary
st.header('Is data is Stationary?')
st.write('**Note: if p-value is less than 0.05  than data is stationary')
st.write(adfuller(data[column])[1]<0.05)
st.header("Decomposition of data")
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

trend_fig = px.line(x=data['Date'], y=decomposition.trend, title='Trend', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue')
st.plotly_chart(trend_fig)
seasonality_fig = px.line( x=data['Date'], y=decomposition.seasonal, title='Seasonality',  width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green')
st.plotly_chart(seasonality_fig)
residual_fig = px.line(x=data['Date'], y=decomposition.resid, title='Residuals', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='red', line_dash='dot')
st.plotly_chart(residual_fig)
# Lets run the model 
# User input for ARIMA parameters
p = st.slider("Select the AR term (p):", 0, 5, 2)
d = st.slider("Select the Differencing term (d):", 0, 5, 1)
q = st.slider("Select the MA term (q):", 0, 5, 2)
seasonal_order = st.number_input("Select the the value of seasional p:", 0,24,12)

# Fit the SARIMAX model
model = sm.tsa.statespace.SARIMAX(data[column], 
                                  order=(p, d, q), 
                                  seasonal_order=(p, d, q, seasonal_order))
model = model.fit()

# Display model summary
st.header("Model Summary")
st.write(model.summary())
st.markdown("""
<p style="color: red; font-weight: bold; font-size: 50px;">Forecasting the Data</p>
""", unsafe_allow_html=True)

# Forecast future values
forecast_period = st.number_input("Select the number of days to forecast:", 1, 365, 10)

predictions = model.get_prediction(start=len(data), end=len(data) +forecast_period)
predictions = predictions.predicted_mean


predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, "Date", predictions.index, True)
predictions.reset_index(drop=True,inplace=True)

# Display predictions
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
# Show the figure
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
st.markdown(
    """
    <p style="color:green; font-weight:bold; font-size:30px; text-align:center;">
        Thank you for using this app! 😊<br>
        Share it with your friends! 🤝
    </p>
    """,
    unsafe_allow_html=True
)
st.write("### About the auther:")
st.markdown(
    """
    <p style="color:blue; font-weight:bold; font-size:50px;">Muhammad Shoaib</p>
    """,
    unsafe_allow_html=True
)


    



