# 📈 Stock Price Prediction App

This is an interactive **Stock Price Prediction Web App** built using **Streamlit**. It allows users to visualize historical stock data and forecast future stock prices using **ARIMA/ARIMAX** models.

---

## 🚀 Features

- 📅 Select date range for stock data  
- 🏢 Choose any listed company (via ticker symbol)
- 📊 Visualize historical stock trends using Plotly  
- 🔮 Predict future prices using ARIMA  
- 📈 See forecasts plotted alongside actual prices  
- 📥 Built with an intuitive sidebar for parameter selection  

---
## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/)
- [pmdarima](https://alkaline-ml.com/pmdarima/)
- [Plotly](https://plotly.com/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [statsmodels](https://www.statsmodels.org/)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Shoaib1-coder/StockPricePrediction.git
cd StockPricePrediction
```

2. **Create a virtual environment (optional but recommended)**

```bash
# Make sure Anaconda is installed: https://www.anaconda.com/products/distribution

# Create a new environment named 'stockprice' with Python 3.10
conda create --name stockprice python=3.10

# Activate the environment
conda activate stockprice

```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```bash
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # This file
└── ...
```

---

## 📌 Usage Instructions

1. Choose a **company ticker symbol** from the sidebar.
2. Select a **date range** to analyze historical stock data.
3. Choose the **data column** (e.g., `Close`, `Open`) for forecasting.
4. View:
   - Interactive line charts for selected data
   - Forecast plots based on ARIMA model

---

## 🧠 ARIMA Model Details

The app uses `pmdarima.auto_arima()` to automatically find the best `(p, d, q)` parameters and generate forecasts.

- `p`: Number of lag observations (AR)
- `d`: Degree of differencing
- `q`: Size of moving average window (MA)

---

## 📬 Feedback and Contributions

Feel free to fork the repo, open issues, or submit pull requests.  
For suggestions or feedback, contact: **[mshoaib3393@gmail.com]**

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
