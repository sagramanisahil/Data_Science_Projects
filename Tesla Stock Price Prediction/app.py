import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset for visualization & metrics
df = pd.read_csv("TESLA.csv")
df['Date'] = pd.to_datetime(df['Date']).astype(int) / 10**9

X = df[['Date', 'Open', 'High', 'Low', 'Volume']]
y = df['Close']

# App Title
st.title("📈 Tesla Stock Closing Price Prediction")
st.write("End-to-End Data Science Project | Deployed ML App")

st.divider()

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.write("**Model:** Linear Regression")
st.sidebar.write("**Features:** Date, Open, High, Low, Volume")
st.sidebar.write("**Target:** Close Price")

# User Input
st.subheader("🔢 Enter Stock Details")

date = st.date_input("Select Date")
open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
high_price = st.number_input("High Price", min_value=0.0, format="%.2f")
low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
volume = st.number_input("Volume", min_value=0.0, format="%.0f")

# Prediction
if st.button("🔮 Predict Closing Price"):
    date_ts = pd.to_datetime(date).timestamp()

    input_data = np.array([[date_ts, open_price, high_price, low_price, volume]])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Tesla Closing Price: **${prediction[0]:.2f}**")

st.divider()

# Model Evaluation Section
st.subheader("📊 Model Evaluation Metrics")

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

st.divider()

# Visualization
st.subheader("📉 Actual vs Predicted Closing Prices")

fig, ax = plt.subplots()
ax.plot(y.values[:100], label="Actual", linewidth=2)
ax.plot(y_pred[:100], label="Predicted", linestyle="--")
ax.set_xlabel("Samples")
ax.set_ylabel("Closing Price")
ax.legend()

st.pyplot(fig)

st.caption("Showing first 100 data points for clarity")
