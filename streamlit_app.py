import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import OneClassSVM
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.decomposition import PCA
from twilio.rest import Client

# Set page title and icon
st.set_page_config(page_title="Advanced Power Grid Analytics", page_icon="⚡")

st.sidebar.title("Power Grid Fault Monitoring Dashboard")
st.sidebar.text("AI-powered system for fault detection, predictive maintenance, and real-time alerts")

# User input for the number of data points
n_points = st.sidebar.slider("Number of data points", 100, 2000, 1000)
forecast_period = st.sidebar.slider("Forecast period (hours)", 1, 72, 30)

# Generate random data for demonstration purposes
np.random.seed(42)

def generate_random_data(n=1000):
    voltage = np.random.normal(loc=230, scale=10, size=n)
    current = np.random.normal(loc=5, scale=0.5, size=n)
    temperature = np.random.normal(loc=25, scale=5, size=n)
    humidity = np.random.normal(loc=60, scale=10, size=n)
    fault_occurred = np.random.choice([0, 1], size=n)
    
    # Simulating timestamp as hourly data
    timestamps = pd.date_range(start="2024-12-01", periods=n, freq='H')

    data = pd.DataFrame({
        'timestamp': timestamps,
        'voltage': voltage,
        'current': current,
        'temperature': temperature,
        'humidity': humidity,
        'fault_occurred': fault_occurred
    })
    
    return data

data = generate_random_data(n=n_points)

# Display sample data
st.subheader('Sample Data')
st.write(data.head())

# Feature engineering
data['voltage_diff'] = data['voltage'].diff().fillna(0)
data['current_diff'] = data['current'].diff().fillna(0)
data['temp_diff'] = data['temperature'].diff().fillna(0)

# Prepare features and target
X = data[['voltage', 'current', 'temperature', 'humidity', 'voltage_diff', 'current_diff', 'temp_diff']]
y = data['fault_occurred']

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Select model dynamically
model_selection = st.sidebar.selectbox("Choose a Model", ["Gradient Boosting", "Random Forest"])

# Set model parameters dynamically using sliders or input fields
if model_selection == "Gradient Boosting":
    n_estimators = st.sidebar.slider("Number of estimators", 50, 200, 100)
    learning_rate = st.sidebar.slider("Learning rate", 0.01, 0.5, 0.1)
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
elif model_selection == "Random Forest":
    n_estimators = st.sidebar.slider("Number of estimators", 50, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

# Show performance metrics dynamically
st.subheader("Model Performance")
report = classification_report(y_test, y_pred, output_dict=True)
st.write("### Precision, Recall, F1-Score:")
metrics_df = pd.DataFrame(report).transpose()
st.write(metrics_df)

# Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis", labels=dict(x="Predicted", y="True"))
fig.update_layout(title=f"{model_selection} Confusion Matrix")
st.plotly_chart(fig)

# Feature Importance visualization
feature_importance = model.feature_importances_
fig = go.Figure([go.Bar(x=X.columns, y=feature_importance)])
fig.update_layout(title=f"{model_selection} Feature Importance", xaxis_title="Features", yaxis_title="Importance")
st.plotly_chart(fig)

# Anomaly detection using One-Class SVM
anomaly_model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
anomaly_model.fit(X_scaled)
y_anomaly = anomaly_model.predict(X_scaled)
y_anomaly = np.where(y_anomaly == 1, 0, 1)

# Display anomalies detected
data['anomaly'] = y_anomaly
st.subheader("Anomaly Detection: Fault Predictions")
st.write(data[['timestamp', 'voltage', 'current', 'temperature', 'humidity', 'anomaly']].head(20))

# Time Series Forecasting for Voltage and Current
st.subheader(f"Time Series Forecasting for Voltage and Current ({forecast_period} hours)")

def time_series_forecasting(df, column, periods=30):
    model = ExponentialSmoothing(df[column], trend="add", seasonal="add", seasonal_periods=24)
    model_fit = model.fit()
    forecast = model_fit.forecast(periods)
    return forecast

forecast_voltage = time_series_forecasting(data, 'voltage', periods=forecast_period)
forecast_current = time_series_forecasting(data, 'current', periods=forecast_period)

# Plot forecasts
forecast_df = pd.DataFrame({
    'timestamp': pd.date_range(start="2024-12-01", periods=forecast_period, freq="H"),
    'forecast_voltage': forecast_voltage,
    'forecast_current': forecast_current
})

fig = px.line(forecast_df, x='timestamp', y=['forecast_voltage', 'forecast_current'], title="Voltage and Current Forecast")
fig.update_layout(xaxis_title="Time", yaxis_title="Value")
st.plotly_chart(fig)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Fault'] = y
fig = px.scatter(pca_df, x='PC1', y='PC2', color='Fault', title="PCA: Fault Detection")
st.plotly_chart(fig)

# Your Twilio credentials
account_sid = 'AC380698afe7bff07b2604f308d451d462'  # Your Account SID
auth_token = '5a95e6f575078f152e7dea7ba7486db0'   # Your Auth Token

# Twilio client initialization
client = Client(account_sid, auth_token)

# Message details
from_phone = '+17756187409'  # Your Twilio phone number
to_phone = '+919000633061'   # Recipient's phone number

# Function to send SMS
def send_alert(message_body):
    try:
        message = client.messages.create(
            body=message_body,
            from_=from_phone,
            to=to_phone
        )
        st.success(f"Alert Sent: {message.sid}")
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Check if any fault is detected and trigger alert
fault_data = data[data['fault_occurred'] == 1]
if not fault_data.empty:
    st.subheader("Fault Detected at the Following Timestamps:")
    st.write(fault_data[['timestamp', 'fault_occurred']])
    
    # Send an alert message based on the fault timestamp
    alert_message = f"Alert: Fault detected at {fault_data['timestamp'].iloc[0]}"
    send_alert(alert_message)

# Manual alert sending option
manual_alert = st.sidebar.button("Send Manual Alert")
if manual_alert:
    alert_message = "Manual Alert: A potential fault has been detected."
    send_alert(alert_message)