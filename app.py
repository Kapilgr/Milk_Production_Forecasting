import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

#  Configuration 
st.set_page_config(page_title="Milk Production Forecast", layout="wide")


# 1. Loading Data and Model
@st.cache_resource
def load_sarima_model():
    """Load the trained SARIMAX model."""
    try:
        with open("sarima_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ 'sarima_model.pkl' not found.")
        st.stop()

@st.cache_data
def load_historical_data():
    """Load the specific milk production dataset."""
    try:
        df = pd.read_csv('monthly_milk_production.csv')
        df.columns = ['Date', 'monthly_milk_production'] 
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    except Exception as e:
        st.error(f"❌ Could not load 'monthly_milk_production.csv'.")
        st.stop()

model_results = load_sarima_model()
hist_df = load_historical_data()


# 2. UI Heading
st.markdown("<h1 style='text-align: center; color:#008080;'>Milk Production Forecast</h1>", unsafe_allow_html=True)
st.markdown("---")


# 3. Sidebar Settings
st.sidebar.header("Settings")
n_periods = st.sidebar.number_input("Months to Forecast:", min_value=1, value=24)
predict_button = st.sidebar.button("Generate Forecast", type='primary')


# 4. SARIMAX Forecasting & Plotting
if predict_button:
    forecast_mean = model_results.forecast(steps=n_periods)

    # Create dates for the forecast
    last_date = hist_df['Date'].max()
    forecast_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq='MS')[1:]

    # Plotly Visualization 
    fig = go.Figure()

    # 1. Historical Data 
    fig.add_trace(go.Scatter(
        x=hist_df['Date'], 
        y=hist_df['monthly_milk_production'],
        name='Historical Data',
        mode='lines',
        line=dict(color='#333333', width=2)
    ))

    # 2. Forecast Mean 
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=forecast_mean,
        name='Forecasted Trend',
        mode='lines',
        line=dict(color='#008080', width=3)
    ))

    fig.update_layout(
        title="Milk Production Forecasted Trend",
        xaxis_title="Year",
        yaxis_title="monthly_milk_production",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Simple table for the numbers
    with st.expander("View Forecast Table"):
        st.dataframe(pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Production': forecast_mean.values
        }))


# 5. Footer (Updated Signature)
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Project | Developed by Kapil Adhikari | © 2026</div>", unsafe_allow_html=True)