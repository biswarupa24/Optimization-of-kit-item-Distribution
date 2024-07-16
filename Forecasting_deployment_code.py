#Deployment

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\senap\Downloads\Projects\Live Project 1\preprocessed_data.csv") # Update with your dataset path

# Load the best models
best_models = {}

# Load the best models from their saved files
for column in df.columns:
    # Load the model for this column
    model_path = f"C:\\Users\\senap\\best_models_1\\{column}.pkl" # Update with your model directory path
    try:
        loaded_model = joblib.load(model_path)
        best_models[column] = loaded_model
    except FileNotFoundError:
        pass #st.write(f"Model file not found for column: {column}")

# Streamlit App
st.title('Time Series Forecasting App')

# Allow user to select the column
selected_column = st.selectbox('Select Column:', df.columns)

# Allow user to input number of points for forecasting
num_future_points = st.slider('Number of Future Points', min_value=1, max_value=30, value=12, step=1,
                              format="%d", help="Select the number of future points for forecasting.")

# Forecast future values
if st.button('Forecast'):
    future_predictions = {}
    try:
        # Forecast future values
        model = best_models[selected_column]
        future_forecast =model.forecast(steps=num_future_points)
        future_predictions[selected_column] = future_forecast
        
        # Plotting the forecast
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[selected_column], label='Original Data')
        plt.plot(range(len(df), len(df) + len(future_forecast)), future_forecast, label='Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Forecast')
        plt.legend()
        
        # Save the plot
        plt_path = "forecast_plot.png"
        plt.savefig(plt_path)
        st.image(plt_path, caption='Forecast Plot')
        
        # Display predictions table
        if future_predictions:
            st.write("Future Predictions:")
            forecast_df = pd.DataFrame(future_predictions)
            st.table(forecast_df)
        else:
            st.write("No predictions available.")
        
        # Save the predictions table
        forecast_df.to_csv("future_predictions.csv", index=False)
        
    except Exception as e:
        st.write(f"An error occurred while forecasting for column {selected_column}: {e}")
