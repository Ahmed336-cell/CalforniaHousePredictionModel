import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_resource
def load_data():
    df = pd.read_csv('output.csv')  # Update with your dataset path
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert boolean columns to integers
    bool_cols = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    df[bool_cols] = df[bool_cols].astype(int)

    # Perform feature engineering with error handling
    df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['households_rooms'] = df['total_rooms'] / df['households']

    # Handle potential division by zero
    df['bedrooms_ratio'].fillna(0, inplace=True)  # Replace NaN with 0
    df['households_rooms'].fillna(0, inplace=True)  # Replace NaN with 0

    return df

# Train a Random Forest model
def train_model(df):
    # Prepare features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def main():
    st.title('California Housing Prices Prediction')

    # Load the data
    df = load_data()

    # Preprocess the data
    df = preprocess_data(df)

    # Train the model and get model and scaler
    model, scaler = train_model(df)

    # Prediction section
    st.header('Make a Prediction')
    st.markdown('Enter values for the features to predict house prices.')

    # Collect user input
    longitude = st.number_input('Longitude')
    latitude = st.number_input('Latitude')
    housing_median_age = st.number_input('Housing Median Age')
    total_rooms = st.number_input('Total Rooms')
    total_bedrooms = st.number_input('Total Bedrooms')
    population = st.number_input('Population')
    households = st.number_input('Households')
    median_income = st.number_input('Median Income')

    # Perform feature engineering with error handling
    if total_rooms > 0 and households > 0:
        bedrooms_ratio = total_bedrooms / total_rooms
        households_rooms = total_rooms / households
    else:
        bedrooms_ratio = 0
        households_rooms = 0

    # Create user input DataFrame
    user_input = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        '<1H OCEAN': 0,  # Placeholder values for categorical features
        'INLAND': 0,
        'ISLAND': 0,
        'NEAR BAY': 0,
        'NEAR OCEAN': 0,
        'bedrooms_ratio': [bedrooms_ratio],  # Add engineered features
        'households_rooms': [households_rooms]
    })

    # Scale user input using the same scaler used for training
    user_input_scaled = scaler.transform(user_input)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(user_input_scaled)
        st.success(f'Predicted House Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
