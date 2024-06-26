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

    return model, scaler, X_train.columns  # Return scaler and training feature names


def main():
    st.title('California Housing Prices Prediction')

    # Load the data
    df = load_data()

    # Preprocess the data
    df = preprocess_data(df)

    # Train the model and get model, scaler, and feature names
    model, scaler, feature_names = train_model(df)

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

    # Select location options
    location_options = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    selected_locations = st.multiselect('Select Locations', location_options)

    # Convert selected locations to categorical features
    location_features = {location: 1 if location in selected_locations else 0 for location in location_options}

    # Create user input DataFrame including location features
    user_input = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        **location_features,  # Include location features in the DataFrame
        'bedrooms_ratio': [total_bedrooms / total_rooms if total_rooms > 0 else 0],  # Calculate bedrooms_ratio
        'households_rooms': [total_rooms / households if households > 0 else 0]  # Calculate households_rooms
    })

    # Ensure user input DataFrame has the same columns as the training features
    user_input = user_input.reindex(columns=feature_names, fill_value=0)

    # Scale user input using the same scaler used for training
    user_input_scaled = scaler.transform(user_input)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(user_input_scaled)
        st.success(f'Predicted House Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
