import pandas as pd
import pickle as pk
import streamlit as st

# Load model
# model = pk.load(open('/Users/pratik/anaconda_projects/HousePricePredictions/House_Prediction_model.pkl', 'rb'))
with open('House_/Users/pratik/anaconda_projects/HousePricePredictions/House_Prediction_model.pkl', 'rb') as f:
    model = pk.load(f)


st.header('🏠 Bangalore House Price Predictor')

# Load dataset
data = pd.read_csv('/Users/pratik/anaconda_projects/HousePricePredictions/Bengaluru_House_Data.csv')

# Input fields
loc = st.selectbox('Choose The Location', data['location'].unique())
sqft = st.number_input('Enter Total sqft', min_value=100.0, step=10.0)
bath = st.number_input('Enter number Of Bathrooms', min_value=1, step=1)
balc = st.number_input('Enter number Of Balconys', min_value=0, step=1)
beds = st.number_input('Enter number Of Bedrooms', min_value=1, step=1)

# Create input DataFrame
input_df = pd.DataFrame([[loc, sqft, bath, balc, beds]],
                        columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

# Predict
if st.button("Predict Price"):
    output = model.predict(input_df)
    out_str = '💰 Price of House is ₹' + str(round(output[0] * 100000, 2))
    st.success(out_str)  
