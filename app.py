import streamlit as st
import pickle
import json
import numpy as np

# Load model and columns
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    data = json.load(f)
    columns = data["data_columns"]
    locations = [c for c in columns[3:]]  # first 3 are sqft, bath, bhk

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    loc = location.lower()
    if loc in columns:
        x[columns.index(loc)] = 1
    return round(model.predict([x])[0], 2)

# UI
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Estimate property prices in Bangalore based on location and features.")

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", sorted(locations))
    sqft = st.number_input("Total Square Feet", min_value=300, max_value=30000, value=1000)
with col2:
    bhk = st.slider("BHK (Bedrooms)", 1, 10, 2)
    bath = st.slider("Bathrooms", 1, 10, 2)

if st.button("Predict Price", type="primary"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"### Estimated Price: ₹ {price} Lakhs")
    st.caption(f"≈ ₹ {round(price * 100000 / sqft):,} per sq ft")
