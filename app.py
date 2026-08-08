import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and feature list
model = joblib.load(r"C:\Users\linji\OneDrive\Desktop\IDXExchange\ca-price-prediction\data\best_model.joblib")
feature_cols = joblib.load(r"C:\Users\linji\OneDrive\Desktop\IDXExchange\ca-price-prediction\data\feature_cols.joblib")

# App title
st.title("🏠 California Home Price Predictor")
st.markdown("Enter property details below to get an estimated close price.")
st.divider()

# User inputs
col1, col2 = st.columns(2)

with col1:
    living_area    = st.number_input("Living Area (sq ft)",    min_value=100,   max_value=20000, value=1500)
    bedrooms       = st.number_input("Bedrooms",               min_value=0,     max_value=20,    value=3)
    bathrooms      = st.number_input("Bathrooms",              min_value=0,     max_value=20,    value=2)
    lot_size       = st.number_input("Lot Size (sq ft)",       min_value=100,   max_value=500000,value=5000)
    year_built     = st.number_input("Year Built",             min_value=1800,  max_value=2025,  value=1990)
    stories        = st.number_input("Stories",                min_value=1,     max_value=5,     value=1)

with col2:
    garage_spaces  = st.number_input("Garage Spaces",          min_value=0,     max_value=10,    value=2)
    parking_total  = st.number_input("Total Parking Spaces",   min_value=0,     max_value=20,    value=2)
    days_on_market = st.number_input("Days on Market",         min_value=0,     max_value=500,   value=30)
    assoc_fee      = st.number_input("HOA Fee ($/month)",      min_value=0,     max_value=5000,  value=0)
    latitude       = st.number_input("Latitude",               min_value=32.0,  max_value=42.0,  value=34.05)
    longitude      = st.number_input("Longitude",              min_value=-125.0,max_value=-114.0,value=-118.25)

st.divider()

# Boolean features
st.subheader("Property Features")
col3, col4, col5 = st.columns(3)
with col3:
    has_garage  = st.checkbox("Attached Garage")
    has_fireplace = st.checkbox("Fireplace")
with col4:
    has_pool    = st.checkbox("Private Pool")
    has_view    = st.checkbox("View")
with col5:
    new_construction = st.checkbox("New Construction")
    has_hoa     = st.checkbox("Has HOA")

st.divider()

# Predict button
if st.button("🔍 Predict Price", type="primary"):

    # Build input row with all features
    input_data = {col: 0 for col in feature_cols}

    # Fill in user inputs
    input_data["LivingArea"]              = living_area
    input_data["BedroomsTotal"]           = bedrooms
    input_data["BathroomsTotalInteger"]   = bathrooms
    input_data["LotSizeSquareFeet"]       = lot_size
    input_data["YearBuilt"]               = year_built
    input_data["Stories"]                 = stories
    input_data["GarageSpaces"]            = garage_spaces
    input_data["ParkingTotal"]            = parking_total
    input_data["DaysOnMarket"]            = days_on_market
    input_data["AssociationFee"]          = assoc_fee
    input_data["Latitude"]                = latitude
    input_data["Longitude"]               = longitude
    input_data["AttachedGarageYN"]        = int(has_garage)
    input_data["FireplaceYN"]             = int(has_fireplace)
    input_data["PoolPrivateYN"]           = int(has_pool)
    input_data["ViewYN"]                  = int(has_view)
    input_data["NewConstructionYN"]       = int(new_construction)
    input_data["HasHOA"]                  = int(has_hoa)
    input_data["LotLivingRatio"]          = lot_size / max(living_area, 1)

    # Create dataframe for prediction
    input_df = pd.DataFrame([input_data])[feature_cols]

    # Predict
    predicted_price = model.predict(input_df)[0]

    # Display result
    st.success(f"### 💰 Estimated Close Price: ${predicted_price:,.0f}")

    # Show confidence range (±10%)
    low  = predicted_price * 0.90
    high = predicted_price * 1.10
    st.info(f"Typical range: ${low:,.0f} — ${high:,.0f}")

    # Show input summary
    with st.expander("View input summary"):
        st.write({
            "Living Area": f"{living_area} sq ft",
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Lot Size": f"{lot_size} sq ft",
            "Year Built": year_built,
            "Location": f"{latitude}, {longitude}",
        })