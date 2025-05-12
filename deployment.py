import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and label encoder
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Water Quality Predictor", layout="centered")

# Custom CSS to remove blank space above title
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💧 Water Quality Predictor")
st.subheader("Enter water quality parameters to predict quality and detect fake reports")

# Input form
with st.form(key='water_quality_form'):
    st.markdown("### Water Quality Parameters")
    
    # Row 1: Type Water Body, State Name, Temp Min, Temp Max, DO Min
    row1 = st.columns(5)
    with row1[0]:
        type_water_body = st.selectbox("Water Body", ["STP", "CANAL", "WTP", "DRAIN"])
    with row1[1]:
        state_name = st.selectbox("State", [
            "ANDHRA PRADESH", "GOA", "GUJARAT", "KERALA", "TRIPURA", "WEST BENGAL", 
            "KARNATAKA", "MAHARASHTRA", "TAMIL NADU", "TELANGANA", "UTTAR PRADESH",
            "BIHAR", "JHARKHAND", "ODISHA", "PUNJAB", "RAJASTHAN", "HARYANA",
            "DELHI", "MADHYA PRADESH", "CHHATTISGARH", "ASSAM", "MEGHALAYA", "MANIPUR"
        ])
    with row1[2]:
        temp_min = st.number_input("Temp Min (°C)", min_value=0.0, max_value=50.0, value=25.0)
    with row1[3]:
        temp_max = st.number_input("Temp Max (°C)", min_value=0.0, max_value=50.0, value=30.0)
    with row1[4]:
        do_min = st.number_input("DO Min (mg/L)", min_value=0.0, max_value=20.0, value=5.0)
    
    # Row 2: DO Max, pH Min, pH Max, Cond Min, Cond Max
    row2 = st.columns(5)
    with row2[0]:
        do_max = st.number_input("DO Max (mg/L)", min_value=0.0, max_value=20.0, value=7.0)
    with row2[1]:
        ph_min = st.number_input("pH Min", min_value=0.0, max_value=14.0, value=7.0)
    with row2[2]:
        ph_max = st.number_input("pH Max", min_value=0.0, max_value=14.0, value=7.5)
    with row2[3]:
        cond_min = st.number_input("Cond Min (µS/cm)", min_value=0.0, max_value=10000.0, value=200.0)
    with row2[4]:
        cond_max = st.number_input("Cond Max (µS/cm)", min_value=0.0, max_value=10000.0, value=500.0)
    
    # Row 3: BOD Min, BOD Max, Nitrate Min, Nitrate Max, Fecal Coliform Min
    row3 = st.columns(5)
    with row3[0]:
        bod_min = st.number_input("BOD Min (mg/L)", min_value=0.0, max_value=500.0, value=5.0)
    with row3[1]:
        bod_max = st.number_input("BOD Max (mg/L)", min_value=0.0, max_value=500.0, value=10.0)
    with row3[2]:
        nitrate_min = st.number_input("Nitrate Min (mg/L)", min_value=0.0, max_value=200.0, value=0.5)
    with row3[3]:
        nitrate_max = st.number_input("Nitrate Max (mg/L)", min_value=0.0, max_value=200.0, value=1.0)
    with row3[4]:
        fecal_coliform_min = st.number_input("Fecal Col Min", min_value=0.0, max_value=1000000.0, value=50.0)
    
    # Row 4: Fecal Coliform Max, Total Coliform Min, Total Coliform Max, Fecal Strep Min, Fecal Strep Max
    row4 = st.columns(5)
    with row4[0]:
        fecal_coliform_max = st.number_input("Fecal Col Max", min_value=0.0, max_value=1000000.0, value=100.0)
    with row4[1]:
        total_coliform_min = st.number_input("Total Col Min", min_value=0.0, max_value=1000000.0, value=100.0)
    with row4[2]:
        total_coliform_max = st.number_input("Total Col Max", min_value=0.0, max_value=1000000.0, value=200.0)
    with row4[3]:
        fecal_strep_min = st.number_input("Fecal Strep Min", min_value=0.0, max_value=100000.0, value=10.0)
    with row4[4]:
        fecal_strep_max = st.number_input("Fecal Strep Max", min_value=0.0, max_value=100000.0, value=20.0)

    submit_button = st.form_submit_button(label="Predict Water Quality")

# Process inputs and predict
if submit_button:
    # Encode inputs
    type_water_body_encoded = le.transform([type_water_body])[0]
    
    # Frequency encoding for State Name
    state_freq = {
        "ANDHRA PRADESH": 0.164103, "GOA": 0.015385, "GUJARAT": 0.076923, "KERALA": 0.025641,
        "TRIPURA": 0.035897, "WEST BENGAL": 0.051282, "KARNATAKA": 0.087179, "MAHARASHTRA": 0.158974,
        "TAMIL NADU": 0.092308, "TELANGANA": 0.046154, "UTTAR PRADESH": 0.041026, "BIHAR": 0.005128,
        "JHARKHAND": 0.010256, "ODISHA": 0.020513, "PUNJAB": 0.005128, "RAJASTHAN": 0.005128,
        "HARYANA": 0.005128, "DELHI": 0.005128, "MADHYA PRADESH": 0.005128, "CHHATTISGARH": 0.005128,
        "ASSAM": 0.005128, "MEGHALAYA": 0.005128, "MANIPUR": 0.005128
    }
    state_name_encoded = state_freq.get(state_name, 0.005128)

    # Create input dataframe
    input_data = pd.DataFrame({
        'Type Water Body': [type_water_body_encoded],
        'State Name': [state_name_encoded],
        'Temperature Min': [temp_min],
        'Temperature (°C) Max': [temp_max],
        'Dissolved O2 (mg/L) Min': [do_min],
        'Dissolved O2 (mg/L) Max': [do_max],
        'pH Min': [ph_min],
        'pH Max': [ph_max],
        'Conductivity (µmhos/cm) Min': [cond_min],
        'Conductivity (µmhos/cm) Max': [cond_max],
        'BOD\n(mg/L) Min': [bod_min],
        'BOD\n(mg/L) Max': [bod_max],
        'Nitrate N + Nitrite N\n(mg/L) Min': [nitrate_min],
        'Nitrate N + Nitrite N\n(mg/L) Max': [nitrate_max],
        'Fecal Coliform (MPN/100ml) Min': [fecal_coliform_min],
        'Fecal Coliform (MPN/100ml) Max': [fecal_coliform_max],
        'Total Coliform (MPN/100ml) Min': [total_coliform_min],
        'Total Coliform (MPN/100ml) Max': [total_coliform_max],
        'Fecal Streptococci\n(MPN/100ml) Min': [fecal_strep_min],
        'Fecal Streptococci\n(MPN/100ml) Max': [fecal_strep_max]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    prediction = np.clip(prediction, 0, 1)  # Ensure score is between 0 and 1

    # Determine water quality label and fake report status
    if prediction <= 0.4:
        quality_label = "Poor"
        fake_status = "⚠️ Potentially Fake Report"
        st.error(fake_status)
    elif prediction <= 0.7:
        quality_label = "Moderate"
        fake_status = "✅ Valid Report"
        st.success(fake_status)
    else:
        quality_label = "Good"
        fake_status = "✅ Valid Report"
        st.success(fake_status)

    # Display results
    st.markdown("### Prediction Results")
    st.write(f"**Water Quality Score**: {prediction:.2f}")
    st.write(f"**Water Quality Label**: {quality_label}")
