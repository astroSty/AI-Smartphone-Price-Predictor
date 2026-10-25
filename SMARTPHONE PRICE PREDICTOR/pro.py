import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="📱 Smartphone Price Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("smartphones.csv")
    df.replace("", np.nan, inplace=True)
    df.dropna(subset=['price'], inplace=True)

    bool_cols = ['has_5g', 'has_nfc', 'has_ir_blaster', 'fast_charging_available', 'extended_memory_available']
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    numeric_cols = ['rating', 'fast_charging', 'ram_capacity', 'internal_memory',
                    'refresh_rate', 'num_rear_cameras', 'num_front_cameras',
                    'primary_camera_rear', 'primary_camera_front',
                    'extended_upto', 'resolution_width', 'resolution_height']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    label_cols = ['brand_name', 'model', 'processor_brand', 'os']
    le_dict = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict

# Load the data
df, le_dict = load_data()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Visualization", "Train Model", "Predict"])

# ====================== PAGE 1 ======================
if page == "Dataset Overview":
    st.title("📱 Smartphone Dataset Overview")
    st.write(df.head(15))

    st.metric("📦 Total Samples", len(df))
    st.metric("🧬 Features", df.shape[1])

    st.subheader("🛠️ Filter Dataset")
    with st.expander("🔎 Filter Options"):
        brand_filter = st.multiselect("Select Brands", options=le_dict['brand_name'].classes_, default=le_dict['brand_name'].classes_)
        ram_range = st.slider("Select RAM Range (GB)", int(df.ram_capacity.min()), int(df.ram_capacity.max()), (6, 12))
        price_range = st.slider("Select Price Range", int(df.price.min()), int(df.price.max()), (10000, 70000))

    brand_idx = le_dict['brand_name'].transform(brand_filter)
    filtered_df = df[
        (df['brand_name'].isin(brand_idx)) &
        (df['ram_capacity'] >= ram_range[0]) &
        (df['ram_capacity'] <= ram_range[1]) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ]

    st.dataframe(filtered_df)

# ====================== PAGE 2 ======================
elif page == "Visualization":
    st.title("📊 Visual Data Analysis")

    st.subheader("💰 Price Distribution")
    fig = px.histogram(df, x='price', nbins=30, color_discrete_sequence=['indianred'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Price vs RAM")
    fig = px.scatter(df, x='ram_capacity', y='price', color='brand_name', title="Price vs RAM (by Brand)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📶 5G Support Distribution")
    fig = px.pie(df, names='has_5g', title="5G Availability")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📱 Screen Size vs Price")
    fig = px.scatter(df, x='screen_size', y='price', color='refresh_rate', title="Screen Size vs Price")
    st.plotly_chart(fig, use_container_width=True)

# ====================== PAGE 3 ======================
elif page == "Train Model":
    st.title("🎯 Model Training and Evaluation")

    X = df.drop(columns=['price', 'model'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.success(f"✅ Model Trained Successfully")
    st.metric("🔢 R² Score", f"{r2:.2f}")
    st.metric("📉 RMSE", f"₹{rmse:,.0f}")

    # Feature importance
    st.subheader("📊 Feature Importance")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# ====================== PAGE 4 ======================
elif page == "Predict":
    st.title("🔮 Predict Smartphone Price")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Brand", le_dict['brand_name'].classes_)
            processor = st.selectbox("Processor", le_dict['processor_brand'].classes_)
            os = st.selectbox("Operating System", le_dict['os'].classes_)
            ram = st.slider("RAM (GB)", 2, 16, 8)
            internal = st.slider("Storage (GB)", 32, 512, 128)
            battery = st.slider("Battery (mAh)", 3000, 6000, 5000)
            fast_charge = st.slider("Fast Charging (W)", 10, 150, 33)

        with col2:
            screen = st.slider("Screen Size (in)", 5.0, 7.0, 6.5)
            refresh = st.slider("Refresh Rate (Hz)", 60, 144, 120)
            rear_cam = st.slider("Rear Camera (MP)", 12, 200, 50)
            front_cam = st.slider("Front Camera (MP)", 5, 50, 16)
            rating = st.slider("User Rating", 50, 100, 85)
            has_5g = st.checkbox("5G", True)
            has_nfc = st.checkbox("NFC", False)

        submitted = st.form_submit_button("Predict Price")

        if submitted:
            # Prepare input
            input_df = pd.DataFrame([{
                'brand_name': le_dict['brand_name'].transform([brand])[0],
                'processor_brand': le_dict['processor_brand'].transform([processor])[0],
                'os': le_dict['os'].transform([os])[0],
                'ram_capacity': ram,
                'internal_memory': internal,
                'battery_capacity': battery,
                'fast_charging': fast_charge,
                'fast_charging_available': True,
                'screen_size': screen,
                'refresh_rate': refresh,
                'primary_camera_rear': rear_cam,
                'primary_camera_front': front_cam,
                'rating': rating,
                'has_5g': has_5g,
                'has_nfc': has_nfc,
                'has_ir_blaster': False,
                'num_cores': 8,
                'processor_speed': 2.5,
                'num_rear_cameras': 3,
                'num_front_cameras': 1,
                'extended_memory_available': False,
                'extended_upto': 1024,
                'resolution_width': 1080,
                'resolution_height': 2400
            }])

            # Train model and predict
            X = df.drop(columns=['price', 'model'])
            y = df['price']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Ensure input columns match model input
            input_df = input_df[X.columns]

            prediction = model.predict(input_df)[0]
            st.success(f"📱 Predicted Price: ₹{prediction:,.0f}")
