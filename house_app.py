import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. بیک اینڈ پر رینڈم فارسٹ ماڈل کو ٹرین کریں
@st.cache_resource
def train_housing_model():
    housing = fetch_california_housing()
    # وقت بچانے اور ایپ کو تیز رکھنے کے لیے ہم 20 درخت (trees) استعمال کر رہے ہیں
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(housing.data, housing.target)
    return model

model = train_housing_model()

# 2. ویب ایپ کا انٹرفیس (UI Design)
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 کیلیفورنیا ہاؤس پرائس پریڈکٹر")
st.write("گھر اور علاقے کی خصوصیات درج کریں، اور ہمارا مشین لرننگ ماڈل گھر کی متوقع قیمت بتائے گا۔")

st.markdown("---")
st.subheader("📋 گھر اور علاقے کی معلومات:")

# ڈیٹا سیٹ کی رینج کے مطابق سلائیڈرز اور ان پٹ باکسز
med_inc = st.number_input("علاقے کی اوسط آمدنی (MedInc - لاکھوں ڈالرز میں)", 0.5, 15.0, 3.8)
house_age = st.slider("گھر کی عمر (House Age - سالوں میں)", 1, 52, 28)
ave_rooms = st.slider("اوسط کمرے (Avg Rooms)", 1, 10, 5)
ave_bedrms = st.slider("اوسط بیڈ رومز (Avg Bedrooms)", 1, 5, 1)
population = st.number_input("علاقے کی آبادی (Population)", 3, 35000, 1400)
ave_occup = st.number_input("گھر میں اوسط افراد (Avg Occupancy)", 1.0, 10.0, 3.0)
latitude = st.number_input("طول بلد (Latitude)", 32.5, 42.5, 35.6)
longitude = st.number_input("عرض بلد (Longitude)", -124.3, -114.3, -119.5)

st.markdown("---")

# 3. پیشگوئی کا بٹن (Prediction Button)
if st.button("گھر کی متوقع قیمت معلوم کریں ✨", type="primary"):
    # تمام 8 فیچرز کو ایک لسٹ میں اکٹھا کریں
    user_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    
    # ماڈل سے قیمت کی پیشگوئی کروائیں
    predicted_value = model.predict(user_data)[0]
    
    # چونکہ ڈیٹا سیٹ میں 1 یونٹ = $100,000 ہے، اس لیے ہم اسے اصل رقم میں بدلیں گے
    actual_price = predicted_value * 100000
    
    # رزلٹ دکھائیں
    st.balloons()
    st.success(f"### 💵 اس گھر کی متوقع قیمت: **${actual_price:,.2f}** ہے!")
