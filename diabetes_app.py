import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. بیک اینڈ پر رینڈم فارسٹ ماڈل کو ٹرین کریں
@st.cache_resource
def train_diabetes_model():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    # ٹارگٹ کو 0 اور 1 میں بدلیں
    y_binary = np.where(y > np.median(y), 1, 0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_binary)
    return model, diabetes.feature_names

model, feature_names = train_diabetes_model()

# 2. ویب ایپ کا انٹرفیس (UI Design)
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩸", layout="centered")

st.title("🩸 شوگر کے خطرے کی پیشگوئی کرنے والی ایپ")
st.write("اپنے جسمانی اور طبی ٹیسٹ کی معلومات درج کریں، اور ہمارا ماڈل شوگر کے امکانی خطرے کی پیشگوئی کرے گا۔")

st.markdown("---")
st.subheader("📋 مریض کا ڈیٹا درج کریں (نارملائزڈ ان پٹ):")

# پبلک ڈیٹا سیٹ میں ڈیٹا پہلے سے نارملائزڈ (-0.2 سے 0.2) ہوتا ہے، اس لیے ہم سلائیڈرز کی رینج اسی حساب سے رکھیں گے
age = st.slider("عمر (Age)", -0.15, 0.15, 0.0)
sex = st.selectbox("جنس (Sex)", options=[-0.04464164, 0.05068012], format_func=lambda x: "مرد (Male)" if x > 0 else "خواتین (Female)")
bmi = st.slider("باڈی ماس انڈیکس / وزن (BMI)", -0.1, 0.15, 0.0)
bp  = st.slider("بلڈ پریشر (Blood Pressure)", -0.1, 0.15, 0.0)

st.markdown("##### دیگر طبی ٹیسٹ (Blood Serum Measurements):")
s1 = st.slider("T-Cells (s1)", -0.1, 0.15, 0.0)
s2 = st.slider("Low-density Lipoproteins (s2)", -0.1, 0.15, 0.0)
s3 = st.slider("High-density Lipoproteins (s3)", -0.1, 0.15, 0.0)
s4 = st.slider("Triglycerides (s4)", -0.1, 0.15, 0.0)
s5 = st.slider("Serum Triglycerides Level (s5)", -0.1, 0.15, 0.0)
s6 = st.slider("Blood Sugar Level (s6)", -0.1, 0.15, 0.0)

st.markdown("---")

# 3. پیشگوئی کا بٹن (Prediction Button)
if st.button("شوگر کا خطرہ چیک کریں ✨", type="primary"):
    # تمام 10 فیچرز کو ایک لسٹ میں اکٹھا کریں
    user_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    
    # ماڈل سے پریڈکشن کروائیں
    prediction = model.predict(user_data)
    
    # رزلٹ دکھائیں
    if prediction[0] == 1:
        st.error("⚠️ **انتباہ (Warning):** ماڈل کے مطابق آپ کو شوگر (Diabetes) کا اعلٰی خطرہ ہو سکتا ہے۔ ڈاکٹر سے رجوع کریں۔")
    else:
        st.success("🎉 **مبارک ہو!** ماڈل کے مطابق آپ محفوظ زون میں ہیں اور شوگر کا خطرہ کم ہے۔")
        st.balloons()
