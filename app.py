import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. ماڈل کو بیک اینڈ پر لوڈ اور ٹرین کریں
@st.cache_resource # اس سے ماڈل بار بار ٹرین نہیں ہوگا اور ایپ تیز چلے گی
def load_and_train_model():
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(iris.data, iris.target)
    return model, iris.target_names

model, target_names = load_and_train_model()

# 2. ویب ایپ کا فرنٹ اینڈ (UI Design)
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸", layout="centered")

st.title("🌸 پھولوں کی پہچان کرنے والی سمارٹ ایپ")
st.write("پھول کی پتیوں کا سائز درج کریں، اور ہمارا مشین لرننگ ماڈل آپ کو پھول کا نام بتائے گا۔")

st.markdown("---")

# 3. صارف سے ان پٹ لینے کے لیے سلائیڈرز (Sliders)
st.subheader("📏 پتیوں کا سائز منتخب کریں:")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

st.markdown("---")

# 4. بٹن اور پیشگوئی (Prediction Button)
if st.button("پھول کی قسم معلوم کریں ✨", type="primary"):
    # صارف کے ڈیٹا کو ایرے میں بدلیں
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # ماڈل سے پیشگوئی کروائیں
    prediction = model.predict(user_input)[0]
    predicted_flower = target_names[prediction]
    
    # رزلٹ کو خوبصورت انداز میں دکھائیں
    st.balloons() # اسکرین پر غبارے اڑانے کے لیے 🎉
    st.success(f"### 🎉 یہ پھول **{predicted_flower.upper()}** ہے!")
