import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. پہلے سے ٹرین شدہ MobileNetV2 ماڈل لوڈ کریں (جو 1000 چیزیں پہچان سکتا ہے)
@st.cache_resource
def load_model():
    # ہم ImageNet کے بنے بنائے وزن (weights) استعمال کر رہے ہیں
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

# 2. ویب ایپ کا انٹرفیس
st.set_page_config(page_title="AI Image Classifier", page_icon="📷")
st.title("📷 سمارٹ امیج کلاسیفائر ایپ")
st.write("کسی بھی چیز (مثلاً بلی، کتا، گاڑی، یا پھل) کی تصویر اپ لوڈ کریں، اور AI اسے پہچان کر دکھائے گا!")

st.markdown("---")

# 3. صارف سے تصویر اپ لوڈ کروانا
uploaded_file = st.file_uploader("تصویر کا انتخاب کریں...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # تصویر کو اسکرین پر دکھائیں
    image = Image.open(uploaded_file)
    st.image(image, caption='آپ کی اپ لوڈ کردہ تصویر', use_container_width=True)
    
    st.write("⏳ AI تصویر کا تجزیہ کر رہا ہے...")
    
    # 4. تصویر کو ماڈل کے سائز (224x224) کے مطابق تیار کرنا
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # 5. ماڈل سے پیشگوئی کروانا
    predictions = model.predict(img_array)
    # رزلٹ کو انسانوں کے سمجھنے والے ناموں میں بدلنا
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    st.markdown("---")
    st.subheader("🎯 AI کے مطابق یہ چیزیں ہو سکتی ہیں:")
    
    # ٹاپ 3 رزلٹس دکھانا
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        # اسکور کو فیصد میں بدلیں
        percentage = score * 100
        st.write(f"**{i+1}. {label.replace('_', ' ').title()}** — {percentage:.2f}% یقین")
        st.progress(int(percentage))
