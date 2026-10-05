import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="ای میل اسپیم کلاسیفائر", page_icon="✉️", layout="centered")

st.title("✉️ ای میل اسپیم کلاسیفائر (Spam Classifier)")
st.write("نیچے دیے گئے باکس میں اپنی ای میل کا متن پیسٹ کریں اور چیک کریں کہ وہ اسپیم ہے یا جائز۔")

@st.cache_resource
def train_spam_model():
    # پائیتھون فائل کا موجودہ فولڈر تلاش کرنا تاکہ سٹریملٹ کنفیوز نہ ہو
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'spam.csv')
    
    # فائل لوڈ کرنا
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    df = df.dropna(how="any", axis=1)
    if 'v1' in df.columns and 'v2' in df.columns:
        df.columns = ['label', 'text']
        
    X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, tfidf

# ماڈل لوڈ کرنے کی کوشش
try:
    model, tfidf = train_spam_model()
    st.sidebar.success("✅ اصلی Kaggle ڈیٹا سیٹ کامیابی سے لوڈ ہو گیا ہے!")
except FileNotFoundError:
    st.error("❌ غلطی: 'spam.csv' فائل نہیں ملی! براہ کرم چیک کریں کہ فائل اسی فولڈر میں موجود ہے اور اس کا نام درست ہے۔")
    st.stop()

# یوزر انٹرفیس
user_input = st.text_area("ای میل کا متن یہاں لکھیں (انگریزی میں):", height=150, placeholder="Type your email here...")

if st.button("چیک کریں (Check Email)"):
    if user_input.strip() == "":
        st.warning("⚠️ براہ کرم پہلے باکس میں کوئی متن لکھیں!")
    else:
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)
        
        st.subheader("نتيجہ (Result):")
        if prediction[0].strip().lower() == 'spam':

            st.error("❌ یہ ایک اسپیم (Spam) ای میل لگ رہی ہے! محتاط رہیں۔")
        else:
            st.success("✅ یہ ایک جائز اور محفوظ ای میل (Ham) ہے۔")

st.markdown("---")
st.caption("Developed with ❤️ using Python & Streamlit")
