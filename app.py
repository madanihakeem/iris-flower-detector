import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="ای میل اسپیم کلاسیفائر", page_icon="✉️", layout="centered")

st.title("✉️ ای میل اسپیم کلاسیفائر (Spam Classifier)")
st.write("نیچے دیے گئے باکس میں اپنی ای میل کا متن پیسٹ کریں اور چیک کریں کہ وہ اسپیم ہے یا جائز۔")

# بغیر کسی بیک اپ کے براہِ راست اصلی فائل لوڈ کرنا
@st.cache_resource
def train_spam_model():
    # اگر فائل کا نام مختلف ہے تو یہاں تبدیل کریں
    df = pd.read_csv(r'C:\Users\dell\Documents\py\spam.csv', encoding='latin-1')

    
    # فالتو کالمز صاف کرنا
    df = df.dropna(how="any", axis=1)
    if 'v1' in df.columns and 'v2' in df.columns:
        df.columns = ['label', 'text']
        
    X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # الفاظ اور ہندسوں دونوں کو بہتر طریقے سے پڑھنے کے لیے
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
    st.stop() # اگر فائل نہ ملے تو ایپ یہیں رک جائے گی

# یوزر انٹرفیس
user_input = st.text_area("ای میل کا متن یہاں لکھیں (انگریزی میں):", height=150, placeholder="Type your email here...")

if st.button("چیک کریں (Check Email)"):
    if user_input.strip() == "":
        st.warning("⚠️ براہ کرم پہلے باکس میں کوئی متن لکھیں!")
    else:
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)
        
        st.subheader("نتيجہ (Result):")
        # سٹرنگ کو صاف کر کے میچ کرنا تاکہ کوئی اسپیس کا مسئلہ نہ ہو
        if prediction[0].strip().lower() == 'spam':
            st.error("❌ یہ ایک اسپیم (Spam) ای میل لگ رہی ہے! محتاط رہیں۔")
        else:
            st.success("✅ یہ ایک جائز اور محفوظ ای میل (Ham) ہے۔")

st.markdown("---")
st.caption("Developed with ❤️ using Python & Streamlit")
