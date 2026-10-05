import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ویب پیج کی بنیادی سیٹنگز
st.set_page_config(page_title="ای میل اسپیم کلاسیفائر", page_icon="✉️", layout="centered")

# ایپ کا عنوان (Title)
st.title("✉️ ای میل اسپیم کلاسیفائر (Spam Classifier)")
st.write("نیچے دیے گئے باکس میں اپنی ای میل کا متن پیسٹ کریں اور چیک کریں کہ وہ اسپیم ہے یا جائز۔")

# 1. ماڈل اور ڈیٹا لوڈ کرنے کا فنکشن (Cache استعمال کیا ہے تاکہ ایپ بار بار لوڈ نہ ہو)
@st.cache_resource
def train_spam_model():
    # اگر آپ کے پاس 'spam.csv' فائل موجود ہے تو وہ لوڈ ہو جائے گی، ورنہ فرضی ڈیٹا چلے گا
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.dropna(how="any", axis=1)
        if 'v1' in df.columns and 'v2' in df.columns:
            df.columns = ['label', 'text']
    except FileNotFoundError:
        # اگر فائل نہ ملے تو ایپ کو چلتا رکھنے کے لیے بیک اپ ڈیٹا
        data = {
            'text': [
                'Win a free iPhone now! Click here.', 'Hey, are we still meeting tomorrow?',
                'URGENT: Verify your password.', 'Can you send me the report?',
                'Get rich quick! Make money now.', 'The project deadline is extended.'
            ],
            'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
        }
        df = pd.DataFrame(data)
    
    # ماڈل کی ٹریننگ
    X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    tfidf = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, tfidf

# ماڈل کو بیک اینڈ پر تیار کرنا
model, tfidf = train_spam_model()

# ========================================================
# 2. یوزر انٹرفیس (UI) اور ان پٹ
# ========================================================

# ٹیکسٹ ایریا جہاں صارف ای میل لکھے گا
user_input = st.text_area("ای میل کا متن یہاں لکھیں (انگریزی میں):", height=150, placeholder="Type your email here...")

# بٹن دبانے پر کارروائی
if st.button("چیک کریں (Check Email)"):
    if user_input.strip() == "":
        st.warning("⚠️ براہ کرم پہلے باکس میں کوئی متن لکھیں!")
    else:
        # ان پٹ ٹیکسٹ کو نمبرز میں بدلنا اور پریڈکشن کرنا
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        
        # رزلٹ کو خوبصورت انداز میں دکھانا
        st.subheader("نتيجہ (Result):")
        if prediction.lower() == 'spam':
            st.error("❌ یہ ایک اسپیم (Spam) ای میل لگ رہی ہے! محتاط رہیں۔")
        else:
            st.success("✅ یہ ایک جائز اور محفوظ ای میل (Ham) ہے۔")

# ایپ کے نیچے کریڈٹ لائن
st.markdown("---")
st.caption("Developed with ❤️ using Python & Streamlit")
