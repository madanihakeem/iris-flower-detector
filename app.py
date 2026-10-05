@st.cache_resource
def train_spam_model():
    try:
        # اصلی کینگل فائل لوڈ کرنا
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.dropna(how="any", axis=1)
        if 'v1' in df.columns and 'v2' in df.columns:
            df.columns = ['label', 'text']
    except FileNotFoundError:
        # اگر فائل نہ ملے تو صارف کو وارننگ دکھانا
        st.sidebar.error("❌ 'spam.csv' فائل نہیں ملی! فرضی ڈیٹا استعمال ہو رہا ہے۔")
        data = {
            'text': [
                'Win a free iPhone now! Click here to claim your prize cash money.', 
                'Hey, are we still meeting tomorrow for lunch?',
                'URGENT: Verify your password and account details immediately.', 
                'Can you please send me the final project report by tomorrow?',
                'Get rich quick! Make money now. Cash reward winner!', 
                'The project deadline is extended to next Friday.'
            ],
            'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
        }
        df = pd.DataFrame(data)
    
    X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # token_pattern شامل کیا ہے تاکہ نمبرز اور پرائز منی (جیسے 900) کو بھی ماڈل نوٹ کرے
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    return model, tfidf
