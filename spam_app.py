from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. ٹریننگ ڈیٹا (0 = نارمل میسج، 1 = اسپیم میسج)
emails = [
    "Hey, are we still meeting for lunch today?",
    "CONGRATULATIONS! You have won a free lottery prize of $1000!",
    "Can you send me the project  by tomorrow?",
    "URgent! Click this link to claim your free cash offer now.",
    "Dear friend, let's play football this evening."
]
labels = [0, 1, 0, 1, 0]

# 2. ایڈوانسڈ TF-IDF Vectorizer کا استعمال
# یہاں ہم نے 'stop_words' بھی شامل کیے ہیں تاکہ فالتو انگریزی الفاظ خودکار طور پر نکل جائیں
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(emails)

# 3. Naive Bayes ماڈل بنائیں اور ٹرین کریں
model = MultinomialNB()
model.fit(X_train, labels)

# 4. بالکل نئے میسجز پر ٹیسٹ کریں
new_emails = [
    "Claim your free lottery prize money now!", 
    "Are you free to talk about the project?"
]

# نئے میسجز کو TF-IDF فارمیٹ میں بدلیں
X_test = vectorizer.transform(new_emails)

# پیشگوئی (Prediction) کریں
predictions = model.predict(X_test)

# 5. نتائج پرنٹ کریں
print("🛡️ TF-IDF اور Naive Bayes اسپیم ڈیٹیکٹر تیار ہے!\n")
for email, prediction in zip(new_emails, predictions):
    result = "Spam (اسپیم/فراڈ)" if prediction == 1 else "Ham (نارمل میسج)"
    print(f"📧 میسج: '{email}'\n🎯 تشخیص: یہ ایک {result} ہے۔\n")
