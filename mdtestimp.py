import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# تمام غیر ضروری وارننگز کو اسکرین پر چھپانے کے لیے
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 1. ایپ کا ٹائٹل اور تعارف (تاکہ براؤزر میں نظر آئے)
st.title("🩸 Diabetes Feature Importance")
st.write("یہ گراف دکھاتا ہے کہ شوگر کی پیشگوئی کے لیے کون سا میڈیکل ٹیسٹ سب سے زیادہ اہم ہے۔")

# 2. ڈیٹا لوڈ کریں اور ماڈل ٹرین کریں
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
y_binary = np.where(y > np.median(y), 1, 0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_binary)

# 3. فیچرز کی اہمیت حاصل کریں
importances = model.ensemble_classifier_ if hasattr(model, 'ensemble_classifier_') else model.feature_importances_
feature_names = diabetes.feature_names

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 4. خوبصورت اور وارننگ سے پاک گراف بنائیں
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# یہاں ہم نے وارننگ ختم کرنے کے لیے hue اور legend سیٹ کر دیا ہے
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_df, 
    hue='Feature', 
    palette='viridis', 
    legend=False,
    ax=ax
)

# ایموجی ہٹا دیے تاکہ فونٹ کا مسئلہ نہ آئے
plt.title('Random Forest: Feature Importance for Diabetes', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Medical Tests / Features', fontsize=12)

# 5. اسٹریم لٹ کے طریقے سے گراف کو اسکرین پر دکھائیں
st.pyplot(fig)
