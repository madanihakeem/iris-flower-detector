from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. ڈیٹا لوڈ کریں اور ماڈل ٹرین کریں
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
y_binary = np.where(y > np.median(y), 1, 0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_binary)

# 2. ماڈل سے فیچرز کی اہمیت (Feature Importances) حاصل کریں
importances = model.feature_importances_
feature_names = diabetes.feature_names

# 3. اسے ایک خوبصورت ٹیبل (DataFrame) میں تبدیل کریں اور ترتیب دیں
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 4. رزلٹ کو پرنٹ کریں
print("📋 ماڈل کے نزدیک تمام ٹیسٹس کی اہمیت کی لسٹ:")
print(feature_df.to_string(index=False))

# 5. خوبصورت گراف بنائیں
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('🌲 Random Forest: Feature Importance for Diabetes 🌲', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Medical Tests / Features', fontsize=12)
plt.show()
