import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib  # ใช้บันทึกโมเดล

#โหลดข้อมูล Titanic (สามารถเปลี่ยนเป็นชุดข้อมูลอื่นได้)
df = pd.read_csv("Dataset/Titanic/train.csv")

#เลือก Feature ที่ใช้
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df.dropna()

#แปลงข้อมูลตัวอักษรเป็นตัวเลข
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

#แยกตัวแปรต้น (X) และตัวแปรเป้าหมาย (y)
X = df[features]
y = df["Survived"]

#แบ่งชุดข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#สร้างโมเดล Random Forest
model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

#การทำ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean())

#ทดสอบโมเดล
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy ของโมเดล: {accuracy:.2f}")

#บันทึกโมเดลเป็น .pkl
joblib.dump(model, "models/titanic_model.pkl")

print("✅ เทรนและบันทึกโมเดลเรียบร้อยแล้ว!")

from sklearn.preprocessing import StandardScaler

