# =========================
# 1️⃣ IMPORTS
# =========================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 2️⃣ LOAD DATA
# =========================

df = pd.read_csv("trends_monthly_v2_clean.csv")

df["date"] = pd.to_datetime(df["date"])

# =========================
# 3️⃣ FEATURE ENGINEERING
# =========================

# şehir encode
df["city_code"] = df["city"].astype("category").cat.codes

# ay bilgisi
df["month"] = df["date"].dt.month

# =========================
# 4️⃣ TARGET (FUTURE)
# =========================

# t → t+1
df["future_trend"] = df.groupby("city")["trend"].shift(-1)

# sınıflandırma (3 class)
df["future_class"] = pd.qcut(
    df["future_trend"],
    q=3,
    labels=[0,1,2]
)

# NaN temizle
df = df.dropna()

# =========================
# 5️⃣ FEATURES / TARGET
# =========================

X = df[["trend", "city_code", "month"]]
y = df["future_class"]

# =========================
# 6️⃣ TIME-BASED SPLIT
# =========================

train = df[df["date"] < "2025-01-01"]
test  = df[df["date"] >= "2025-01-01"]

X_train = train[["trend", "city_code", "month"]]
y_train = train["future_class"]

X_test = test[["trend", "city_code", "month"]]
y_test = test["future_class"]

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# =========================
# 7️⃣ MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 8️⃣ PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# 9️⃣ EVALUATION
# =========================

print("\n🎯 ACCURACY:", accuracy_score(y_test, y_pred))

print("\n📊 CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))
