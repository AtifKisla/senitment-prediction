
!pip install pytrends scikit-learn pandas

from pytrends.request import TrendReq
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

cities = [
    "Paris","Rome","Barcelona","Amsterdam","Prague",
    "Vienna","Budapest","Lisbon","Berlin","Athens"
]

pytrends = TrendReq()

all_data = []

for city in cities:

    print("Fetching:", city)

    try:

        pytrends.build_payload([city], timeframe="today 5-y")

        df_city = pytrends.interest_over_time()

        if not df_city.empty:

            df_city = df_city.drop(columns=["isPartial"])

            df_city = df_city.rename(columns={city:"trend_score"})

            df_city["city"] = city

            all_data.append(df_city)

    except Exception as e:

        print("Error:", e)

    time.sleep(5)

df = pd.concat(all_data)

df = df.reset_index()

df["popularity_class"] = pd.qcut(
    df["trend_score"],
    q=3,
    labels=["low","medium","high"]
)

df["city_code"] = df["city"].astype("category").cat.codes

df["month"] = pd.to_datetime(df["date"]).dt.month

df["popularity_class"] = df["popularity_class"].map({
    "low":0,
    "medium":1,
    "high":2
})

df["future_trend"] = df.groupby("city")["trend_score"].shift(-4)

df["future_class"] = pd.qcut(
    df["future_trend"],
    q=3,
    labels=[0,1,2]
)

df = df.dropna()
X = df[["trend_score","city_code","month"]]
y = df["future_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

df.to_csv("google_trends_tourism_dataset.csv", index=False)

