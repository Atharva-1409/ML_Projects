import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Placement_Data_Full_Class.csv")

print(df.isnull().sum())

le = LabelEncoder()

df["gender"] = le.fit_transform(df["gender"])
df["workex"] = le.fit_transform(df["workex"])
df["status"] = le.fit_transform(df["status"])

X = df[["ssc_p", "hsc_p", "degree_p", "mba_p"]]
y = df["status"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Predicted Values:")
print(y_pred)

print("Actual Values:")
print(y_test.values)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))