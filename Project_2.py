import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("loan_prediction.csv")

print(df.isnull().sum())

df = df.dropna()

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Education"] = le.fit_transform(df["Education"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

X = df[["ApplicantIncome", "LoanAmount", "Credit_History"]]
y = df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

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
