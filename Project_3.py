import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("insurance.csv")

print(df.isnull().sum())

le = LabelEncoder()

df["smoker"] = le.fit_transform(df["smoker"])

X = df[["age", "bmi", "children", "smoker"]]
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Predicted Values:")
print(y_pred)

print("Actual Values:")
print(y_test.values)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))