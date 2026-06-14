import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Score': [20, 30, 35, 50, 55, 65, 75, 85]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted Scores:", predictions)
print("MSE:", mean_squared_error(y_test, predictions))
