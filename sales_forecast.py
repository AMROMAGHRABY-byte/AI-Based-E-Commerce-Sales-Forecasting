import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ----------------------------------------------------------
#  AI-Based E-Commerce Sales Forecasting
#  Author: Amr Al Maghraby
#  Description:
#  This model predicts future e-commerce sales based on
#  product price, discount, rating, and month data.
# ----------------------------------------------------------

# --- Load the dataset ---
data = pd.read_csv("sales_data.csv")

# --- Define features and target variable ---
X = data[["price", "discount", "rating", "month"]]
y = data["sales"]

# --- Train a Linear Regression model ---
model = LinearRegression()
model.fit(X, y)

# --- Make predictions on the training data ---
y_pred = model.predict(X)

# --- Evaluate model performance ---
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# --- Predict future sales (example data for testing) ---
future_data = pd.DataFrame({
    "price": [700, 800, 900],
    "discount": [15, 10, 20],
    "rating": [4.8, 4.7, 4.9],
    "month": [1, 2, 3]
})

future_pred = model.predict(future_data)
print("\n🔮 Predicted Future Sales:")
for i, val in enumerate(future_pred, 1):
    print(f"Month {i}: {val:.2f} units")

# --- Plot actual vs predicted sales ---
plt.figure(figsize=(8, 5))
plt.plot(y.values, label="Actual Sales", marker='o')
plt.plot(y_pred, label="Predicted Sales", linestyle='--', marker='x')
plt.title("Actual vs Predicted Sales (E-Commerce Forecasting)")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save plot as an image ---
plt.savefig("sales_forecast_plot.png", dpi=300)
plt.show()

print("\n📈 Plot saved as 'sales_forecast_plot.png'")
