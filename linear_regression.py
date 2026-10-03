import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLR

# ==========================================
# 1. FROM-SCRATCH MODEL IMPLEMENTATION
# ==========================================
class NumpyLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.cost_history = []

    def fit(self, X, y):
        # m = number of samples, n = number of features
        m, n = X.shape
        # Initialize weights (theta) to zeros
        self.weights = np.zeros(n)
        
        for i in range(self.iterations):
            # Hypothesis: h(X) = Xθ
            predictions = np.dot(X, self.weights)
            
            # Error = h(X) - y
            error = predictions - y
            
            # Cost Function (MSE / 2): J(θ) = 1/(2m) * sum(error^2)
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # Gradient: (1/m) * X^T * error
            gradients = (1 / m) * np.dot(X.T, error)
            
            # Update Rule: θ = θ - α * gradient
            self.weights -= self.learning_rate * gradients
            
    def predict(self, X):
        return np.dot(X, self.weights)

# ==========================================
# 2. METRICS (MANUAL IMPLEMENTATION)
# ==========================================
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# ==========================================
# 3. DATA PREPARATION
# ==========================================
# Load California Housing Dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for Gradient Descent stability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add intercept term (column of 1s) to X for the bias weight (theta_0)
X_train_final = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_final = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
# Train Custom Model
custom_model = NumpyLinearRegression(learning_rate=0.1, iterations=1000)
custom_model.fit(X_train_final, y_train)
custom_predictions = custom_model.predict(X_test_final)

# Manual Metrics
custom_mse = calculate_mse(y_test, custom_predictions)
custom_r2 = calculate_r2(y_test, custom_predictions)

# Train Scikit-Learn Model for Comparison
# (Note: fit_intercept=False because we already added the column of 1s)
sk_model = SklearnLR(fit_intercept=False)
sk_model.fit(X_train_final, y_train)
sk_predictions = sk_model.predict(X_test_final)

sk_mse = calculate_mse(y_test, sk_predictions)
sk_r2 = calculate_r2(y_test, sk_predictions)

# ==========================================
# 5. RESULTS & VISUALIZATION
# ==========================================
print("=== Performance Comparison ===")
print(f"Custom Model -> MSE: {custom_mse:.6f}, R²: {custom_r2:.6f}")
print(f"Sklearn Model -> MSE: {sk_mse:.6f}, R²: {sk_r2:.6f}")

print("\n=== Coefficients Comparison ===")
print(f"Custom Weights:  {np.round(custom_model.weights, 4)}")
print(f"Sklearn Weights: {np.round(sk_model.coef_, 4)}")

# Plotting the Cost Function Convergence
plt.figure(figsize=(8, 5))
plt.plot(range(custom_model.iterations), custom_model.cost_history, color='blue', linewidth=2)
plt.title('Cost Function Convergence (Gradient Descent)')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE / 2)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()