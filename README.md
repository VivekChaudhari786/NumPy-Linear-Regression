# Linear Regression from Scratch using NumPy

This project implements a Multiple Linear Regression model entirely from scratch using only NumPy. It includes the cost function, gradient descent optimization loop, and mathematical derivations without relying on machine learning libraries like scikit-learn for the core training logic.

## 🎯 Project Requirements Fulfilled

* \[x\] Implemented hypothesis function, MSE cost function, and gradient descent using matrix operations.

* \[x\] Supports both single-feature and multi-feature linear regression.

* \[x\] Trained on the real-world California Housing Dataset.

* \[x\] Evaluated using manually computed Mean Squared Error (MSE) and R² metrics.

* \[x\] Performance compared side-by-side with scikit-learn's `LinearRegression`.

* \[x\] Plotted the cost function decreasing over training iterations to prove convergence.

## 📊 Results & Comparison

Here is the performance comparison between the custom NumPy model and Scikit-Learn's implementation on the test set:

| Model | Mean Squared Error (MSE) | R² Score | 
| ----- | ----- | ----- | 
| **Custom NumPy Model** | 0.555998 | 0.575706 | 
| **Scikit-Learn Model** | 0.555892 | 0.575788 | 

*(Note: The coefficients and R² scores of both models are nearly identical, proving the correctness of the custom implementation.)*

**Coefficients Comparison:**
```text
Custom Weights:  [ 2.0719  0.8554  0.1228 -0.2962  0.3407 -0.0022 -0.0409 -0.8942 -0.8672]
Sklearn Weights: [ 2.0719  0.8544  0.1225 -0.2944  0.3393 -0.0023 -0.0408 -0.8969 -0.8698]
```

## 📖 The Math Behind the Model

By avoiding iterative loops over individual data points, NumPy calculates updates efficiently across the entire dataset simultaneously using matrix operations.

### 1. The Hypothesis Function

For a dataset with $n$ features, the prediction is the weighted sum of its features, plus a bias term:

$$
h_\theta(X) = X\theta
$$

Where $X$ is an $m \times (n+1)$ matrix (with a column of 1s prepended for the bias), and $\theta$ is an $(n+1) \times 1$ vector of weights.

### 2. The Cost Function (Mean Squared Error)

To measure the error, we use the Mean Squared Error (MSE), conventionally halved in gradient descent to cancel out the exponent during the derivative calculation.

$$
J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)
$$

### 3. The Gradient

To minimize the cost function, we calculate the partial derivative of the cost function with respect to the weights:

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$

### 4. Gradient Descent Update Rule

We update the weights by taking a step determined by the learning rate $\alpha$ in the opposite direction of the gradient:

$$
\theta = \theta - \alpha \left( \frac{1}{m} X^T (X\theta - y) \right)
$$

## 🛠️ Tech Stack

* **NumPy**: Matrix operations, cost function, and gradient calculations.
* **Pandas & Scikit-learn**: Data loading, preprocessing, train/test splitting, and final baseline comparison.
* **Matplotlib**: Visualizing the cost function convergence.

## 🚀 How to Run

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd <repo-folder-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python linear_regression.py
```