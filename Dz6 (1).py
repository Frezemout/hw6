import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def hypothesis(X, w):
    return X @ w

def compute_loss(X, y, w):
    m = len(y)
    predictions = hypothesis(X, w)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)
def gradient_step(X, y, w, learning_rate):
    m = len(y)
    predictions = hypothesis(X, w)
    gradient = (1 / m) * X.T @ (predictions - y)
    return w - learning_rate * gradient


def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    w = np.zeros(X.shape[1])  # початкові значення ваг
    for i in range(epochs):
        w = gradient_step(X, y, w, learning_rate)
    return w


def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y



np.random.seed(0)
X = np.random.rand(100, 3)
true_w = np.array([150, 200, 300, 50000])
X_b = np.c_[np.ones((X.shape[0], 1)), X]
y = X_b @ true_w + np.random.randn(100) * 1000


w_gradient_descent = gradient_descent(X_b, y, learning_rate=0.01, epochs=1000)


w_normal_eq = normal_equation(X_b, y)


lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_sklearn = lin_reg.predict(X)


y_pred_gradient_descent = hypothesis(X_b, w_gradient_descent)
y_pred_normal_eq = hypothesis(X_b, w_normal_eq)

mse_gradient_descent = mean_squared_error(y, y_pred_gradient_descent)
mse_normal_eq = mean_squared_error(y, y_pred_normal_eq)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)


print("Оптимальні ваги (Градієнтний спуск):", w_gradient_descent)
print("Оптимальні ваги (Аналітичне рішення):", w_normal_eq)
print("Оптимальні ваги (Scikit-learn):", np.r_[lin_reg.intercept_, lin_reg.coef_])

print("MSE (Градієнтний спуск):", mse_gradient_descent)
print("MSE (Аналітичне рішення):", mse_normal_eq)
print("MSE (Scikit-learn):", mse_sklearn)
