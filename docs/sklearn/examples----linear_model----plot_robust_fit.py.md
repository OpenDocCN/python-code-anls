# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_robust_fit.py`

```
"""
Robust linear estimator fitting
===============================

Here a sine function is fit with a polynomial of order 3, for values
close to zero.

Robust fitting is demoed in different situations:

- No measurement errors, only modelling errors (fitting a sine with a
  polynomial)

- Measurement errors in X

- Measurement errors in y

The median absolute deviation to non corrupt new data is used to judge
the quality of the prediction.

What we can see that:

- RANSAC is good for strong outliers in the y direction

- TheilSen is good for small outliers, both in direction X and y, but has
  a break point above which it performs worse than OLS.

- The scores of HuberRegressor may not be compared directly to both TheilSen
  and RANSAC because it does not attempt to completely filter the outliers
  but lessen their effect.
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# Generate random data following a normal distribution
X = np.random.normal(size=400)
# Compute sine values for the generated X values
y = np.sin(X)

# Ensure X is 2D
X = X[:, np.newaxis]

# Generate random test data
X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

# Introduce errors in y data
y_errors = y.copy()
y_errors[::3] = 3

# Introduce errors in X data
X_errors = X.copy()
X_errors[::3] = 3

# Introduce larger errors in y data
y_errors_large = y.copy()
y_errors_large[::3] = 10

# Introduce larger errors in X data
X_errors_large = X.copy()
X_errors_large[::3] = 10

# Define a list of estimators for linear regression
estimators = [
    ("OLS", LinearRegression()),  # Ordinary Least Squares regression
    ("Theil-Sen", TheilSenRegressor(random_state=42)),  # Theil-Sen robust estimator
    ("RANSAC", RANSACRegressor(random_state=42)),  # RANdom SAmple Consensus estimator
    ("HuberRegressor", HuberRegressor()),  # Huber robust regression estimator
]

# Define colors for each estimator
colors = {
    "OLS": "turquoise",
    "Theil-Sen": "gold",
    "RANSAC": "lightgreen",
    "HuberRegressor": "black",
}

# Define linestyle for each estimator
linestyle = {
    "OLS": "-",
    "Theil-Sen": "-.",
    "RANSAC": "--",
    "HuberRegressor": "--",
}

# Define line width for plotting
lw = 3

# Generate a range of X values for plotting
x_plot = np.linspace(X.min(), X.max())

# Iterate through different scenarios for plotting
for title, this_X, this_y in [
    ("Modeling Errors Only", X, y),  # No measurement errors
    ("Corrupt X, Small Deviants", X_errors, y),  # Measurement errors in X
    ("Corrupt y, Small Deviants", X, y_errors),  # Measurement errors in y
    ("Corrupt X, Large Deviants", X_errors_large, y),  # Larger measurement errors in X
    ("Corrupt y, Large Deviants", X, y_errors_large),  # Larger measurement errors in y
]:
    # Create a new figure for each scenario
    plt.figure(figsize=(5, 4))
    # Plot original data points
    plt.plot(this_X[:, 0], this_y, "b+")

    # Iterate through each estimator and fit the model
    for name, estimator in estimators:
        # Create a pipeline with PolynomialFeatures and the current estimator
        model = make_pipeline(PolynomialFeatures(3), estimator)
        # Fit the model using current data
        model.fit(this_X, this_y)
        # Compute mean squared error on test data
        mse = mean_squared_error(model.predict(X_test), y_test)
        # Predict y values for the range of X_plot
        y_plot = model.predict(x_plot[:, np.newaxis])
        # Plot the predicted values with respective color, linestyle, and label
        plt.plot(
            x_plot,
            y_plot,
            color=colors[name],
            linestyle=linestyle[name],
            linewidth=lw,
            label="%s: error = %.3f" % (name, mse),
        )

    # Set legend title
    legend_title = "Error of Mean\nAbsolute Deviation\nto Non-corrupt Data"
    # 创建图例对象并设置其位置为右上角，去除边框，设置标题和字体大小
    legend = plt.legend(
        loc="upper right", frameon=False, title=legend_title, prop=dict(size="x-small")
    )
    
    # 设置 X 轴的显示范围从 -4 到 10.2
    plt.xlim(-4, 10.2)
    
    # 设置 Y 轴的显示范围从 -2 到 10.2
    plt.ylim(-2, 10.2)
    
    # 设置图表的标题
    plt.title(title)
# 显示当前绘图的窗口或者保存图形到文件
plt.show()
```