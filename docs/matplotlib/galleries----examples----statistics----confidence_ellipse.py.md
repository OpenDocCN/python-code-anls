# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\confidence_ellipse.py`

```
#
# This section imports necessary libraries for plotting and numerical operations.
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# %%
#
# The plotting function itself
# """"""""""""""""""""""""""""
#
# This function plots the confidence ellipse of the covariance of the given
# array-like variables x and y. The ellipse is plotted into the given
# Axes object *ax*.
#
# The radiuses of the ellipse can be controlled by n_std which is the number
# of standard deviations. The default value is 3 which makes the ellipse
# enclose 98.9% of the points if the data is normally distributed
# like in these examples (3 standard deviations in 1-D contain 99.7%
# of the data, which is 98.9% of the data in 2-D).
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
        The ellipse object representing the confidence ellipse.
    """
    # Check if the sizes of x and y are the same
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # Calculate the covariance matrix of x and y
    cov = np.cov(x, y)
    # Calculate the Pearson correlation coefficient
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Semi-major and semi-minor axes lengths of the ellipse
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Create the Ellipse object representing the confidence ellipse
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Standard deviation of x and y
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    # Transformation for the ellipse based on covariance and mean values
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    # Apply the transformation and add the ellipse to the plot
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# %%
#
# A helper function to create a correlated dataset
# """"""""""""""""""""""""""""""""""""""""""""""""
#
# Creates a random two-dimensional dataset with the specified
# two-dimensional mean (mu) and dimensions (scale).
# The correlation can be controlled by the param 'dependency',
# a 2x2 matrix.
def get_correlated_dataset(n, dependency, mu, scale):
    # Generate a matrix of random numbers with shape (n, 2)
    latent = np.random.randn(n, 2)
    # Compute the dependent variable by matrix multiplication
    dependent = latent.dot(dependency)
    # Scale the dependent variables
    scaled = dependent * scale
    # Offset the scaled variables by the mean vector mu
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


# %%
#
# Positive, negative and weak correlation
# """""""""""""""""""""""""""""""""""""""
#
# Note that the shape for the weak correlation (right) is an ellipse,
# not a circle because x and y are differently scaled.
# However, the fact that x and y are uncorrelated is shown by
# the axes of the ellipse being aligned with the x- and y-axis
# of the coordinate system.

np.random.seed(0)

PARAMETERS = {
    'Positive correlation': [[0.85, 0.35],
                             [0.15, -0.65]],
    'Negative correlation': [[0.9, -0.4],
                             [0.1, -0.6]],
    'Weak correlation': [[1, 0],
                         [0, 1]],
}

mu = 2, 4
scale = 3, 5

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for ax, (title, dependency) in zip(axs, PARAMETERS.items()):
    # Generate correlated dataset using get_correlated_dataset function
    x, y = get_correlated_dataset(800, dependency, mu, scale)
    # Scatter plot of x and y with small markers
    ax.scatter(x, y, s=0.5)

    # Add grey lines representing x and y axes
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)

    # Plot confidence ellipse using a helper function
    confidence_ellipse(x, y, ax, edgecolor='red')

    # Scatter plot of the mean point in red
    ax.scatter(mu[0], mu[1], c='red', s=3)
    # Set title for each subplot
    ax.set_title(title)

plt.show()


# %%
#
# Different number of standard deviations
# """""""""""""""""""""""""""""""""""""""
#
# A plot with n_std = 3 (blue), 2 (purple) and 1 (red)

fig, ax_nstd = plt.subplots(figsize=(6, 6))

dependency_nstd = [[0.8, 0.75],
                   [-0.2, 0.35]]
mu = 0, 0
scale = 8, 5

# Add grey lines representing x and y axes
ax_nstd.axvline(c='grey', lw=1)
ax_nstd.axhline(c='grey', lw=1)

# Generate correlated dataset using get_correlated_dataset function
x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
# Scatter plot of x and y with small markers
ax_nstd.scatter(x, y, s=0.5)

# Plot confidence ellipses with different number of standard deviations
confidence_ellipse(x, y, ax_nstd, n_std=1,
                   label=r'$1\sigma$', edgecolor='firebrick')
confidence_ellipse(x, y, ax_nstd, n_std=2,
                   label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
confidence_ellipse(x, y, ax_nstd, n_std=3,
                   label=r'$3\sigma$', edgecolor='blue', linestyle=':')

# Scatter plot of the mean point in red
ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
# Set title for the subplot
ax_nstd.set_title('Different standard deviations')
# Add legend to the plot
ax_nstd.legend()
plt.show()


# %%
#
# Using the keyword arguments
# """""""""""""""""""""""""""
#
# Use the keyword arguments specified for `matplotlib.patches.Patch` in order
# to have the ellipse rendered in different ways.

fig, ax_kwargs = plt.subplots(figsize=(6, 6))
dependency_kwargs = [[-0.8, 0.5],
                     [-0.2, 0.5]]
mu = 2, -3
scale = 6, 5

# Add grey lines representing x and y axes
ax_kwargs.axvline(c='grey', lw=1)
ax_kwargs.axhline(c='grey', lw=1)

# Generate correlated dataset using get_correlated_dataset function
x, y = get_correlated_dataset(800, dependency_kwargs, mu, scale)
# Scatter plot of x and y with small markers
ax_kwargs.scatter(x, y, s=0.5)

# The code for rendering ellipse with different keyword arguments for `matplotlib.patches.Patch`
# 从函数 `get_correlated_dataset` 中获取包含 500 个样本的相关数据集，使用给定的依赖关键字参数、均值和标度
x, y = get_correlated_dataset(500, dependency_kwargs, mu, scale)

# 调用 `confidence_ellipse` 函数，在图中绘制椭圆表示透明度，使用 alpha=0.5 控制透明度，
# 面颜色为粉红色，边框颜色为紫色，zorder=0 用于控制图层顺序
confidence_ellipse(x, y, ax_kwargs,
                   alpha=0.5, facecolor='pink', edgecolor='purple', zorder=0)

# 在 `ax_kwargs` 对象上绘制散点图，点的大小为 0.5
ax_kwargs.scatter(x, y, s=0.5)

# 在 `ax_kwargs` 对象上绘制一个红色的小点，坐标为 `mu` 的第一个和第二个元素
ax_kwargs.scatter(mu[0], mu[1], c='red', s=3)

# 设置图的标题为 'Using keyword arguments'
ax_kwargs.set_title('Using keyword arguments')

# 调整子图之间的水平间距为 0.25
fig.subplots_adjust(hspace=0.25)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    在这个示例中展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.transforms.Affine2D`
#    - `matplotlib.patches.Ellipse`
```