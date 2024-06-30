# `D:\src\scipysrc\scikit-learn\examples\svm\plot_rbf_parameters.py`

```
"""
==================
RBF SVM parameters
==================

This example illustrates the effect of the parameters ``gamma`` and ``C`` of
the Radial Basis Function (RBF) kernel SVM.

Intuitively, the ``gamma`` parameter defines how far the influence of a single
training example reaches, with low values meaning 'far' and high values meaning
'close'. The ``gamma`` parameters can be seen as the inverse of the radius of
influence of samples selected by the model as support vectors.

The ``C`` parameter trades off correct classification of training examples
against maximization of the decision function's margin. For larger values of
``C``, a smaller margin will be accepted if the decision function is better at
classifying all training points correctly. A lower ``C`` will encourage a
larger margin, therefore a simpler decision function, at the cost of training
accuracy. In other words ``C`` behaves as a regularization parameter in the
SVM.

The first plot is a visualization of the decision function for a variety of
parameter values on a simplified classification problem involving only 2 input
features and 2 possible target classes (binary classification). Note that this
kind of plot is not possible to do for problems with more features or target
classes.

The second plot is a heatmap of the classifier's cross-validation accuracy as a
function of ``C`` and ``gamma``. For this example we explore a relatively large
grid for illustration purposes. In practice, a logarithmic grid from
:math:`10^{-3}` to :math:`10^3` is usually sufficient. If the best parameters
lie on the boundaries of the grid, it can be extended in that direction in a
subsequent search.

Note that the heat map plot has a special colorbar with a midpoint value close
to the score values of the best performing models so as to make it easy to tell
them apart in the blink of an eye.

The behavior of the model is very sensitive to the ``gamma`` parameter. If
``gamma`` is too large, the radius of the area of influence of the support
vectors only includes the support vector itself and no amount of
regularization with ``C`` will be able to prevent overfitting.

When ``gamma`` is very small, the model is too constrained and cannot capture
the complexity or "shape" of the data. The region of influence of any selected
support vector would include the whole training set. The resulting model will
behave similarly to a linear model with a set of hyperplanes that separate the
centers of high density of any pair of two classes.

For intermediate values, we can see on the second plot that good models can
be found on a diagonal of ``C`` and ``gamma``. Smooth models (lower ``gamma``
values) can be made more complex by increasing the importance of classifying
each point correctly (larger ``C`` values) hence the diagonal of good
performing models.

Finally, one can also observe that for some intermediate values of ``gamma`` we
get equally performing models when ``C`` becomes very large. This suggests that
"""
# %%
# Utility class to adjust the midpoint of a colormap normalization around
# specific values of interest.

import numpy as np
from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        # Initialize the normalization with specified parameters
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Masked array interpolation to map input values to the range [0, 1]
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# %%
# Load and prepare dataset
# -------------------------
#
# Load the Iris dataset for grid search

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # Features of the dataset
y = iris.target  # Target labels

# %%
# Dataset for decision function visualization: retaining only the first two
# features from X and sub-sampling to form a binary classification problem.

X_2d = X[:, :2]  # Select only the first two features
X_2d = X_2d[y > 0]  # Subset where target is not zero
y_2d = y[y > 0]  # Subset target accordingly
y_2d -= 1  # Adjust target labels to be binary (0 or 1)

# %%
# Scaling the data is recommended for SVM training.
# In this example, scaling is applied to all data for simplicity, although
# typically scaling should be fit on training data and applied to test data.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)  # Fit and transform all data
X_2d = scaler.fit_transform(X_2d)  # Fit and transform 2D subset

# %%
# Train classifiers
# -----------------
#
# Initialize a grid search over logarithmically spaced values for C and gamma.
# Using a StratifiedShuffleSplit cross-validation strategy.

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

C_range = np.logspace(-2, 10, 13)  # Range of C values (regularization parameter)
gamma_range = np.logspace(-9, 3, 13)  # Range of gamma values (kernel coefficient)
param_grid = dict(gamma=gamma_range, C=C_range)  # Grid of parameters to search
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)  # Cross-validation strategy
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)  # Grid search using SVM classifier
grid.fit(X, y)  # Fit the grid search on the scaled dataset X and target y

print(
    # 输出最佳参数和其对应的得分，格式化字符串使用两个占位符
    "The best parameters are %s with a score of %0.2f" 
    % (grid.best_params_, grid.best_score_)
# %%
# 现在我们需要为二维版本中的所有参数拟合分类器
# （这里使用较小的参数集合，因为训练时间较长）

C_2d_range = [1e-2, 1, 1e2]  # 设定用于C的取值范围
gamma_2d_range = [1e-1, 1, 1e1]  # 设定用于gamma的取值范围
classifiers = []  # 初始化分类器列表
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)  # 创建一个SVC分类器对象
        clf.fit(X_2d, y_2d)  # 使用二维数据集(X_2d, y_2d)训练分类器
        classifiers.append((C, gamma, clf))  # 将分类器及其参数(C, gamma)存入列表

# %%
# 可视化
# -------------
#
# 绘制参数效果的可视化图表

import matplotlib.pyplot as plt  # 导入matplotlib库

plt.figure(figsize=(8, 6))  # 创建一个8x6英寸大小的图表
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))  # 创建网格点坐标
for k, (C, gamma, clf) in enumerate(classifiers):
    # 在网格中评估决策函数
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 可视化这些参数下的决策函数
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size="medium")

    # 可视化参数对决策函数的影响
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis("tight")

scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

# %%
# 绘制gamma和C的验证准确率热图
#
# 分数以热图的形式编码，使用热色图(colormap)从深红到亮黄变化。由于最有趣的分数都位于0.92到0.97之间，
# 我们使用自定义的归一化器将中点设置为0.92，这样更容易在有趣的范围内可视化分数的小变化，同时不会强行将所有低分数值折叠到同一颜色中。

plt.figure(figsize=(8, 6))  # 创建一个8x6英寸大小的图表
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)  # 调整子图的布局
plt.imshow(
    scores,
    interpolation="nearest",
    cmap=plt.cm.hot,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
)  # 显示验证准确率的热图
plt.xlabel("gamma")  # 设置x轴标签
plt.ylabel("C")  # 设置y轴标签
plt.colorbar()  # 显示颜色条
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)  # 设置x轴刻度
plt.yticks(np.arange(len(C_range)), C_range)  # 设置y轴刻度
plt.title("Validation accuracy")  # 设置图表标题
plt.show()  # 显示图表
```