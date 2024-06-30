# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_ridge_coeffs.py`

```
# 本例展示了如何使用 Ridge 回归中的 L2 正则化来影响模型的性能，通过在损失函数中添加一个惩罚项，该惩罚项随着系数 beta 的增加而增加。

# %%
# 本例的目的
# -----------------------
# 本例旨在说明过拟合问题及其解决方法之一：正则化。正则化通过对线性模型中的大权重（系数）施加惩罚，迫使模型收缩所有系数，从而减少模型对训练样本特定信息的依赖性。
# 展示 Ridge 正则化工作原理的目的是创建一个非嘈杂数据集。然后我们将在一系列正则化强度（:math:`\alpha`）上训练一个正则化模型，并绘制训练后的系数和它们与原始值之间的均方误差随正则化强度变化的情况。

# 创建非嘈杂数据集
# ******************
# 我们创建一个玩具数据集，包含 100 个样本和 10 个特征，适合进行回归分析。其中，有 8 个信息量丰富的特征对回归有贡献，而剩下的 2 个特征对目标变量没有影响（它们的真实系数为 0）。请注意，在这个例子中数据是非嘈杂的，因此我们可以期望我们的回归模型能够完全恢复真实系数 w。
from sklearn.datasets import make_regression

X, y, w = make_regression(
    n_samples=100, n_features=10, n_informative=8, coef=True, random_state=1
)

# 获取真实系数
print(f"The true coefficient of this regression problem are:\n{w}")

# %%
# 训练 Ridge 回归器
# ******************
# 我们使用 :class:`~sklearn.linear_model.Ridge`，一个带有 L2 正则化的线性模型。我们训练多个模型，每个模型的参数 `alpha` 不同，它是一个正数常数，用于乘以惩罚项，控制正则化强度。对于每个训练好的模型，我们计算真实系数 `w` 与模型 `clf` 找到的系数之间的误差。我们将识别的系数和对应的误差存储在列表中，便于后续绘图分析。
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

clf = Ridge()

# 在对数尺度上生成均匀分布的 `alpha` 值
alphas = np.logspace(-3, 4, 200)
coefs = []
errors_coefs = []

# 使用不同的正则化强度训练模型
for a in alphas:
    clf.set_params(alpha=a).fit(X, y)
    coefs.append(clf.coef_)
    errors_coefs.append(mean_squared_error(clf.coef_, w))

# %%
# 绘制训练后的系数和均方误差
# ***************************
# 我们现在将 10 个不同正则化系数作为 `alpha` 参数的函数进行绘制，每种颜色代表一个不同的系数。
#
# 在右侧，我们绘制估计器中系数的误差随正则化强度变化的情况。
import matplotlib.pyplot as plt
import pandas as pd

alphas = pd.Index(alphas, name="alpha")
coefs = pd.DataFrame(coefs, index=alphas, columns=[f"Feature {i}" for i in range(10)])
errors = pd.Series(errors_coefs, index=alphas, name="Mean squared error")

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

coefs.plot(
    ax=axs[0],
    logx=True,
    # 定义一个字符串变量，标题为“Ridge coefficients as a function of the regularization strength”
    title="Ridge coefficients as a function of the regularization strength",
# %%
# Interpreting the plots
# **********************
# 解释绘图结果
# **********************
# The plot on the left-hand side shows how the regularization strength (`alpha`)
# affects the Ridge regression coefficients.
# 左侧图表展示了正则化强度 (`alpha`) 对岭回归系数的影响。
# Smaller values of `alpha` (weak
# regularization), allow the coefficients to closely resemble the true
# coefficients (`w`) used to generate the data set.
# 较小的 `alpha` 值（弱正则化）使得系数更接近于生成数据集时的真实系数 (`w`)。
# This is because no
# additional noise was added to our artificial data set.
# 这是因为我们的人工数据集中没有添加额外的噪声。
# As `alpha` increases,
# the coefficients shrink towards zero, gradually reducing the impact of the
# features that were formerly more significant.
# 随着 `alpha` 的增加，系数向零收缩，逐渐减少了先前更重要特征的影响。
#
# The right-hand side plot shows the mean squared error (MSE) between the
# coefficients found by the model and the true coefficients (`w`).
# 右侧图表显示了模型找到的系数与真实系数 (`w`) 之间的均方误差 (MSE)。
# It provides a
# measure that relates to how exact our ridge model is in comparison to the true
# generative model.
# 它提供了一个度量，用于比较我们的岭回归模型与真实生成模型之间的精确度。
# A low error means that it found coefficients closer to the
# ones of the true generative model.
# 低误差意味着模型找到的系数更接近真实生成模型的系数。
# In this case, since our toy data set was
# non-noisy, we can see that the least regularized model retrieves coefficients
# closest to the true coefficients (`w`) (error is close to 0).
# 在本例中，由于我们的玩具数据集没有噪声，我们可以看到最不正则化的模型得到的系数最接近真实系数 (`w`)（误差接近 0）。
#
# When `alpha` is small, the model captures the intricate details of the
# training data, whether those were caused by noise or by actual information.
# 当 `alpha` 较小时，模型捕捉了训练数据的复杂细节，无论是由噪声还是实际信息引起的。
# As `alpha` increases, the highest coefficients shrink more rapidly, rendering
# their corresponding features less influential in the training process.
# 随着 `alpha` 的增加，最大的系数收缩更快，使得其对应的特征在训练过程中影响减小。
# This
# can enhance a model's ability to generalize to unseen data (if there was a lot
# of noise to capture),
# 这可以增强模型对未见数据的泛化能力（如果需要捕捉大量噪声），
# but it also poses the risk of losing performance if the
# regularization becomes too strong compared to the amount of noise the data
# contained (as in this example).
# 但如果正则化变得过于强大，超过数据中噪声的量，也会面临性能下降的风险（如本例所示）。
#
# In real-world scenarios where data typically includes noise, selecting an
# appropriate `alpha` value becomes crucial in striking a balance between an
# overfitting and an underfitting model.
# 在现实世界中，数据通常包含噪声，选择合适的 `alpha` 值对于在过拟合和欠拟合模型之间取得平衡至关重要。
#
# Here, we saw that :class:`~sklearn.linear_model.Ridge` adds a penalty to the
# coefficients to fight overfitting.
# 在这里，我们看到 :class:`~sklearn.linear_model.Ridge` 为系数增加了惩罚项来抑制过拟合。
# Another problem that occurs is linked to
# the presence of outliers in the training dataset.
# 另一个问题与训练数据集中存在的异常值相关联。
# An outlier is a data point
# that differs significantly from other observations.
# 异常值是与其他观察结果显著不同的数据点。
# Concretely, these outliers
# impact the left-hand side term of the loss function that we showed earlier.
# 具体而言，这些异常值会影响我们之前展示的损失函数的左侧项。
# Some other linear models are formulated to be robust to outliers such as the
# :class:`~sklearn.linear_model.HuberRegressor`.
# 一些其他线性模型设计成对异常值更加稳健，例如 :class:`~sklearn.linear_model.HuberRegressor`。
# You can learn more about it in
# the :ref:`sphx_glr_auto_examples_linear_model_plot_huber_vs_ridge.py` example.
# 您可以在 :ref:`sphx_glr_auto_examples_linear_model_plot_huber_vs_ridge.py` 示例中了解更多相关内容。
```