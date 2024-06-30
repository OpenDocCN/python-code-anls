# `D:\src\scipysrc\scikit-learn\examples\miscellaneous\plot_roc_curve_visualization_api.py`

```
# %%
# Load Data and Train a SVC
# -------------------------
# 首先，加载葡萄酒数据集并将其转换为二元分类问题。然后，我们在训练数据集上训练支持向量分类器（SVC）。
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)
y = y == 2  # 将数据集转换为二元分类，判断是否为类别2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

# %%
# Plotting the ROC Curve
# ----------------------
# 接下来，我们使用单个调用 :func:`sklearn.metrics.RocCurveDisplay.from_estimator` 绘制ROC曲线。
# 返回的 `svc_disp` 对象允许我们在未来的绘图中继续使用已计算的SVC的ROC曲线。
svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
plt.show()

# %%
# Training a Random Forest and Plotting the ROC Curve
# ---------------------------------------------------
# 我们训练一个随机森林分类器，并创建一个绘图来比较它与SVC的ROC曲线。
# 注意 `svc_disp` 使用 :func:`~sklearn.metrics.RocCurveDisplay.plot` 绘制SVC的ROC曲线，而无需重新计算ROC曲线的值。
# 此外，我们通过将 `alpha=0.8` 传递给绘图函数来调整曲线的透明度。
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
```