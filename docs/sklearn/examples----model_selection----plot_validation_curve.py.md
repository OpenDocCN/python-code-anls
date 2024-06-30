# `D:\src\scipysrc\scikit-learn\examples\model_selection\plot_validation_curve.py`

```
"""
==========================
Plotting Validation Curves
==========================

In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from sklearn.datasets import load_digits  # 导入 load_digits 函数，用于加载手写数字数据集
from sklearn.model_selection import ValidationCurveDisplay  # 导入 ValidationCurveDisplay 类，用于显示验证曲线
from sklearn.svm import SVC  # 导入支持向量机分类器 SVM

# 加载手写数字数据集，返回特征矩阵 X 和目标向量 y
X, y = load_digits(return_X_y=True)

# 创建一个子集掩码，选择目标向量 y 中值为 1 或 2 的样本，进行二元分类任务
subset_mask = np.isin(y, [1, 2])
X, y = X[subset_mask], y[subset_mask]

# 从估计器（SVC()）创建一个 ValidationCurveDisplay 对象，展示 gamma 参数的验证曲线
disp = ValidationCurveDisplay.from_estimator(
    SVC(),  # 使用默认参数创建 SVC 分类器对象
    X,  # 特征矩阵
    y,  # 目标向量
    param_name="gamma",  # 验证曲线中变化的参数名称为 gamma
    param_range=np.logspace(-6, -1, 5),  # gamma 参数的取值范围为 10^-6 到 10^-1 对数间隔的五个值
    score_type="both",  # 显示训练分数和验证分数
    n_jobs=2,  # 使用两个并行工作进程
    score_name="Accuracy",  # 评分名称为 Accuracy，表示准确率
)

# 设置图表标题
disp.ax_.set_title("Validation Curve for SVM with an RBF kernel")
# 设置 x 轴标签
disp.ax_.set_xlabel(r"gamma (inverse radius of the RBF kernel)")
# 设置 y 轴范围
disp.ax_.set_ylim(0.0, 1.1)

# 显示绘制的验证曲线图
plt.show()
```