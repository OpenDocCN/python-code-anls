# `numpy-ml\numpy_ml\trees\losses.py`

```py
import numpy as np
# 导入 NumPy 库

#######################################################################
#                           Base Estimators                           #
#######################################################################

# 定义 ClassProbEstimator 类，用于计算类别概率
class ClassProbEstimator:
    # 训练函数，计算类别概率
    def fit(self, X, y):
        self.class_prob = y.sum() / len(y)

    # 预测函数，返回类别概率
    def predict(self, X):
        pred = np.empty(X.shape[0], dtype=np.float64)
        pred.fill(self.class_prob)
        return pred

# 定义 MeanBaseEstimator 类，用于计算均值
class MeanBaseEstimator:
    # 训练函数，计算均值
    def fit(self, X, y):
        self.avg = np.mean(y)

    # 预测函数，返回均值
    def predict(self, X):
        pred = np.empty(X.shape[0], dtype=np.float64)
        pred.fill(self.avg)
        return pred

#######################################################################
#                           Loss Functions                            #
#######################################################################

# 定义 MSELoss 类，用于计算均方误差
class MSELoss:
    # 计算均方误差
    def __call__(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    # 返回基本估计器
    def base_estimator(self):
        return MeanBaseEstimator()

    # 计算梯度
    def grad(self, y, y_pred):
        return -2 / len(y) * (y - y_pred)

    # 线搜索函数
    def line_search(self, y, y_pred, h_pred):
        # TODO: revise this
        Lp = np.sum((y - y_pred) * h_pred)
        Lpp = np.sum(h_pred * h_pred)

        # if we perfectly fit the residuals, use max step size
        return 1 if np.sum(Lpp) == 0 else Lp / Lpp

# 定义 CrossEntropyLoss 类，用于计算交叉熵损失
class CrossEntropyLoss:
    # 计算交叉熵损失
    def __call__(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))

    # 返回基本估计器
    def base_estimator(self):
        return ClassProbEstimator()

    # 计算梯度
    def grad(self, y, y_pred):
        eps = np.finfo(float).eps
        return -y * 1 / (y_pred + eps)

    # 线搜索函数
    def line_search(self, y, y_pred, h_pred):
        raise NotImplementedError
```