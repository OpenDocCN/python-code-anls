# `numpy-ml\numpy_ml\trees\gbdt.py`

```py
# 导入 numpy 库并重命名为 np
import numpy as np

# 从当前目录下的 dt 模块中导入 DecisionTree 类
# 从当前目录下的 losses 模块中导入 MSELoss 和 CrossEntropyLoss 类
from .dt import DecisionTree
from .losses import MSELoss, CrossEntropyLoss

# 定义一个函数，将标签转换为 one-hot 编码
def to_one_hot(labels, n_classes=None):
    # 如果标签的维度大于 1，则抛出异常
    if labels.ndim > 1:
        raise ValueError("labels must have dimension 1, but got {}".format(labels.ndim))

    # 获取标签的数量
    N = labels.size
    # 计算 one-hot 编码的列数
    n_cols = np.max(labels) + 1 if n_classes is None else n_classes
    # 创建一个全零矩阵，用于存储 one-hot 编码
    one_hot = np.zeros((N, n_cols))
    # 将对应位置的值设为 1.0，实现 one-hot 编码
    one_hot[np.arange(N), labels] = 1.0
    return one_hot

# 定义一个梯度提升决策树的类
class GradientBoostedDecisionTree:
    def __init__(
        self,
        n_iter,
        max_depth=None,
        classifier=True,
        learning_rate=1,
        loss="crossentropy",
        step_size="constant",
    # 定义一个预测方法，用于对输入数据进行分类或预测
    def predict(self, X):
        """
        Use the trained model to classify or predict the examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The integer class labels predicted for each example in `X` if
            ``self.classifier = True``, otherwise the predicted target values.
        """
        # 初始化预测结果矩阵
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        # 遍历每个迭代器
        for i in range(self.n_iter):
            # 遍历每个输出维度
            for k in range(self.out_dims):
                # 根据权重和学习器的预测结果更新预测结果矩阵
                Y_pred[:, k] += self.weights[i, k] * self.learners[i, k].predict(X)

        # 如果是分类器，则取预测结果矩阵中每行最大值的索引作为预测结果
        if self.classifier:
            Y_pred = Y_pred.argmax(axis=1)

        return Y_pred
```