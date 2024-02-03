# `numpy-ml\numpy_ml\nonparametric\knn.py`

```
# 一个用于分类和回归的 k-最近邻（KNN）模型
from collections import Counter
import numpy as np
from ..utils.data_structures import BallTree

class KNN:
    def __init__(
        self, k=5, leaf_size=40, classifier=True, metric=None, weights="uniform",
    ):
        """
        一个基于球树的 `k`-最近邻（kNN）模型，用于高效计算。

        参数
        ----------
        k : int
            预测时要使用的邻居数量。默认为 5。
        leaf_size : int
            球树中每个叶子节点上的最大数据点数量。默认为 40。
        classifier : bool
            是否将 Y 中的值视为类标签（classifier = True）或实值目标（classifier = False）。默认为 True。
        metric : :doc:`距离度量 <numpy_ml.utils.distance_metrics>` 或 None
            用于计算最近邻的距离度量。如果为 None，则默认使用 :func:`~numpy_ml.utils.distance_metrics.euclidean` 距离度量。默认为 None。
        weights : {'uniform', 'distance'}
            如何对每个邻居的预测进行加权。'uniform' 为每个邻居分配均匀权重，而 'distance' 分配的权重与查询点的距离的倒数成比例。默认为 'uniform'。
        """
        # 使用 leaf_size 和 metric 创建一个球树对象
        self._ball_tree = BallTree(leaf_size=leaf_size, metric=metric)
        # 存储模型的超参数
        self.hyperparameters = {
            "id": "KNN",
            "k": k,
            "leaf_size": leaf_size,
            "classifier": classifier,
            "metric": str(metric),
            "weights": weights,
        }
    # 将模型拟合到数据和目标值`X`和`y`中
    def fit(self, X, y):
        r"""
        Fit the model to the data and targets in `X` and `y`

        Parameters
        ----------
        X : numpy array of shape `(N, M)`
            An array of `N` examples to generate predictions on.
        y : numpy array of shape `(N, *)`
            Targets for the `N` rows in `X`.
        """
        # 检查输入数据`X`的维度是否为2
        if X.ndim != 2:
            raise Exception("X must be two-dimensional")
        # 使用BallTree模型拟合数据`X`和目标值`y`
        self._ball_tree.fit(X, y)
    # 为给定输入数据集 X 生成预测结果
    def predict(self, X):
        r"""
        Generate predictions for the targets associated with the rows in `X`.

        Parameters
        ----------
        X : numpy array of shape `(N', M')`
            An array of `N'` examples to generate predictions on.

        Returns
        -------
        y : numpy array of shape `(N', *)`
            Predicted targets for the `N'` rows in `X`.
        """
        # 初始化一个空列表用于存储预测结果
        predictions = []
        # 获取超参数
        H = self.hyperparameters
        # 遍历输入数据集 X 中的每一行
        for x in X:
            # 初始化预测值为 None
            pred = None
            # 获取最近的 k 个邻居
            nearest = self._ball_tree.nearest_neighbors(H["k"], x)
            # 获取邻居的目标值
            targets = [n.val for n in nearest]

            # 如果是分类器
            if H["classifier"]:
                # 如果权重为 "uniform"
                if H["weights"] == "uniform":
                    # 对于与 sklearn / scipy.stats.mode 一致性，返回在出现平局时最小的类别 ID
                    counts = Counter(targets).most_common()
                    pred, _ = sorted(counts, key=lambda x: (-x[1], x[0]))[0]
                # 如果权重为 "distance"
                elif H["weights"] == "distance":
                    best_score = -np.inf
                    for label in set(targets):
                        scores = [1 / n.distance for n in nearest if n.val == label]
                        pred = label if np.sum(scores) > best_score else pred
            # 如果不是分类器
            else:
                # 如果权重为 "uniform"
                if H["weights"] == "uniform":
                    pred = np.mean(targets)
                # 如果权重为 "distance"
                elif H["weights"] == "distance":
                    weights = [1 / n.distance for n in nearest]
                    pred = np.average(targets, weights=weights)
            # 将预测结果添加到列表中
            predictions.append(pred)
        # 将预测结果转换为 numpy 数组并返回
        return np.array(predictions)
```