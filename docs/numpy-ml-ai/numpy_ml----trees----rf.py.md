# `numpy-ml\numpy_ml\trees\rf.py`

```py
# 导入 numpy 库并重命名为 np
import numpy as np
# 从当前目录下的 dt 模块中导入 DecisionTree 类
from .dt import DecisionTree

# 定义一个函数 bootstrap_sample，用于生成数据集的自助采样
def bootstrap_sample(X, Y):
    # 获取数据集 X 的行数和列数
    N, M = X.shape
    # 从 0 到 N 中随机选择 N 个数，允许重复
    idxs = np.random.choice(N, N, replace=True)
    # 返回自助采样后的数据集 X 和 Y
    return X[idxs], Y[idxs]

# 定义一个类 RandomForest，表示随机森林模型
class RandomForest:
    def __init__(
        self, n_trees, max_depth, n_feats, classifier=True, criterion="entropy"
    ):
        """
        An ensemble (forest) of decision trees where each split is calculated
        using a random subset of the features in the input.

        Parameters
        ----------
        n_trees : int
            The number of individual decision trees to use within the ensemble.
        max_depth: int or None
            The depth at which to stop growing each decision tree. If None,
            grow each tree until the leaf nodes are pure.
        n_feats : int
            The number of features to sample on each split.
        classifier : bool
            Whether `Y` contains class labels or real-valued targets. Default
            is True.
        criterion : {'entropy', 'gini', 'mse'}
            The error criterion to use when calculating splits for each weak
            learner. When ``classifier = False``, valid entries are {'mse'}.
            When ``classifier = True``, valid entries are {'entropy', 'gini'}.
            Default is 'entropy'.
        """
        # 初始化随机森林模型的属性
        self.trees = []
        self.n_trees = n_trees
        self.n_feats = n_feats
        self.max_depth = max_depth
        self.criterion = criterion
        self.classifier = classifier
    def fit(self, X, Y):
        """
        Create `n_trees`-worth of bootstrapped samples from the training data
        and use each to fit a separate decision tree.
        """
        # 初始化一个空列表用于存放决策树
        self.trees = []
        # 循环创建n_trees个决策树
        for _ in range(self.n_trees):
            # 从训练数据中创建一个自助采样样本
            X_samp, Y_samp = bootstrap_sample(X, Y)
            # 创建一个决策树对象
            tree = DecisionTree(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                criterion=self.criterion,
                classifier=self.classifier,
            )
            # 使用自助采样样本拟合决策树
            tree.fit(X_samp, Y_samp)
            # 将拟合好的决策树添加到列表中
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target value for each entry in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            Model predictions for each entry in `X`.
        """
        # 对每个决策树在输入数据X上进行预测
        tree_preds = np.array([[t._traverse(x, t.root) for x in X] for t in self.trees])
        # 对每个决策树的预测结果进行投票，返回最终的预测结果
        return self._vote(tree_preds)
    # 对于每个问题，返回随机森林中所有树的预测结果的聚合值
    def _vote(self, predictions):
        """
        Return the aggregated prediction across all trees in the RF for each problem.

        Parameters
        ----------
        predictions : :py:class:`ndarray <numpy.ndarray>` of shape `(n_trees, N)`
            The array of predictions from each decision tree in the RF for each
            of the `N` problems in `X`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            If classifier is True, the class label predicted by the majority of
            the decision trees for each problem in `X`. If classifier is False,
            the average prediction across decision trees on each problem.
        """
        # 如果是分类器，则对每个问题返回大多数决策树预测的类标签
        if self.classifier:
            out = [np.bincount(x).argmax() for x in predictions.T]
        # 如果不是分类器，则对每个问题返回决策树预测的平均值
        else:
            out = [np.mean(x) for x in predictions.T]
        # 将结果转换为数组并返回
        return np.array(out)
```