# `numpy-ml\numpy_ml\trees\dt.py`

```py
# 导入 NumPy 库
import numpy as np

# 定义节点类
class Node:
    def __init__(self, left, right, rule):
        # 初始化节点的左子节点
        self.left = left
        # 初始化节点的右子节点
        self.right = right
        # 获取节点的特征
        self.feature = rule[0]
        # 获取节点的阈值
        self.threshold = rule[1]

# 定义叶子节点类
class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        # 初始化叶子节点的值，如果是分类器，则为类别概率数组，否则为区域的均值
        self.value = value

# 定义决策树类
class DecisionTree:
    def __init__(
        self,
        classifier=True,
        max_depth=None,
        n_feats=None,
        criterion="entropy",
        seed=None,
        """
        用于回归和分类问题的决策树模型。

        参数
        ----------
        classifier : bool
            是否将目标值视为分类（classifier = True）或连续（classifier = False）。默认为True。
        max_depth: int or None
            停止生长树的深度。如果为None，则生长树直到所有叶子节点都是纯的。默认为None。
        n_feats : int
            指定在每次分裂时要采样的特征数量。如果为None，则在每次分裂时使用所有特征。默认为None。
        criterion : {'mse', 'entropy', 'gini'}
            计算分裂时要使用的错误标准。当`classifier`为False时，有效输入为{'mse'}。当`classifier`为True时，有效输入为{'entropy', 'gini'}。默认为'entropy'。
        seed : int or None
            随机数生成器的种子。默认为None。
        """
        如果有种子值，则设置随机数种子
        if seed:
            np.random.seed(seed)

        初始化深度为0
        初始化根节点为None

        设置特征数量为n_feats
        设置错误标准为criterion
        设置分类器为classifier
        设置最大深度为max_depth，如果max_depth为None，则设置为无穷大

        如果不是分类问题且错误标准为["gini", "entropy"]之一，则引发值错误
        if not classifier and criterion in ["gini", "entropy"]:
            raise ValueError(
                "{} is a valid criterion only when classifier = True.".format(criterion)
            )
        如果是分类问题且错误标准为"mse"，则引发值错误
        if classifier and criterion == "mse":
            raise ValueError("`mse` is a valid criterion only when classifier = False.")
    # 适应二叉决策树到数据集
    def fit(self, X, Y):
        """
        Fit a binary decision tree to a dataset.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            An array of integer class labels for each example in `X` if
            self.classifier = True, otherwise the set of target values for
            each example in `X`.
        """
        # 如果是分类器，则确定类别数量，否则为 None
        self.n_classes = max(Y) + 1 if self.classifier else None
        # 确定特征数量，如果未指定则为 X 的特征数量，否则取最小值
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # 生成决策树的根节点
        self.root = self._grow(X, Y)

    # 使用训练好的决策树对 X 中的示例进行分类或预测
    def predict(self, X):
        """
        Use the trained decision tree to classify or predict the examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The integer class labels predicted for each example in `X` if
            self.classifier = True, otherwise the predicted target values.
        """
        # 对 X 中的每个示例进行预测，返回预测的类别或目标值
        return np.array([self._traverse(x, self.root) for x in X])
    def predict_class_probs(self, X):
        """
        使用训练好的决策树来返回`X`中每个示例的类别概率。

        参数
        ----------
        X : :py:class:`ndarray <numpy.ndarray>`，形状为`(N, M)`
            `N`个示例的训练数据，每个示例有`M`个特征

        返回
        -------
        preds : :py:class:`ndarray <numpy.ndarray>`，形状为`(N, n_classes)`
            针对`X`中每个示例预测的类别概率。
        """
        # 检查分类器是否已定义
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        # 对`X`中的每个示例调用`_traverse`方法，返回类别概率数组
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y, cur_depth=0):
        # 如果所有标签都相同，则返回一个叶节点
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])

        # 如果达到最大深度，则返回一个叶节点
        if cur_depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v)

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        N, M = X.shape
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # 根据`criterion`贪婪地选择最佳分割
        feat, thresh = self._segment(X, Y, feat_idxs)
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        # 递归生长左右子树
        left = self._grow(X[l, :], Y[l], cur_depth)
        right = self._grow(X[r, :], Y[r], cur_depth)
        return Node(left, right, (feat, thresh))
    def _segment(self, X, Y, feat_idxs):
        """
        Find the optimal split rule (feature index and split threshold) for the
        data according to `self.criterion`.
        """
        # 初始化最佳增益为负无穷
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        # 遍历特征索引
        for i in feat_idxs:
            # 获取特征值
            vals = X[:, i]
            # 获取唯一值
            levels = np.unique(vals)
            # 计算阈值
            thresholds = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels
            # 计算增益
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            # 更新最佳增益
            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]

        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """
        Compute the impurity gain associated with a given split.

        IG(split) = loss(parent) - weighted_avg[loss(left_child), loss(right_child)]
        """
        # 根据不同的准则选择损失函数
        if self.criterion == "entropy":
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse

        # 计算父节点的损失
        parent_loss = loss(Y)

        # 生成分裂
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        # 计算子节点损失的加权平均
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # 计算增益是分裂前后损失的差异
        ig = parent_loss - child_loss
        return ig
    # 递归遍历决策树节点，根据输入数据 X 和节点信息 node 进行预测
    def _traverse(self, X, node, prob=False):
        # 如果节点是叶子节点
        if isinstance(node, Leaf):
            # 如果是分类器模型
            if self.classifier:
                # 返回节点的值（概率值或类别值）
                return node.value if prob else node.value.argmax()
            # 如果是回归模型，直接返回节点的值
            return node.value
        # 如果输入数据 X 在节点的特征上的取值小于等于节点的阈值
        if X[node.feature] <= node.threshold:
            # 递归遍历左子树节点
            return self._traverse(X, node.left, prob)
        # 如果输入数据 X 在节点的特征上的取值大于节点的阈值
        return self._traverse(X, node.right, prob)
# 计算决策树（即均值）预测的均方误差
def mse(y):
    # 返回 y 与 y 的均值之差的平方的均值
    return np.mean((y - np.mean(y)) ** 2)


# 计算标签序列的熵
def entropy(y):
    # 统计标签序列中每个标签出现的次数
    hist = np.bincount(y)
    # 计算每个标签出现的概率
    ps = hist / np.sum(hist)
    # 计算熵值
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


# 计算标签序列的基尼不纯度（局部熵）
def gini(y):
    # 统计标签序列中每个标签出现的次数
    hist = np.bincount(y)
    # 计算总样本数
    N = np.sum(hist)
    # 计算基尼不纯度
    return 1 - sum([(i / N) ** 2 for i in hist])
```