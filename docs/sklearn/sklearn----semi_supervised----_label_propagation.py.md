# `D:\src\scipysrc\scikit-learn\sklearn\semi_supervised\_label_propagation.py`

```
# 设置 Python 文件编码为 UTF-8
# 
# 模块说明文档字符串，介绍了标签传播算法在半监督分类中的应用，算法通过形成全连接图解决每个点上标签的稳态分布。
# 
# 该算法在实践中表现良好，但运行成本高，大约为 O(N^3)，其中 N 是（标记和未标记）点的数量。
# 理论上的解释源于随机游走算法和数据中的几何关系。详细信息可参考下面的参考文献。
# 
# 模型特点
# --------------
# 标签夹持（Label clamping）:
#   算法尝试在给定初始子集上学习标签在数据集上的分布。在一种变体中，算法不允许初始分配中的任何错误（硬夹持），
#   而在另一种变体中，算法允许初始分配有一定的余地，允许它们在每次迭代中变动一个分数 alpha（软夹持）。
# 
# 核函数（Kernel）:
#   将向量投影到更高维空间的函数。这个实现支持 RBF 和 KNN 核函数。
#   使用 RBF 核函数会生成一个大小为 O(N^2) 的密集矩阵。
#   使用 KNN 核函数会生成一个大小为 O(k*N) 的稀疏矩阵，运行速度更快。
#   更多有关核函数的信息，请参阅 SVM 文档。
# 
# 示例
# --------
# >>> import numpy as np
# >>> from sklearn import datasets
# >>> from sklearn.semi_supervised import LabelPropagation
# >>> label_prop_model = LabelPropagation()
# >>> iris = datasets.load_iris()
# >>> rng = np.random.RandomState(42)
# >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
# >>> labels = np.copy(iris.target)
# >>> labels[random_unlabeled_points] = -1
# >>> label_prop_model.fit(iris.data, labels)
# LabelPropagation(...)
# 
# 注释
# -----
# 参考文献：
# [1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
# Learning (2006), pp. 193-216
# 
# [2] Olivier Delalleau, Yoshua Bengio, Nicolas Le Roux. Efficient
# Non-Parametric Function Induction in Semi-Supervised Learning. AISTAT 2005
# 
# 作者：scikit-learn 开发者
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入抽象基类元类、整数和实数验证函数
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

# 导入 NumPy 库
import numpy as np
# 导入 SciPy 中的稀疏矩阵模块
from scipy import sparse

# 导入 scikit-learn 的基础估算器、分类器混合、拟合上下文
from ..base import BaseEstimator, ClassifierMixin, _fit_context
# 导入收敛警告异常类
from ..exceptions import ConvergenceWarning
# 导入度量学习中的径向基核函数
from ..metrics.pairwise import rbf_kernel
# 导入最近邻模块
from ..neighbors import NearestNeighbors
# 导入参数验证工具：区间、字符串选项验证
from ..utils._param_validation import Interval, StrOptions
# 导入扩展数学工具，安全稀疏点积
from ..utils.extmath import safe_sparse_dot
# 导入修复工具：图拉普拉斯函数
from ..utils.fixes import laplacian as csgraph_laplacian
# 导入多类分类验证函数
from ..utils.multiclass import check_classification_targets
# 导入检查是否已拟合的验证函数
from ..utils.validation import check_is_fitted

# 定义基础标签传播类，继承分类器混合、基础估算器，并设定元类为抽象基类元类
class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for label propagation module.

     Parameters
     ----------
     kernel : {'knn', 'rbf'} or callable, default='rbf'
         String identifier for kernel function to use or the kernel function
         itself. Only 'rbf' and 'knn' strings are valid inputs. The function
         passed should take two inputs, each of shape (n_samples, n_features),
         and return a (n_samples, n_samples) shaped weight matrix.

     gamma : float, default=20
         Parameter for rbf kernel.

     n_neighbors : int, default=7
         Parameter for knn kernel. Need to be strictly positive.

     alpha : float, default=1.0
         Clamping factor.

     max_iter : int, default=30
         Change maximum number of iterations allowed.

     tol : float, default=1e-3
         Convergence tolerance: threshold to consider the system at steady
         state.

    n_jobs : int, default=None
         The number of parallel jobs to run.
         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
         for more details.
    """



    _parameter_constraints: dict = {
        "kernel": [StrOptions({"knn", "rbf"}), callable],
        "gamma": [Interval(Real, 0, None, closed="left")],
        "n_neighbors": [Interval(Integral, 0, None, closed="neither")],
        "alpha": [None, Interval(Real, 0, 1, closed="neither")],
        "max_iter": [Interval(Integral, 0, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "n_jobs": [None, Integral],
    }



    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        alpha=1,
        max_iter=30,
        tol=1e-3,
        n_jobs=None,
    ):
        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.n_jobs = n_jobs
    ```


    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                # Compute the RBF kernel matrix for the input data X
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                # Compute the RBF kernel matrix between X and y
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                # Fit a nearest neighbors model on X for later use
                self.nn_fit = NearestNeighbors(
                    n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
                ).fit(X)
            if y is None:
                # Construct a sparse graph of k-neighbors for each sample in X
                return self.nn_fit.kneighbors_graph(
                    self.nn_fit._fit_X, self.n_neighbors, mode="connectivity"
                )
            else:
                # Find k-neighbors of y and return indices without distances
                return self.nn_fit.kneighbors(y, return_distance=False)
        elif callable(self.kernel):
            if y is None:
                # Compute the kernel matrix using the provided callable kernel for X
                return self.kernel(X, X)
            else:
                # Compute the kernel matrix using the callable kernel between X and y
                return self.kernel(X, y)
    ```
    # 定义一个方法 `_build_graph`，用于构建图结构，但是该方法目前只是抛出一个未实现错误
    def _build_graph(self):
        raise NotImplementedError(
            "Graph construction must be implemented to fit a label propagation model."
        )

    # 定义一个预测方法 `predict`，用于执行模型的归纳推理
    def predict(self, X):
        """Perform inductive inference across the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predictions for input data.
        """
        # 注意：由于 `predict` 方法不接受半监督标签作为输入，
        # 因此 `fit(X, y).predict(X) != fit(X, y).transduction_`。
        # 因此，`fit_predict` 方法没有实现。
        # 参见 https://github.com/scikit-learn/scikit-learn/pull/24898
        # 使用 `predict_proba` 方法预测概率值
        probas = self.predict_proba(X)
        # 根据最大概率值确定每个样本的预测类别，将结果展平为一维数组
        return self.classes_[np.argmax(probas, axis=1)].ravel()

    # 定义一个预测概率方法 `predict_proba`，用于计算每个可能结果的概率
    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Compute the probability estimates for each single sample in X
        and each possible outcome seen during training (categorical
        distribution).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Normalized probability distributions across
            class labels.
        """
        # 确保模型已经拟合过
        check_is_fitted(self)

        # 验证输入数据 X，并根据需要重置格式
        X_2d = self._validate_data(
            X,
            accept_sparse=["csc", "csr", "coo", "dok", "bsr", "lil", "dia"],
            reset=False,
        )
        # 获取核函数生成的权重矩阵
        weight_matrices = self._get_kernel(self.X_, X_2d)
        
        # 根据核函数类型计算概率值
        if self.kernel == "knn":
            probabilities = np.array(
                [
                    np.sum(self.label_distributions_[weight_matrix], axis=0)
                    for weight_matrix in weight_matrices
                ]
            )
        else:
            weight_matrices = weight_matrices.T
            probabilities = safe_sparse_dot(weight_matrices, self.label_distributions_)
        
        # 对概率值进行归一化处理
        normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
        probabilities /= normalizer
        return probabilities

    # 应用装饰器 `_fit_context`，参数设置为 `prefer_skip_nested_validation=True`
    @_fit_context(prefer_skip_nested_validation=True)
class LabelPropagation(BaseLabelPropagation):
    """Label Propagation classifier.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf'} or callable, default='rbf'
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape (n_samples, n_features),
        and return a (n_samples, n_samples) shaped weight matrix.

    gamma : float, default=20
        Parameter for rbf kernel.

    n_neighbors : int, default=7
        Parameter for knn kernel which need to be strictly positive.

    max_iter : int, default=1000
        Change maximum number of iterations allowed.

    tol : float, 1e-3
        Convergence tolerance: threshold to consider the system at steady
        state.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input array.

    classes_ : ndarray of shape (n_classes,)
        The distinct labels used in classifying instances.

    label_distributions_ : ndarray of shape (n_samples, n_classes)
        Categorical distribution for each item.

    transduction_ : ndarray of shape (n_samples)
        Label assigned to each item during :term:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    LabelSpreading : Alternate label propagation strategy more robust to noise.

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelPropagation
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelPropagation(...)
    """

    _variant = "propagation"

    # 继承自 BaseLabelPropagation 类的参数约束字典，作为类变量
    _parameter_constraints: dict = {**BaseLabelPropagation._parameter_constraints}
    # 从参数约束中移除 "alpha" 参数
    _parameter_constraints.pop("alpha")

    # 初始化方法，设置各种参数并调用父类的初始化方法
    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        max_iter=1000,
        tol=1e-3,
        n_jobs=None,
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            alpha=None,  # 设置 alpha 参数为 None
        )

    # 构建图方法，生成一个表示每个样本之间完全连接的非随机亲和力矩阵
    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        # 如果 kernel 是 "knn"，则初始化 nn_fit 为 None
        if self.kernel == "knn":
            self.nn_fit = None
        # 获取核函数计算的亲和力矩阵
        affinity_matrix = self._get_kernel(self.X_)
        # 计算每列的和，用于后续归一化
        normalizer = affinity_matrix.sum(axis=0)
        # 如果亲和力矩阵是稀疏矩阵，则进行归一化处理
        if sparse.issparse(affinity_matrix):
            affinity_matrix.data /= np.diag(np.array(normalizer))
        else:
            affinity_matrix /= normalizer[:, np.newaxis]
        # 返回归一化后的亲和力矩阵
        return affinity_matrix

    # 拟合方法，将半监督标签传播模型拟合到数据 X 上
    def fit(self, X, y):
        """Fit a semi-supervised label propagation model to X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target class values with unlabeled points marked as -1.
            All unlabeled samples will be transductively assigned labels
            internally, which are stored in `transduction_`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 调用父类的 fit 方法进行拟合
        return super().fit(X, y)
# 继承自 BaseLabelPropagation 的 LabelSpreading 模型，用于半监督学习。

class LabelSpreading(BaseLabelPropagation):
    """LabelSpreading model for semi-supervised learning.

    This model is similar to the basic Label Propagation algorithm,
    but uses affinity matrix based on the normalized graph Laplacian
    and soft clamping across the labels.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf'} or callable, default='rbf'
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape (n_samples, n_features),
        and return a (n_samples, n_samples) shaped weight matrix.
    
    gamma : float, default=20
      Parameter for rbf kernel.
    
    n_neighbors : int, default=7
      Parameter for knn kernel which is a strictly positive integer.
    
    alpha : float, default=0.2
      Clamping factor. A value in (0, 1) that specifies the relative amount
      that an instance should adopt the information from its neighbors as
      opposed to its initial label.
      alpha=0 means keeping the initial label information; alpha=1 means
      replacing all initial information.
    
    max_iter : int, default=30
      Maximum number of iterations allowed.
    
    tol : float, default=1e-3
      Convergence tolerance: threshold to consider the system at steady
      state.
    
    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Input array.
    
    classes_ : ndarray of shape (n_classes,)
        The distinct labels used in classifying instances.
    
    label_distributions_ : ndarray of shape (n_samples, n_classes)
        Categorical distribution for each item.
    
    transduction_ : ndarray of shape (n_samples,)
        Label assigned to each item during :term:`fit`.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    n_iter_ : int
        Number of iterations run.
    
    See Also
    --------
    LabelPropagation : Unregularized graph based semi-supervised learning.
    
    References
    ----------
    `Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston,
    Bernhard Schoelkopf. Learning with local and global consistency (2004)
    <https://citeseerx.ist.psu.edu/doc_view/pid/d74c37aabf2d5cae663007cbd8718175466aea8c>`_
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelSpreading
    >>> label_prop_model = LabelSpreading()
    # 创建一个 LabelSpreading 模型的实例

    >>> iris = datasets.load_iris()
    # 载入经典数据集 iris 数据

    >>> rng = np.random.RandomState(42)
    # 创建一个种子为 42 的随机数生成器实例

    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    # 生成一个布尔数组，表示是否为未标记点，条件是随机数小于 0.3

    >>> labels = np.copy(iris.target)
    # 复制 iris 数据集中的目标标签到 labels 变量

    >>> labels[random_unlabeled_points] = -1
    # 将随机选择的未标记点的标签设置为 -1

    >>> label_prop_model.fit(iris.data, labels)
    # 使用 iris 数据集的数据和修改后的标签进行 Label Spreading 模型的拟合
    LabelSpreading(...)
    """

    _variant = "spreading"
    # 设定变量 _variant 为字符串 "spreading"

    _parameter_constraints: dict = {**BaseLabelPropagation._parameter_constraints}
    # 使用 BaseLabelPropagation 类的参数约束字典创建一个新的参数约束字典

    _parameter_constraints["alpha"] = [Interval(Real, 0, 1, closed="neither")]
    # 向参数约束字典中添加 alpha 参数的约束条件，即 alpha 应为介于 0 和 1 之间的实数，不包括边界值

    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        alpha=0.2,
        max_iter=30,
        tol=1e-3,
        n_jobs=None,
    ):
        # 构造函数，初始化 LabelSpreading 类的实例
        # 以核函数为 rbf 开始，设定默认参数值 gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3, n_jobs=None

        # 调用父类的构造函数，传递相同的参数
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
        )

    def _build_graph(self):
        """Graph matrix for Label Spreading computes the graph laplacian"""
        # 为 Label Spreading 构建图矩阵，计算图拉普拉斯矩阵

        # 计算亲和力矩阵（或 Gram 矩阵）
        if self.kernel == "knn":
            self.nn_fit = None
        n_samples = self.X_.shape[0]
        affinity_matrix = self._get_kernel(self.X_)
        laplacian = csgraph_laplacian(affinity_matrix, normed=True)
        laplacian = -laplacian
        # 将拉普拉斯矩阵的所有元素取负

        if sparse.issparse(laplacian):
            diag_mask = laplacian.row == laplacian.col
            laplacian.data[diag_mask] = 0.0
        else:
            laplacian.flat[:: n_samples + 1] = 0.0  # 将对角线元素设为 0.0
        return laplacian
    # 返回构建好的拉普拉斯矩阵
```