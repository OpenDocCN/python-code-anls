# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_estimator_checks.py`

```
# 导入必要的库和模块
import importlib      # 动态导入模块的功能
import sys            # 提供与Python解释器相关的功能
import unittest       # 单元测试框架
import warnings       # 控制警告的显示

from numbers import Integral, Real  # 导入数值类型

import joblib         # 提供用于序列化Python对象的工具
import numpy as np    # 数值计算库
import scipy.sparse as sp  # 稀疏矩阵库

from sklearn import config_context, get_config  # sklearn配置管理
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin  # sklearn基类
from sklearn.cluster import MiniBatchKMeans  # MiniBatchKMeans聚类器
from sklearn.datasets import make_multilabel_classification  # 创建多标签分类数据集
from sklearn.decomposition import PCA   # 主成分分析
from sklearn.ensemble import ExtraTreesClassifier  # 极端随机树分类器
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning  # sklearn异常
from sklearn.linear_model import (
    LinearRegression,    # 线性回归模型
    LogisticRegression,  # 逻辑回归模型
    MultiTaskElasticNet, # 多任务弹性网络模型
    SGDClassifier,       # 随机梯度下降分类器
)
from sklearn.mixture import GaussianMixture  # 高斯混合模型
from sklearn.neighbors import KNeighborsRegressor  # K近邻回归器
from sklearn.svm import SVC, NuSVC  # 支持向量机模型
from sklearn.utils import _array_api, all_estimators, deprecated  # sklearn工具函数
from sklearn.utils._param_validation import Interval, StrOptions  # 参数验证工具
from sklearn.utils._testing import (
    MinimalClassifier,   # 最小分类器用于测试
    MinimalRegressor,    # 最小回归器用于测试
    MinimalTransformer,  # 最小变换器用于测试
    SkipTest,            # 用于测试时跳过的异常类
    ignore_warnings,     # 忽略警告的装饰器
    raises,              # 测试中用于验证异常抛出的装饰器
)
from sklearn.utils.estimator_checks import (
    _NotAnArray,  # 自定义异常类，表示不是数组
    _set_checking_parameters,  # 设置验证参数
    _yield_all_checks,         # 生成所有检查的生成器
    check_array_api_input,     # 检查数组API输入
    check_class_weight_balanced_linear_classifier,  # 检查线性分类器的平衡类权重
    check_classifier_data_not_an_array,            # 检查分类器的数据不是数组
    check_classifiers_multilabel_output_format_decision_function,  # 检查多标签输出格式的决策函数
    check_classifiers_multilabel_output_format_predict,            # 检查多标签输出格式的预测函数
    check_classifiers_multilabel_output_format_predict_proba,      # 检查多标签输出格式的预测概率函数
    check_dataframe_column_names_consistency,      # 检查数据框列名的一致性
    check_decision_proba_consistency,              # 检查决策和概率一致性
    check_estimator,                              # 检查评估器
    check_estimator_get_tags_default_keys,         # 检查评估器的默认标签键
    check_estimators_unfitted,                    # 检查评估器是否未拟合
    check_fit_check_is_fitted,                    # 检查拟合检查是否已拟合
    check_fit_score_takes_y,                      # 检查拟合分数是否接受y参数
    check_methods_sample_order_invariance,         # 检查方法对样本顺序不变性
    check_methods_subset_invariance,               # 检查方法对子集不变性
    check_no_attributes_set_in_init,               # 检查初始化时是否未设置属性
    check_outlier_contamination,                   # 检查异常值污染
    check_outlier_corruption,                      # 检查异常值破坏
    check_regressor_data_not_an_array,             # 检查回归器的数据不是数组
    check_requires_y_none,                         # 检查是否需要y为None
    set_random_state,                              # 设置随机状态
)
from sklearn.utils.fixes import CSR_CONTAINERS, SPARRAY_PRESENT  # sklearn修复
from sklearn.utils.metaestimators import available_if   # 如果可用，则导入元估计器
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y  # 验证工具

class CorrectNotFittedError(ValueError):
    """如果估计器在拟合之前使用，则抛出的异常类。

    类似于NotFittedError，继承自ValueError，但不继承自AttributeError。仅用于测试。
    """

class BaseBadClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        return self  # 返回自身作为拟合后的估计器

    def predict(self, X):
        return np.ones(X.shape[0])  # 返回全为1的预测结果数组

class ChangesDict(BaseEstimator):
    def __init__(self, key=0):
        self.key = key

    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)  # 验证输入数据X和y
        return self  # 返回自身作为拟合后的估计器
    # 定义一个预测方法，该方法属于类的实例方法
    def predict(self, X):
        # 调用辅助函数 check_array 对输入的 X 进行检查和转换
        X = check_array(X)
        # 设置实例变量 key 的值为 1000
        self.key = 1000
        # 返回一个由 1 组成的数组，数组的形状与输入 X 的行数相同
        return np.ones(X.shape[0])
class SetsWrongAttribute(BaseEstimator):
    # 初始化方法，设置接受的参数 acceptable_key，默认为0
    def __init__(self, acceptable_key=0):
        self.acceptable_key = acceptable_key

    # 拟合方法，设置 wrong_attribute 为0，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        self.wrong_attribute = 0
        X, y = self._validate_data(X, y)
        return self


class ChangesWrongAttribute(BaseEstimator):
    # 初始化方法，设置 wrong_attribute 参数，默认为0
    def __init__(self, wrong_attribute=0):
        self.wrong_attribute = wrong_attribute

    # 拟合方法，设置 wrong_attribute 为1，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        self.wrong_attribute = 1
        X, y = self._validate_data(X, y)
        return self


class ChangesUnderscoreAttribute(BaseEstimator):
    # 拟合方法，设置 _good_attribute 为1，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        self._good_attribute = 1
        X, y = self._validate_data(X, y)
        return self


class RaisesErrorInSetParams(BaseEstimator):
    # 初始化方法，设置参数 p，默认为0
    def __init__(self, p=0):
        self.p = p

    # 设置参数方法，接收关键字参数 kwargs，并在参数 p 小于0时抛出 ValueError 异常
    def set_params(self, **kwargs):
        if "p" in kwargs:
            p = kwargs.pop("p")
            if p < 0:
                raise ValueError("p can't be less than 0")
            self.p = p
        return super().set_params(**kwargs)

    # 拟合方法，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        return self


class HasMutableParameters(BaseEstimator):
    # 初始化方法，设置参数 p，默认为一个对象
    def __init__(self, p=object()):
        self.p = p

    # 拟合方法，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        return self


class HasImmutableParameters(BaseEstimator):
    # 初始化方法，设置参数 p 默认为 42，q 默认为 np.int32 类型的 42，r 默认为对象类型
    def __init__(self, p=42, q=np.int32(42), r=object):
        self.p = p
        self.q = q
        self.r = r

    # 拟合方法，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        return self


class ModifiesValueInsteadOfRaisingError(BaseEstimator):
    # 初始化方法，设置参数 p，默认为0
    def __init__(self, p=0):
        self.p = p

    # 设置参数方法，接收关键字参数 kwargs，并在参数 p 小于0时将其设置为0
    def set_params(self, **kwargs):
        if "p" in kwargs:
            p = kwargs.pop("p")
            if p < 0:
                p = 0
            self.p = p
        return super().set_params(**kwargs)

    # 拟合方法，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        return self


class ModifiesAnotherValue(BaseEstimator):
    # 初始化方法，设置参数 a 默认为0，参数 b 默认为 "method1"
    def __init__(self, a=0, b="method1"):
        self.a = a
        self.b = b

    # 设置参数方法，接收关键字参数 kwargs，根据参数 a 是否为 None 设置参数 a 和 b 的值
    def set_params(self, **kwargs):
        if "a" in kwargs:
            a = kwargs.pop("a")
            self.a = a
            if a is None:
                kwargs.pop("b")
                self.b = "method2"
        return super().set_params(**kwargs)

    # 拟合方法，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y=None):
        X, y = self._validate_data(X, y)
        return self


class NoCheckinPredict(BaseBadClassifier):
    # 拟合方法，验证数据 X 和 y，并返回对象自身
    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        return self


class NoSparseClassifier(BaseBadClassifier):
    # 初始化方法，设置参数 raise_for_type，默认为 None
    # raise_for_type : str, 期望值为 "sparse_array" 或 "sparse_matrix"
    def __init__(self, raise_for_type=None):
        self.raise_for_type = raise_for_type
    # 定义模型的训练方法，接收输入数据 X 和目标数据 y
    def fit(self, X, y):
        # 调用私有方法 _validate_data 对输入数据 X 和目标数据 y 进行验证和转换，确保数据格式正确
        X, y = self._validate_data(X, y, accept_sparse=["csr", "csc"])
        
        # 根据设置决定要引发异常的情况：如果设置为 "sparse_array"，则检查 X 是否为稀疏数组类型 sp.sparray
        if self.raise_for_type == "sparse_array":
            correct_type = isinstance(X, sp.sparray)
        # 如果设置为 "sparse_matrix"，则检查 X 是否为稀疏矩阵类型 sp.spmatrix
        elif self.raise_for_type == "sparse_matrix":
            correct_type = isinstance(X, sp.spmatrix)
        
        # 如果 X 的类型符合预期，引发 ValueError 异常
        if correct_type:
            raise ValueError("Nonsensical Error")
        
        # 返回模型自身，用于方法链式调用或其他用途
        return self

    # 定义模型的预测方法，接收输入数据 X
    def predict(self, X):
        # 调用 check_array 对输入数据 X 进行检查和转换，确保其为数组形式
        X = check_array(X)
        
        # 返回一个形状与 X 行数相同的全为 1 的预测结果数组
        return np.ones(X.shape[0])
class CorrectNotFittedErrorClassifier(BaseBadClassifier):
    # 继承自 BaseBadClassifier 类的一个分类器，用于处理未正确拟合错误
    def fit(self, X, y):
        # 对输入数据进行验证和处理
        X, y = self._validate_data(X, y)
        # 设置模型的系数为全为1的数组
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 对输入数据进行验证和处理
        X = check_array(X)
        # 返回一个预测结果全为1的数组，长度与输入数据行数相同
        return np.ones(X.shape[0])


class NoSampleWeightPandasSeriesType(BaseEstimator):
    # 继承自 BaseEstimator 类的一个估计器，用于处理不支持 pandas.Series 作为样本权重的情况
    def fit(self, X, y, sample_weight=None):
        # 转换数据类型并验证数据
        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        # 导入 pandas 中的 Series 类
        from pandas import Series

        # 如果样本权重是 pandas.Series 类型，抛出错误
        if isinstance(sample_weight, Series):
            raise ValueError(
                "Estimator does not accept 'sample_weight'of type pandas.Series"
            )
        return self

    def predict(self, X):
        # 对输入数据进行验证和处理
        X = check_array(X)
        # 返回一个预测结果全为1的数组，长度与输入数据行数相同
        return np.ones(X.shape[0])


class BadBalancedWeightsClassifier(BaseBadClassifier):
    # 继承自 BaseBadClassifier 类的一个分类器，处理坏的类权重计算情况
    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def fit(self, X, y):
        # 导入需要的库和函数
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import compute_class_weight

        # 对目标变量进行标签编码
        label_encoder = LabelEncoder().fit(y)
        # 获取所有类别
        classes = label_encoder.classes_
        # 计算类别权重
        class_weight = compute_class_weight(self.class_weight, classes=classes, y=y)

        # 故意修改平衡的类别权重，模拟一个 bug 并抛出异常
        if self.class_weight == "balanced":
            class_weight += 1.0

        # 将系数 coef_ 赋值为计算得到的类别权重
        self.coef_ = class_weight
        return self


class BadTransformerWithoutMixin(BaseEstimator):
    # 继承自 BaseEstimator 类的一个估计器，没有正确实现混合类
    def fit(self, X, y=None):
        # 验证和处理输入数据
        X = self._validate_data(X)
        return self

    def transform(self, X):
        # 对输入数据进行验证和处理
        X = check_array(X)
        return X


class NotInvariantPredict(BaseEstimator):
    # 继承自 BaseEstimator 类的一个估计器，处理不具备预测不变性的情况
    def fit(self, X, y):
        # 转换数据类型并验证数据
        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        return self

    def predict(self, X):
        # 对输入数据进行验证和处理
        X = check_array(X)
        # 如果输入数据的行数大于1，返回全为1的预测结果数组；否则返回全为0的数组
        if X.shape[0] > 1:
            return np.ones(X.shape[0])
        return np.zeros(X.shape[0])


class NotInvariantSampleOrder(BaseEstimator):
    # 继承自 BaseEstimator 类的一个估计器，处理不具备样本顺序不变性的情况
    def fit(self, X, y):
        # 转换数据类型并验证数据
        X, y = self._validate_data(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )
        # 将原始的输入数据保存在属性 _X 中，以便后续检查样本顺序
        self._X = X
        return self
    # 定义一个预测方法，接受输入参数 X
    def predict(self, X):
        # 调用 check_array 函数，确保输入 X 是一个可用的数组格式
        X = check_array(X)
        # 如果输入的 X 和存储在对象属性 _X 中的数据在元素上排序后相等，
        # 但其样本顺序不同，则返回一个全零数组。
        if (
            np.array_equiv(np.sort(X, axis=0), np.sort(self._X, axis=0))
            and (X != self._X).any()
        ):
            # 返回一个形状与 X 相同的全零数组
            return np.zeros(X.shape[0])
        # 否则，返回 X 的第一列数据
        return X[:, 0]
class OneClassSampleErrorClassifier(BaseBadClassifier):
    """Classifier allowing to trigger different behaviors when `sample_weight` reduces
    the number of classes to 1."""

    # 初始化分类器，设置是否在单一类时触发异常
    def __init__(self, raise_when_single_class=False):
        self.raise_when_single_class = raise_when_single_class

    # 拟合模型，接受特征矩阵 X，标签 y，可选的样本权重 sample_weight
    def fit(self, X, y, sample_weight=None):
        # 检查并转换 X, y 数据类型
        X, y = check_X_y(
            X, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True
        )

        # 初始化变量，判断是否存在单一类别
        self.has_single_class_ = False
        # 对标签进行唯一化处理，并返回转换后的标签
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes_ = self.classes_.shape[0]
        # 如果类别数小于 2 且设置了在单一类别时触发异常，则设置标志位并抛出异常
        if n_classes_ < 2 and self.raise_when_single_class:
            self.has_single_class_ = True
            raise ValueError("normal class error")

        # 根据样本权重确定经过处理后的类别数
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray) and len(sample_weight) > 0:
                n_classes_ = np.count_nonzero(np.bincount(y, sample_weight))
            # 如果类别数小于 2，则设置标志位并抛出异常
            if n_classes_ < 2:
                self.has_single_class_ = True
                raise ValueError("Nonsensical Error")

        return self

    # 使用已拟合的模型进行预测
    def predict(self, X):
        # 检查模型是否已经拟合
        check_is_fitted(self)
        # 检查并转换输入特征 X
        X = check_array(X)
        # 如果存在单一类别，返回全零数组；否则返回全一数组
        if self.has_single_class_:
            return np.zeros(X.shape[0])
        return np.ones(X.shape[0])


class LargeSparseNotSupportedClassifier(BaseEstimator):
    """Estimator that claims to support large sparse data
    (accept_large_sparse=True), but doesn't"""

    # 初始化分类器，设置是否在特定稀疏类型时触发异常
    def __init__(self, raise_for_type=None):
        # raise_for_type : str, expects "sparse_array" or "sparse_matrix"
        self.raise_for_type = raise_for_type

    # 拟合模型，接受特征矩阵 X 和标签 y
    def fit(self, X, y):
        # 验证并转换输入数据 X, y
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=("csr", "csc", "coo"),
            accept_large_sparse=True,
            multi_output=True,
            y_numeric=True,
        )
        # 根据设置的稀疏类型，检查 X 是否符合预期类型
        if self.raise_for_type == "sparse_array":
            correct_type = isinstance(X, sp.sparray)
        elif self.raise_for_type == "sparse_matrix":
            correct_type = isinstance(X, sp.spmatrix)
        if correct_type:
            # 如果 X 的格式为 COO，且行或列的数据类型为 int64，则抛出异常
            if X.format == "coo":
                if X.row.dtype == "int64" or X.col.dtype == "int64":
                    raise ValueError("Estimator doesn't support 64-bit indices")
            # 如果 X 的格式为 CSC 或 CSR，确保索引不是 int64 类型，否则断言异常
            elif X.format in ["csc", "csr"]:
                assert "int64" not in (
                    X.indices.dtype,
                    X.indptr.dtype,
                ), "Estimator doesn't support 64-bit indices"

        return self


class SparseTransformer(BaseEstimator):
    # 初始化转换器，设置稀疏容器参数
    def __init__(self, sparse_container=None):
        self.sparse_container = sparse_container

    # 拟合转换器，接受特征矩阵 X，可选的标签 y
    def fit(self, X, y=None):
        # 验证并转换输入数据 X，并记录其形状
        self.X_shape_ = self._validate_data(X).shape
        return self

    # 拟合并转换特征矩阵 X
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    # 定义一个类方法 `transform`，接收 self 和 X 作为参数
    def transform(self, X):
        # 调用 `check_array` 函数，将 X 转换为合适的数组格式
        X = check_array(X)
        # 检查 X 的列数是否与模型已知的特征数相匹配，如果不匹配则引发 ValueError 异常
        if X.shape[1] != self.X_shape_[1]:
            raise ValueError("Bad number of features")
        # 调用对象的 `sparse_container` 方法，对输入的 X 进行稀疏容器处理，并返回结果
        return self.sparse_container(X)
class EstimatorInconsistentForPandas(BaseEstimator):
    # 定义一个类，继承自 BaseEstimator，用于处理与 Pandas 数据结构不一致的估算器
    def fit(self, X, y):
        try:
            from pandas import DataFrame

            # 如果输入数据 X 是 DataFrame 类型
            if isinstance(X, DataFrame):
                # 从 DataFrame 中获取第一行第一列的数值，并存储在 self.value_ 中
                self.value_ = X.iloc[0, 0]
            else:
                # 否则，将 X 转换为数组，并从中获取第二行第一列的数值，并存储在 self.value_ 中
                X = check_array(X)
                self.value_ = X[1, 0]
            # 返回当前对象自身
            return self

        except ImportError:
            # 如果导入 pandas 失败，将 X 转换为数组，并从中获取第二行第一列的数值，并存储在 self.value_ 中
            X = check_array(X)
            self.value_ = X[1, 0]
            # 返回当前对象自身
            return self

    # 预测方法，接受输入 X，将其转换为数组，返回一个数组，其元素为 self.value_ 的重复
    def predict(self, X):
        X = check_array(X)
        return np.array([self.value_] * X.shape[0])


class UntaggedBinaryClassifier(SGDClassifier):
    # 玩具分类器，仅支持二分类，不会通过所有测试
    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        # 调用父类的 fit 方法，进行模型拟合
        super().fit(X, y, coef_init, intercept_init, sample_weight)
        # 如果类别数量大于2，抛出值错误异常
        if len(self.classes_) > 2:
            raise ValueError("Only 2 classes are supported")
        # 返回当前对象自身
        return self

    # 部分拟合方法，调用父类的 partial_fit 方法，进行模型部分拟合
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        super().partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
        # 如果类别数量大于2，抛出值错误异常
        if len(self.classes_) > 2:
            raise ValueError("Only 2 classes are supported")
        # 返回当前对象自身
        return self


class TaggedBinaryClassifier(UntaggedBinaryClassifier):
    # 玩具分类器，仅支持二分类
    def _more_tags(self):
        # 返回一个包含 "binary_only": True 的字典
        return {"binary_only": True}


class EstimatorMissingDefaultTags(BaseEstimator):
    # 继承自 BaseEstimator 的估算器，去除默认标签
    def _get_tags(self):
        # 调用父类的 _get_tags 方法，复制其结果到 tags 变量中
        tags = super()._get_tags().copy()
        # 删除 tags 中的 "allow_nan" 键
        del tags["allow_nan"]
        # 返回修改后的 tags 字典
        return tags


class RequiresPositiveXRegressor(LinearRegression):
    # 要求输入 X 为正的回归器，继承自 LinearRegression
    def fit(self, X, y):
        # 调用父类的 _validate_data 方法，验证输入数据 X 和 y，要求多输出
        X, y = self._validate_data(X, y, multi_output=True)
        # 如果 X 中有任意元素小于0，抛出值错误异常
        if (X < 0).any():
            raise ValueError("negative X values not supported!")
        # 调用父类的 fit 方法，进行模型拟合
        return super().fit(X, y)

    # 返回一个包含 "requires_positive_X": True 的字典
    def _more_tags(self):
        return {"requires_positive_X": True}


class RequiresPositiveYRegressor(LinearRegression):
    # 要求输入 y 为正的回归器，继承自 LinearRegression
    def fit(self, X, y):
        # 调用父类的 _validate_data 方法，验证输入数据 X 和 y，要求多输出
        X, y = self._validate_data(X, y, multi_output=True)
        # 如果 y 中有任意元素小于等于0，抛出值错误异常
        if (y <= 0).any():
            raise ValueError("negative y values not supported!")
        # 调用父类的 fit 方法，进行模型拟合
        return super().fit(X, y)

    # 返回一个包含 "requires_positive_y": True 的字典
    def _more_tags(self):
        return {"requires_positive_y": True}


class PoorScoreLogisticRegression(LogisticRegression):
    # 返回调用父类 decision_function 方法的结果加1
    def decision_function(self, X):
        return super().decision_function(X) + 1

    # 返回一个包含 "poor_score": True 的字典
    def _more_tags(self):
        return {"poor_score": True}


class PartialFitChecksName(BaseEstimator):
    # 部分拟合检查名称的估算器，继承自 BaseEstimator
    def fit(self, X, y):
        # 调用 _validate_data 方法，验证输入数据 X 和 y
        self._validate_data(X, y)
        # 返回当前对象自身
        return self

    # 部分拟合方法，首次拟合时重置 _fitted 属性，再次拟合时不重置
    def partial_fit(self, X, y):
        reset = not hasattr(self, "_fitted")
        # 调用 _validate_data 方法，验证输入数据 X 和 y，根据 reset 决定是否重置
        self._validate_data(X, y, reset=reset)
        # 设置 _fitted 属性为 True
        self._fitted = True
        # 返回当前对象自身
        return self


class BrokenArrayAPI(BaseEstimator):
    """Make different predictions when using Numpy and the Array API"""
    # 定义一个类，继承自 BaseEstimator，描述了在使用 Numpy 和 Array API 时进行不同预测的情况
    def fit(self, X, y):
        # 返回当前对象自身，未定义具体的拟合操作
        return self
    # 定义一个预测函数，接受输入 X
    def predict(self, X):
        # 获取配置中的数组 API 调度信息
        enabled = get_config()["array_api_dispatch"]
        # 获取 X 的数组命名空间
        xp, _ = _array_api.get_namespace(X)
        # 如果数组 API 调度已启用
        if enabled:
            # 使用 xp 将列表 [1, 2, 3] 转换为数组并返回
            return xp.asarray([1, 2, 3])
        # 如果数组 API 调度未启用
        else:
            # 使用 np 将列表 [3, 2, 1] 转换为数组并返回
            return np.array([3, 2, 1])
# 测试函数，检查是否可以导入 array_api_compat 模块
def test_check_array_api_input():
    try:
        importlib.import_module("array_api_compat")
    except ModuleNotFoundError:
        # 如果模块不存在，则跳过测试，并显示相应的错误信息
        raise SkipTest("array_api_compat is required to run this test")
    
    try:
        importlib.import_module("array_api_strict")
    except ModuleNotFoundError:  # pragma: nocover
        # 如果模块不存在，则跳过测试，并显示相应的错误信息
        raise SkipTest("array-api-strict is required to run this test")

    # 使用 raises 断言来检查函数是否会抛出 AssertionError 异常，并匹配特定的错误消息
    with raises(AssertionError, match="Not equal to tolerance"):
        check_array_api_input(
            "BrokenArrayAPI",  # 第一个参数：字符串 "BrokenArrayAPI"
            BrokenArrayAPI(),   # 第二个参数：BrokenArrayAPI 类的实例
            array_namespace="array_api_strict",  # array_namespace 参数设置为 "array_api_strict"
            check_values=True,  # check_values 参数设置为 True
        )


# 测试函数，检查非数组的数组函数
def test_not_an_array_array_function():
    # 创建一个 _NotAnArray 的实例，传入参数为 np.ones(10)
    not_array = _NotAnArray(np.ones(10))
    msg = "Don't want to call array_function sum!"
    # 使用 raises 断言来检查函数是否会抛出 TypeError 异常，并匹配特定的错误消息
    with raises(TypeError, match=msg):
        np.sum(not_array)
    
    # 使用 assert 断言来验证 np.may_share_memory(not_array, None) 的返回值为 True
    assert np.may_share_memory(not_array, None)  # always returns True


# 测试函数，检查 check_fit_score_takes_y 在具有过时 fit 方法的类上是否有效
def test_check_fit_score_takes_y_works_on_deprecated_fit():
    # 定义一个具有过时 fit 方法的测试类
    class TestEstimatorWithDeprecatedFitMethod(BaseEstimator):
        @deprecated("Deprecated for the purpose of testing check_fit_score_takes_y")
        def fit(self, X, y):
            return self

    # 调用 check_fit_score_takes_y 函数，传入测试名称和 TestEstimatorWithDeprecatedFitMethod 类的实例
    check_fit_score_takes_y("test", TestEstimatorWithDeprecatedFitMethod())


# 测试函数，检查检查器是否可以成功识别 "坏" 估算器
def test_check_estimator():
    # 测试估算器实际上在 "坏" 估算器上失败的情况
    # 这并非是所有检查的完整测试，因为检查非常广泛。

    # 检查是否具有 set_params 方法并且可以克隆
    msg = "Passing a class was deprecated"
    with raises(TypeError, match=msg):
        check_estimator(object)

    msg = (
        "Parameter 'p' of estimator 'HasMutableParameters' is of type "
        "object which is not allowed"
    )
    # 检查 "default_constructible" 测试是否检查可变参数
    check_estimator(HasImmutableParameters())  # 应该通过
    with raises(AssertionError, match=msg):
        check_estimator(HasMutableParameters())

    # 检查 get_params 返回的值是否与 set_params 的参数匹配
    msg = "get_params result does not match what was passed to set_params"
    with raises(AssertionError, match=msg):
        check_estimator(ModifiesValueInsteadOfRaisingError())

    with warnings.catch_warnings(record=True) as records:
        check_estimator(RaisesErrorInSetParams())
    assert UserWarning in [rec.category for rec in records]

    with raises(AssertionError, match=msg):
        check_estimator(ModifiesAnotherValue())

    # 检查是否具有 fit 方法
    msg = "object has no attribute 'fit'"
    with raises(AttributeError, match=msg):
        check_estimator(BaseEstimator())

    # 检查 fit 方法是否进行输入验证
    msg = "Did not raise"
    with raises(AssertionError, match=msg):
        check_estimator(BaseBadClassifier())

    # 检查 fit 方法是否接受 pandas.Series 类型的 sample_weights 参数
    try:
        from pandas import Series  # noqa
        # 引入 pandas 的 Series 类型用于测试
        msg = (
            "Estimator NoSampleWeightPandasSeriesType raises error if "
            "'sample_weight' parameter is of type pandas.Series"
        )
        # 设置期望的异常消息，用于检查 estimator 的行为
        with raises(ValueError, match=msg):
            # 检查 NoSampleWeightPandasSeriesType 的行为是否符合预期
            check_estimator(NoSampleWeightPandasSeriesType())
    except ImportError:
        pass

    # 检查 predict 方法是否进行了输入验证（不接受字典作为输入）
    msg = "Estimator NoCheckinPredict doesn't check for NaN and inf in predict"
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 NoCheckinPredict 的行为是否符合预期
        check_estimator(NoCheckinPredict())

    # 检查 estimator 在 transform/predict/predict_proba 时是否会改变其状态
    msg = "Estimator changes __dict__ during predict"
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 ChangesDict 的行为是否符合预期
        check_estimator(ChangesDict())

    # 检查 `fit` 方法是否只会修改私有属性（以 _ 开头或以 _ 结尾）
    msg = (
        "Estimator ChangesWrongAttribute should not change or mutate  "
        "the parameter wrong_attribute from 0 to 1 during fit."
    )
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 ChangesWrongAttribute 的行为是否符合预期
        check_estimator(ChangesWrongAttribute())

    # 检查 `fit` 方法是否不会添加任何公共属性
    msg = (
        r"Estimator adds public attribute\(s\) during the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but wrong_attribute added"
    )
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 SetsWrongAttribute 的行为是否符合预期
        check_estimator(SetsWrongAttribute())

    # 检查样本顺序的不变性
    name = NotInvariantSampleOrder.__name__
    method = "predict"
    msg = (
        "{method} of {name} is not invariant when applied to a dataset"
        "with different sample order."
    ).format(method=method, name=name)
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 NotInvariantSampleOrder 的行为是否符合预期
        check_estimator(NotInvariantSampleOrder())

    # 检查方法的不变性
    name = NotInvariantPredict.__name__
    method = "predict"
    msg = ("{method} of {name} is not invariant when applied to a subset.").format(
        method=method, name=name
    )
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 NotInvariantPredict 的行为是否符合预期
        check_estimator(NotInvariantPredict())

    # 检查稀疏数据输入处理能力
    name = NoSparseClassifier.__name__
    msg = "Estimator %s doesn't seem to fail gracefully on sparse data" % name
    # 设置期望的异常消息，用于检查 estimator 的行为
    with raises(AssertionError, match=msg):
        # 检查 NoSparseClassifier 的行为是否符合预期
        check_estimator(NoSparseClassifier("sparse_matrix"))

    if SPARRAY_PRESENT:
        # 如果存在 SPARRAY_PRESENT，则检查稀疏数组输入处理能力
        with raises(AssertionError, match=msg):
            # 检查 NoSparseClassifier 的行为是否符合预期
            check_estimator(NoSparseClassifier("sparse_array"))

    # 检查分类器在通过样本权重导致分类少于两类时的行为
    name = OneClassSampleErrorClassifier.__name__
    # 构造错误消息，用于检查模型在单个标签上拟合后的样本权重修剪时是否失败
    msg = (
        f"{name} failed when fitted on one label after sample_weight "
        "trimming. Error message is not explicit, it should have "
        "'class'."
    )
    # 使用 pytest 的 raises 断言来验证抛出 AssertionError，并匹配预期的错误消息
    with raises(AssertionError, match=msg):
        check_estimator(OneClassSampleErrorClassifier())

    # 测试在不支持大型稀疏矩阵的估计器上是否会引发 AssertionError
    msg = (
        "Estimator LargeSparseNotSupportedClassifier doesn't seem to "
        r"support \S{3}_64 matrix, and is not failing gracefully.*"
    )
    with raises(AssertionError, match=msg):
        check_estimator(LargeSparseNotSupportedClassifier("sparse_matrix"))

    # 如果 SPARRAY_PRESENT 为 True，同样测试在不支持大型稀疏数组的估计器上是否会引发 AssertionError
    if SPARRAY_PRESENT:
        with raises(AssertionError, match=msg):
            check_estimator(LargeSparseNotSupportedClassifier("sparse_array"))

    # 验证只支持二进制分类的估计器是否会在支持的类别数不是 2 时引发 ValueError
    msg = "Only 2 classes are supported"
    with raises(ValueError, match=msg):
        check_estimator(UntaggedBinaryClassifier())

    # 对于每种 CSR_CONTAINERS，验证能否正确处理将稠密数据转换为稀疏数据的估计器
    for csr_container in CSR_CONTAINERS:
        # 非回归测试，确保估计器能够正确转换稀疏数据
        check_estimator(SparseTransformer(sparse_container=csr_container))

    # 验证是否在 LogisticRegression 估计器上没有引发异常
    check_estimator(LogisticRegression())
    check_estimator(LogisticRegression(C=0.01))
    check_estimator(MultiTaskElasticNet())

    # 验证是否在二进制分类标记的估计器上没有引发异常
    check_estimator(TaggedBinaryClassifier())
    check_estimator(RequiresPositiveXRegressor())

    # 验证具有 requires_positive_y 估计器标记的回归器是否在负值 y 引发异常
    msg = "negative y values not supported!"
    with raises(ValueError, match=msg):
        check_estimator(RequiresPositiveYRegressor())

    # 验证带有 poor_score 标记的分类器是否没有引发异常
    check_estimator(PoorScoreLogisticRegression())
def test_check_outlier_corruption():
    # should raise AssertionError
    # 准备一个包含异常值的决策数组
    decision = np.array([0.0, 1.0, 1.5, 2.0])
    # 使用 raises 上下文，期望捕获 AssertionError 异常
    with raises(AssertionError):
        # 调用检查异常值的函数，传入参数 1, 2, decision
        check_outlier_corruption(1, 2, decision)
    # should pass
    # 准备一个没有异常值的决策数组
    decision = np.array([0.0, 1.0, 1.0, 2.0])
    # 调用检查异常值的函数，传入参数 1, 2, decision
    check_outlier_corruption(1, 2, decision)


def test_check_estimator_transformer_no_mixin():
    # check that TransformerMixin is not required for transformer tests to run
    # 使用 raises 上下文，期望捕获 AttributeError 异常，并且异常消息中包含 "fit_transform"
    with raises(AttributeError, ".*fit_transform.*"):
        # 调用检查评估器的函数，传入一个没有混合类的错误的转换器
        check_estimator(BadTransformerWithoutMixin())


def test_check_estimator_clones():
    # check that check_estimator doesn't modify the estimator it receives
    # 导入鸢尾花数据集
    from sklearn.datasets import load_iris

    iris = load_iris()

    for Estimator in [
        GaussianMixture,
        LinearRegression,
        SGDClassifier,
        PCA,
        ExtraTreesClassifier,
        MiniBatchKMeans,
    ]:
        # without fitting
        # 使用 ignore_warnings 上下文，忽略 ConvergenceWarning 警告
        with ignore_warnings(category=ConvergenceWarning):
            # 创建估计器对象
            est = Estimator()
            # 设置检查参数
            _set_checking_parameters(est)
            # 设置随机状态
            set_random_state(est)
            # 记录估计器的哈希值
            old_hash = joblib.hash(est)
            # 调用检查评估器的函数
            check_estimator(est)
        # 断言检查前后估计器的哈希值未发生变化
        assert old_hash == joblib.hash(est)

        # with fitting
        # 使用 ignore_warnings 上下文，忽略 ConvergenceWarning 警告
        with ignore_warnings(category=ConvergenceWarning):
            # 创建估计器对象
            est = Estimator()
            # 设置检查参数
            _set_checking_parameters(est)
            # 设置随机状态
            set_random_state(est)
            # 使用鸢尾花数据进行拟合
            est.fit(iris.data + 10, iris.target)
            # 记录估计器的哈希值
            old_hash = joblib.hash(est)
            # 调用检查评估器的函数
            check_estimator(est)
        # 断言检查前后估计器的哈希值未发生变化
        assert old_hash == joblib.hash(est)


def test_check_estimators_unfitted():
    # check that a ValueError/AttributeError is raised when calling predict
    # on an unfitted estimator
    # 设置错误消息
    msg = "Did not raise"
    # 使用 raises 上下文，期望捕获 AssertionError 异常，并且异常消息匹配 msg
    with raises(AssertionError, match=msg):
        # 调用检查未拟合评估器的函数，传入字符串 "estimator" 和一个未拟合的分类器
        check_estimators_unfitted("estimator", NoSparseClassifier())

    # check that CorrectNotFittedError inherit from either ValueError
    # or AttributeError
    # 调用检查未拟合评估器的函数，传入字符串 "estimator" 和一个正确的未拟合异常分类器
    check_estimators_unfitted("estimator", CorrectNotFittedErrorClassifier())


def test_check_no_attributes_set_in_init():
    class NonConformantEstimatorPrivateSet(BaseEstimator):
        def __init__(self):
            # 定义一个不应该在初始化期间设置的属性
            self.you_should_not_set_this_ = None

    class NonConformantEstimatorNoParamSet(BaseEstimator):
        def __init__(self, you_should_set_this_=None):
            pass

    class ConformantEstimatorClassAttribute(BaseEstimator):
        # making sure our __metadata_request__* class attributes are okay!
        # 确保我们的 __metadata_request__* 类属性是正确的！
        __metadata_request__fit = {"foo": True}

    # 设置错误消息
    msg = (
        "Estimator estimator_name should not set any"
        " attribute apart from parameters during init."
        r" Found attributes \['you_should_not_set_this_'\]."
    )
    # 使用 raises 上下文，期望捕获 AssertionError 异常，并且异常消息匹配 msg
    with raises(AssertionError, match=msg):
        # 调用检查初始化函数中没有设置额外属性的函数，传入字符串 "estimator_name" 和一个非一致的未设置私有属性的估计器
        check_no_attributes_set_in_init(
            "estimator_name", NonConformantEstimatorPrivateSet()
        )

    # 设置错误消息
    msg = (
        "Estimator estimator_name should store all parameters as an attribute"
        " during init"
    )
    # 使用 pytest 的 raises 断言，验证是否会引发 AttributeError 并匹配特定错误消息
    with raises(AttributeError, match=msg):
        check_no_attributes_set_in_init(
            "estimator_name", NonConformantEstimatorNoParamSet()
        )

    # 检查在初始化中设置类属性是否符合规范
    check_no_attributes_set_in_init(
        "estimator_name", ConformantEstimatorClassAttribute()
    )

    # 在启用元数据路由的配置环境下，检查克隆具有非默认设置请求的估算器是否符合规范。
    # 通过 `set_{method}_request` 设置非默认值会设置私有属性 _metadata_request，
    # 该属性在 `clone` 方法中被复制。
    with config_context(enable_metadata_routing=True):
        check_no_attributes_set_in_init(
            "estimator_name",
            ConformantEstimatorClassAttribute().set_fit_request(foo=True),
        )
def test_check_estimator_pairwise():
    # check that check_estimator() works on estimator with _pairwise
    # kernel or metric

    # test precomputed kernel
    est = SVC(kernel="precomputed")
    check_estimator(est)

    # test precomputed metric
    est = KNeighborsRegressor(metric="precomputed")
    check_estimator(est)


def test_check_classifier_data_not_an_array():
    # Verify that check_classifier_data_not_an_array raises an AssertionError
    # with message matching "Not equal to tolerance"
    with raises(AssertionError, match="Not equal to tolerance"):
        check_classifier_data_not_an_array(
            "estimator_name", EstimatorInconsistentForPandas()
        )


def test_check_regressor_data_not_an_array():
    # Verify that check_regressor_data_not_an_array raises an AssertionError
    # with message matching "Not equal to tolerance"
    with raises(AssertionError, match="Not equal to tolerance"):
        check_regressor_data_not_an_array(
            "estimator_name", EstimatorInconsistentForPandas()
        )


def test_check_estimator_get_tags_default_keys():
    # Verify that EstimatorMissingDefaultTags raises an AssertionError
    # with specific error message when default tags are missing
    estimator = EstimatorMissingDefaultTags()
    err_msg = (
        r"EstimatorMissingDefaultTags._get_tags\(\) is missing entries"
        r" for the following default tags: {'allow_nan'}"
    )
    with raises(AssertionError, match=err_msg):
        check_estimator_get_tags_default_keys(estimator.__class__.__name__, estimator)

    # Verify no error occurs when _get_tags is not available
    estimator = MinimalTransformer()
    check_estimator_get_tags_default_keys(estimator.__class__.__name__, estimator)


def test_check_dataframe_column_names_consistency():
    # Verify that ValueError is raised with specific error message when
    # Estimator does not have feature_names_in_ attribute
    err_msg = "Estimator does not have a feature_names_in_"
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency("estimator_name", BaseBadClassifier())

    # Verify no error occurs when using PartialFitChecksName()
    check_dataframe_column_names_consistency("estimator_name", PartialFitChecksName())

    # Verify no error occurs with LogisticRegression() and valid docstring
    lr = LogisticRegression()
    check_dataframe_column_names_consistency(lr.__class__.__name__, lr)

    # Verify ValueError is raised with specific error message when
    # LogisticRegression does not document feature_names_in_ attribute
    lr.__doc__ = "Docstring that does not document the estimator's attributes"
    err_msg = (
        "Estimator LogisticRegression does not document its feature_names_in_ attribute"
    )
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency(lr.__class__.__name__, lr)


class _BaseMultiLabelClassifierMock(ClassifierMixin, BaseEstimator):
    def __init__(self, response_output):
        self.response_output = response_output

    def fit(self, X, y):
        return self

    def _more_tags(self):
        return {"multilabel": True}


def test_check_classifiers_multilabel_output_format_predict():
    # Test case setup
    n_samples, test_size, n_outputs = 100, 25, 5
    _, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    y_test = y[-test_size:]

    # 1. inconsistent array type
    # Create MultiLabelClassifierPredict instance with specified response_output
    # which is expected to raise an issue regarding inconsistent array type
    clf = MultiLabelClassifierPredict(response_output=y_test.tolist())
    # 定义错误消息，指出预期 MultiLabelClassifierPredict.predict 输出 NumPy 数组，但得到了 list 类型。
    err_msg = (
        r"MultiLabelClassifierPredict.predict is expected to output a "
        r"NumPy array. Got <class 'list'> instead."
    )
    # 使用 pytest 的 raises 断言，检查是否抛出 AssertionError，且错误消息匹配指定的 err_msg。
    with raises(AssertionError, match=err_msg):
        # 调用函数 check_classifiers_multilabel_output_format_predict 检查分类器的多标签输出格式。
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
        
    # 2. inconsistent shape
    # 创建 MultiLabelClassifierPredict 对象，传入 y_test 的部分切片作为 response_output 参数。
    clf = MultiLabelClassifierPredict(response_output=y_test[:, :-1])
    # 定义错误消息，指出预期 MultiLabelClassifierPredict.predict 输出形状为 (25, 4)，但实际输出形状为 (25, 5)。
    err_msg = (
        r"MultiLabelClassifierPredict.predict outputs a NumPy array of "
        r"shape \(25, 4\) instead of \(25, 5\)."
    )
    # 使用 pytest 的 raises 断言，检查是否抛出 AssertionError，且错误消息匹配指定的 err_msg。
    with raises(AssertionError, match=err_msg):
        # 再次调用函数检查分类器的多标签输出格式。
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
        
    # 3. inconsistent dtype
    # 创建 MultiLabelClassifierPredict 对象，传入经过 np.float64 类型转换后的 y_test 作为 response_output 参数。
    clf = MultiLabelClassifierPredict(response_output=y_test.astype(np.float64))
    # 定义错误消息，指出预期 MultiLabelClassifierPredict.predict 输出的数据类型与目标数据类型不匹配。
    err_msg = (
        r"MultiLabelClassifierPredict.predict does not output the same "
        r"dtype than the targets."
    )
    # 使用 pytest 的 raises 断言，检查是否抛出 AssertionError，且错误消息匹配指定的 err_msg。
    with raises(AssertionError, match=err_msg):
        # 第三次调用函数检查分类器的多标签输出格式。
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
def test_check_classifiers_multilabel_output_format_predict_proba():
    # 设置测试样本数量、测试集大小、输出标签数量
    n_samples, test_size, n_outputs = 100, 25, 5
    # 生成多标签分类的数据集，返回特征和标签
    _, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    # 从标签中选取测试集
    y_test = y[-test_size:]

    # 定义一个多标签分类器，用于测试预测概率的输出格式
    class MultiLabelClassifierPredictProba(_BaseMultiLabelClassifierMock):
        def predict_proba(self, X):
            return self.response_output

    # 针对不同的 CSR 容器进行测试
    for csr_container in CSR_CONTAINERS:
        # 1. 未知的输出类型
        clf = MultiLabelClassifierPredictProba(response_output=csr_container(y_test))
        err_msg = (
            f"Unknown returned type .*{csr_container.__name__}.* by "
            r"MultiLabelClassifierPredictProba.predict_proba. A list or a Numpy "
            r"array is expected."
        )
        # 确保抛出值错误，并检查错误消息
        with raises(ValueError, match=err_msg):
            check_classifiers_multilabel_output_format_predict_proba(
                clf.__class__.__name__,
                clf,
            )
    
    # 2. 对于列表输出
    # 2.1. 不一致的长度
    clf = MultiLabelClassifierPredictProba(response_output=y_test.tolist())
    err_msg = (
        "When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        "the list should be of length n_outputs and contain NumPy arrays. Got "
        f"length of {test_size} instead of {n_outputs}."
    )
    # 确保抛出断言错误，并检查错误消息
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    
    # 2.2. 不一致形状的数组
    response_output = [np.ones_like(y_test) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        r"this list should contain NumPy arrays of shape \(n_samples, 2\). Got "
        r"NumPy arrays of shape \(25, 5\) instead of \(25, 2\)."
    )
    # 确保抛出断言错误，并检查错误消息
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    
    # 2.3. 不一致的数据类型的数组
    response_output = [
        np.ones(shape=(y_test.shape[0], 2), dtype=np.int64) for _ in range(n_outputs)
    ]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = (
        "When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        "it should contain NumPy arrays with floating dtype."
    )
    # 确保抛出断言错误，并检查错误消息
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    
    # 2.4. 数组不包含概率（每行应该总和为1）
    # 创建一个包含 n_outputs 个二维数组的列表，每个数组形状为 (y_test.shape[0], 2)，数据类型为 np.float64，每个元素都是 1.0
    response_output = [
        np.ones(shape=(y_test.shape[0], 2), dtype=np.float64) for _ in range(n_outputs)
    ]
    # 创建 MultiLabelClassifierPredictProba 的实例 clf，使用上面创建的 response_output 作为参数
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    # 定义错误信息，用于断言异常，指出当 predict_proba 返回一个列表时，每个 NumPy 数组应包含每个类的概率，每行概率之和应为 1
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a list, "
        r"each NumPy array should contain probabilities for each class and "
        r"thus each row should sum to 1"
    )
    # 断言异常，检查 clf 是否符合上述错误信息定义的格式
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    
    # 3 for array output
    # 3.1. 数组形状不一致的情况
    # 创建一个 MultiLabelClassifierPredictProba 的实例 clf，使用 y_test[:, :-1] 作为 response_output 参数
    clf = MultiLabelClassifierPredictProba(response_output=y_test[:, :-1])
    # 定义错误信息，指出当 predict_proba 返回一个 NumPy 数组时，期望的形状是 (n_samples, n_outputs)，但得到的形状是 (25, 4)，而不是期望的 (25, 5)
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a NumPy "
        r"array, the expected shape is \(n_samples, n_outputs\). Got \(25, 4\)"
        r" instead of \(25, 5\)."
    )
    # 断言异常，检查 clf 是否符合上述错误信息定义的格式
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    
    # 3.2. 数组数据类型不一致的情况
    # 将 response_output 设置为与 y_test 相同形状的零数组，数据类型为 np.int64
    response_output = np.zeros_like(y_test, dtype=np.int64)
    # 创建一个 MultiLabelClassifierPredictProba 的实例 clf，使用上面创建的 response_output 作为参数
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    # 定义错误信息，指出当 predict_proba 返回一个 NumPy 数组时，期望的数据类型是浮点型
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a NumPy "
        r"array, the expected data type is floating."
    )
    # 断言异常，检查 clf 是否符合上述错误信息定义的格式
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
    
    # 4. 数组不包含概率值的情况
    # 创建一个 MultiLabelClassifierPredictProba 的实例 clf，使用 y_test * 2.0 作为 response_output 参数
    clf = MultiLabelClassifierPredictProba(response_output=y_test * 2.0)
    # 定义错误信息，指出当 predict_proba 返回一个 NumPy 数组时，该数组应提供正类的概率，因此应包含在 0 到 1 之间的值
    err_msg = (
        r"When MultiLabelClassifierPredictProba.predict_proba returns a NumPy "
        r"array, this array is expected to provide probabilities of the "
        r"positive class and should therefore contain values between 0 and 1."
    )
    # 断言异常，检查 clf 是否符合上述错误信息定义的格式
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(
            clf.__class__.__name__,
            clf,
        )
def test_check_classifiers_multilabel_output_format_decision_function():
    # 设置测试数据的样本数量、测试集大小和输出数量
    n_samples, test_size, n_outputs = 100, 25, 5
    # 生成多标签分类数据集
    _, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    # 从生成的数据中获取测试集
    y_test = y[-test_size:]

    # 定义一个模拟多标签分类器的类，重写 decision_function 方法
    class MultiLabelClassifierDecisionFunction(_BaseMultiLabelClassifierMock):
        def decision_function(self, X):
            return self.response_output

    # 1. 测试不一致的数组类型
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test.tolist())
    err_msg = (
        r"MultiLabelClassifierDecisionFunction.decision_function is expected "
        r"to output a NumPy array. Got <class 'list'> instead."
    )
    # 断言错误消息是否符合预期
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(
            clf.__class__.__name__,
            clf,
        )
    # 2. 测试不一致的形状
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test[:, :-1])
    err_msg = (
        r"MultiLabelClassifierDecisionFunction.decision_function is expected "
        r"to provide a NumPy array of shape \(n_samples, n_outputs\). Got "
        r"\(25, 4\) instead of \(25, 5\)"
    )
    # 断言错误消息是否符合预期
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(
            clf.__class__.__name__,
            clf,
        )
    # 3. 测试不一致的数据类型
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test)
    err_msg = (
        r"MultiLabelClassifierDecisionFunction.decision_function is expected "
        r"to output a floating dtype."
    )
    # 断言错误消息是否符合预期
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(
            clf.__class__.__name__,
            clf,
        )


def run_tests_without_pytest():
    """Runs the tests in this file without using pytest."""
    # 获取当前模块
    main_module = sys.modules["__main__"]
    # 获取所有以'test_'开头的测试函数
    test_functions = [
        getattr(main_module, name)
        for name in dir(main_module)
        if name.startswith("test_")
    ]
    # 创建测试用例
    test_cases = [unittest.FunctionTestCase(fn) for fn in test_functions]
    # 创建测试套件
    suite = unittest.TestSuite()
    suite.addTests(test_cases)
    # 运行测试
    runner = unittest.TextTestRunner()
    runner.run(suite)


def test_check_class_weight_balanced_linear_classifier():
    # 检查计算不正确的平衡权重是否引发异常
    msg = "Classifier estimator_name is not computing class_weight=balanced properly"
    # 断言错误消息是否符合预期
    with raises(AssertionError, match=msg):
        check_class_weight_balanced_linear_classifier(
            "estimator_name", BadBalancedWeightsClassifier
        )


def test_all_estimators_all_public():
    # 当未安装 pytest 时，all_estimator 不应失败，并且应返回仅公共估计器
    # 使用 `warnings` 模块捕获所有警告，并记录到 `record` 中
    with warnings.catch_warnings(record=True) as record:
        # 获取所有可用的估计器类的列表
        estimators = all_estimators()
    # 确保没有警告被触发，即 `record` 应为空
    assert not record
    # 遍历所有估计器类的列表
    for est in estimators:
        # 确保估计器类的名称不是以下划线开头的内部类
        assert not est.__class__.__name__.startswith("_")
if __name__ == "__main__":
    # 如果当前模块被作为脚本直接执行，运行测试函数以检查是否依赖于 pytest 进行估计器检查。
    run_tests_without_pytest()


def test_xfail_ignored_in_check_estimator():
    # 确保标记为 xfail 的检查在 check_estimator() 中被忽略而不会运行，但仍会引发警告。
    with warnings.catch_warnings(record=True) as records:
        check_estimator(NuSVC())
    # 断言是否在警告记录中存在 SkipTestWarning 类别的记录
    assert SkipTestWarning in [rec.category for rec in records]


# FIXME: 当检查变得足够详细时，应取消注释此测试。在 0.24 版本中，由于估计器性能不佳，这些测试会失败。
def test_minimal_class_implementation_checks():
    # 检查第三方库可以在不继承 BaseEstimator 的情况下运行测试。
    # FIXME
    raise SkipTest
    minimal_estimators = [MinimalTransformer(), MinimalRegressor(), MinimalClassifier()]
    for estimator in minimal_estimators:
        check_estimator(estimator)


def test_check_fit_check_is_fitted():
    class Estimator(BaseEstimator):
        def __init__(self, behavior="attribute"):
            self.behavior = behavior

        def fit(self, X, y, **kwargs):
            if self.behavior == "attribute":
                self.is_fitted_ = True
            elif self.behavior == "method":
                self._is_fitted = True
            return self

        @available_if(lambda self: self.behavior in {"method", "always-true"})
        def __sklearn_is_fitted__(self):
            if self.behavior == "always-true":
                return True
            return hasattr(self, "_is_fitted")

    with raises(Exception, match="passes check_is_fitted before being fit"):
        check_fit_check_is_fitted("estimator", Estimator(behavior="always-true"))

    check_fit_check_is_fitted("estimator", Estimator(behavior="method"))
    check_fit_check_is_fitted("estimator", Estimator(behavior="attribute"))


def test_check_requires_y_none():
    class Estimator(BaseEstimator):
        def fit(self, X, y):
            X, y = check_X_y(X, y)

    with warnings.catch_warnings(record=True) as record:
        check_requires_y_none("estimator", Estimator())

    # 断言没有引发任何警告
    assert not [r.message for r in record]


def test_non_deterministic_estimator_skip_tests():
    # 检查标记为 non_deterministic=True 的估计器将跳过某些测试，详情请参阅 issue #22313。
    for est in [MinimalTransformer, MinimalRegressor, MinimalClassifier]:
        all_tests = list(_yield_all_checks(est()))
        assert check_methods_sample_order_invariance in all_tests
        assert check_methods_subset_invariance in all_tests

        class Estimator(est):
            def _more_tags(self):
                return {"non_deterministic": True}

        all_tests = list(_yield_all_checks(Estimator()))
        assert check_methods_sample_order_invariance not in all_tests
        assert check_methods_subset_invariance not in all_tests
# 检查异常检测器中的污染参数的测试。
def test_check_outlier_contamination():
    """Check the test for the contamination parameter in the outlier detectors."""

    # 没有任何参数约束时，异常检测器会通过返回 None 来尽早退出测试。
    class OutlierDetectorWithoutConstraint(OutlierMixin, BaseEstimator):
        """Outlier detector without parameter validation."""

        def __init__(self, contamination=0.1):
            self.contamination = contamination

        def fit(self, X, y=None, sample_weight=None):
            return self  # pragma: no cover

        def predict(self, X, y=None):
            return np.ones(X.shape[0])

    detector = OutlierDetectorWithoutConstraint()
    # 断言异常检测器未通过污染参数测试时返回 None
    assert check_outlier_contamination(detector.__class__.__name__, detector) is None

    # 现在，我们检查参数约束是否正确，测试只有在提供了界于 [0, 1] 的区间约束时才有效。
    class OutlierDetectorWithConstraint(OutlierDetectorWithoutConstraint):
        _parameter_constraints = {"contamination": [StrOptions({"auto"})]}

    detector = OutlierDetectorWithConstraint()
    err_msg = "contamination constraints should contain a Real Interval constraint."
    # 使用错误的参数约束，断言会引发 AssertionError，并匹配错误信息
    with raises(AssertionError, match=err_msg):
        check_outlier_contamination(detector.__class__.__name__, detector)

    # 添加正确的区间约束，并检查测试是否通过。
    OutlierDetectorWithConstraint._parameter_constraints["contamination"] = [
        Interval(Real, 0, 0.5, closed="right")
    ]
    detector = OutlierDetectorWithConstraint()
    check_outlier_contamination(detector.__class__.__name__, detector)

    # 错误的区间约束列表
    incorrect_intervals = [
        Interval(Integral, 0, 1, closed="right"),  # 不是整数区间
        Interval(Real, -1, 1, closed="right"),    # 下界为负数
        Interval(Real, 0, 2, closed="right"),     # 上界大于1
        Interval(Real, 0, 0.5, closed="left"),    # 下界包括0
    ]

    err_msg = r"contamination constraint should be an interval in \(0, 0.5\]"
    # 遍历错误的区间约束列表，逐个设置约束并检查是否引发预期的 AssertionError
    for interval in incorrect_intervals:
        OutlierDetectorWithConstraint._parameter_constraints["contamination"] = [
            interval
        ]
        detector = OutlierDetectorWithConstraint()
        with raises(AssertionError, match=err_msg):
            check_outlier_contamination(detector.__class__.__name__, detector)


def test_decision_proba_tie_ranking():
    """Check that in case with some probabilities ties, we relax the
    ranking comparison with the decision function.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24025
    """
    estimator = SGDClassifier(loss="log_loss")
    # 检查在某些概率出现并列情况下，通过决策函数放宽排名比较的一致性。
    check_decision_proba_consistency("SGDClassifier", estimator)
```