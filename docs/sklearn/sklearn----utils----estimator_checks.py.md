# `D:\src\scipysrc\scikit-learn\sklearn\utils\estimator_checks.py`

```
# 导入所需的模块和库
"""Various utilities to check the compatibility of estimators with scikit-learn API."""

import pickle  # 导入 pickle 模块，用于序列化和反序列化 Python 对象
import re  # 导入 re 模块，用于正则表达式的操作
import warnings  # 导入 warnings 模块，用于处理警告
from contextlib import nullcontext  # 从 contextlib 模块导入 nullcontext 上下文管理器
from copy import deepcopy  # 从 copy 模块导入 deepcopy 函数，用于深拷贝对象
from functools import partial, wraps  # 从 functools 模块导入 partial 和 wraps 装饰器
from inspect import isfunction, signature  # 从 inspect 模块导入 isfunction 函数和 signature 函数
from numbers import Integral, Real  # 从 numbers 模块导入 Integral 和 Real 类型

import joblib  # 导入 joblib 库，用于高效处理 Python 对象持久化
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy import sparse  # 从 scipy 导入 sparse 子模块，用于稀疏矩阵操作
from scipy.stats import rankdata  # 从 scipy.stats 导入 rankdata 函数，用于排名数据

# 导入 scikit-learn 中的各种模块和类
from .. import config_context  # 从当前包的父级包导入 config_context 模块
from ..base import (  # 从当前包的父级包导入多个基础模块和函数
    ClusterMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_outlier_detector,
    is_regressor,
)
from ..datasets import (  # 从当前包的父级包导入数据集生成函数
    load_iris,
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from ..exceptions import (  # 从当前包的父级包导入异常处理相关类
    DataConversionWarning,
    NotFittedError,
    SkipTestWarning,
)
from ..feature_selection import SelectFromModel, SelectKBest  # 从当前包的父级包导入特征选择相关类
from ..linear_model import (  # 从当前包的父级包导入线性模型相关类
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
)
from ..metrics import accuracy_score, adjusted_rand_score, f1_score  # 从当前包的父级包导入评估指标函数
from ..metrics.pairwise import (  # 从当前包的父级包导入成对指标计算函数
    linear_kernel,
    pairwise_distances,
    rbf_kernel,
)
from ..model_selection import ShuffleSplit, train_test_split  # 从当前包的父级包导入模型选择相关类和函数
from ..model_selection._validation import _safe_split  # 从当前包的父级包导入验证数据划分函数
from ..pipeline import make_pipeline  # 从当前包的父级包导入管道构建函数
from ..preprocessing import StandardScaler, scale  # 从当前包的父级包导入数据预处理函数和类
from ..random_projection import BaseRandomProjection  # 从当前包的父级包导入基础随机投影类
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor  # 从当前包的父级包导入决策树分类和回归类
from ..utils._array_api import (  # 从当前包的父级包导入数组操作函数和类
    _atol_for_type,
    _convert_to_numpy,
    get_namespace,
    yield_namespace_device_dtype_combinations,
)
from ..utils._array_api import device as array_device  # 从当前包的父级包导入数组设备标识符
from ..utils._param_validation import (  # 从当前包的父级包导入参数验证相关函数和类
    InvalidParameterError,
    generate_invalid_param_val,
    make_constraint,
)
from . import shuffle  # 从当前包导入 shuffle 模块
from ._missing import is_scalar_nan  # 从当前包导入判断是否为标量 NaN 的函数
from ._param_validation import Interval  # 从当前包导入参数区间类
from ._tags import (  # 从当前包导入标签相关模块
    _DEFAULT_TAGS,
    _safe_tags,
)
from ._testing import (  # 从当前包导入测试相关函数和类
    SkipTest,
    _array_api_for_tests,
    _get_args,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_raise_message,
    create_memmap_backed_data,
    ignore_warnings,
    raises,
    set_random_state,
)
from .fixes import SPARSE_ARRAY_PRESENT, parse_version, sp_version  # 从当前包导入修复相关函数和常量
from .validation import _num_samples, check_is_fitted, has_fit_parameter  # 从当前包导入验证函数和常量

# 定义全局变量
REGRESSION_DATASET = None  # 初始化回归数据集为空
CROSS_DECOMPOSITION = ["PLSCanonical", "PLSRegression", "CCA", "PLSSVD"]  # 定义交叉分解方法列表


def _yield_checks(estimator):
    # 获取估计器类名
    name = estimator.__class__.__name__
    # 获取估计器的安全标签
    tags = _safe_tags(estimator)

    # 生成估计器的检查器迭代器
    yield check_no_attributes_set_in_init  # 检查初始化时是否设置了任何属性
    yield check_estimators_dtypes  # 检查估计器的数据类型
    yield check_fit_score_takes_y  # 检查 fit 和 score 方法是否接受 y 参数
    # 如果评估器支持样本权重参数，则执行以下检查
    if has_fit_parameter(estimator, "sample_weight"):
        # 检查样本权重参数为 Pandas Series 类型的情况
        yield check_sample_weights_pandas_series
        # 检查样本权重参数不是数组的情况
        yield check_sample_weights_not_an_array
        # 检查样本权重参数为列表的情况
        yield check_sample_weights_list
        # 如果评估器不是成对计算（pairwise），则执行以下检查
        if not tags["pairwise"]:
            # 跳过成对计算，因为数据不是成对的
            yield check_sample_weights_shape
            # 检查样本权重参数没有被覆盖
            yield check_sample_weights_not_overwritten
            # 检查样本权重对模型的影响，保持不变性，设置权重为全部为1
            yield partial(check_sample_weights_invariance, kind="ones")
            # 检查样本权重对模型的影响，保持不变性，设置权重为全部为0
            yield partial(check_sample_weights_invariance, kind="zeros")
    # 检查评估器的fit方法返回自身对象
    yield check_estimators_fit_returns_self
    # 检查评估器的fit方法在使用只读内存映射时返回自身对象
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)

    # 检查所有评估器在空数据集上训练时是否给出了信息性消息
    if not tags["no_validation"]:
        # 检查复杂数据的处理
        yield check_complex_data
        # 检查数据类型为对象类型的处理
        yield check_dtype_object
        # 检查所有评估器在空数据集上训练时是否给出了适当的消息
        yield check_estimators_empty_data_messages

    # 如果评估器名称不在交叉分解中
    if name not in CROSS_DECOMPOSITION:
        # 检查管道的一致性
        yield check_pipeline_consistency

    # 如果不允许出现NaN且不跳过验证
    if not tags["allow_nan"] and not tags["no_validation"]:
        # 测试所有评估器是否检查其输入中的NaN和无穷大
        yield check_estimators_nan_inf

    # 如果评估器需要成对计算
    if tags["pairwise"]:
        # 检查成对计算的评估器在非方阵输入时是否会抛出错误
        yield check_nonsquare_error

    # 检查评估器是否覆盖参数
    yield check_estimators_overwrite_params
    # 如果评估器具有"sparsify"属性，检查其系数的稀疏性
    if hasattr(estimator, "sparsify"):
        yield check_sparsify_coefficients

    # 检查评估器是否能够处理稀疏数组输入
    yield check_estimator_sparse_array
    # 检查评估器是否能够处理稀疏矩阵输入
    yield check_estimator_sparse_matrix

    # 测试评估器是否能够正确序列化并返回相同结果
    yield check_estimators_pickle
    # 测试评估器在使用只读内存映射并进行序列化后是否返回相同结果
    yield partial(check_estimators_pickle, readonly_memmap=True)

    # 检查评估器的默认标签键
    yield check_estimator_get_tags_default_keys

    # 如果评估器支持数组API，则执行以下数组API相关的检查
    if tags["array_api_support"]:
        # 为评估器生成所有数组API相关的检查
        for check in _yield_array_api_checks(estimator):
            yield check
# 生成器函数，用于生成分类器检查函数
def _yield_classifier_checks(classifier):
    # 调用 _safe_tags 函数获取分类器的安全标签
    tags = _safe_tags(classifier)

    # 测试分类器能否处理非数组数据和 pandas 对象
    yield check_classifier_data_not_an_array
    # 测试分类器在单标签训练时是否始终返回此标签
    yield check_classifiers_one_label
    # 测试分类器在使用样本权重时是否仍然返回单标签
    yield check_classifiers_one_label_sample_weights
    # 检查分类器是否正确返回类别信息
    yield check_classifiers_classes
    # 检查分类器部分拟合时特征数量是否符合预期
    yield check_estimators_partial_fit_n_features
    # 如果分类器支持多输出，则进行多输出测试
    if tags["multioutput"]:
        yield check_classifier_multioutput
    # 基本一致性测试
    yield check_classifiers_train
    # 对只读内存映射数据进行分类器训练测试
    yield partial(check_classifiers_train, readonly_memmap=True)
    # 对指定数据类型为 float32 的只读内存映射数据进行分类器训练测试
    yield partial(check_classifiers_train, readonly_memmap=True, X_dtype="float32")
    # 检查分类器是否正确处理回归目标
    yield check_classifiers_regression_target
    # 如果分类器支持多标签，则进行多标签不变性测试
    if tags["multilabel"]:
        yield check_classifiers_multilabel_representation_invariance
        # 测试分类器预测输出的多标签格式是否正确
        yield check_classifiers_multilabel_output_format_predict
        # 测试分类器预测概率输出的多标签格式是否正确
        yield check_classifiers_multilabel_output_format_predict_proba
        # 测试分类器决策函数输出的多标签格式是否正确
        yield check_classifiers_multilabel_output_format_decision_function
    # 如果分类器要求进行非 NaN 的验证，则执行验证
    if not tags["no_validation"]:
        # 检查监督学习的目标 y 是否不包含 NaN
        yield check_supervised_y_no_nan
        # 如果不仅支持多输出，则进行二维监督学习目标 y 的测试
        if not tags["multioutput_only"]:
            yield check_supervised_y_2d
    # 如果分类器要求进行拟合，则检查未拟合的分类器
    if tags["requires_fit"]:
        yield check_estimators_unfitted
    # 如果分类器参数中包含 'class_weight'，则进行分类器类别权重检查
    if "class_weight" in classifier.get_params().keys():
        yield check_class_weight_classifiers

    # 检查非变换器估计器的迭代次数
    yield check_non_transformer_estimators_n_iter
    # 测试 predict_proba 是否是 decision_function 的单调变换
    yield check_decision_proba_consistency


# 忽略未来警告的装饰器，检查监督学习的目标 y 是否不包含 NaN
def check_supervised_y_no_nan(name, estimator_orig):
    # 检查估计器的目标值 y 是否不包含 NaN
    estimator = clone(estimator_orig)
    # 创建随机数生成器
    rng = np.random.RandomState(888)
    # 创建大小为 (10, 5) 的标准正态分布随机数作为输入 X
    X = rng.standard_normal(size=(10, 5))

    # 针对 NaN 和无穷大值分别进行测试
    for value in [np.nan, np.inf]:
        # 创建元素全为 value 的大小为 10 的数组作为目标值 y
        y = np.full(10, value)
        # 调用 _enforce_estimator_tags_y 函数处理目标值 y
        y = _enforce_estimator_tags_y(estimator, y)

        # 获取估计器的模块名称
        module_name = estimator.__module__
        # 如果模块名称以 "sklearn." 开头且不包含 "test_" 或以 "_testing" 结尾，则执行以下逻辑
        if module_name.startswith("sklearn.") and not (
            "test_" in module_name or module_name.endswith("_testing")
        ):
            # 在 scikit-learn 中，期望错误信息提到输入名称及其未预期值的具体类型
            if np.isinf(value):
                match = (
                    r"Input (y|Y) contains infinity or a value too large for"
                    r" dtype\('float64'\)."
                )
            else:
                match = r"Input (y|Y) contains NaN."
        else:
            # 对于第三方库，不强制特定的错误消息
            match = None
        # 错误消息
        err_msg = (
            f"Estimator {name} should have raised error on fitting array y with inf"
            " value."
        )
        # 使用 pytest 的 raises 断言来验证是否抛出 ValueError 异常
        with raises(ValueError, match=match, err_msg=err_msg):
            estimator.fit(X, y)
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing

    # 生成器函数中的各个测试用例，用于回归器的训练检查
    yield check_regressors_train

    # 生成带有部分参数的回归器训练检查，使用只读内存映射
    yield partial(check_regressors_train, readonly_memmap=True)

    # 生成带有部分参数的回归器训练检查，使用只读内存映射和特定的X数据类型
    yield partial(check_regressors_train, readonly_memmap=True, X_dtype="float32")

    # 检查回归器的数据不是数组的情况
    yield check_regressor_data_not_an_array

    # 检查估计器的偏序适合特征数
    yield check_estimators_partial_fit_n_features

    # 如果模型支持多输出，则进行多输出回归器的检查
    if tags["multioutput"]:
        yield check_regressor_multioutput

    # 检查回归器没有决策函数的情况
    yield check_regressors_no_decision_function

    # 如果不是没有验证和仅多输出的标签，则检查有监督的y数据是2D的情况
    if not tags["no_validation"] and not tags["multioutput_only"]:
        yield check_supervised_y_2d

    # 检查有监督的y数据没有NaN的情况
    yield check_supervised_y_no_nan

    # 获取回归器的类名
    name = regressor.__class__.__name__

    # 如果回归器不是CCA类型，则检查回归器处理整数输入的情况
    if name != "CCA":
        yield check_regressors_int

    # 如果模型需要拟合，则检查估计器未拟合的情况
    if tags["requires_fit"]:
        yield check_estimators_unfitted

    # 检查非变换器估计器的迭代次数
    yield check_non_transformer_estimators_n_iter
# 生成并返回与给定转换器相关的检查函数生成器
def _yield_transformer_checks(transformer):
    # 获取转换器的安全标签
    tags = _safe_tags(transformer)
    # 所有转换器应该处理稀疏数据或引发 TypeError 类型的异常，并包含清晰的错误消息
    if not tags["no_validation"]:
        yield check_transformer_data_not_an_array
    # 这些转换器实际上不适合数据，因此不会引发错误
    yield check_transformer_general
    # 如果转换器保持数据类型，则引发相应检查
    if tags["preserves_dtype"]:
        yield check_transformer_preserve_dtypes
    # 引发通用转换器检查，并将 readonly_memmap 设置为 True
    yield partial(check_transformer_general, readonly_memmap=True)
    # 如果转换器不具有 stateless 属性，则引发未安装转换器的检查
    if not _safe_tags(transformer, key="stateless"):
        yield check_transformers_unfitted
    else:
        # 否则引发 stateless 转换器未安装的检查
        yield check_transformers_unfitted_stateless
    # 外部求解器依赖，因此访问 iter 参数是非平凡的
    external_solver = [
        "Isomap",
        "KernelPCA",
        "LocallyLinearEmbedding",
        "RandomizedLasso",
        "LogisticRegressionCV",
        "BisectingKMeans",
    ]
    # 获取转换器的类名
    name = transformer.__class__.__name__
    # 如果类名不在外部求解器列表中，则引发 n_iter 转换器检查
    if name not in external_solver:
        yield check_transformer_n_iter


# 生成并返回与给定聚类器相关的检查函数生成器
def _yield_clustering_checks(clusterer):
    # 引发聚类器计算标签预测的检查
    yield check_clusterer_compute_labels_predict
    # 获取聚类器的类名
    name = clusterer.__class__.__name__
    # 如果类名不是 "WardAgglomeration" 或 "FeatureAgglomeration"，则执行以下操作
    if name not in ("WardAgglomeration", "FeatureAgglomeration"):
        # 这是对特征进行聚类，这里不进行测试
        yield check_clustering
        # 引发聚类检查，并将 readonly_memmap 设置为 True
        yield partial(check_clustering, readonly_memmap=True)
        # 引发偏部分拟合估算器的特征数检查
        yield check_estimators_partial_fit_n_features
    # 如果聚类器没有 "transform" 属性，则引发非转换器估算器的 n_iter 检查
    if not hasattr(clusterer, "transform"):
        yield check_non_transformer_estimators_n_iter


# 生成并返回与给定异常值检测器相关的检查函数生成器
def _yield_outliers_checks(estimator):
    # 检查异常值检测器是否具有 contamination 参数
    if hasattr(estimator, "contamination"):
        yield check_outlier_contamination
    # 检查具有 fit_predict 方法的异常值检测器
    if hasattr(estimator, "fit_predict"):
        yield check_outliers_fit_predict
    # 检查能够在测试集上使用的估算器
    if hasattr(estimator, "predict"):
        # 引发异常值检测训练的检查，并将 readonly_memmap 设置为 True
        yield check_outliers_train
        # 测试异常值检测器是否能处理非数组数据
        yield check_classifier_data_not_an_array
        # 测试是否引发 NotFittedError
        if _safe_tags(estimator, key="requires_fit"):
            yield check_estimators_unfitted
    # 引发非转换器估算器的 n_iter 检查
    yield check_non_transformer_estimators_n_iter


# 生成并返回与给定数组 API 相关的检查函数生成器
def _yield_array_api_checks(estimator):
    # 遍历 yield_namespace_device_dtype_combinations() 的结果
    for (
        array_namespace,
        device,
        dtype_name,
    ) in yield_namespace_device_dtype_combinations():
        # 生成数组 API 输入检查的偏函数
        yield partial(
            check_array_api_input,
            array_namespace=array_namespace,
            dtype_name=dtype_name,
            device=device,
        )


# 生成并返回与给定估算器相关的所有检查函数生成器
def _yield_all_checks(estimator):
    # 获取估算器的类名
    name = estimator.__class__.__name__
    # 获取估算器的安全标签
    tags = _safe_tags(estimator)
    # 检查是否在标签中指定的输入类型中包含 "2darray"，如果不包含，则发出警告并返回
    if "2darray" not in tags["X_types"]:
        warnings.warn(
            "Can't test estimator {} which requires input of type {}".format(
                name, tags["X_types"]
            ),
            SkipTestWarning,
        )
        return
    
    # 如果标记指定跳过测试，则发出相应警告并返回
    if tags["_skip_test"]:
        warnings.warn(
            "Explicit SKIP via _skip_test tag for estimator {}.".format(name),
            SkipTestWarning,
        )
        return

    # 使用 _yield_checks 函数逐个获取并生成各种检查项
    for check in _yield_checks(estimator):
        yield check
    
    # 如果估计器是分类器，则使用 _yield_classifier_checks 函数逐个获取并生成分类器特定的检查项
    if is_classifier(estimator):
        for check in _yield_classifier_checks(estimator):
            yield check
    
    # 如果估计器是回归器，则使用 _yield_regressor_checks 函数逐个获取并生成回归器特定的检查项
    if is_regressor(estimator):
        for check in _yield_regressor_checks(estimator):
            yield check
    
    # 如果估计器具有 "transform" 属性，则使用 _yield_transformer_checks 函数逐个获取并生成转换器特定的检查项
    if hasattr(estimator, "transform"):
        for check in _yield_transformer_checks(estimator):
            yield check
    
    # 如果估计器是 ClusterMixin 类的实例，则使用 _yield_clustering_checks 函数逐个获取并生成聚类特定的检查项
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(estimator):
            yield check
    
    # 如果估计器是异常检测器，则使用 _yield_outliers_checks 函数逐个获取并生成异常检测特定的检查项
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(estimator):
            yield check
    
    # 生成默认可构造参数的检查项
    yield check_parameters_default_constructible
    
    # 如果不指定为非确定性，则生成样本顺序不变性和子集不变性的方法检查项
    if not tags["non_deterministic"]:
        yield check_methods_sample_order_invariance
        yield check_methods_subset_invariance
    
    # 生成适合 2D 单样本的拟合检查项
    yield check_fit2d_1sample
    
    # 生成适合 2D 单特征的拟合检查项
    yield check_fit2d_1feature
    
    # 生成获取参数不变性的检查项
    yield check_get_params_invariance
    
    # 生成设置参数的检查项
    yield check_set_params
    
    # 生成字典不变性的检查项
    yield check_dict_unchanged
    
    # 生成不要覆盖参数的检查项
    yield check_dont_overwrite_parameters
    
    # 生成拟合幂等性的检查项
    yield check_fit_idempotent
    
    # 生成检查拟合后是否已拟合的检查项
    yield check_fit_check_is_fitted
    
    # 如果不指定为无验证，则生成特征数验证的检查项
    if not tags["no_validation"]:
        yield check_n_features_in
        
        # 生成适合 1D 数据的拟合检查项
        yield check_fit1d
        
        # 生成适合 2D 预测 1D 的拟合检查项
        yield check_fit2d_predict1d
        
        # 如果需要 y，则生成不需要 y 的检查项
        if tags["requires_y"]:
            yield check_requires_y_none
    
    # 如果需要正 X，则生成拟合非负 X 的检查项
    if tags["requires_positive_X"]:
        yield check_fit_non_negative
# 创建用于 pytest 检查的标识符字符串。

# 如果 `obj` 是一个函数，则返回函数的名称。
# 如果 `obj` 是一个偏函数（partial），且没有关键字参数，则返回函数的名称。
# 如果 `obj` 是一个偏函数且有关键字参数，则返回函数名称及其关键字参数的字符串表示。
# 如果 `obj` 是一个具有 `get_params` 方法的对象，则返回其用 `print_changed_only=True` 打印后去除空白字符的字符串表示。

# 用作 `pytest.mark.parametrize` 的 `id`，在 `check_estimator(..., generate_only=True)` 生成评估器和检查时使用。

# 参数：
# - obj : 评估器或函数，由 `check_estimator` 生成的项目。

# 返回：
# - str or None，用作 `pytest` 的标识符。

# 参见：
# - check_estimator
def _get_check_estimator_ids(obj):
    if isfunction(obj):
        return obj.__name__
    if isinstance(obj, partial):
        if not obj.keywords:
            return obj.func.__name__
        kwstring = ",".join(["{}={}".format(k, v) for k, v in obj.keywords.items()])
        return "{}({})".format(obj.func.__name__, kwstring)
    if hasattr(obj, "get_params"):
        with config_context(print_changed_only=True):
            return re.sub(r"\s", "", str(obj))


# 构造评估器实例（Estimator），如果可能的话。

# 获取 Estimator 的 `_required_parameters` 属性作为必需参数列表。
def _construct_instance(Estimator):
    required_parameters = getattr(Estimator, "_required_parameters", [])
    # 检查是否有必需的参数
    if len(required_parameters):
        # 检查 required_parameters 是否为 ["estimator"] 或 ["base_estimator"]
        if required_parameters in (["estimator"], ["base_estimator"]):
            # 如果 Estimator 是 RANSACRegressor 的子类，则使用 LinearRegression 作为默认估计器
            # 因为 RANSACRegressor 除了 LinearRegression 外的模型会引发错误
            if issubclass(Estimator, RANSACRegressor):
                estimator = Estimator(LinearRegression())
            # 如果 Estimator 是 RegressorMixin 的子类，则使用 Ridge 作为估计器
            elif issubclass(Estimator, RegressorMixin):
                estimator = Estimator(Ridge())
            # 如果 Estimator 是 SelectFromModel 的子类，则使用 SGDRegressor 作为估计器
            elif issubclass(Estimator, SelectFromModel):
                # 增加覆盖率，因为 SGDRegressor 有 partial_fit 方法
                estimator = Estimator(SGDRegressor(random_state=0))
            else:
                # 其他情况下使用 LogisticRegression 作为估计器
                estimator = Estimator(LogisticRegression(C=1))
        # 如果 required_parameters 是 ["estimators"]
        elif required_parameters in (["estimators"],):
            # 对于异质集成类（如 stacking, voting），根据 Estimator 是否为 RegressorMixin 决定使用不同的估计器集合
            if issubclass(Estimator, RegressorMixin):
                estimator = Estimator(
                    estimators=[
                        ("est1", DecisionTreeRegressor(max_depth=3, random_state=0)),
                        ("est2", DecisionTreeRegressor(max_depth=3, random_state=1)),
                    ]
                )
            else:
                estimator = Estimator(
                    estimators=[
                        ("est1", DecisionTreeClassifier(max_depth=3, random_state=0)),
                        ("est2", DecisionTreeClassifier(max_depth=3, random_state=1)),
                    ]
                )
        else:
            # 如果没有匹配到任何必需参数组合，生成错误消息
            msg = (
                f"Can't instantiate estimator {Estimator.__name__} "
                f"parameters {required_parameters}"
            )
            # 发出额外的警告，供 pytest 显示
            warnings.warn(msg, SkipTestWarning)
            # 抛出 SkipTest 异常
            raise SkipTest(msg)
    else:
        # 如果没有必需参数，则直接实例化 Estimator
        estimator = Estimator()
    # 返回选择的估计器对象
    return estimator
# 根据需要标记 (estimator, check) 对为 XFAIL，若需要的话（参见 _should_be_skipped_or_marked() 中的条件）
# 这与 _maybe_skip() 类似，但此函数由 @parametrize_with_checks() 使用，而非 check_estimator()

def _maybe_mark_xfail(estimator, check, pytest):
    # 调用 _should_be_skipped_or_marked() 检查是否应该标记为 XFAIL，并获取原因
    should_be_marked, reason = _should_be_skipped_or_marked(estimator, check)
    if not should_be_marked:
        # 如果不需要标记为 XFAIL，则直接返回 estimator 和 check
        return estimator, check
    else:
        # 否则，使用 pytest.param() 标记为 XFAIL，并传入原因
        return pytest.param(estimator, check, marks=pytest.mark.xfail(reason=reason))


def _maybe_skip(estimator, check):
    # 封装一个检查函数，如果需要的话将其跳过（参见 _should_be_skipped_or_marked() 中的条件）
    # 这与 _maybe_mark_xfail() 类似，但此函数由 check_estimator() 使用，而非 @parametrize_with_checks()，后者需要 pytest
    should_be_skipped, reason = _should_be_skipped_or_marked(estimator, check)
    if not should_be_skipped:
        # 如果不需要跳过，则直接返回原始的 check
        return check

    # 获取 check 的名称，支持 partial 函数
    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

    @wraps(check)
    def wrapped(*args, **kwargs):
        # 抛出 SkipTest 异常，指示跳过该检查
        raise SkipTest(
            f"Skipping {check_name} for {estimator.__class__.__name__}: {reason}"
        )

    return wrapped


def _should_be_skipped_or_marked(estimator, check):
    # 返回是否应该跳过检查（当使用 check_estimator() 时），或者标记为 XFAIL（当使用 @parametrize_with_checks() 时），以及其原因
    # 目前，如果检查在 estimator 的 _xfail_checks 标签中，则应该跳过或标记为 XFAIL

    # 获取检查的名称，支持 partial 函数
    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

    # 获取 estimator 的 _xfail_checks 标签，如果不存在则为空字典
    xfail_checks = _safe_tags(estimator, key="_xfail_checks") or {}

    # 如果检查名称在 _xfail_checks 中，则返回需要跳过，并返回其原因
    if check_name in xfail_checks:
        return True, xfail_checks[check_name]

    # 否则，返回不需要跳过，并提供一个占位原因
    return False, "placeholder reason that will never be used"


def parametrize_with_checks(estimators):
    """用于为估计器检查参数化的 pytest 特定装饰器。

    每个检查的 `id` 设置为估计器的 pprint 版本，以及检查名称及其关键字参数。
    这允许使用 `pytest -k` 来指定要运行的测试::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : 估计器实例的列表
        要为其生成检查的估计器。

        .. versionchanged:: 0.24
           0.23 版本中不再支持传递类，从 0.24 版本开始移除了对类的支持。现在应传递实例。

        .. versionadded:: 0.24

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : 检查估计器是否符合 scikit-learn 约定。

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor
"""
    ```python`
    # 导入 pytest 库，用于测试框架
    import pytest
    
    # 检查是否有传入类对象的情况，如果有则抛出异常
    if any(isinstance(est, type) for est in estimators):
        msg = (
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)
    
    # 定义生成器函数，用于生成参数化测试所需的参数
    def checks_generator():
        for estimator in estimators:
            name = type(estimator).__name__
            # 遍历获取给定估计器的所有检查函数，并为每个检查函数生成一个参数化测试用例
            for check in _yield_all_checks(estimator):
                check = partial(check, name)
                # 使用 pytest 的标记函数，为每个检查函数生成一个参数化测试用例
                yield _maybe_mark_xfail(estimator, check, pytest)
    
    # 返回使用 pytest 的参数化标记函数，将生成器函数的结果应用于测试函数
    return pytest.mark.parametrize(
        "estimator, check", checks_generator(), ids=_get_check_estimator_ids
    )
# 定义一个函数，用于检查估计器是否符合 scikit-learn 的约定和规范
def check_estimator(estimator=None, generate_only=False):
    """Check if estimator adheres to scikit-learn conventions.

    This function will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `scikit-learn`
    conventions as detailed in :ref:`rolling_your_own_estimator`.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    Setting `generate_only=True` returns a generator that yields (estimator,
    check) tuples where the check can be called independently from each
    other, i.e. `check(estimator)`. This allows all checks to be run
    independently and report the checks that are failing.

    scikit-learn provides a pytest specific decorator,
    :func:`~sklearn.utils.estimator_checks.parametrize_with_checks`, making it
    easier to test multiple estimators.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance to check.

        .. versionadded:: 1.1
           Passing a class was deprecated in version 0.23, and support for
           classes was removed in 0.24.

    generate_only : bool, default=False
        When `False`, checks are evaluated when `check_estimator` is called.
        When `True`, `check_estimator` returns a generator that yields
        (estimator, check) tuples. The check is run by calling
        `check(estimator)`.

        .. versionadded:: 0.22

    Returns
    -------
    checks_generator : generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> check_estimator(LogisticRegression(), generate_only=True)
    <generator object ...>
    """

    # 如果传入的 estimator 是一个类而不是实例，则抛出 TypeError
    if isinstance(estimator, type):
        msg = (
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)

    # 获取 estimator 的类名
    name = type(estimator).__name__

    # 定义一个生成器函数，用于生成 (estimator, check) 元组
    def checks_generator():
        # 遍历所有的检查函数
        for check in _yield_all_checks(estimator):
            # 对于特定的 estimator 和检查函数，可能会根据条件跳过
            check = _maybe_skip(estimator, check)
            # 返回 (estimator, partial(check, name)) 的元组
            yield estimator, partial(check, name)

    # 如果 generate_only=True，则返回生成器对象
    if generate_only:
        return checks_generator()

    # 否则，遍历生成器对象并逐个执行检查
    for estimator, check in checks_generator():
        try:
            check(estimator)
        except SkipTest as exception:
            # 当遇到 SkipTest 异常时，通常是由于无法导入 pandas 或者由于标记了 xfail_checks
            warnings.warn(str(exception), SkipTestWarning)


def _regression_dataset():
    global REGRESSION_DATASET
    # 如果 REGRESSION_DATASET 变量为 None，则生成一个回归数据集并赋值给 X, y
    if REGRESSION_DATASET is None:
        # 使用 make_regression 函数生成回归数据集
        X, y = make_regression(
            n_samples=200,        # 样本数为 200
            n_features=10,        # 特征数为 10
            n_informative=1,      # 有信息特征数为 1
            bias=5.0,             # 偏置设为 5.0
            noise=20,             # 噪声设为 20
            random_state=42,      # 随机种子设为 42，以确保可重现性
        )
        # 使用 StandardScaler 对特征 X 进行标准化处理
        X = StandardScaler().fit_transform(X)
        # 将生成的 X, y 组成的元组赋值给 REGRESSION_DATASET 变量
        REGRESSION_DATASET = X, y
    # 返回 REGRESSION_DATASET 变量，其中包含了生成的回归数据集
    return REGRESSION_DATASET
# 设置检查参数以加快某些估计器的速度并避免不推荐的行为
def _set_checking_parameters(estimator):
    # 获取当前估计器的参数
    params = estimator.get_params()
    # 获取估计器的类名
    name = estimator.__class__.__name__

    # 如果估计器是 TSNE 类，则设置 perplexity 参数为 2
    if name == "TSNE":
        estimator.set_params(perplexity=2)

    # 如果参数中包含 'n_iter'，并且估计器不是 TSNE 类，则设置 n_iter 参数为 5
    if "n_iter" in params and name != "TSNE":
        estimator.set_params(n_iter=5)

    # 如果参数中包含 'max_iter'
    if "max_iter" in params:
        # 如果 estimator 的 max_iter 不为 None，则将 max_iter 设置为 5 和 estimator.max_iter 中较小的值
        if estimator.max_iter is not None:
            estimator.set_params(max_iter=min(5, estimator.max_iter))
        
        # 对于 LinearSVR 和 LinearSVC 类，将 max_iter 设置为 20
        if name in ["LinearSVR", "LinearSVC"]:
            estimator.set_params(max_iter=20)
        
        # 对于 NMF 类，将 max_iter 设置为 500
        if name == "NMF":
            estimator.set_params(max_iter=500)
        
        # 对于 DictionaryLearning 类，将 max_iter 设置为 20，transform_algorithm 设置为 "lasso_lars"
        if name == "DictionaryLearning":
            estimator.set_params(max_iter=20, transform_algorithm="lasso_lars")
        
        # 对于 MiniBatchNMF 类，将 max_iter 设置为 20，fresh_restarts 设置为 True
        if estimator.__class__.__name__ == "MiniBatchNMF":
            estimator.set_params(max_iter=20, fresh_restarts=True)
        
        # 对于 MLPClassifier 和 MLPRegressor 类，将 max_iter 设置为 100
        if name in ["MLPClassifier", "MLPRegressor"]:
            estimator.set_params(max_iter=100)
        
        # 对于 MiniBatchDictionaryLearning 类，将 max_iter 设置为 5
        if name == "MiniBatchDictionaryLearning":
            estimator.set_params(max_iter=5)

    # 如果参数中包含 'n_resampling'，将 n_resampling 设置为 5（例如对 randomized lasso）
    if "n_resampling" in params:
        estimator.set_params(n_resampling=5)
    
    # 如果参数中包含 'n_estimators'，将 n_estimators 设置为 5 和 estimator.n_estimators 中较小的值
    if "n_estimators" in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    
    # 如果参数中包含 'max_trials'，将 max_trials 设置为 10（例如对 RANSAC）
    if "max_trials" in params:
        estimator.set_params(max_trials=10)
    
    # 如果参数中包含 'n_init'，将 n_init 设置为 2（例如对 K-Means）
    if "n_init" in params:
        estimator.set_params(n_init=2)
    
    # 如果参数中包含 'batch_size'，并且估计器类名不以 "MLP" 开头，则将 batch_size 设置为 10
    if "batch_size" in params and not name.startswith("MLP"):
        estimator.set_params(batch_size=10)

    # 如果估计器类名为 "MeanShift"，将 bandwidth 设置为 1.0，用于 check_fit2d_1sample 的特殊情况处理
    if name == "MeanShift":
        estimator.set_params(bandwidth=1.0)

    # 如果估计器类名为 "TruncatedSVD"，将 n_components 设置为 1，因为 TruncatedSVD 不能运行 n_components = n_features
    if name == "TruncatedSVD":
        estimator.n_components = 1

    # 如果估计器类名为 "LassoLarsIC"，将 noise_variance 设置为 1.0，因为噪声方差估计不适用于 n_samples < n_features 的情况
    if name == "LassoLarsIC":
        estimator.set_params(noise_variance=1.0)

    # 如果估计器具有属性 'n_clusters'，将 estimator.n_clusters 设置为 estimator.n_clusters 和 2 中较小的值
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = min(estimator.n_clusters, 2)

    # 如果估计器具有属性 'n_best'，将 estimator.n_best 设置为 1
    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    # 如果估计器类名为 "SelectFdr"，将 alpha 设置为 0.5，以容忍嘈杂的数据集
    if name == "SelectFdr":
        estimator.set_params(alpha=0.5)

    # 如果估计器类名为 "TheilSenRegressor"，将 max_subpopulation 设置为 100
    if name == "TheilSenRegressor":
        estimator.max_subpopulation = 100
    # 如果 estimator 是 BaseRandomProjection 的实例
    if isinstance(estimator, BaseRandomProjection):
        # 根据 Johnson-Lindenstrauss 引理和样本数通常较少的情况，
        # 随机投影矩阵的组件数量可能会大于特征数量。
        # 因此，我们设定一个较小的数量 (避免使用 "auto" 模式)
        estimator.set_params(n_components=2)

    # 如果 estimator 是 SelectKBest 的实例
    if isinstance(estimator, SelectKBest):
        # SelectKBest 的默认 k 值为 10
        # 大多数情况下，这比我们的特征数要多。
        estimator.set_params(k=1)

    # 如果算法名字为 "HistGradientBoostingClassifier" 或 "HistGradientBoostingRegressor"
    if name in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        # 默认的 min_samples_leaf (20) 对于小数据集来说不合适
        # （只会构建非常浅的树），因此我们设定一个更小的值。
        estimator.set_params(min_samples_leaf=5)

    # 如果算法名字为 "DummyClassifier"
    if name == "DummyClassifier":
        # 默认的策略会输出常数预测并且无法通过 check_classifiers_predictions 检查
        estimator.set_params(strategy="stratified")

    # 通过减少交叉验证的次数来加速 CV 或使用 CV 的估算器
    loo_cv = ["RidgeCV", "RidgeClassifierCV"]
    if name not in loo_cv and hasattr(estimator, "cv"):
        estimator.set_params(cv=3)
    if hasattr(estimator, "n_splits"):
        estimator.set_params(n_splits=3)

    # 如果算法名字为 "OneHotEncoder"
    if name == "OneHotEncoder":
        # 处理未知值时忽略它们
        estimator.set_params(handle_unknown="ignore")

    # 如果算法名字为 "QuantileRegressor"
    if name == "QuantileRegressor":
        # 避免因为 Scipy 放弃 interior-point solver 而出现警告
        solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
        estimator.set_params(solver=solver)

    # 如果算法名字在 CROSS_DECOMPOSITION 中
    if name in CROSS_DECOMPOSITION:
        # 设置分量数量为 1
        estimator.set_params(n_components=1)

    # 对于 "SpectralEmbedding" 算法，默认的 "auto" 参数可能导致在 Windows 上特征值排序不同
    if name == "SpectralEmbedding":
        estimator.set_params(eigen_tol=1e-5)

    # 如果算法名字为 "HDBSCAN"
    if name == "HDBSCAN":
        # 设置最小样本数为 1
        estimator.set_params(min_samples=1)
class _NotAnArray:
    """An object that is convertible to an array.

    Parameters
    ----------
    data : array-like
        The data.
    """

    def __init__(self, data):
        self.data = np.asarray(data)  # 将输入数据转换为 NumPy 数组

    def __array__(self, dtype=None, copy=None):
        return self.data  # 返回存储在对象中的 NumPy 数组

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "may_share_memory":
            return True  # 如果调用的函数是 `may_share_memory`，则返回 True
        raise TypeError("Don't want to call array_function {}!".format(func.__name__))  # 如果调用了其他 array_function 函数，则抛出 TypeError


def _is_pairwise_metric(estimator):
    """Returns True if estimator accepts pairwise metric.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    metric = getattr(estimator, "metric", None)  # 获取估算器对象的 metric 属性

    return bool(metric == "precomputed")  # 返回是否 metric 属性值为 "precomputed"


def _generate_sparse_data(X_csr):
    """Generate sparse matrices or arrays with {32,64}bit indices of diverse format.

    Parameters
    ----------
    X_csr: scipy.sparse.csr_matrix or scipy.sparse.csr_array
        Input in CSR format.

    Returns
    -------
    out: iter(Matrices) or iter(Arrays)
        In format['dok', 'lil', 'dia', 'bsr', 'csr', 'csc', 'coo',
        'coo_64', 'csc_64', 'csr_64']
    """

    assert X_csr.format == "csr"  # 断言输入的稀疏矩阵格式为 CSR

    yield "csr", X_csr.copy()  # 返回原始的 CSR 格式矩阵副本

    for sparse_format in ["dok", "lil", "dia", "bsr", "csc", "coo"]:
        yield sparse_format, X_csr.asformat(sparse_format)  # 生成并返回其他稀疏矩阵格式的副本

    # Generate large indices matrix only if its supported by scipy
    X_coo = X_csr.asformat("coo")
    X_coo.row = X_coo.row.astype("int64")  # 将 COO 格式的行索引转换为 int64 类型
    X_coo.col = X_coo.col.astype("int64")  # 将 COO 格式的列索引转换为 int64 类型
    yield "coo_64", X_coo  # 返回 int64 类型的 COO 格式矩阵副本

    for sparse_format in ["csc", "csr"]:
        X = X_csr.asformat(sparse_format)
        X.indices = X.indices.astype("int64")  # 将 CSC/CSR 格式的索引数组转换为 int64 类型
        X.indptr = X.indptr.astype("int64")    # 将 CSC/CSR 格式的指针数组转换为 int64 类型
        yield sparse_format + "_64", X  # 返回 int64 类型的 CSC/CSR 格式矩阵副本


def check_array_api_input(
    name,
    estimator_orig,
    array_namespace,
    device=None,
    dtype_name="float64",
    check_values=False,
):
    """Check that the estimator can work consistently with the Array API

    By default, this just checks that the types and shapes of the arrays are
    consistent with calling the same estimator with numpy arrays.

    When check_values is True, it also checks that calling the estimator on the
    array_api Array gives the same results as ndarrays.
    """
    xp = _array_api_for_tests(array_namespace, device)

    X, y = make_classification(random_state=42)  # 生成分类数据集 X 和 y
    X = X.astype(dtype_name, copy=False)  # 将数据类型转换为指定类型，不进行复制操作

    X = _enforce_estimator_tags_X(estimator_orig, X)  # 根据估算器的要求调整 X 的标签
    y = _enforce_estimator_tags_y(estimator_orig, y)  # 根据估算器的要求调整 y 的标签

    est = clone(estimator_orig)  # 克隆估算器对象

    X_xp = xp.asarray(X, device=device)  # 使用指定的设备将 X 转换为 array_api 的数组
    y_xp = xp.asarray(y, device=device)  # 使用指定的设备将 y 转换为 array_api 的数组

    est.fit(X, y)  # 在原始数据上拟合估算器

    array_attributes = {
        key: value for key, value in vars(est).items() if isinstance(value, np.ndarray)
    }  # 收集估算器中所有的 NumPy 数组属性

    est_xp = clone(est)  # 克隆已拟合的估算器对象
    # 设置上下文，确保使用数组 API 调度
    with config_context(array_api_dispatch=True):
        # 使用经验估计器拟合数据
        est_xp.fit(X_xp, y_xp)
        # 获取输入数据的命名空间
        input_ns = get_namespace(X_xp)[0].__name__

    # 检查拟合后的属性数组必须与训练数据的命名空间相同
    for key, attribute in array_attributes.items():
        # 获取经验估计器的属性
        est_xp_param = getattr(est_xp, key)
        # 设置上下文，确保使用数组 API 调度
        with config_context(array_api_dispatch=True):
            # 获取经验估计器属性的命名空间
            attribute_ns = get_namespace(est_xp_param)[0].__name__
        # 断言属性的命名空间与输入数据的命名空间相同
        assert attribute_ns == input_ns, (
            f"'{key}' attribute is in wrong namespace, expected {input_ns} "
            f"got {attribute_ns}"
        )

        # 断言经验估计器属性的设备与输入数据的设备相同
        assert array_device(est_xp_param) == array_device(X_xp)

        # 将经验估计器属性转换为 NumPy 数组
        est_xp_param_np = _convert_to_numpy(est_xp_param, xp=xp)
        if check_values:
            # 如果需要检查数值，则使用 assert_allclose 检查数值是否相近
            assert_allclose(
                attribute,
                est_xp_param_np,
                err_msg=f"{key} not the same",
                atol=_atol_for_type(X.dtype),
            )
        else:
            # 否则，断言属性的形状和数据类型与转换后的 NumPy 数组相同
            assert attribute.shape == est_xp_param_np.shape
            assert attribute.dtype == est_xp_param_np.dtype

    # 要检查的估计器方法，如果支持的话，应该给出相同的结果
    methods = (
        "score",
        "score_samples",
        "decision_function",
        "predict",
        "predict_log_proba",
        "predict_proba",
        "transform",
    )
    # 遍历方法列表中的每一个方法名
    for method_name in methods:
        # 获取估算器对象（estimator）中对应的方法对象
        method = getattr(est, method_name, None)
        # 如果方法对象不存在，则继续下一个方法的处理
        if method is None:
            continue

        # 如果当前方法名为"score"
        if method_name == "score":
            # 调用该方法并计算结果
            result = method(X, y)
            # 使用配置上下文，确保在数组API中调度
            with config_context(array_api_dispatch=True):
                # 调用对应的XP版本估算器的相同方法，并计算结果
                result_xp = getattr(est_xp, method_name)(X_xp, y_xp)
            # 断言结果是一个Python浮点数
            assert isinstance(result, float)
            assert isinstance(result_xp, float)
            # 如果需要检查数值精度，则断言两个结果的差值小于指定的数值精度
            if check_values:
                assert abs(result - result_xp) < _atol_for_type(X.dtype)
            # 继续处理下一个方法
            continue
        else:
            # 对于非"score"方法，调用该方法并计算结果
            result = method(X)
            # 使用配置上下文，确保在数组API中调度
            with config_context(array_api_dispatch=True):
                # 调用对应的XP版本估算器的相同方法，并计算结果
                result_xp = getattr(est_xp, method_name)(X_xp)

        # 使用配置上下文，确保在数组API中调度
        with config_context(array_api_dispatch=True):
            # 获取XP版本结果的命名空间，并获取其第一个元素的名称
            result_ns = get_namespace(result_xp)[0].__name__
        # 断言XP版本结果的命名空间与输入命名空间一致
        assert result_ns == input_ns, (
            f"'{method}' output is in wrong namespace, expected {input_ns}, "
            f"got {result_ns}."
        )

        # 断言XP版本结果的设备与输入数据的设备一致
        assert array_device(result_xp) == array_device(X_xp)
        # 将XP版本结果转换为NumPy数组
        result_xp_np = _convert_to_numpy(result_xp, xp=xp)

        # 如果需要检查数值精度
        if check_values:
            # 断言原始结果与XP版本结果在指定数值精度下相等
            assert_allclose(
                result,
                result_xp_np,
                err_msg=f"{method} did not the return the same result",
                atol=_atol_for_type(X.dtype),
            )
        else:
            # 如果结果具有形状属性，则断言形状和数据类型相同
            if hasattr(result, "shape"):
                assert result.shape == result_xp_np.shape
                assert result.dtype == result_xp_np.dtype

        # 如果当前方法名为"transform"且估算器对象具有"inverse_transform"方法
        if method_name == "transform" and hasattr(est, "inverse_transform"):
            # 对原始结果进行逆变换
            inverse_result = est.inverse_transform(result)
            # 使用配置上下文，确保在数组API中调度
            with config_context(array_api_dispatch=True):
                # 调用对应的XP版本估算器的逆变换方法，并计算结果
                invese_result_xp = est_xp.inverse_transform(result_xp)
                # 获取XP版本逆变换结果的命名空间，并获取其第一个元素的名称
                inverse_result_ns = get_namespace(invese_result_xp)[0].__name__
            # 断言XP版本逆变换结果的命名空间与输入命名空间一致
            assert inverse_result_ns == input_ns, (
                "'inverse_transform' output is in wrong namespace, expected"
                f" {input_ns}, got {inverse_result_ns}."
            )

            # 断言XP版本逆变换结果的设备与输入数据的设备一致
            assert array_device(invese_result_xp) == array_device(X_xp)

            # 将XP版本逆变换结果转换为NumPy数组
            invese_result_xp_np = _convert_to_numpy(invese_result_xp, xp=xp)
            # 如果需要检查数值精度
            if check_values:
                # 断言原始逆变换结果与XP版本逆变换结果在指定数值精度下相等
                assert_allclose(
                    inverse_result,
                    invese_result_xp_np,
                    err_msg="inverse_transform did not the return the same result",
                    atol=_atol_for_type(X.dtype),
                )
            else:
                # 断言逆变换结果的形状和数据类型相同
                assert inverse_result.shape == invese_result_xp_np.shape
                assert inverse_result.dtype == invese_result_xp_np.dtype
# 检查数组 API 的输入和数值，通过调用 check_array_api_input 函数完成
def check_array_api_input_and_values(
    name,
    estimator_orig,
    array_namespace,
    device=None,
    dtype_name="float64",
):
    # 调用 check_array_api_input 函数，设置 check_values 参数为 True，并返回结果
    return check_array_api_input(
        name,
        estimator_orig,
        array_namespace=array_namespace,
        device=device,
        dtype_name=dtype_name,
        check_values=True,
    )


# 检查稀疏估计器的容器
def _check_estimator_sparse_container(name, estimator_orig, sparse_type):
    # 使用种子为 0 的随机数生成器创建一个大小为 (40, 3) 的随机数组 X
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(40, 3))
    # 将 X 中小于 0.8 的元素设为 0
    X[X < 0.8] = 0
    # 调用 _enforce_estimator_tags_X 函数，以确保 X 符合估计器的标签要求
    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 创建一个大小为 40 的整数数组 y，数组元素为 4 倍随机数生成器生成的随机数
    y = (4 * rng.uniform(size=40)).astype(int)
    # 忽略未来警告
    with ignore_warnings(category=FutureWarning):
        # 克隆原始估计器，捕获 deprecation warnings
        estimator = clone(estimator_orig)
    # 调用 _enforce_estimator_tags_y 函数，以确保 y 符合估计器的标签要求
    y = _enforce_estimator_tags_y(estimator, y)
    # 获取安全标签 tags
    tags = _safe_tags(estimator_orig)
    # 使用 _generate_sparse_data 函数生成稀疏数据，并遍历 matrix_format 和 X
    for matrix_format, X in _generate_sparse_data(sparse_type(X)):
        # 忽略未来警告
        with ignore_warnings(category=FutureWarning):
            # 克隆原始估计器
            estimator = clone(estimator_orig)
            # 如果 name 在 ["Scaler", "StandardScaler"] 中
            if name in ["Scaler", "StandardScaler"]:
                # 设置估计器的参数 with_mean=False
                estimator.set_params(with_mean=False)
        # 拟合和预测
        # 如果 matrix_format 中包含 "64"
        if "64" in matrix_format:
            # 错误消息 err_msg，指出估计器不支持 matrix_format 矩阵，未能通过使用 check_array(X, accept_large_sparse=False) 优雅地失败
            err_msg = (
                f"Estimator {name} doesn't seem to support {matrix_format} "
                "matrix, and is not failing gracefully, e.g. by using "
                "check_array(X, accept_large_sparse=False)."
            )
        else:
            # 错误消息 err_msg，指出估计器在稀疏数据上未能优雅地失败，应明确指出稀疏输入不受支持，例如使用 check_array(X, accept_sparse=False)
            err_msg = (
                f"Estimator {name} doesn't seem to fail gracefully on sparse "
                "data: error message should state explicitly that sparse "
                "input is not supported if this is not the case, e.g. by using "
                "check_array(X, accept_sparse=False)."
            )
        # 使用 raises 函数，检查是否引发 TypeError 或 ValueError 异常，匹配关键词 ["sparse", "Sparse"]，允许通过，并指定错误消息为 err_msg
        with raises(
            (TypeError, ValueError),
            match=["sparse", "Sparse"],
            may_pass=True,
            err_msg=err_msg,
        ):
            # 忽略未来警告
            with ignore_warnings(category=FutureWarning):
                # 对数据 X 和 y 进行拟合
                estimator.fit(X, y)
            # 如果估计器具有 predict 方法
            if hasattr(estimator, "predict"):
                # 进行预测 pred
                pred = estimator.predict(X)
                # 如果 tags["multioutput_only"] 为真
                if tags["multioutput_only"]:
                    # 断言 pred 的形状为 (X.shape[0], 1)
                    assert pred.shape == (X.shape[0], 1)
                else:
                    # 断言 pred 的形状为 (X.shape[0],)
                    assert pred.shape == (X.shape[0],)
            # 如果估计器具有 predict_proba 方法
            if hasattr(estimator, "predict_proba"):
                # 进行预测概率 probs
                probs = estimator.predict_proba(X)
                # 如果 tags["binary_only"] 为真
                if tags["binary_only"]:
                    # 期望的概率形状为 (X.shape[0], 2)
                    expected_probs_shape = (X.shape[0], 2)
                else:
                    # 期望的概率形状为 (X.shape[0], 4)
                    expected_probs_shape = (X.shape[0], 4)
                # 断言 probs 的形状为 expected_probs_shape
                assert probs.shape == expected_probs_shape


# 检查估计器在稀疏矩阵上的行为
def check_estimator_sparse_matrix(name, estimator_orig):
    # 调用 _check_estimator_sparse_container 函数，稀疏类型为 sparse.csr_matrix
    _check_estimator_sparse_container(name, estimator_orig, sparse.csr_matrix)


# 检查估计器在稀疏数组上的行为
def check_estimator_sparse_array(name, estimator_orig):
    # 如果 SPARSE_ARRAY_PRESENT 存在
    if SPARSE_ARRAY_PRESENT:
        # 调用 _check_estimator_sparse_container 函数，稀疏类型为 sparse.csr_array
        _check_estimator_sparse_container(name, estimator_orig, sparse.csr_array)


# 忽略未来警告
@ignore_warnings(category=FutureWarning)
# 检查估算器是否接受 'sample_weight' 参数类型为 pandas.Series 在 'fit' 函数中
def check_sample_weights_pandas_series(name, estimator_orig):
    # 克隆原始估算器
    estimator = clone(estimator_orig)
    try:
        import pandas as pd
        
        # 构造特征矩阵 X
        X = np.array(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 1],
                [2, 2],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]
        )
        # 将 X 转换为 pandas.DataFrame 并强制遵循估算器的标签
        X = pd.DataFrame(_enforce_estimator_tags_X(estimator_orig, X), copy=False)
        
        # 构造目标向量 y
        y = pd.Series([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
        
        # 构造样本权重 weights 为 pandas.Series 类型
        weights = pd.Series([1] * 12)
        
        # 如果估算器要求 'multioutput_only' 标签，则将 y 转换为 DataFrame
        if _safe_tags(estimator, key="multioutput_only"):
            y = pd.DataFrame(y, copy=False)
        
        # 尝试拟合估算器
        try:
            estimator.fit(X, y, sample_weight=weights)
        except ValueError:
            # 如果抛出 ValueError，则说明估算器不接受 'sample_weight' 参数为 pandas.Series 类型
            raise ValueError(
                "Estimator {0} raises error if "
                "'sample_weight' parameter is of "
                "type pandas.Series".format(name)
            )
    except ImportError:
        # 如果导入 pandas 失败，则跳过测试
        raise SkipTest(
            "pandas is not installed: not testing for "
            "input of type pandas.Series to class weight."
        )


@ignore_warnings(category=(FutureWarning))
# 检查估算器是否接受 'sample_weight' 参数类型为 _NotAnArray 在 'fit' 函数中
def check_sample_weights_not_an_array(name, estimator_orig):
    # 克隆原始估算器
    estimator = clone(estimator_orig)
    
    # 构造特征矩阵 X
    X = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    )
    # 将 X 转换为 _NotAnArray 类型并强制遵循估算器的标签
    X = _NotAnArray(_enforce_estimator_tags_X(estimator_orig, X))
    
    # 构造目标向量 y
    y = _NotAnArray([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
    
    # 构造样本权重 weights 为 _NotAnArray 类型
    weights = _NotAnArray([1] * 12)
    
    # 如果估算器要求 'multioutput_only' 标签，则将 y 转换为二维数组
    if _safe_tags(estimator, key="multioutput_only"):
        y = _NotAnArray(y.data.reshape(-1, 1))
    
    # 尝试拟合估算器
    estimator.fit(X, y, sample_weight=weights)


@ignore_warnings(category=(FutureWarning))
# 检查估算器是否接受 'sample_weight' 参数类型为 list 在 'fit' 函数中
def check_sample_weights_list(name, estimator_orig):
    # 克隆原始估算器
    estimator = clone(estimator_orig)
    
    # 使用随机数种子生成器生成特征矩阵 X
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = _enforce_estimator_tags_X(estimator_orig, rnd.uniform(size=(n_samples, 3)))
    
    # 构造目标向量 y
    y = np.arange(n_samples) % 3
    y = _enforce_estimator_tags_y(estimator, y)
    
    # 构造样本权重 sample_weight 为列表类型
    sample_weight = [3] * n_samples
    
    # 测试确保估算器不会引发任何异常
    estimator.fit(X, y, sample_weight=sample_weight)


@ignore_warnings(category=FutureWarning)
# 检查估算器是否在 sample_weight 的形状与输入数据不匹配时引发错误
def check_sample_weights_shape(name, estimator_orig):
    # 克隆原始估算器以便进行操作
    estimator = clone(estimator_orig)
    
    # 创建一个包含特征数据的 NumPy 数组 X
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ]
    )
    
    # 创建一个包含目标变量数据的 NumPy 数组 y
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2])
    
    # 根据估算器的需求，调整目标变量 y 的标签
    y = _enforce_estimator_tags_y(estimator, y)

    # 使用样本权重为每个样本进行拟合
    estimator.fit(X, y, sample_weight=np.ones(len(y)))

    # 使用断言检查当样本权重维度为 2 * len(y) 时是否触发 ValueError
    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones(2 * len(y)))

    # 使用断言检查当样本权重的形状为 (len(y), 2) 时是否触发 ValueError
    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones((len(y), 2)))
@ignore_warnings(category=FutureWarning)
def check_sample_weights_invariance(name, estimator_orig, kind="ones"):
    # 检查样本权重是否不变的函数
    # 对于 kind="ones"，检查估算器在单位权重和无权重时是否产生相同结果
    # 对于 kind="zeros"，检查设置样本权重为0是否等效于删除对应样本

    # 克隆原始估算器，生成两个新的估算器对象
    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    # 设置两个估算器的随机状态为0
    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)

    # 创建输入数据 X1 和标签 y1
    X1 = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    if kind == "ones":
        # 对于 kind="ones"，X2 和 y2 与 X1、y1 相同，样本权重 sw2 全为1
        X2 = X1
        y2 = y1
        sw2 = np.ones(shape=len(y1))
        # 错误消息
        err_msg = (
            f"For {name} sample_weight=None is not equivalent to sample_weight=ones"
        )
    elif kind == "zeros":
        # 对于 kind="zeros"，构建一个数据集，如果忽略权重，它与 (X, y) 非常不同，但给定权重时与 (X, y) 相同
        X2 = np.vstack([X1, X1 + 1])
        y2 = np.hstack([y1, 3 - y1])
        sw2 = np.ones(shape=len(y1) * 2)
        sw2[len(y1) :] = 0
        # 将 X2、y2、sw2 随机打乱
        X2, y2, sw2 = shuffle(X2, y2, sw2, random_state=0)

        # 错误消息
        err_msg = (
            f"For {name}, a zero sample_weight is not equivalent to removing the sample"
        )
    else:  # pragma: no cover
        raise ValueError

    # 对 y1 和 y2 进行标签强制规范化
    y1 = _enforce_estimator_tags_y(estimator1, y1)
    y2 = _enforce_estimator_tags_y(estimator2, y2)

    # 使用样本权重为 None 的 X1 拟合 estimator1，使用样本权重为 sw2 的 X2 拟合 estimator2
    estimator1.fit(X1, y=y1, sample_weight=None)
    estimator2.fit(X2, y=y2, sample_weight=sw2)

    # 对于每种方法（预测、预测概率、决策函数、转换函数），检查原始估算器是否具有该方法
    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator_orig, method):
            # 分别调用 estimator1 和 estimator2 的 method 方法进行预测或转换，并检查它们是否接近
            X_pred1 = getattr(estimator1, method)(X1)
            X_pred2 = getattr(estimator2, method)(X1)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)


def check_sample_weights_not_overwritten(name, estimator_orig):
    # 检查估算器是否没有覆盖传递的样本权重参数

    # 克隆原始估算器，生成新的估算器对象
    estimator = clone(estimator_orig)
    # 设置估算器的随机状态为0
    set_random_state(estimator, random_state=0)

    # 创建输入数据 X 和标签 y
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    # 对 y 进行标签强制规范化
    y = _enforce_estimator_tags_y(estimator, y)
    # 创建一个包含所有元素为1的数组，长度与y的行数相同
    sample_weight_original = np.ones(y.shape[0])
    # 将数组中第一个元素的值设为10.0
    sample_weight_original[0] = 10.0
    
    # 复制原始的样本权重数组
    sample_weight_fit = sample_weight_original.copy()
    
    # 使用给定的X、y和调整后的样本权重数组进行估计器的拟合
    estimator.fit(X, y, sample_weight=sample_weight_fit)
    
    # 准备错误消息，指示在拟合期间估算器已经覆盖了在fit期间给定的原始样本权重
    err_msg = f"{name} overwrote the original `sample_weight` given during fit"
    # 断言调整后的样本权重数组与原始的样本权重数组几乎相等，若不相等则抛出错误消息err_msg
    assert_allclose(sample_weight_fit, sample_weight_original, err_msg=err_msg)
@ignore_warnings(category=(FutureWarning, UserWarning))
# 使用装饰器忽略指定类型的警告

def check_dtype_object(name, estimator_orig):
    # 检查估算器在可能的情况下将 dtype 为 object 的数据视为数值型

    rng = np.random.RandomState(0)
    # 创建随机数生成器对象

    X = _enforce_estimator_tags_X(estimator_orig, rng.uniform(size=(40, 10)))
    # 强制将数据 X 标记为估算器所需类型

    X = X.astype(object)
    # 将 X 的数据类型转换为 object

    tags = _safe_tags(estimator_orig)
    # 获取估算器的安全标签信息

    y = (X[:, 0] * 4).astype(int)
    # 创建目标变量 y，将第一列数据乘以 4 并转换为整数类型

    estimator = clone(estimator_orig)
    # 克隆估算器对象

    y = _enforce_estimator_tags_y(estimator, y)
    # 强制将目标变量 y 标记为估算器所需类型

    estimator.fit(X, y)
    # 使用 X 和 y 对估算器进行拟合

    if hasattr(estimator, "predict"):
        estimator.predict(X)
        # 如果估算器支持预测操作，则对 X 进行预测

    if hasattr(estimator, "transform"):
        estimator.transform(X)
        # 如果估算器支持变换操作，则对 X 进行变换

    with raises(Exception, match="Unknown label type", may_pass=True):
        estimator.fit(X, y.astype(object))
        # 期望捕获到标签类型未知的异常，如果捕获到则通过测试

    if "string" not in tags["X_types"]:
        X[0, 0] = {"foo": "bar"}
        # 如果输入数据不包含字符串类型，则将 X 的第一个元素改为字典类型
        # 引发错误由以下操作引起：
        # - `check_array` 中的 `np.asarray`
        # - 编码器中的 `_unique_python`

        msg = "argument must be .* string.* number"
        with raises(TypeError, match=msg):
            estimator.fit(X, y)
            # 期望捕获到类型错误异常，匹配指定的错误消息

    else:
        # 如果支持字符串的估算器不会调用 np.asarray 将数据转换为数值型，
        # 因此不会引发错误。检查输入数组中每个元素的数据类型将是昂贵的操作。
        # 参考 #11401 进行详细讨论。
        estimator.fit(X, y)
        # 对估算器进行拟合操作


def check_complex_data(name, estimator_orig):
    # 检查估算器在提供复杂数据时是否引发异常

    rng = np.random.RandomState(42)
    # 创建随机数生成器对象

    X = rng.uniform(size=10) + 1j * rng.uniform(size=10)
    # 创建包含复数的数据 X

    X = X.reshape(-1, 1)
    # 将 X 重新整形为列向量

    y = rng.randint(low=0, high=2, size=10) + 1j
    # 创建包含复数的目标变量 y

    estimator = clone(estimator_orig)
    # 克隆估算器对象

    set_random_state(estimator, random_state=0)
    # 设置估算器的随机状态为 0

    with raises(ValueError, match="Complex data not supported"):
        estimator.fit(X, y)
        # 期望捕获到值错误异常，匹配指定的错误消息


@ignore_warnings
# 使用装饰器忽略所有警告

def check_dict_unchanged(name, estimator_orig):
    # 检查估算器在某些情况下是否引发特定的数值错误异常

    # 对于 "SpectralCoclustering" 类型的估算器，直接返回
    if name in ["SpectralCoclustering"]:
        return

    rnd = np.random.RandomState(0)
    # 创建随机数生成器对象

    if name in ["RANSACRegressor"]:
        X = 3 * rnd.uniform(size=(20, 3))
    else:
        X = 2 * rnd.uniform(size=(20, 3))
    # 根据估算器类型选择不同的数据 X

    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 强制将数据 X 标记为估算器所需类型

    y = X[:, 0].astype(int)
    # 创建目标变量 y，将 X 的第一列数据转换为整数类型

    estimator = clone(estimator_orig)
    # 克隆估算器对象

    y = _enforce_estimator_tags_y(estimator, y)
    # 强制将目标变量 y 标记为估算器所需类型

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
        # 如果估算器支持 n_components 属性，则设置其值为 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
        # 如果估算器支持 n_clusters 属性，则设置其值为 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1
        # 如果估算器支持 n_best 属性，则设置其值为 1

    set_random_state(estimator, 1)
    # 设置估算器的随机状态为 1

    estimator.fit(X, y)
    # 对估算器进行拟合操作
    # 对于给定的方法列表 ["predict", "transform", "decision_function", "predict_proba"]，逐一检查是否存在于 estimator 对象中
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        # 如果 estimator 对象具有当前遍历到的方法
        if hasattr(estimator, method):
            # 备份当前 estimator 对象的 __dict__ 属性，即对象的所有属性和状态的字典表示
            dict_before = estimator.__dict__.copy()
            # 调用当前方法（method），传递参数 X 进行预测、转换或者决策函数计算等操作
            getattr(estimator, method)(X)
            # 断言：验证在调用方法后，estimator 对象的 __dict__ 属性未发生变化
            assert estimator.__dict__ == dict_before, (
                "Estimator changes __dict__ during %s" % method
            )
# 检查属性是否为公共参数，返回布尔值
def _is_public_parameter(attr):
    return not (attr.startswith("_") or attr.endswith("_"))


# 忽略未来警告类别的装饰器，用于检查不覆盖参数
@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    # 检查是否有 deprecated_original 属性，用于跳过已废弃的类
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        return

    # 克隆原始的 estimator 对象
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    # 创建一个 20x3 的随机数组，用作特征矩阵 X
    X = 3 * rnd.uniform(size=(20, 3))
    # 根据 estimator 原始对象的要求，对 X 进行标签强制操作
    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 从 X 的第一列创建目标向量 y，并强制类型转换为整数
    y = X[:, 0].astype(int)
    # 根据 estimator 对象的要求，对 y 进行标签强制操作
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果 estimator 具有 n_components 属性，则将其设置为 1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果 estimator 具有 n_clusters 属性，则将其设置为 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    # 设置随机状态为 1
    set_random_state(estimator, 1)
    # 在调用 fit 方法前，获取 estimator 的当前属性字典的副本
    dict_before_fit = estimator.__dict__.copy()
    # 调用 estimator 的 fit 方法，拟合数据 X 和目标 y
    estimator.fit(X, y)
    # 获取 fit 方法执行后的 estimator 属性字典
    dict_after_fit = estimator.__dict__

    # 筛选出 fit 方法执行后新增的公共属性键列表
    public_keys_after_fit = [
        key for key in dict_after_fit.keys() if _is_public_parameter(key)
    ]

    # 计算 fit 方法执行后新增的属性中不在 fit 前属性中的键列表
    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit.keys()
    ]

    # 检查 fit 方法是否添加了公共属性，如果有则引发 AssertionError
    assert not attrs_added_by_fit, (
        "Estimator adds public attribute(s) during"
        " the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but %s added" % ", ".join(attrs_added_by_fit)
    )

    # 计算 fit 方法后公共属性中发生变化的键列表
    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]

    # 检查 fit 方法是否修改了公共属性，如果有则引发 AssertionError
    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during"
        " the fit method. Estimators are only allowed"
        " to change attributes started"
        " or ended with _, but"
        " %s changed" % ", ".join(attrs_changed_by_fit)
    )


# 忽略未来警告类别的装饰器，用于检查拟合 2D 数组和预测 1D 数组的方法
@ignore_warnings(category=FutureWarning)
def check_fit2d_predict1d(name, estimator_orig):
    rnd = np.random.RandomState(0)
    # 创建一个 20x3 的随机数组，用作特征矩阵 X
    X = 3 * rnd.uniform(size=(20, 3))
    # 根据 estimator 原始对象的要求，对 X 进行标签强制操作
    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 从 X 的第一列创建目标向量 y，并强制类型转换为整数
    y = X[:, 0].astype(int)
    # 克隆原始的 estimator 对象
    estimator = clone(estimator_orig)
    # 根据 estimator 对象的要求，对 y 进行标签强制操作
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果 estimator 具有 n_components 属性，则将其设置为 1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果 estimator 具有 n_clusters 属性，则将其设置为 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    # 设置随机状态为 1
    set_random_state(estimator, 1)
    # 调用 estimator 的 fit 方法，拟合数据 X 和目标 y
    estimator.fit(X, y)

    # 检查预测、转换、决策函数和预测概率方法是否存在，并对其参数进行断言检查
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(
                ValueError, "Reshape your data", getattr(estimator, method), X[0]
            )


# 将函数应用于整个数据集和小批量数据的函数
def _apply_on_subsets(func, X):
    # 对整个数据集 X 应用给定函数 func，返回结果
    result_full = func(X)
    # 获取数据集 X 的特征数量
    n_features = X.shape[1]
    # 对输入数据 X 中的每个批次应用函数 func，并将结果存储在列表中
    result_by_batch = [func(batch.reshape(1, n_features)) for batch in X]

    # 如果 func 的输出是一个元组（例如 score_samples），则只使用元组的第一个元素
    if type(result_full) == tuple:
        result_full = result_full[0]
        # 对 result_by_batch 中的每个元素，使用 lambda 函数获取元组的第一个元素
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    # 如果 result_full 是稀疏矩阵，则将其转换为密集数组
    if sparse.issparse(result_full):
        result_full = result_full.toarray()
        # 将 result_by_batch 中每个稀疏矩阵也转换为密集数组
        result_by_batch = [x.toarray() for x in result_by_batch]

    # 返回扁平化后的 result_full 和 result_by_batch 数组
    return np.ravel(result_full), np.ravel(result_by_batch)
# 使用装饰器忽略未来警告类别，定义一个函数来检查方法的子集不变性
@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    # 检查方法是否在应用于小批量或整个集合时能给出不变的结果
    rnd = np.random.RandomState(0)
    # 创建一个形状为 (20, 3) 的随机数组，并乘以 3
    X = 3 * rnd.uniform(size=(20, 3))
    # 根据评估器的要求处理 X
    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 从 X 中选择第一列作为标签 y，并将其转换为整数类型
    y = X[:, 0].astype(int)
    # 克隆评估器对象
    estimator = clone(estimator_orig)
    # 根据评估器的要求处理 y
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果评估器具有 "n_components" 属性，则将其设置为 1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果评估器具有 "n_clusters" 属性，则将其设置为 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    # 设置评估器的随机状态为 1
    set_random_state(estimator, 1)
    # 使用 X 和 y 进行拟合
    estimator.fit(X, y)

    # 遍历方法列表，对每个方法进行检查
    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        # 构造错误消息，指出方法在应用于子集时不变的情况
        msg = ("{method} of {name} is not invariant when applied to a subset.").format(
            method=method, name=name
        )

        # 如果评估器具有当前方法，则在子集上应用该方法并进行检查
        if hasattr(estimator, method):
            # 调用评估器对象的当前方法，分别在整体数据集和子集上应用
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X
            )
            # 使用 assert_allclose 检查两种应用结果的近似性
            assert_allclose(result_full, result_by_batch, atol=1e-7, err_msg=msg)


# 使用装饰器忽略未来警告类别，定义一个函数来检查方法在不同样本顺序下的不变性
@ignore_warnings(category=FutureWarning)
def check_methods_sample_order_invariance(name, estimator_orig):
    # 检查方法是否在应用于具有不同样本顺序的子集时能给出不变的结果
    rnd = np.random.RandomState(0)
    # 创建一个形状为 (20, 3) 的随机数组，并乘以 3
    X = 3 * rnd.uniform(size=(20, 3))
    # 根据评估器的要求处理 X
    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 从 X 中选择第一列作为标签 y，并将其转换为 int64 类型
    y = X[:, 0].astype(np.int64)
    # 如果评估器要求仅支持二元分类，则将标签 y 中的值 2 替换为 1
    if _safe_tags(estimator_orig, key="binary_only"):
        y[y == 2] = 1
    # 克隆评估器对象
    estimator = clone(estimator_orig)
    # 根据评估器的要求处理 y
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果评估器具有 "n_components" 属性，则将其设置为 1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果评估器具有 "n_clusters" 属性，则将其设置为 2
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 2

    # 设置评估器的随机状态为 1
    set_random_state(estimator, 1)
    # 使用 X 和 y 进行拟合
    estimator.fit(X, y)

    # 随机排列 X 的样本索引
    idx = np.random.permutation(X.shape[0])

    # 遍历方法列表，对每个方法进行检查
    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        # 构造错误消息，指出方法在应用于具有不同样本顺序的数据集时不变的情况
        msg = (
            "{method} of {name} is not invariant when applied to a dataset"
            "with different sample order."
        ).format(method=method, name=name)

        # 如果评估器具有当前方法，则在不同样本顺序的数据集上应用该方法并进行检查
        if hasattr(estimator, method):
            # 使用 assert_allclose_dense_sparse 检查两种应用结果的近似性（适用于密集或稀疏数据）
            assert_allclose_dense_sparse(
                getattr(estimator, method)(X)[idx],
                getattr(estimator, method)(X[idx]),
                atol=1e-9,
                err_msg=msg,
            )


# 使用装饰器忽略警告，定义一个函数来检查仅包含一个样本的 2d 数组的拟合行为
@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # 检查使用仅包含一个样本的 2d 数组进行拟合时是否能正常工作或返回信息性消息，
    # 错误消息应提到样本数量或类别数量。
    rnd = np.random.RandomState(0)
    # 创建一个形状为 (1, 10) 的随机数组
    X = 3 * rnd.uniform(size=(1, 10))
    # 根据评估器的要求处理 X
    X = _enforce_estimator_tags_X(estimator_orig, X)

    # 从 X 中选择第一列作为标签 y，并将其转换为整数类型
    y = X[:, 0].astype(int)
    # 复制原始的评估器对象，以避免在原对象上进行修改
    estimator = clone(estimator_orig)
    # 根据评估器的标签要求，确保目标 y 符合标签要求
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果评估器具有属性 "n_components"，将其设置为 1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果评估器具有属性 "n_clusters"，将其设置为 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    # 使用固定的随机状态（random_state）设置评估器的随机数种子为 1
    set_random_state(estimator, 1)

    # 对于 OPTICS 算法，设置 min_samples 参数为 1.0，确保 min_cluster_size 不小于数据集大小
    if name == "OPTICS":
        estimator.set_params(min_samples=1.0)

    # 对于 TSNE 算法，设置 perplexity 参数为 0.5，确保 perplexity 不超过样本数
    if name == "TSNE":
        estimator.set_params(perplexity=0.5)

    # 准备用于测试的异常消息列表，用于验证是否引发了特定异常
    msgs = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]

    # 使用 pytest 的 raises 函数，验证 estimator.fit(X, y) 是否会引发 ValueError 异常，
    # 并且异常消息匹配 msgs 列表中的任意一项
    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)
# 忽略警告的装饰器，用于检查在特定条件下拟合失败时是否引发异常
@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # 使用种子为0的随机数生成器创建一个大小为(10, 1)的二维数组 X
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    # 将 X 进行特定估计器标签的强制化处理
    X = _enforce_estimator_tags_X(estimator_orig, X)
    # 从 X 中提取第一列作为整数类型的 y
    y = X[:, 0].astype(int)
    # 克隆估计器对象
    estimator = clone(estimator_orig)
    # 对 y 进行特定估计器标签的强制化处理
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果估计器具有属性 "n_components"，将其设为1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果估计器具有属性 "n_clusters"，将其设为1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # 对于 RandomizedLogisticRegression，确保在子样本中有两个标签
    if name == "RandomizedLogisticRegression":
        estimator.sample_fraction = 1
    # 对于 RANSACRegressor，确保残差阈值不超过0.5
    if name == "RANSACRegressor":
        estimator.residual_threshold = 0.5

    # 再次对 y 进行特定估计器标签的强制化处理
    y = _enforce_estimator_tags_y(estimator, y)
    # 设定估计器的随机状态为1
    set_random_state(estimator, 1)

    # 准备用于匹配的异常消息列表
    msgs = [r"1 feature\(s\)", "n_features = 1", "n_features=1"]

    # 使用 pytest 的 raises 函数检查是否引发 ValueError 异常，并匹配消息列表
    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


# 忽略警告的装饰器，用于检查在拟合1维 X 数组时是否引发 ValueError 异常
@ignore_warnings
def check_fit1d(name, estimator_orig):
    # 使用种子为0的随机数生成器创建一个大小为(20,)的一维数组 X
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    # 将 X 进行特定估计器标签的强制化处理
    y = X.astype(int)
    # 克隆估计器对象
    estimator = clone(estimator_orig)
    # 对 y 进行特定估计器标签的强制化处理
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果估计器具有属性 "n_components"，将其设为1
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    # 如果估计器具有属性 "n_clusters"，将其设为1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    # 设定估计器的随机状态为1
    set_random_state(estimator, 1)

    # 使用 pytest 的 raises 函数检查是否引发 ValueError 异常
    with raises(ValueError):
        estimator.fit(X, y)


# 忽略 FutureWarning 警告的装饰器，用于检查转换器的一般性功能
@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    # 使用 make_blobs 生成一个包含 30 个样本的数据集
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    # 对 X 应用 StandardScaler 进行标准化，并进行特定转换器标签的强制化处理
    X = StandardScaler().fit_transform(X)
    X = _enforce_estimator_tags_X(transformer, X)

    # 如果设置了 readonly_memmap，则创建支持内存映射的数据
    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    # 调用 _check_transformer 函数检查转换器的转换功能
    _check_transformer(name, transformer, X, y)


# 忽略 FutureWarning 警告的装饰器，用于检查数据不是数组的转换器
@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    # 使用 make_blobs 生成一个包含 30 个样本的数据集
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    # 对 X 应用 StandardScaler 进行标准化，并进行特定转换器标签的强制化处理
    X = StandardScaler().fit_transform(X)
    X = _enforce_estimator_tags_X(transformer, X)
    # 创建一个非数组的包装器 this_X 和 this_y
    this_X = _NotAnArray(X)
    this_y = _NotAnArray(np.asarray(y))
    # 调用 _check_transformer 函数检查转换器的转换功能
    _check_transformer(name, transformer, this_X, this_y)
    # 尝试将 X 和 y 转换为列表后再次调用 _check_transformer 函数
    _check_transformer(name, transformer, X.tolist(), y.tolist())


# 忽略 FutureWarning 警告的装饰器，用于检查未拟合的转换器
@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted(name, transformer):
    # 使用 _regression_dataset 获取回归数据集 X 和 y
    X, y = _regression_dataset()
    # 克隆转换器对象
    transformer = clone(transformer)
    # 使用 pytest 的 raises 上下文管理器来检测是否抛出指定的异常
    with raises(
        (AttributeError, ValueError),  # 捕获 AttributeError 和 ValueError 异常
        err_msg=(  # 错误消息的格式化字符串
            "The unfitted "  # 当未拟合的转换器调用 transform 方法时，抛出的错误消息的一部分
            f"transformer {name} does not raise an error when "  # 格式化字符串，显示未拟合的转换器的名称
            "transform is called. Perhaps use "  # 继续错误消息，建议使用 check_is_fitted 在 transform 方法中使用
            "check_is_fitted in transform."  # 最后的错误消息部分，建议在 transform 方法中使用 check_is_fitted
        ),
    ):
        transformer.transform(X)  # 调用 transformer 的 transform 方法并期望抛出异常
# 忽略未来警告类别的警告，通常用于装饰函数或方法
@ignore_warnings(category=FutureWarning)
# 检查状态无关变换器在未拟合状态下使用 transform 方法不会因为未拟合而引发 NotFittedError 异常
def check_transformers_unfitted_stateless(name, transformer):
    """Check that using transform without prior fitting
    doesn't raise a NotFittedError for stateless transformers.
    """
    # 使用固定的随机种子创建随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个随机生成的矩阵 X，形状为 (20, 5)
    X = rng.uniform(size=(20, 5))
    # 根据转换器的要求，对输入的 X 进行标准化处理
    X = _enforce_estimator_tags_X(transformer, X)

    # 克隆转换器对象，以防止修改原始对象
    transformer = clone(transformer)
    # 对输入的 X 进行转换操作，获取转换后的结果 X_trans
    X_trans = transformer.transform(X)

    # 断言转换后的数据行数与原始数据 X 的行数相同
    assert X_trans.shape[0] == X.shape[0]


# 检查转换器对象在给定数据集 X 和 y 上的拟合和转换操作的一致性
def _check_transformer(name, transformer_orig, X, y):
    # 将输入数据 X 转换为 NumPy 数组，获取样本数和特征数
    n_samples, n_features = np.asarray(X).shape
    # 克隆转换器对象，以防止修改原始对象
    transformer = clone(transformer_orig)
    # 设置转换器对象的随机状态

    # 拟合转换器对象
    # 如果转换器名称存在于 CROSS_DECOMPOSITION 中，对目标 y 进行处理以适应特定要求
    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2
        # 如果输入数据 X 是 _NotAnArray 类型，则将 y_ 也转换为 _NotAnArray 类型
        if isinstance(X, _NotAnArray):
            y_ = _NotAnArray(y_)
    else:
        y_ = y

    # 在给定数据集 X 和处理后的目标 y_ 上拟合转换器对象
    transformer.fit(X, y_)

    # 克隆转换器对象，以测试 fit_transform 方法在未拟合的估算器上的工作情况
    transformer_clone = clone(transformer)
    # 调用 fit_transform 方法，获取处理后的预测结果 X_pred
    X_pred = transformer_clone.fit_transform(X, y=y_)

    # 如果 X_pred 是一个元组，则分别检查每个元素的样本数是否与 n_samples 相同
    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert x_pred.shape[0] == n_samples
    else:
        # 否则，检查 X_pred 的样本数是否与 n_samples 相同
        assert X_pred.shape[0] == n_samples
    # 检查 transformer 对象是否具有 transform 属性
    if hasattr(transformer, "transform"):
        # 如果 transformer 名称在 CROSS_DECOMPOSITION 中
        if name in CROSS_DECOMPOSITION:
            # 使用 transformer 对象对 X 进行 transform 操作，使用 y_ 进行拟合并预测
            X_pred2 = transformer.transform(X, y_)
            # 使用 transformer 对象对 X 进行 fit_transform 操作，使用 y=y_ 进行拟合并返回预测结果
            X_pred3 = transformer.fit_transform(X, y=y_)
        else:
            # 使用 transformer 对象对 X 进行 transform 操作
            X_pred2 = transformer.transform(X)
            # 使用 transformer 对象对 X 进行 fit_transform 操作，使用 y=y_ 进行拟合并返回预测结果
            X_pred3 = transformer.fit_transform(X, y=y_)

        # 检查 transformer_orig 对象是否标记为非确定性
        if _safe_tags(transformer_orig, key="non_deterministic"):
            # 抛出 SkipTest 异常，提示该 transformer 是非确定性的
            msg = name + " is non deterministic"
            raise SkipTest(msg)

        # 如果 X_pred 和 X_pred2 都是元组，则逐个比较其中的元素
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                # 检查两个密集或稀疏数组的元素是否接近
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred2,
                    atol=1e-2,
                    err_msg="fit_transform and transform outcomes not consistent in %s"
                    % transformer,
                )
                # 检查两个密集或稀疏数组的元素是否接近
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred3,
                    atol=1e-2,
                    err_msg="consecutive fit_transform outcomes not consistent in %s"
                    % transformer,
                )
        else:
            # 检查两个密集或稀疏数组的元素是否接近
            assert_allclose_dense_sparse(
                X_pred,
                X_pred2,
                err_msg="fit_transform and transform outcomes not consistent in %s"
                % transformer,
                atol=1e-2,
            )
            # 检查两个密集或稀疏数组的元素是否接近
            assert_allclose_dense_sparse(
                X_pred,
                X_pred3,
                atol=1e-2,
                err_msg="consecutive fit_transform outcomes not consistent in %s"
                % transformer,
            )
            # 检查 X_pred2 的样本数是否等于 n_samples
            assert _num_samples(X_pred2) == n_samples
            # 检查 X_pred3 的样本数是否等于 n_samples

            assert _num_samples(X_pred3) == n_samples

        # 如果 X 具有 shape 属性，并且 transformer 不是无状态的，并且 X 是二维数组，并且 X 的列数大于 1
        if (
            hasattr(X, "shape")
            and not _safe_tags(transformer, key="stateless")
            and X.ndim == 2
            and X.shape[1] > 1
        ):
            # 如果 transformer.transform 对 X[:, :-1] 抛出 ValueError 异常
            with raises(
                ValueError,
                err_msg=(
                    f"The transformer {name} does not raise an error "
                    "when the number of features in transform is different from "
                    "the number of features in fit."
                ),
            ):
                transformer.transform(X[:, :-1])
@ignore_warnings
# 忽略警告并执行函数
def check_pipeline_consistency(name, estimator_orig):
    # 如果 estimator_orig 具有 'non_deterministic' 标签，则跳过测试
    if _safe_tags(estimator_orig, key="non_deterministic"):
        msg = name + " is non deterministic"
        raise SkipTest(msg)

    # 使用 make_blobs 生成样本数据集 X 和标签 y
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    # 对 X 进行必要的处理以符合 estimator_orig 的标签要求
    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)
    # 克隆 estimator_orig 以进行后续的训练和比较
    estimator = clone(estimator_orig)
    # 对 y 进行必要的处理以符合 estimator 的标签要求
    y = _enforce_estimator_tags_y(estimator, y)
    # 设置 estimator 的随机状态
    set_random_state(estimator)
    # 创建 pipeline 对象
    pipeline = make_pipeline(estimator)
    # 使用 X, y 训练 estimator 和 pipeline
    estimator.fit(X, y)
    pipeline.fit(X, y)

    # 需要比较的函数列表
    funcs = ["score", "fit_transform"]

    # 遍历函数列表，并验证 estimator 和 pipeline 返回的结果是否一致
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings
# 忽略警告并执行函数
def check_fit_score_takes_y(name, estimator_orig):
    # 检查所有的估算器是否在 fit 和 score 方法中接受可选的 y 参数，以便在管道中使用
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = rnd.uniform(size=(n_samples, 3))
    # 对 X 进行必要的处理以符合 estimator_orig 的标签要求
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = np.arange(n_samples) % 3
    # 克隆 estimator_orig 以进行后续的训练和比较
    estimator = clone(estimator_orig)
    # 对 y 进行必要的处理以符合 estimator 的标签要求
    y = _enforce_estimator_tags_y(estimator, y)
    # 设置 estimator 的随机状态
    set_random_state(estimator)

    # 需要验证的函数列表
    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            # 获取函数的参数名称列表
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # 如果函数是类方法，需要调整参数列表
                args = args[1:]
            # 断言第二个参数应为 'y' 或 'Y'
            assert args[1] in ["y", "Y"], (
                "Expected y or Y as second argument for method "
                "%s of %s. Got arguments: %r."
                % (func_name, type(estimator).__name__, args)
            )


@ignore_warnings
# 忽略警告并执行函数
def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    # 创建不同类型的训练数据集
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    # 对 X_train_32 进行必要的处理以符合 estimator_orig 的标签要求
    X_train_32 = _enforce_estimator_tags_X(estimator_orig, X_train_32)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    # 从 X_train_int_64 中提取标签 y
    y = X_train_int_64[:, 0]
    # 对 y 进行必要的处理以符合 estimator_orig 的标签要求
    y = _enforce_estimator_tags_y(estimator_orig, y)

    # 需要验证的方法列表
    methods = ["predict", "transform", "decision_function", "predict_proba"]
    # 遍历不同的训练数据集，依次为 X_train_32, X_train_64, X_train_int_64, X_train_int_32
    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        # 克隆原始的估算器对象
        estimator = clone(estimator_orig)
        # 设置估算器对象的随机状态为1
        set_random_state(estimator, 1)
        # 使用当前数据集 X_train 进行估算器的训练
        estimator.fit(X_train, y)

        # 遍历方法列表中的每一个方法
        for method in methods:
            # 检查估算器是否具有当前方法
            if hasattr(estimator, method):
                # 调用 getattr 函数获取并执行估算器对象的当前方法，传入当前训练数据集 X_train
                getattr(estimator, method)(X_train)
# 检查转换器是否保持数据类型的函数
def check_transformer_preserve_dtypes(name, transformer_orig):
    # 生成一个包含两个中心的样本集，每个中心位于三维空间中的原点和 [1, 1, 1]
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        cluster_std=0.1,
    )
    # 对 X 进行标准化处理
    X = StandardScaler().fit_transform(X)
    # 根据转换器的估计标签强制执行 X 的特性
    X = _enforce_estimator_tags_X(transformer_orig, X)

    # 遍历转换器的安全标签，这些标签指示保持的数据类型
    for dtype in _safe_tags(transformer_orig, key="preserves_dtype"):
        # 将 X 转换为指定的数据类型
        X_cast = X.astype(dtype)
        # 克隆原始转换器并设置随机状态
        transformer = clone(transformer_orig)
        set_random_state(transformer)
        # 使用不同的方法来拟合和转换 X_cast
        X_trans1 = transformer.fit_transform(X_cast, y)
        X_trans2 = transformer.fit(X_cast, y).transform(X_cast)

        # 针对每种方法检查输出的数据类型是否保持一致
        for Xt, method in zip([X_trans1, X_trans2], ["fit_transform", "transform"]):
            if isinstance(Xt, tuple):
                # 对于交叉分解，当使用 fit_transform 时返回 (x_scores, y_scores)
                # 只检查第一个元素
                Xt = Xt[0]

            # 断言输出的数据类型与预期的数据类型一致
            assert Xt.dtype == dtype, (
                f"{name} (method={method}) does not preserve dtype. "
                f"Original/Expected dtype={dtype.__name__}, got dtype={Xt.dtype}."
            )


# 忽略未来警告类别的装饰器，检查估计器在空数据下的消息函数
@ignore_warnings(category=FutureWarning)
def check_estimators_empty_data_messages(name, estimator_orig):
    # 克隆估计器并设置随机状态
    e = clone(estimator_orig)
    set_random_state(e, 1)

    # 创建零样本数据 X_zero_samples
    X_zero_samples = np.empty(0).reshape(0, 3)
    # 预期会抛出 ValueError 异常，指示估计器在使用空数据进行训练时未引发异常
    err_msg = (
        f"The estimator {name} does not raise a ValueError when an "
        "empty data is used to train. Perhaps use check_array in train."
    )
    with raises(ValueError, err_msg=err_msg):
        e.fit(X_zero_samples, [])

    # 创建零特征数据 X_zero_features
    X_zero_features = np.empty(0).reshape(12, 0)
    # 预期会抛出 ValueError 异常，指示估计器在使用零特征数据时未引发异常
    msg = r"0 feature\(s\) \(shape=\(\d*, 0\)\) while a minimum of \d* " "is required."
    with raises(ValueError, match=msg):
        e.fit(X_zero_features, y)


# 忽略未来警告类别的装饰器，检查估计器在包含 NaN 或 inf 数据下的行为函数
@ignore_warnings(category=FutureWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # 检查 Estimator 是否不包含 NaN 或 inf 的数据
    rnd = np.random.RandomState(0)
    # 对 X_train_finite 应用 Estimator 的标签，并生成一个有限的数据集
    X_train_finite = _enforce_estimator_tags_X(
        estimator_orig, rnd.uniform(size=(10, 3))
    )
    # 创建包含 NaN 的 X_train_nan 和包含 inf 的 X_train_inf 数据集
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    # 根据 Estimator 的标签对 y 进行强制执行
    y = _enforce_estimator_tags_y(estimator_orig, y)
    # 检查在拟合过程中是否检查了 NaN 和 inf 的情况
    error_string_fit = f"Estimator {name} doesn't check for NaN and inf in fit."
    # 创建错误信息字符串，指示估算器在预测时未检查NaN和inf
    error_string_predict = f"Estimator {name} doesn't check for NaN and inf in predict."
    # 创建错误信息字符串，指示估算器在变换时未检查NaN和inf
    error_string_transform = (
        f"Estimator {name} doesn't check for NaN and inf in transform."
    )
    # 遍历包含NaN和inf的训练数据集列表
    for X_train in [X_train_nan, X_train_inf]:
        # 忽略将来可能弃用的警告
        with ignore_warnings(category=FutureWarning):
            # 克隆原始的估算器对象
            estimator = clone(estimator_orig)
            # 设置估算器对象的随机状态为1
            set_random_state(estimator, 1)
            
            # 尝试拟合模型
            with raises(ValueError, match=["inf", "NaN"], err_msg=error_string_fit):
                estimator.fit(X_train, y)
            
            # 实际进行拟合
            estimator.fit(X_train_finite, y)

            # 如果估算器具有"predict"方法
            if hasattr(estimator, "predict"):
                # 预测阶段，检查是否引发值错误，匹配错误信息字符串
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=error_string_predict,
                ):
                    estimator.predict(X_train)

            # 如果估算器具有"transform"方法
            if hasattr(estimator, "transform"):
                # 变换阶段，检查是否引发值错误，匹配错误信息字符串
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=error_string_transform,
                ):
                    estimator.transform(X_train)
# 忽略警告并装饰函数，检查非方阵数据是否引发错误
@ignore_warnings
def check_nonsquare_error(name, estimator_orig):
    """Test that error is thrown when non-square data provided."""
    
    # 生成包含 20 个样本和 10 个特征的聚类数据
    X, y = make_blobs(n_samples=20, n_features=10)
    # 克隆给定的估计器对象
    estimator = clone(estimator_orig)
    
    # 使用断言验证在提供非方阵数据时是否引发 ValueError 异常
    with raises(
        ValueError,
        err_msg=(
            f"The pairwise estimator {name} does not raise an error on non-square data"
        ),
    ):
        estimator.fit(X, y)


# 忽略警告并装饰函数，检查所有估计器是否可以被序列化
@ignore_warnings
def check_estimators_pickle(name, estimator_orig, readonly_memmap=False):
    """Test that we can pickle all estimators."""
    
    # 要检查的方法列表
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    # 生成包含 30 个样本的聚类数据
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )

    # 根据估计器对象强制执行标签和属性
    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)

    # 获取估计器的安全标签
    tags = _safe_tags(estimator_orig)
    
    # 如果允许处理 NaN 值
    if tags["allow_nan"]:
        # 将随机选定的 10 个元素设置为 NaN
        rng = np.random.RandomState(42)
        mask = rng.choice(X.size, 10, replace=False)
        X.reshape(-1)[mask] = np.nan

    # 克隆给定的估计器对象
    estimator = clone(estimator_orig)
    
    # 根据标签和属性强制执行 y 值
    y = _enforce_estimator_tags_y(estimator, y)

    # 设置估计器的随机状态
    set_random_state(estimator)
    
    # 使用数据拟合估计器
    estimator.fit(X, y)

    # 如果只读内存映射是真的
    if readonly_memmap:
        # 创建内存映射支持的数据
        unpickled_estimator = create_memmap_backed_data(estimator)
    else:
        # 在这种情况下不需要触碰文件系统
        pickled_estimator = pickle.dumps(estimator)
        module_name = estimator.__module__
        
        # 对于不是测试模块中实现的 sklearn 估计器进行严格检查
        if module_name.startswith("sklearn.") and not (
            "test_" in module_name or module_name.endswith("_testing")
        ):
            assert b"_sklearn_version" in pickled_estimator
        
        # 反序列化估计器对象
        unpickled_estimator = pickle.loads(pickled_estimator)

    # 存储结果的字典
    result = dict()
    
    # 对每个方法进行检查
    for method in check_methods:
        if hasattr(estimator, method):
            # 使用估计器的方法生成结果
            result[method] = getattr(estimator, method)(X)

    # 对每个方法的结果进行断言，验证它们是否与反序列化后的估计器的结果相近
    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)


# 忽略警告并装饰函数，检查部分拟合中特征数是否发生变化
@ignore_warnings(category=FutureWarning)
def check_estimators_partial_fit_n_features(name, estimator_orig):
    # 如果估计器不具有 partial_fit 方法，则直接返回
    if not hasattr(estimator_orig, "partial_fit"):
        return
    
    # 克隆给定的估计器对象
    estimator = clone(estimator_orig)
    
    # 生成包含 50 个样本的聚类数据
    X, y = make_blobs(n_samples=50, random_state=1)
    
    # 根据估计器对象强制执行标签和属性
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)

    try:
        # 如果是分类器，则获取唯一的类别数组
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    except NotImplementedError:
        return
    # 使用 pytest 的 raises 断言来检查 ValueError 异常是否被触发
    with raises(
        ValueError,
        # 设置异常消息的格式化字符串，报告特定估算器（estimator）在部分拟合（partial_fit）时，
        # 当特征数量在调用之间变化时未引发错误。
        err_msg=(
            f"The estimator {name} does not raise an error when the "
            "number of features changes between calls to partial_fit."
        ),
    ):
        # 使用估算器的 partial_fit 方法进行部分拟合，传入 X 的除最后一列之外的所有特征和对应的标签 y
        estimator.partial_fit(X[:, :-1], y)
# 忽略未来警告类别的警告信息
@ignore_warnings(category=FutureWarning)
# 检查多输出分类器的功能
def check_classifier_multioutput(name, estimator):
    # 定义样本数、标签数、类数
    n_samples, n_labels, n_classes = 42, 5, 3
    # 获取评估器的安全标签
    tags = _safe_tags(estimator)
    # 克隆评估器对象
    estimator = clone(estimator)
    # 生成多标签分类数据集
    X, y = make_multilabel_classification(
        random_state=42, n_samples=n_samples, n_labels=n_labels, n_classes=n_classes
    )
    # 使用数据拟合评估器
    estimator.fit(X, y)
    # 进行预测
    y_pred = estimator.predict(X)

    # 断言预测结果的形状是否正确
    assert y_pred.shape == (n_samples, n_classes), (
        "多输出数据的预测结果形状不正确。期望得到 {}，实际得到 {}。".format((n_samples, n_labels), y_pred.shape)
    )
    # 断言预测结果的数据类型是否为整数
    assert y_pred.dtype.kind == "i"

    # 如果评估器具有决策函数方法
    if hasattr(estimator, "decision_function"):
        # 获取决策函数的输出
        decision = estimator.decision_function(X)
        # 断言决策函数输出的类型为 ndarray
        assert isinstance(decision, np.ndarray)
        # 断言决策函数输出的形状是否正确
        assert decision.shape == (n_samples, n_classes), (
            "多输出数据的决策函数输出形状不正确。期望得到 {}，实际得到 {}。".format(
                (n_samples, n_classes), decision.shape
            )
        )

        # 将决策函数的输出转换为预测值
        dec_pred = (decision > 0).astype(int)
        # 根据预测值获取对应的类别
        dec_exp = estimator.classes_[dec_pred]
        # 断言预测结果与决策函数输出的一致性
        assert_array_equal(dec_exp, y_pred)

    # 如果评估器具有预测概率方法
    if hasattr(estimator, "predict_proba"):
        # 获取预测概率
        y_prob = estimator.predict_proba(X)

        # 如果预测概率是列表类型且不是“poor_score”标签
        if isinstance(y_prob, list) and not tags["poor_score"]:
            # 遍历每个类别的预测概率
            for i in range(n_classes):
                # 断言预测概率的形状是否正确
                assert y_prob[i].shape == (n_samples, 2), (
                    "多输出数据的概率预测形状不正确。期望得到 {}，实际得到 {}。".format(
                        (n_samples, 2), y_prob[i].shape
                    )
                )
                # 断言预测概率的最大值索引与预测结果的一致性
                assert_array_equal(
                    np.argmax(y_prob[i], axis=1).astype(int), y_pred[:, i]
                )
        # 如果不是“poor_score”标签
        elif not tags["poor_score"]:
            # 断言预测概率的形状是否正确
            assert y_prob.shape == (n_samples, n_classes), (
                "多输出数据的概率预测形状不正确。期望得到 {}，实际得到 {}。".format(
                    (n_samples, n_classes), y_prob.shape
                )
            )
            # 断言预测概率四舍五入后与预测结果的一致性
            assert_array_equal(y_prob.round().astype(int), y_pred)

    # 如果评估器同时具有决策函数和预测概率方法
    if hasattr(estimator, "decision_function") and hasattr(estimator, "predict_proba"):
        # 遍历每个类别
        for i in range(n_classes):
            # 获取每个类别的预测概率和决策函数输出
            y_proba = estimator.predict_proba(X)[:, i]
            y_decision = estimator.decision_function(X)
            # 断言预测概率的排名与决策函数输出的排名一致性
            assert_array_equal(rankdata(y_proba), rankdata(y_decision[:, i]))


# 忽略未来警告类别的警告信息
@ignore_warnings(category=FutureWarning)
# 检查多输出回归器的功能
def check_regressor_multioutput(name, estimator):
    # 克隆评估器对象
    estimator = clone(estimator)
    # 定义样本数和特征数
    n_samples = n_features = 10

    # 如果评估器不是成对度量
    if not _is_pairwise_metric(estimator):
        # 增加样本数
        n_samples = n_samples + 1

    # 生成回归数据集
    X, y = make_regression(
        random_state=42, n_targets=5, n_samples=n_samples, n_features=n_features
    )
    # 对数据集 X 应用评估器的标签
    X = _enforce_estimator_tags_X(estimator, X)

    # 使用数据拟合评估器
    estimator.fit(X, y)
    # 进行预测
    y_pred = estimator.predict(X)
    # 确保 y_pred 的数据类型为 float64，因为回归器的多输出预测应当是浮点数精度的。
    # 如果不是，则抛出异常并显示错误消息，指明实际数据类型。
    assert y_pred.dtype == np.dtype("float64"), (
        "Multioutput predictions by a regressor are expected to be"
        " floating-point precision. Got {} instead".format(y_pred.dtype)
    )
    
    # 确保 y_pred 的形状与 y 的形状相同，以确保多输出数据的预测形状正确。
    # 如果形状不匹配，则抛出异常并显示错误消息，指明期望的形状和实际得到的形状。
    assert y_pred.shape == y.shape, (
        "The shape of the prediction for multioutput data is incorrect."
        " Expected {}, got {}."
    )
@ignore_warnings(category=FutureWarning)
# 定义一个函数，用于检查聚类算法的表现，忽略未来警告

def check_clustering(name, clusterer_orig, readonly_memmap=False):
    # 克隆传入的聚类器对象
    clusterer = clone(clusterer_orig)
    
    # 生成50个样本和对应的标签，用于聚类
    X, y = make_blobs(n_samples=50, random_state=1)
    
    # 打乱数据集
    X, y = shuffle(X, y, random_state=7)
    
    # 标准化数据集
    X = StandardScaler().fit_transform(X)
    
    # 生成一个随机数生成器
    rng = np.random.RandomState(7)
    
    # 在数据集中添加噪声
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])
    
    # 如果需要使用只读内存映射，则创建内存映射数据
    if readonly_memmap:
        X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

    # 获取样本数量和特征数量
    n_samples, n_features = X.shape
    
    # 捕获过时警告和邻居警告
    if hasattr(clusterer, "n_clusters"):
        # 设置聚类器的簇数为3
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)
    
    # 如果聚类器名为"AffinityPropagation"
    if name == "AffinityPropagation":
        # 设置AffinityPropagation的偏好值为-100，最大迭代次数为100
        clusterer.set_params(preference=-100)
        clusterer.set_params(max_iter=100)

    # 对数据进行拟合
    clusterer.fit(X)
    
    # 使用列表形式进行拟合
    clusterer.fit(X.tolist())

    # 获取预测结果
    pred = clusterer.labels_
    
    # 断言预测结果的形状应为(n_samples,)
    assert pred.shape == (n_samples,)
    
    # 断言调整兰德指数大于0.4
    assert adjusted_rand_score(pred, y) > 0.4
    
    # 如果聚类器被标记为非确定性的，则直接返回
    if _safe_tags(clusterer, key="non_deterministic"):
        return
    
    # 重新设置随机状态
    set_random_state(clusterer)
    
    # 捕获警告信息
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    
    # 断言预测结果一致
    assert_array_equal(pred, pred2)

    # 断言预测结果的数据类型应为int32或int64
    assert pred.dtype in [np.dtype("int32"), np.dtype("int64")]
    assert pred2.dtype in [np.dtype("int32"), np.dtype("int64")]

    # 给X添加噪声，以测试标签的可能值
    labels = clusterer.fit_predict(X_noise)

    # 每个聚类中至少应有一个样本。等价地，labels_应包含其最小值和最大值之间的所有连续值
    labels_sorted = np.unique(labels)
    assert_array_equal(
        labels_sorted, np.arange(labels_sorted[0], labels_sorted[-1] + 1)
    )

    # 预期标签应从0开始（无噪声）或-1（如果有噪声）
    assert labels_sorted[0] in [0, -1]
    
    # 标签应小于等于n_clusters - 1
    if hasattr(clusterer, "n_clusters"):
        n_clusters = getattr(clusterer, "n_clusters")
        assert n_clusters - 1 >= labels_sorted[-1]
    # 否则，标签应小于等于max(labels_)，这一点是必然成立
    # 错误预测字符串，用于错误信息显示
    error_string_predict = "Classifier can't predict when only one class is present."
    # 创建一个指定种子的随机数生成器对象
    rnd = np.random.RandomState(0)
    # 生成训练数据集，大小为10x3，数据服从均匀分布
    X_train = rnd.uniform(size=(10, 3))
    # 生成测试数据集，大小为10x3，数据服从均匀分布
    X_test = rnd.uniform(size=(10, 3))
    # 创建一个包含10个元素的数组，所有元素为1
    y = np.ones(10)
    # 忽略未来警告类别的警告信息
    with ignore_warnings(category=FutureWarning):
        # 克隆原始分类器对象
        classifier = clone(classifier_orig)
        # 使用指定的训练数据X_train和标签y来拟合分类器，期望抛出特定的值错误
        with raises(
            ValueError, match="class", may_pass=True, err_msg=error_string_fit
        ) as cm:
            classifier.fit(X_train, y)

        # 如果捕获到预期的错误并成功匹配错误消息
        if cm.raised_and_matched:
            # 返回，表示成功捕获到预期错误
            return

        # 断言分类器对测试数据X_test的预测结果等于标签y，否则抛出特定错误消息
        assert_array_equal(classifier.predict(X_test), y, err_msg=error_string_predict)
# 忽略未来警告类别为 FutureWarning 的装饰器
@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label_sample_weights(name, classifier_orig):
    """检查接受样本权重的分类器是否适合，或者在问题减少为单类时是否抛出 ValueError 错误，
    如果抛出的错误消息不明确，应包含 'class' 关键字。
    """
    # 构建当只有一个标签时通过样本权重修剪后，分类器适配失败的错误信息
    error_fit = (
        f"{name} failed when fitted on one label after sample_weight trimming. Error "
        "message is not explicit, it should have 'class'."
    )
    # 构建预测结果应只输出剩余类的错误信息
    error_predict = f"{name} prediction results should only output the remaining class."
    # 使用种子为 0 的随机状态创建随机数生成器
    rnd = np.random.RandomState(0)
    # 生成大小为 (10, 10) 的均匀分布的训练和测试数据集
    X_train = rnd.uniform(size=(10, 10))
    X_test = rnd.uniform(size=(10, 10))
    # 创建数组 y，其中每个元素都是其索引对 2 取模的结果，以生成两个类别的标签
    y = np.arange(10) % 2
    # 复制 y 以创建样本权重
    sample_weight = y.copy()  # 选择单个类别
    # 克隆原始分类器对象
    classifier = clone(classifier_orig)

    # 检查分类器是否具有样本权重参数
    if has_fit_parameter(classifier, "sample_weight"):
        # 当具有样本权重参数时，匹配错误信息中应包含的模式和错误消息
        match = [r"\bclass(es)?\b", error_predict]
        err_type, err_msg = (AssertionError, ValueError), error_fit
    else:
        # 当没有样本权重参数时，匹配错误信息中应包含的模式
        match = r"\bsample_weight\b"
        err_type, err_msg = (TypeError, ValueError), None

    # 使用 pytest 的 raises 上下文管理器来捕获错误并验证
    with raises(err_type, match=match, may_pass=True, err_msg=err_msg) as cm:
        # 使用样本权重拟合分类器
        classifier.fit(X_train, y, sample_weight=sample_weight)
        if cm.raised_and_matched:
            # 如果捕获到预期的错误类型和匹配的错误消息，则直接返回
            return
        # 对于不会失败的估算器，它们应能够预测在拟合过程中唯一剩余的类别
        # 使用断言来验证预测结果
        assert_array_equal(
            classifier.predict(X_test), np.ones(10), err_msg=error_predict
        )


# 忽略警告，因为由 decision function 引起
@ignore_warnings
def check_classifiers_train(
    name, classifier_orig, readonly_memmap=False, X_dtype="float64"
):
    """检查分类器的训练过程。
    """
    # 生成大小为 (300, n_features) 的数据集和相关标签
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    # 将数据类型转换为 X_dtype 类型
    X_m = X_m.astype(X_dtype)
    # 对数据集进行混洗
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    # 对数据集进行标准化处理
    X_m = StandardScaler().fit_transform(X_m)
    # 从多类问题生成二分类问题
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    # 如果 readonly_memmap 为 True，则使用内存映射创建数据
    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    # 创建问题列表，其中包含二分类和可能的多分类问题
    problems = [(X_b, y_b)]
    # 获取分类器原始对象的标签信息
    tags = _safe_tags(classifier_orig)
    # 如果不是仅支持二分类，则添加多分类问题到问题列表
    if not tags["binary_only"]:
        problems.append((X_m, y_m))


def check_outlier_corruption(num_outliers, expected_outliers, decision):
    """检查异常值破坏情况。
    """
    # 检查由于异常分数中的绑定可能导致与给定异常污染级别的偏差
    if num_outliers < expected_outliers:
        start = num_outliers
        end = expected_outliers + 1
    else:
        start = expected_outliers
        end = num_outliers + 1

    # 确保 'critical area' 中的所有值都是绑定的，
    # 导致提供的异常污染级别与实际污染级别之间的差异。
    # 对决策值进行排序，返回一个按升序排列的 NumPy 数组
    sorted_decision = np.sort(decision)
    # 设置一条错误信息，用于断言失败时的提示
    msg = (
        "The number of predicted outliers is not equal to the expected "
        "number of outliers and this difference is not explained by the "
        "number of ties in the decision_function values"
    )
    # 使用断言确保在指定范围内的决策值是唯一的，否则抛出错误消息
    assert len(np.unique(sorted_decision[start:end])) == 1, msg
# 检查训练集中的异常值
def check_outliers_train(name, estimator_orig, readonly_memmap=True):
    # 创建一个包含300个样本的虚拟数据集，并对其进行随机排序
    n_samples = 300
    X, _ = make_blobs(n_samples=n_samples, random_state=0)
    X = shuffle(X, random_state=7)

    # 如果设置了只读内存映射标志，将数据集转换为内存映射格式
    if readonly_memmap:
        X = create_memmap_backed_data(X)

    # 获取数据集的样本数和特征数
    n_samples, n_features = X.shape

    # 复制给定的估计器对象，以防止对原始对象的修改
    estimator = clone(estimator_orig)
    # 设置估计器的随机状态
    set_random_state(estimator)

    # 拟合估计器模型到数据集
    estimator.fit(X)

    # 将数据集转换为列表形式并重新拟合估计器模型
    estimator.fit(X.tolist())

    # 使用拟合好的模型进行预测
    y_pred = estimator.predict(X)

    # 断言预测结果的形状应为(n_samples,)，即每个样本的预测值
    assert y_pred.shape == (n_samples,)
    # 断言预测值的数据类型应为整数
    assert y_pred.dtype.kind == "i"
    # 断言预测值的唯一值应为[-1, 1]
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    # 获取估计器对数据集的决策函数值和分数
    decision = estimator.decision_function(X)
    scores = estimator.score_samples(X)

    # 断言决策函数值和分数的数据类型应为浮点数
    for output in [decision, scores]:
        assert output.dtype == np.dtype("float")
        # 断言决策函数值和分数的形状应为(n_samples,)，即每个样本的值
        assert output.shape == (n_samples,)

    # 针对预测时的异常输入情况，检查是否会引发值错误异常
    with raises(ValueError):
        estimator.predict(X.T)

    # 验证决策函数值与预测值的一致性
    dec_pred = (decision >= 0).astype(int)
    dec_pred[dec_pred == 0] = -1
    assert_array_equal(dec_pred, y_pred)

    # 针对决策函数异常输入情况，检查是否会引发值错误异常
    with raises(ValueError):
        estimator.decision_function(X.T)

    # 验证决策函数是分数方法的一种转换
    y_dec = scores - estimator.offset_
    assert_allclose(y_dec, decision)

    # 针对分数方法的异常输入情况，检查是否会引发值错误异常
    with raises(ValueError):
        estimator.score_samples(X.T)

    # 检查污染参数（不适用于具有 nu 参数的 OneClassSVM）
    if hasattr(estimator, "contamination") and not hasattr(estimator, "novelty"):
        # 预期的异常值数量为30个
        expected_outliers = 30
        # 计算污染参数，应等于预期异常值占样本总数的比例
        contamination = expected_outliers / n_samples
        # 设置估计器的污染参数并重新拟合
        estimator.set_params(contamination=contamination)
        estimator.fit(X)
        y_pred = estimator.predict(X)

        # 计算实际异常值数量
        num_outliers = np.sum(y_pred != 1)

        # 对于具有决策函数方法的估计器，检查异常值数量是否等于预期值
        if num_outliers != expected_outliers:
            decision = estimator.decision_function(X)
            # 检查异常值情况是否正确
            check_outlier_corruption(num_outliers, expected_outliers, decision)


# 检查异常值污染参数
def check_outlier_contamination(name, estimator_orig):
    # 当估计器实现参数约束时，检查污染参数是否在 (0.0, 0.5] 范围内
    if not hasattr(estimator_orig, "_parameter_constraints"):
        # 只有实现了参数约束的估计器将被检查
        return
    # 检查在原始估计器的参数约束中是否包含 "contamination"
    if "contamination" not in estimator_orig._parameter_constraints:
        # 如果不包含，则直接返回，不进行后续操作
        return

    # 获取 "contamination" 参数的约束条件
    contamination_constraints = estimator_orig._parameter_constraints["contamination"]
    
    # 检查是否至少有一个约束是实数区间（Interval）类型
    if not any([isinstance(c, Interval) for c in contamination_constraints]):
        # 如果没有找到实数区间类型的约束，则抛出断言错误
        raise AssertionError(
            "contamination constraints should contain a Real Interval constraint."
        )

    # 遍历所有的 "contamination" 约束条件
    for constraint in contamination_constraints:
        # 如果当前约束是实数区间类型
        if isinstance(constraint, Interval):
            # 断言约束的类型是实数，并且左边界大于等于0，右边界小于等于0.5
            # 同时，要求左边界大于0或者约束闭合方式是右边或都不闭合
            assert (
                constraint.type == Real
                and constraint.left >= 0.0
                and constraint.right <= 0.5
                and (constraint.left > 0 or constraint.closed in {"right", "neither"})
            ), "contamination constraint should be an interval in (0, 0.5]"
@ignore_warnings(category=FutureWarning)
# 定义一个函数，用于验证多标签分类器在预测方法中的输出格式是否符合预期
def check_classifiers_multilabel_representation_invariance(name, classifier_orig):
    # 生成一个多标签分类的模拟数据集，包括特征和标签
    X, y = make_multilabel_classification(
        n_samples=100,      # 样本数
        n_features=2,       # 特征数
        n_classes=5,        # 类别数
        n_labels=3,         # 每个样本的标签数
        length=50,          # 特征向量的长度
        allow_unlabeled=True,   # 是否允许未标记的样本
        random_state=0,     # 随机数种子，确保可重现性
    )
    # 对特征进行缩放
    X = scale(X)

    # 将数据集分为训练集和测试集
    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    # 将标签转换为不同的列表表示形式
    y_train_list_of_lists = y_train.tolist()       # 转换为列表的列表
    y_train_list_of_arrays = list(y_train)         # 转换为列表的数组

    # 克隆分类器对象，设置随机状态
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    # 使用训练集进行拟合，并预测测试集的标签
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # 分别使用两种不同形式的标签列表进行拟合和预测
    y_pred_list_of_lists = classifier.fit(X_train, y_train_list_of_lists).predict(
        X_test
    )
    y_pred_list_of_arrays = classifier.fit(X_train, y_train_list_of_arrays).predict(
        X_test
    )

    # 断言预测结果的一致性
    assert_array_equal(y_pred, y_pred_list_of_arrays)
    assert_array_equal(y_pred, y_pred_list_of_lists)

    # 断言预测结果的数据类型和原始预测结果一致
    assert y_pred.dtype == y_pred_list_of_arrays.dtype
    assert y_pred.dtype == y_pred_list_of_lists.dtype

    # 断言预测结果的类型和原始预测结果一致
    assert type(y_pred) == type(y_pred_list_of_arrays)
    assert type(y_pred) == type(y_pred_list_of_lists)


@ignore_warnings(category=FutureWarning)
# 定义一个函数，用于验证多标签分类器在支持多标签指示目标的预测方法中的输出格式是否符合预期
def check_classifiers_multilabel_output_format_predict(name, classifier_orig):
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 100, 25, 5
    # 生成一个多标签分类的模拟数据集，包括特征和标签
    X, y = make_multilabel_classification(
        n_samples=n_samples,     # 样本数
        n_features=2,            # 特征数
        n_classes=n_outputs,     # 类别数
        n_labels=3,              # 每个样本的标签数
        length=50,               # 特征向量的长度
        allow_unlabeled=True,    # 是否允许未标记的样本
        random_state=0,          # 随机数种子，确保可重现性
    )
    # 对特征进行缩放
    X = scale(X)

    # 将数据集分为训练集和测试集
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # 使用训练集进行拟合
    classifier.fit(X_train, y_train)

    response_method_name = "predict"
    # 获取分类器的预测方法
    predict_method = getattr(classifier, response_method_name, None)
    if predict_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    # 使用测试集进行预测
    y_pred = predict_method(X_test)

    # 断言预测结果的格式与测试集的标签形状和数据类型相同
    assert isinstance(y_pred, np.ndarray), (
        f"{name}.predict is expected to output a NumPy array. Got "
        f"{type(y_pred)} instead."
    )
    assert y_pred.shape == y_test.shape, (
        f"{name}.predict outputs a NumPy array of shape {y_pred.shape} "
        f"instead of {y_test.shape}."
    )
    assert y_pred.dtype == y_test.dtype, (
        f"{name}.predict does not output the same dtype than the targets. "
        f"Got {y_pred.dtype} instead of {y_test.dtype}."
    )


@ignore_warnings(category=FutureWarning)
# 定义一个函数，用于验证多标签分类器在支持多标签指示目标的预测概率方法中的输出格式是否符合预期
def check_classifiers_multilabel_output_format_predict_proba(name, classifier_orig):
    """Check the output of the `predict_proba` method for classifiers supporting
    multilabel-indicator targets."""
    # 克隆分类器对象，以确保原始分类器不会被修改
    classifier = clone(classifier_orig)
    # 设置分类器的随机状态，确保结果可复现性
    set_random_state(classifier)

    # 定义数据集相关参数：样本数量、测试集大小、输出类别数量
    n_samples, test_size, n_outputs = 100, 25, 5
    # 生成多标签分类的合成数据集 X 和对应标签 y
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    # 对特征数据 X 进行标准化处理
    X = scale(X)

    # 划分训练集和测试集
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    # 使用训练集数据训练分类器
    classifier.fit(X_train, y_train)

    # 定义要调用的分类器方法名称
    response_method_name = "predict_proba"
    # 获取分类器中的预测概率方法
    predict_proba_method = getattr(classifier, response_method_name, None)
    # 如果获取不到预测概率方法，则抛出跳过测试的异常
    if predict_proba_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    # 使用测试集数据进行预测，获取预测的概率值
    y_pred = predict_proba_method(X_test)

    # y_pred 的形状有两种可能性：
    # - 长度为 n_outputs 的列表，每个元素是形状为 (n_samples, 2) 的 NumPy 数组；
    # - 形状为 (n_samples, n_outputs) 的 NumPy 数组。
    # 数据类型应为浮点数
    if isinstance(y_pred, list):
        # 当 y_pred 是列表时，应当保证列表长度为 n_outputs，并且每个元素是 NumPy 数组
        assert len(y_pred) == n_outputs, (
            f"When {name}.predict_proba returns a list, the list should "
            "be of length n_outputs and contain NumPy arrays. Got length "
            f"of {len(y_pred)} instead of {n_outputs}."
        )
        for pred in y_pred:
            # 检查每个预测数组的形状是否为 (test_size, 2)
            assert pred.shape == (test_size, 2), (
                f"When {name}.predict_proba returns a list, this list "
                "should contain NumPy arrays of shape (n_samples, 2). Got "
                f"NumPy arrays of shape {pred.shape} instead of "
                f"{(test_size, 2)}."
            )
            # 检查每个预测数组的数据类型是否为浮点型
            assert pred.dtype.kind == "f", (
                f"When {name}.predict_proba returns a list, it should "
                "contain NumPy arrays with floating dtype. Got "
                f"{pred.dtype} instead."
            )
            # 检查预测概率是否正确，即每行的概率和为 1
            err_msg = (
                f"When {name}.predict_proba returns a list, each NumPy "
                "array should contain probabilities for each class and "
                "thus each row should sum to 1 (or close to 1 due to "
                "numerical errors)."
            )
            assert_allclose(pred.sum(axis=1), 1, err_msg=err_msg)
    # 如果 y_pred 是一个 NumPy 数组
    elif isinstance(y_pred, np.ndarray):
        # 确保 y_pred 的形状为 (test_size, n_outputs)
        assert y_pred.shape == (test_size, n_outputs), (
            f"When {name}.predict_proba returns a NumPy array, the "
            f"expected shape is (n_samples, n_outputs). Got {y_pred.shape}"
            f" instead of {(test_size, n_outputs)}."
        )
        # 确保 y_pred 的数据类型为浮点型
        assert y_pred.dtype.kind == "f", (
            f"When {name}.predict_proba returns a NumPy array, the "
            f"expected data type is floating. Got {y_pred.dtype} instead."
        )
        # 错误消息，指出当 {name}.predict_proba 返回一个 NumPy 数组时，期望它提供正类别的概率，因此应该包含在 0 到 1 之间的值
        err_msg = (
            f"When {name}.predict_proba returns a NumPy array, this array "
            "is expected to provide probabilities of the positive class "
            "and should therefore contain values between 0 and 1."
        )
        # 确保 y_pred 的所有值都大于 0，小于 1
        assert_array_less(0, y_pred, err_msg=err_msg)
        assert_array_less(y_pred, 1, err_msg=err_msg)
    else:
        # 如果 y_pred 的类型不是列表或 NumPy 数组，则抛出 ValueError 异常
        raise ValueError(
            f"Unknown returned type {type(y_pred)} by {name}."
            "predict_proba. A list or a Numpy array is expected."
        )
# 忽略未来警告，用于检查多标签输出格式决策函数的分类器
@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_decision_function(name, classifier_orig):
    """Check the output of the `decision_function` method for classifiers supporting
    multilabel-indicator targets."""
    
    # 克隆分类器对象
    classifier = clone(classifier_orig)
    # 设置分类器的随机状态
    set_random_state(classifier)

    # 创建多标签分类数据集
    n_samples, test_size, n_outputs = 100, 25, 5
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    # 对特征进行缩放
    X = scale(X)

    # 划分训练集和测试集
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    # 在训练集上拟合分类器
    classifier.fit(X_train, y_train)

    # 确定分类器的响应方法名称
    response_method_name = "decision_function"
    decision_function_method = getattr(classifier, response_method_name, None)
    if decision_function_method is None:
        # 如果分类器没有决策函数，则抛出跳过测试异常
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    # 使用测试集计算决策函数的预测值
    y_pred = decision_function_method(X_test)

    # 断言预测结果是一个 NumPy 数组
    assert isinstance(y_pred, np.ndarray), (
        f"{name}.decision_function is expected to output a NumPy array."
        f" Got {type(y_pred)} instead."
    )
    # 断言预测结果的形状符合预期
    assert y_pred.shape == (test_size, n_outputs), (
        f"{name}.decision_function is expected to provide a NumPy array "
        f"of shape (n_samples, n_outputs). Got {y_pred.shape} instead of "
        f"{(test_size, n_outputs)}."
    )
    # 断言预测结果的数据类型是浮点型
    assert y_pred.dtype.kind == "f", (
        f"{name}.decision_function is expected to output a floating dtype."
        f" Got {y_pred.dtype} instead."
    )


# 忽略未来警告，用于检查未拟合估计器在调用 get_feature_names_out 方法时引发的错误
@ignore_warnings(category=FutureWarning)
def check_get_feature_names_out_error(name, estimator_orig):
    """Check the error raised by get_feature_names_out when called before fit.

    Unfitted estimators with get_feature_names_out should raise a NotFittedError.
    """

    # 克隆估计器对象
    estimator = clone(estimator_orig)
    # 错误消息字符串
    err_msg = (
        f"Estimator {name} should have raised a NotFitted error when fit is called"
        " before get_feature_names_out"
    )
    # 使用 pytest 的 raises 函数检查是否引发 NotFittedError 异常
    with raises(NotFittedError, err_msg=err_msg):
        estimator.get_feature_names_out()


# 忽略未来警告，用于检查估计器的 fit 方法是否返回自身
@ignore_warnings(category=FutureWarning)
def check_estimators_fit_returns_self(name, estimator_orig, readonly_memmap=False):
    """Check if self is returned when calling fit."""
    
    # 创建示例数据集
    X, y = make_blobs(random_state=0, n_samples=21)
    # 对数据集进行预处理
    X = _enforce_estimator_tags_X(estimator_orig, X)

    # 克隆估计器对象
    estimator = clone(estimator_orig)
    # 对标签进行预处理
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果使用只读内存映射，则创建内存映射后端数据
    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    # 设置估计器的随机状态
    set_random_state(estimator)
    # 断言调用 fit 方法后返回的是估计器自身
    assert estimator.fit(X, y) is estimator


# 忽略警告，用于检查未拟合的估计器在调用 predict 方法时是否引发异常
@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise a NotFittedError.
    """
    # 定义一个测试用的数据集 X, y，用于回归器、分类器和异常检测估计器的通用测试
    X, y = _regression_dataset()
    
    # 克隆原始估计器，确保每个方法的测试都在一个独立的实例上进行
    estimator = clone(estimator_orig)
    
    # 遍历方法列表，包括 decision_function、predict、predict_proba 和 predict_log_proba
    for method in (
        "decision_function",
        "predict",
        "predict_proba",
        "predict_log_proba",
    ):
        # 如果估计器具有当前方法
        if hasattr(estimator, method):
            # 使用 raises 断言确保当前估计器未经拟合时调用该方法会引发 NotFittedError 异常
            with raises(NotFittedError):
                # 调用当前方法(method)并传入数据集 X，期望引发 NotFittedError
                getattr(estimator, method)(X)
@ignore_warnings(category=FutureWarning)
# 定义一个装饰器，用于忽略未来警告类别的警告

def check_supervised_y_2d(name, estimator_orig):
    # 获取估计器的安全标签
    tags = _safe_tags(estimator_orig)
    
    # 创建随机数生成器对象，种子为0
    rnd = np.random.RandomState(0)
    
    # 设定样本数量为30
    n_samples = 30
    
    # 创建一个随机样本矩阵 X，形状为(n_samples, 3)，强制符合估计器的标签
    X = _enforce_estimator_tags_X(estimator_orig, rnd.uniform(size=(n_samples, 3)))
    
    # 创建一个简单的类别标签 y，强制符合估计器的标签
    y = np.arange(n_samples) % 3
    y = _enforce_estimator_tags_y(estimator_orig, y)
    
    # 克隆原始估计器对象
    estimator = clone(estimator_orig)
    
    # 设定估计器的随机状态
    set_random_state(estimator)
    
    # 拟合估计器到数据 X, y 上
    estimator.fit(X, y)
    
    # 预测结果
    y_pred = estimator.predict(X)

    # 重新设定估计器的随机状态
    set_random_state(estimator)
    
    # 检查当给定一个二维 y 时，是否会引发 DataConversionWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    
    # 使用估计器预测结果
    y_pred_2d = estimator.predict(X)
    
    # 消息字符串，用于描述收到的警告
    msg = "expected 1 DataConversionWarning, got: %s" % ", ".join(
        [str(w_x) for w_x in w]
    )
    
    if not tags["multioutput"]:
        # 如果不支持多输出，则检查是否发出了警告
        assert len(w) > 0, msg
        assert (
            "DataConversionWarning('A column-vector y"
            " was passed when a 1d array was expected" in msg
        )
    
    # 检查预测结果是否全部接近
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())


@ignore_warnings
# 定义一个装饰器，用于忽略所有警告

def check_classifiers_predictions(X, y, name, classifier_orig):
    # 获取类别的唯一值
    classes = np.unique(y)
    
    # 克隆分类器对象
    classifier = clone(classifier_orig)
    
    # 如果分类器为 BernoulliNB，则按照阈值处理 X
    if name == "BernoulliNB":
        X = X > X.mean()
    
    # 设定分类器的随机状态
    set_random_state(classifier)
    
    # 拟合分类器到数据 X, y 上
    classifier.fit(X, y)
    
    # 使用分类器预测结果
    y_pred = classifier.predict(X)

    # 如果分类器具有 decision_function 方法
    if hasattr(classifier, "decision_function"):
        # 获取决策函数的输出
        decision = classifier.decision_function(X)
        
        # 确保输出是 numpy 数组
        assert isinstance(decision, np.ndarray)
        
        # 如果类别数量为2
        if len(classes) == 2:
            # 将决策结果转换为预测结果
            dec_pred = (decision.ravel() > 0).astype(int)
            dec_exp = classifier.classes_[dec_pred]
            
            # 检查预测结果是否符合期望
            assert_array_equal(
                dec_exp,
                y_pred,
                err_msg=(
                    "decision_function does not match "
                    "classifier for %r: expected '%s', got '%s'"
                )
                % (
                    classifier,
                    ", ".join(map(str, dec_exp)),
                    ", ".join(map(str, y_pred)),
                ),
            )
        
        # 如果分类器的 decision_function_shape 为 ovr
        elif getattr(classifier, "decision_function_shape", "ovr") == "ovr":
            # 获取决策结果中每行的最大值索引
            decision_y = np.argmax(decision, axis=1).astype(int)
            y_exp = classifier.classes_[decision_y]
            
            # 检查预测结果是否符合期望
            assert_array_equal(
                y_exp,
                y_pred,
                err_msg=(
                    "decision_function does not match "
                    "classifier for %r: expected '%s', got '%s'"
                )
                % (
                    classifier,
                    ", ".join(map(str, y_exp)),
                    ", ".join(map(str, y_pred)),
                ),
            )

    # 训练集性能
    # 如果分类器名称不是 "ComplementNB"，执行下面的条件语句
    if name != "ComplementNB":
        # 这是一个针对 ComplementNB 的特殊数据集。
        # 在某些特定情况下，'ComplementNB' 预测的类别数少于预期的数量。
        # 使用 numpy 的 assert_array_equal 函数比较 y 和 y_pred 的唯一值是否相等
        assert_array_equal(np.unique(y), np.unique(y_pred))
    
    # 使用 assert_array_equal 函数比较 classes 和 classifier.classes_ 是否相等
    assert_array_equal(
        classes,
        classifier.classes_,
        # 如果不相等，输出错误信息，指明预期的 classes_ 属性和实际得到的 classifier.classes_
        err_msg="Unexpected classes_ attribute for %r: expected '%s', got '%s'"
        % (
            classifier,
            ", ".join(map(str, classes)),
            ", ".join(map(str, classifier.classes_)),
        ),
    )
# 根据分类器名称和标签返回适当的标签集合，半监督分类器使用-1表示未标记样本
def _choose_check_classifiers_labels(name, y, y_names):
    return (
        y
        if name in ["LabelPropagation", "LabelSpreading", "SelfTrainingClassifier"]
        else y_names
    )


# 检查多分类和二分类问题的分类器类别
def check_classifiers_classes(name, classifier_orig):
    # 创建多类别数据集
    X_multiclass, y_multiclass = make_blobs(
        n_samples=30, random_state=0, cluster_std=0.1
    )
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass, random_state=7)
    X_multiclass = StandardScaler().fit_transform(X_multiclass)

    # 创建二元分类数据集
    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    # 对数据集应用估计器标签
    X_multiclass = _enforce_estimator_tags_X(classifier_orig, X_multiclass)
    X_binary = _enforce_estimator_tags_X(classifier_orig, X_binary)

    # 多类别和二分类的标签名称
    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    # 定义问题列表，包含二分类和可能的多分类问题
    problems = [(X_binary, y_binary, y_names_binary)]
    if not _safe_tags(classifier_orig, key="binary_only"):
        problems.append((X_multiclass, y_multiclass, y_names_multiclass))

    # 遍历所有问题，并针对不同的标签名称类型进行分类器标签的选择和检查
    for X, y, y_names in problems:
        for y_names_i in [y_names, y_names.astype("O")]:
            y_ = _choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    # 设置二元分类的标签名称
    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = _choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)


# 忽略未来警告，检查回归器整数
@ignore_warnings(category=FutureWarning)
def check_regressors_int(name, regressor_orig):
    # 获取回归数据集，将估计器标签应用于X
    X, _ = _regression_dataset()
    X = _enforce_estimator_tags_X(regressor_orig, X[:50])
    rnd = np.random.RandomState(0)
    y = rnd.randint(3, size=X.shape[0])
    y = _enforce_estimator_tags_y(regressor_orig, y)

    # 克隆回归器以控制随机种子
    regressor_1 = clone(regressor_orig)
    regressor_2 = clone(regressor_orig)
    set_random_state(regressor_1)
    set_random_state(regressor_2)

    if name in CROSS_DECOMPOSITION:
        # 如果是交叉分解算法，则创建另一种标签
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y

    # 拟合回归器并预测，确保预测结果接近
    regressor_1.fit(X, y_)
    pred1 = regressor_1.predict(X)
    regressor_2.fit(X, y_.astype(float))
    pred2 = regressor_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


# 忽略未来警告，检查回归器训练
@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    # 获取回归数据集，并将数据类型转换为指定的X_dtype
    X, y = _regression_dataset()
    X = X.astype(X_dtype)
    y = scale(y)  # X已经被缩放
    regressor = clone(regressor_orig)
    X = _enforce_estimator_tags_X(regressor, X)
    y = _enforce_estimator_tags_y(regressor, y)
    # 如果模型名在CROSS_DECOMPOSITION中
    if name in CROSS_DECOMPOSITION:
        # 使用随机种子0创建随机数生成器对象
        rnd = np.random.RandomState(0)
        # 构建新的目标变量y_，包括y本身和2*y加上随机生成的0或1
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        # 转置y_
        y_ = y_.T
    else:
        # 否则，直接使用y作为y_
        y_ = y

    # 如果设置了readonly_memmap参数
    if readonly_memmap:
        # 使用create_memmap_backed_data函数创建内存映射数据
        X, y, y_ = create_memmap_backed_data([X, y, y_])

    # 如果回归器(regressor)没有alphas属性但有alpha属性
    if not hasattr(regressor, "alphas") and hasattr(regressor, "alpha"):
        # 对于线性回归器，需要设置alpha参数为0.01，但对于非广义交叉验证的回归器则不需要
        regressor.alpha = 0.01
    # 如果模型名为"PassiveAggressiveRegressor"
    if name == "PassiveAggressiveRegressor":
        # 设置回归器的参数C为0.01

        regressor.C = 0.01

    # 当fit方法对于不正确或者格式不正确的输入抛出错误时
    with raises(
        ValueError,
        err_msg=(
            f"The classifier {name} does not raise an error when "
            "incorrect/malformed input data for fit is passed. The number of "
            "training examples is not the same as the number of labels. Perhaps "
            "use check_X_y in fit."
        ),
    ):
        # 使用X和y的前len(y)-1个数据来拟合回归器，期望会抛出ValueError异常
        regressor.fit(X, y[:-1])
    # 使用set_random_state函数设置回归器的随机状态
    set_random_state(regressor)
    # 使用X和y_来拟合回归器
    regressor.fit(X, y_)
    # 将X和y_转换为列表形式后拟合回归器
    regressor.fit(X.tolist(), y_.tolist())
    # 使用X预测目标变量y，并将结果存储在y_pred中
    y_pred = regressor.predict(X)
    # 断言y_pred的形状与y_的形状相同
    assert y_pred.shape == y_.shape

    # TODO: find out why PLS and CCA fail. RANSAC is random
    # and furthermore assumes the presence of outliers, hence
    # skipped
    # 如果回归器没有标记为"poor_score"
    if not _safe_tags(regressor, key="poor_score"):
        # 断言回归器在X和y_上的得分大于0.5
        assert regressor.score(X, y_) > 0.5
@ignore_warnings
# 使用装饰器 ignore_warnings 来忽略警告信息
def check_regressors_no_decision_function(name, regressor_orig):
    # 检查回归器是否没有 decision_function、predict_proba 或 predict_log_proba 方法
    rng = np.random.RandomState(0)
    regressor = clone(regressor_orig)

    # 创建一个随机正态分布的数据集 X 和相应的标签 y
    X = rng.normal(size=(10, 4))
    # 根据 regressor_orig 的标签强制转换 X 的数据类型
    X = _enforce_estimator_tags_X(regressor_orig, X)
    # 根据 regressor 的标签强制转换 y 的数据类型
    y = _enforce_estimator_tags_y(regressor, X[:, 0])

    # 使用 X 和 y 来拟合回归器
    regressor.fit(X, y)

    # 定义需要检查的方法列表
    funcs = ["decision_function", "predict_proba", "predict_log_proba"]
    for func_name in funcs:
        # 确保 regressor 没有 funcs 中列出的方法
        assert not hasattr(regressor, func_name)


@ignore_warnings(category=FutureWarning)
# 使用 ignore_warnings 装饰器来忽略 FutureWarning 类别的警告信息
def check_class_weight_classifiers(name, classifier_orig):
    # 根据 binary_only 标签检查 classifier_orig 是否有问题
    if _safe_tags(classifier_orig, key="binary_only"):
        problems = [2]
    else:
        problems = [2, 3]

    for n_centers in problems:
        # 创建一个非常嘈杂的数据集
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        # 将数据集分割为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        # 如果 classifier_orig 标签包含 pairwise，则手动设置 gram 矩阵
        if _safe_tags(classifier_orig, key="pairwise"):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        # 计算训练集中唯一标签的数量
        n_centers = len(np.unique(y_train))

        # 根据标签数量设置类别权重
        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        # 克隆 classifier_orig 并设置类别权重
        classifier = clone(classifier_orig).set_params(class_weight=class_weight)
        # 如果 classifier 有 n_iter 属性，则设置为 100
        if hasattr(classifier, "n_iter"):
            classifier.set_params(n_iter=100)
        # 如果 classifier 有 max_iter 属性，则设置为 1000
        if hasattr(classifier, "max_iter"):
            classifier.set_params(max_iter=1000)
        # 如果 classifier 有 min_weight_fraction_leaf 属性，则设置为 0.01
        if hasattr(classifier, "min_weight_fraction_leaf"):
            classifier.set_params(min_weight_fraction_leaf=0.01)
        # 如果 classifier 有 n_iter_no_change 属性，则设置为 20
        if hasattr(classifier, "n_iter_no_change"):
            classifier.set_params(n_iter_no_change=20)

        # 设置随机状态
        set_random_state(classifier)
        # 使用训练集拟合分类器
        classifier.fit(X_train, y_train)
        # 预测测试集的标签
        y_pred = classifier.predict(X_test)
        # 如果 classifier_orig 没有 poor_score 标签，则验证预测精度
        if not _safe_tags(classifier_orig, key="poor_score"):
            assert np.mean(y_pred == 0) > 0.87


@ignore_warnings(category=FutureWarning)
# 使用 ignore_warnings 装饰器来忽略 FutureWarning 类别的警告信息
def check_class_weight_balanced_classifiers(
    name, classifier_orig, X_train, y_train, X_test, y_test, weights
):
    # 克隆 classifier_orig
    classifier = clone(classifier_orig)
    # 如果 classifier 有 n_iter 属性，则设置为 100
    if hasattr(classifier, "n_iter"):
        classifier.set_params(n_iter=100)
    # 如果 classifier 有 max_iter 属性，则设置为 1000
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)

    # 设置随机状态
    set_random_state(classifier)
    # 使用训练集拟合分类器
    classifier.fit(X_train, y_train)
    # 预测测试集的标签
    y_pred = classifier.predict(X_test)

    # 将分类器的类别权重设置为 "balanced"
    classifier.set_params(class_weight="balanced")
    # 使用平衡后的类别权重重新拟合分类器
    classifier.fit(X_train, y_train)
    # 使用平衡后的分类器预测测试集的标签
    y_pred_balanced = classifier.predict(X_test)
    # 断言：验证在平衡后的预测结果（使用加权平均）的 F1 分数大于未平衡的预测结果（使用加权平均）的 F1 分数。
    assert f1_score(y_test, y_pred_balanced, average="weighted") > f1_score(
        y_test, y_pred, average="weighted"
    )
# 使用装饰器忽略未来警告类别
@ignore_warnings(category=FutureWarning)
# 检查带有非连续类标签的类权重的线性分类器
def check_class_weight_balanced_linear_classifier(name, Classifier):
    """Test class weights with non-contiguous class labels."""
    # 创建一个包含五个样本的二维数组
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    # 创建一个包含五个类标签的一维数组
    y = np.array([1, 1, 1, -1, -1])

    # 根据给定的分类器类创建一个分类器实例
    classifier = Classifier()

    # 如果分类器具有属性 "n_iter"
    if hasattr(classifier, "n_iter"):
        # 设置参数 "n_iter" 为 1000，这是一个非常小的数据集，增加迭代次数有助于收敛
        classifier.set_params(n_iter=1000)
    # 如果分类器具有属性 "max_iter"
    if hasattr(classifier, "max_iter"):
        # 设置参数 "max_iter" 为 1000
        classifier.set_params(max_iter=1000)
    # 如果分类器具有属性 "cv"
    if hasattr(classifier, "cv"):
        # 设置参数 "cv" 为 3
        classifier.set_params(cv=3)
    # 设置分类器的随机状态
    set_random_state(classifier)

    # 让模型计算类频率
    classifier.set_params(class_weight="balanced")
    # 使用平衡类权重拟合模型并复制其系数
    coef_balanced = classifier.fit(X, y).coef_.copy()

    # 计算每个标签的出现次数以手动重新加权
    n_samples = len(y)
    n_classes = float(len(np.unique(y)))

    # 创建一个类权重字典，用于手动加权
    class_weight = {
        1: n_samples / (np.sum(y == 1) * n_classes),
        -1: n_samples / (np.sum(y == -1) * n_classes),
    }
    # 使用手动定义的类权重重新设置分类器的类权重参数
    classifier.set_params(class_weight=class_weight)
    # 使用手动定义的类权重拟合模型并复制其系数
    coef_manual = classifier.fit(X, y).coef_.copy()

    # 断言平衡和手动加权后的系数是否相等，用于验证分类器是否正确计算了 class_weight=balanced
    assert_allclose(
        coef_balanced,
        coef_manual,
        err_msg="Classifier %s is not computing class_weight=balanced properly." % name,
    )


# 使用装饰器忽略未来警告类别
@ignore_warnings(category=FutureWarning)
# 检查估计器是否在参数上进行了覆写
def check_estimators_overwrite_params(name, estimator_orig):
    # 创建一个包含 21 个样本的数据集
    X, y = make_blobs(random_state=0, n_samples=21)
    # 强制估计器 X 标签的特定类型和属性
    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)
    # 克隆给定的估计器
    estimator = clone(estimator_orig)
    # 强制估计器 y 标签的特定类型和属性
    y = _enforce_estimator_tags_y(estimator, y)

    # 设置估计器的随机状态
    set_random_state(estimator)

    # 在拟合之前，创建原始估计器参数的物理副本
    params = estimator.get_params()
    original_params = deepcopy(params)

    # 拟合模型
    estimator.fit(X, y)

    # 比较模型参数的状态与原始参数的状态
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # 我们不应该默认改变或突变输入参数的内部状态。
        # 我们使用 joblib.hash 函数递归检查任何子对象以计算校验和。
        # 在此检查中，唯一不可变的构造参数的例外情况是可能的 RandomState 实例，
        # 但在此检查中，我们明确地固定了 random_state 参数以递归方式为整数种子。
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (name, param_name, original_value, new_value)
        )
# 检查在初始化期间的设置
def check_no_attributes_set_in_init(name, estimator_orig):
    try:
        # 克隆可能会失败，如果估算器在初始化期间没有将所有参数存储为属性
        estimator = clone(estimator_orig)
    except AttributeError:
        raise AttributeError(
            f"Estimator {name} should store all parameters as an attribute during init."
        )

    if hasattr(type(estimator).__init__, "deprecated_original"):
        return

    # 获取初始化函数的参数列表
    init_params = _get_args(type(estimator).__init__)
    # 获取所有父类的初始化参数列表
    parents_init_params = [
        param
        for params_parent in (_get_args(parent) for parent in type(estimator).__mro__)
        for param in params_parent
    ]

    # 测试在初始化期间除了参数以外没有设置任何属性
    invalid_attr = set(vars(estimator)) - set(init_params) - set(parents_init_params)
    # 忽略私有属性
    invalid_attr = set([attr for attr in invalid_attr if not attr.startswith("_")])
    assert not invalid_attr, (
        "Estimator %s should not set any attribute apart"
        " from parameters during init. Found attributes %s."
        % (name, sorted(invalid_attr))
    )


@ignore_warnings(category=FutureWarning)
def check_sparsify_coefficients(name, estimator_orig):
    X = np.array(
        [
            [-2, -1],
            [-1, -1],
            [-1, -2],
            [1, 1],
            [1, 2],
            [2, 1],
            [-1, -2],
            [2, 2],
            [-2, -2],
        ]
    )
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = _enforce_estimator_tags_y(estimator_orig, y)
    est = clone(estimator_orig)

    # 使用训练数据拟合估算器
    est.fit(X, y)
    pred_orig = est.predict(X)

    # 测试在稠密输入中使用稀疏化
    est.sparsify()
    assert sparse.issparse(est.coef_)
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)

    # 使用带稀疏 coef_ 的 pickle 进行序列化和反序列化
    est = pickle.loads(pickle.dumps(est))
    assert sparse.issparse(est.coef_)
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)


@ignore_warnings(category=FutureWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X = np.array(
        [
            [3, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [1, 2],
            [2, 1],
            [0, 3],
            [1, 0],
            [2, 0],
            [4, 4],
            [2, 3],
            [3, 2],
        ]
    )
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
    y = _enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        # 检查分类器数据不是数组的情况
        check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)


@ignore_warnings(category=FutureWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = _regression_dataset()
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    # 对于每个指定的对象类型，调用函数检查估算器的数据是否不是数组
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)
# 忽略未来警告类别的警告信息
@ignore_warnings(category=FutureWarning)
# 检查给定估算器的数据是否不是数组
def check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type):
    # 如果估算器名称在CROSS_DECOMPOSITION中，则跳过此测试
    if name in CROSS_DECOMPOSITION:
        raise SkipTest(
            "Skipping check_estimators_data_not_an_array "
            "for cross decomposition module as estimators "
            "are not deterministic."
        )
    
    # 复制原始估算器以控制随机种子
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    # 设置估算器的随机状态
    set_random_state(estimator_1)
    set_random_state(estimator_2)

    # 如果数据类型不是"NotAnArray"或"PandasDataframe"，则引发值错误
    if obj_type not in ["NotAnArray", "PandasDataframe"]:
        raise ValueError("Data type {0} not supported".format(obj_type))

    # 如果数据类型是"NotAnArray"
    if obj_type == "NotAnArray":
        # 将y和X转换为_NotAnArray类型的对象
        y_ = _NotAnArray(np.asarray(y))
        X_ = _NotAnArray(np.asarray(X))
    else:
        # 这里显式测试pandas对象（Series和DataFrame），因为某些估算器可能会特别处理它们（特别是它们的索引）。
        try:
            import pandas as pd

            # 将y转换为numpy数组
            y_ = np.asarray(y)
            # 如果y是一维的，则转换为Pandas Series对象；否则转换为Pandas DataFrame对象
            if y_.ndim == 1:
                y_ = pd.Series(y_, copy=False)
            else:
                y_ = pd.DataFrame(y_, copy=False)
            # 将X转换为Pandas DataFrame对象
            X_ = pd.DataFrame(np.asarray(X), copy=False)

        except ImportError:
            # 如果导入pandas失败，则跳过此测试
            raise SkipTest(
                "pandas is not installed: not checking estimators for pandas objects."
            )

    # 使用X_和y_拟合估算器1，并预测结果
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    # 使用原始X和y拟合估算器2，并预测结果
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)
    # 断言两个预测结果在一定的误差范围内相等
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


# 检查参数是否支持默认构造
def check_parameters_default_constructible(name, Estimator):
    # 测试默认构造的估算器
    # 消除过时警告

def _enforce_estimator_tags_y(estimator, y):
    # 具有'requires_positive_y'标签的估算器仅接受严格正数的数据
    if _safe_tags(estimator, key="requires_positive_y"):
        # 创建严格正数的y。大于0的最小增量为1，因为y可能是整数类型。
        y += 1 + abs(y.min())
    # 具有'binary_only'标签的估算器且y.size > 0时，将y转换为只有一个元素的数组
    if _safe_tags(estimator, key="binary_only") and y.size > 0:
        y = np.where(y == y.flat[0], y, y.flat[0] + 1)
    # 具有'multioutput_only'标签的估算器将一维y转换为二维数组
    if _safe_tags(estimator, key="multioutput_only"):
        return np.reshape(y, (-1, 1))
    return y


def _enforce_estimator_tags_X(estimator, X, kernel=linear_kernel):
    # 具有'1darray'标签的估算器仅接受形状为（n_samples，）的X数组
    if "1darray" in _safe_tags(estimator, key="X_types"):
        X = X[:, 0]
    # 具有'requires_positive_X'标签的估算器仅接受严格正数的数据
    if _safe_tags(estimator, key="requires_positive_X"):
        X = X - X.min()
    # 如果评估器允许使用"categorical"，则根据"allow_nan"的值选择数据类型为np.float64或np.int32，并对X进行四舍五入处理
    if "categorical" in _safe_tags(estimator, key="X_types"):
        dtype = np.float64 if _safe_tags(estimator, key="allow_nan") else np.int32
        X = np.round((X - X.min())).astype(dtype)

    # 如果评估器的类名为"SkewedChi2Sampler"，则要求X中的每个值都大于-skewdness
    if estimator.__class__.__name__ == "SkewedChi2Sampler":
        # 调整X使得每个值大于-skewdness
        X = X - X.min()

    # 如果评估器是基于对成对样本距离的度量
    if _is_pairwise_metric(estimator):
        # 使用欧氏距离计算X中每对样本之间的距离矩阵
        X = pairwise_distances(X, metric="euclidean")
    # 如果评估器支持成对计算
    elif _safe_tags(estimator, key="pairwise"):
        # 使用核函数计算X和自身的相似度
        X = kernel(X, X)
    # 返回处理后的X
    return X
@ignore_warnings(category=FutureWarning)
# 检查不是转换器的估算器是否具有参数max_iter，并返回n_iter_属性至少为1
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # 测试不是转换器且具有max_iter参数的估算器，确保返回n_iter_属性至少为1

    # 这些模型依赖于像libsvm这样的外部求解器，访问iter参数并非易事。
    # SelfTrainingClassifier如果所有样本都标记，则不执行迭代，因此n_iter_ = 0是有效的。
    not_run_check_n_iter = [
        "Ridge",
        "RidgeClassifier",
        "RandomizedLasso",
        "LogisticRegressionCV",
        "LinearSVC",
        "LogisticRegression",
        "SelfTrainingClassifier",
    ]

    # 在test_transformer_n_iter中已经测试过
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # 对于LassoLars，默认alpha = 1.0时会提前停止，例如在鸢尾花数据集上。
    if name == "LassoLars":
        estimator = clone(estimator_orig).set_params(alpha=0.0)
    else:
        estimator = clone(estimator_orig)

    # 如果估算器具有max_iter属性
    if hasattr(estimator, "max_iter"):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = _enforce_estimator_tags_y(estimator, y_)

        set_random_state(estimator, 0)

        X = _enforce_estimator_tags_X(estimator_orig, X)

        estimator.fit(X, y_)

        # 断言所有的n_iter_属性都至少为1
        assert np.all(estimator.n_iter_ >= 1)


@ignore_warnings(category=FutureWarning)
# 检查具有max_iter参数的转换器是否返回至少1的n_iter_属性
def check_transformer_n_iter(name, estimator_orig):
    estimator = clone(estimator_orig)
    # 如果估算器具有max_iter属性
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # 使用默认数据进行检查
            X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 5.0, 4.0]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(
                n_samples=30,
                centers=[[0, 0, 0], [1, 1, 1]],
                random_state=0,
                n_features=2,
                cluster_std=0.1,
            )
            X = _enforce_estimator_tags_X(estimator_orig, X)
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # 这些返回每个组件的n_iter。
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert iter_ >= 1
        else:
            assert estimator.n_iter_ >= 1


@ignore_warnings(category=FutureWarning)
# 检查get_params(deep=False)是否是get_params(deep=True)的子集
def check_get_params_invariance(name, estimator_orig):
    e = clone(estimator_orig)

    # 浅层参数和深层参数的获取
    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    # 断言浅层参数是深层参数的子集
    assert all(item in deep_params.items() for item in shallow_params.items())


@ignore_warnings(category=FutureWarning)
def check_set_params(name, estimator_orig):
    # TODO: This function is incomplete and requires further implementation.
    pass
    # 创建估算器的克隆副本
    estimator = clone(estimator_orig)

    # 获取原始估算器的参数（不深度复制）
    orig_params = estimator.get_params(deep=False)
    msg = "get_params 的结果在调用 set_params 后与传递的参数不匹配"

    # 使用原始参数设置估算器
    estimator.set_params(**orig_params)
    # 获取当前估算器的参数（不深度复制）
    curr_params = estimator.get_params(deep=False)
    assert set(orig_params.keys()) == set(curr_params.keys()), msg

    # 验证每个参数是否与原始参数相同
    for k, v in curr_params.items():
        assert orig_params[k] is v, msg

    # 一些测试模糊值
    test_values = [-np.inf, np.inf, None]

    # 深度复制原始参数，用于测试
    test_params = deepcopy(orig_params)
    for param_name in orig_params.keys():
        # 获取默认值
        default_value = orig_params[param_name]
        # 对于每个模糊值进行测试
        for value in test_values:
            test_params[param_name] = value
            try:
                # 尝试使用测试参数设置估算器
                estimator.set_params(**test_params)
            except (TypeError, ValueError) as e:
                e_type = e.__class__.__name__
                # 发生异常，可能是参数验证错误
                warnings.warn(
                    "{0} 在参数 {1} 的 set_params 中发生。建议延迟参数验证直到 fit。".format(e_type, param_name, name)
                )

                change_warning_msg = (
                    "在 set_params 抛出 {} 异常后，估算器的参数发生了变化".format(
                        e_type
                    )
                )
                # 获取异常前的参数状态
                params_before_exception = curr_params
                curr_params = estimator.get_params(deep=False)
                try:
                    assert set(params_before_exception.keys()) == set(
                        curr_params.keys()
                    )
                    for k, v in curr_params.items():
                        assert params_before_exception[k] is v
                except AssertionError:
                    warnings.warn(change_warning_msg)
            else:
                # 没有异常发生，获取当前估算器的参数
                curr_params = estimator.get_params(deep=False)
                assert set(test_params.keys()) == set(curr_params.keys()), msg
                for k, v in curr_params.items():
                    assert test_params[k] is v, msg

        # 恢复默认值
        test_params[param_name] = default_value
@ignore_warnings(category=FutureWarning)
# 忽略未来警告类别的警告

def check_classifiers_regression_target(name, estimator_orig):
    # 检查分类器是否在使用回归目标时抛出异常

    X, y = _regression_dataset()

    # 确保使用的分类器符合预期的标签要求
    X = _enforce_estimator_tags_X(estimator_orig, X)
    
    # 克隆原始估算器
    e = clone(estimator_orig)
    
    msg = "Unknown label type: "
    if not _safe_tags(e, key="no_validation"):
        # 使用断言验证是否抛出特定的 ValueError 异常
        with raises(ValueError, match=msg):
            e.fit(X, y)


@ignore_warnings(category=FutureWarning)
# 忽略未来警告类别的警告

def check_decision_proba_consistency(name, estimator_orig):
    # 检查具有 decision_function 和 predict_proba 方法的估算器的输出是否具有完美的等级相关性

    centers = [(2, 2), (4, 4)]
    X, y = make_blobs(
        n_samples=100,
        random_state=0,
        n_features=4,
        centers=centers,
        cluster_std=1.0,
        shuffle=True,
    )
    
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    
    # 克隆原始估算器
    estimator = clone(estimator_orig)

    if hasattr(estimator, "decision_function") and hasattr(estimator, "predict_proba"):
        # 使用训练集拟合估算器
        estimator.fit(X_train, y_train)
        
        # 因为 decision_function() 到 predict_proba() 的链接函数有时不够精确（通常是 expit 函数），
        # 我们四舍五入到第十位小数以避免数值问题：我们比较等级而不是平台特定的等级反转。
        a = estimator.predict_proba(X_test)[:, 1].round(decimals=10)
        b = estimator.decision_function(X_test).round(decimals=10)

        # 计算排名数据
        rank_proba, rank_score = rankdata(a), rankdata(b)
        try:
            # 使用数组几乎相等的函数来验证排名的一致性
            assert_array_almost_equal(rank_proba, rank_score)
        except AssertionError:
            # 有时，应用于概率的四舍五入会产生在分数中不存在的并列值，因为它在数值上更精确。
            # 在这种情况下，通过基于概率等级对决策函数分数进行分组来放宽测试，并检查分数是否单调递增。
            grouped_y_score = np.array(
                [b[rank_proba == group].mean() for group in np.unique(rank_proba)]
            )
            sorted_idx = np.argsort(grouped_y_score)
            assert_array_equal(sorted_idx, np.arange(len(sorted_idx)))


def check_outliers_fit_predict(name, estimator_orig):
    # 检查异常值检测器的 fit_predict 方法

    n_samples = 300
    X, _ = make_blobs(n_samples=n_samples, random_state=0)
    X = shuffle(X, random_state=7)
    n_samples, n_features = X.shape
    
    # 克隆原始估算器
    estimator = clone(estimator_orig)

    # 设置随机种子
    set_random_state(estimator)

    # 使用 fit_predict 方法进行预测
    y_pred = estimator.fit_predict(X)
    
    # 断言预测结果的形状和类型
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == "i"
    
    # 断言预测结果的唯一值为 [-1, 1]
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))
    # 检查当估算器（estimator）既有 predict 方法又有 fit_predict 方法时是否一致。
    # 这里假设估算器已经有 fit_predict 方法。
    if hasattr(estimator, "predict"):
        # 使用估算器进行拟合并预测，将预测结果存储在 y_pred_2 中
        y_pred_2 = estimator.fit(X).predict(X)
        # 断言 y_pred 和 y_pred_2 是否相等
        assert_array_equal(y_pred, y_pred_2)

    # 检查估算器是否有 contamination 属性
    if hasattr(estimator, "contamination"):
        # 预期的异常值数量
        expected_outliers = 30
        # 计算异常值比例，设置为 contamination 参数
        contamination = float(expected_outliers) / n_samples
        estimator.set_params(contamination=contamination)
        # 使用估算器进行拟合和预测，将结果存储在 y_pred 中
        y_pred = estimator.fit_predict(X)

        # 计算预测中的异常值数量
        num_outliers = np.sum(y_pred != 1)
        # 如果 num_outliers 不等于 expected_outliers，并且估算器有 decision_function 方法，
        # 则检查 decision_function 的值以确认异常值是否正确分离
        if num_outliers != expected_outliers and hasattr(
            estimator, "decision_function"
        ):
            decision = estimator.decision_function(X)
            # 检查异常值的正确性
            check_outlier_corruption(num_outliers, expected_outliers, decision)
# 检查对于要求正值 X 的标签是否正确引发警告
def check_fit_non_negative(name, estimator_orig):
    X = np.array([[-1.0, 1], [-1.0, 1]])
    y = np.array([1, 2])
    # 克隆原始的估算器对象
    estimator = clone(estimator_orig)
    # 使用 pytest 的 raises 方法确保 ValueError 被触发
    with raises(ValueError):
        estimator.fit(X, y)


# 检查 est.fit(X) 是否与 est.fit(X).fit(X) 相同
def check_fit_idempotent(name, estimator_orig):
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]
    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    # 设置估算器的随机状态
    set_random_state(estimator)
    # 如果估算器支持 warm_start 参数，则设置为 False
    if "warm_start" in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    n_samples = 100
    # 从正态分布中生成数据作为训练集 X
    X = rng.normal(loc=100, size=(n_samples, 2))
    # 根据估算器的标签强制处理 X
    X = _enforce_estimator_tags_X(estimator, X)
    # 如果是回归器，生成正态分布的 y，否则生成二元分类的 y
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    # 根据估算器的标签强制处理 y
    y = _enforce_estimator_tags_y(estimator, y)

    # 使用 ShuffleSplit 划分训练集和测试集
    train, test = next(ShuffleSplit(test_size=0.2, random_state=rng).split(X))
    # 使用安全的方式拆分数据集
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # 第一次拟合
    estimator.fit(X_train, y_train)

    # 收集预测方法的结果
    result = {
        method: getattr(estimator, method)(X_test)
        for method in check_methods
        if hasattr(estimator, method)
    }

    # 再次拟合
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    # 检查每个预测方法的结果是否与之前一致
    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            # 根据结果类型设置公差
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2 * np.finfo(new_result.dtype).eps
            else:
                tol = 2 * np.finfo(np.float64).eps
            # 使用 assert_allclose_dense_sparse 检查结果是否相似
            assert_allclose_dense_sparse(
                result[method],
                new_result,
                atol=max(tol, 1e-9),
                rtol=max(tol, 1e-7),
                err_msg="Idempotency check failed for method {}".format(method),
            )


# 确保在调用 fit 之前估算器未通过 check_is_fitted 检查，并且在拟合后通过检查
def check_fit_check_is_fitted(name, estimator_orig):
    rng = np.random.RandomState(42)

    estimator = clone(estimator_orig)
    # 设置估算器的随机状态
    set_random_state(estimator)
    # 如果估算器支持 warm_start 参数，则设置为 False
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_samples = 100
    # 从正态分布中生成数据作为训练集 X
    X = rng.normal(loc=100, size=(n_samples, 2))
    # 根据估算器的标签强制处理 X
    X = _enforce_estimator_tags_X(estimator, X)
    # 如果估算器是回归器，生成一个服从正态分布的随机数组作为目标变量 y
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        # 否则，生成一个由 0 和 1 组成的随机整数数组作为目标变量 y
        y = rng.randint(low=0, high=2, size=n_samples)
    
    # 根据估算器的要求对目标变量 y 进行处理
    y = _enforce_estimator_tags_y(estimator, y)

    # 如果估算器不是"stateless"（无状态）的，需要进行状态检查
    if not _safe_tags(estimator).get("stateless", False):
        # 尝试检查估算器是否已经被拟合
        try:
            check_is_fitted(estimator)
            # 如果检查通过，抛出断言错误，说明估算器在拟合之前已经被标记为已拟合
            raise AssertionError(
                f"{estimator.__class__.__name__} passes check_is_fitted before being"
                " fit!"
            )
        except NotFittedError:
            # 如果估算器没有被拟合，直接通过
            pass
    
    # 使用输入的特征 X 和目标变量 y 来拟合估算器
    estimator.fit(X, y)
    
    # 再次检查估算器是否已经被拟合
    try:
        check_is_fitted(estimator)
    except NotFittedError as e:
        # 如果估算器仍然没有被标记为已拟合，抛出一个新的 NotFittedError
        raise NotFittedError(
            "Estimator fails to pass `check_is_fitted` even though it has been fit."
        ) from e
# 确保在调用 fit 方法之前，不存在 n_features_in_ 属性，并且其值正确。
def check_n_features_in(name, estimator_orig):
    # 创建随机数生成器对象，用于生成随机数据
    rng = np.random.RandomState(0)

    # 克隆原始的评估器对象，确保不修改原始对象
    estimator = clone(estimator_orig)
    # 设置评估器对象的随机状态
    set_random_state(estimator)
    # 如果评估器支持 'warm_start' 参数，则设置为 False
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    # 设定生成数据的样本数
    n_samples = 100
    # 生成服从正态分布的随机样本作为特征矩阵 X
    X = rng.normal(loc=100, size=(n_samples, 2))
    # 将特征矩阵 X 根据评估器的要求进行适配处理
    X = _enforce_estimator_tags_X(estimator, X)
    # 如果评估器为回归器，则生成服从正态分布的随机目标值 y；否则生成二元随机目标值 y
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    # 将目标值 y 根据评估器的要求进行适配处理
    y = _enforce_estimator_tags_y(estimator, y)

    # 断言评估器对象不具有属性 "n_features_in_"
    assert not hasattr(estimator, "n_features_in_")
    # 调用 fit 方法，训练评估器
    estimator.fit(X, y)
    # 断言评估器对象具有属性 "n_features_in_"，并且其值等于特征矩阵 X 的列数
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == X.shape[1]


# 确保当 requires_y=True 时，当 y=None 时评估器能够优雅地失败
def check_requires_y_none(name, estimator_orig):
    # 创建随机数生成器对象，用于生成随机数据
    rng = np.random.RandomState(0)

    # 克隆原始的评估器对象，确保不修改原始对象
    estimator = clone(estimator_orig)
    # 设置评估器对象的随机状态
    set_random_state(estimator)

    # 设定生成数据的样本数
    n_samples = 100
    # 生成服从正态分布的随机样本作为特征矩阵 X
    X = rng.normal(loc=100, size=(n_samples, 2))
    # 将特征矩阵 X 根据评估器的要求进行适配处理
    X = _enforce_estimator_tags_X(estimator, X)

    # 预期的错误信息列表，用于检查是否捕获到预期的 ValueError 异常
    expected_err_msgs = (
        "requires y to be passed, but the target y is None",
        "Expected array-like (array or non-string sequence), got None",
        "y should be a 1d array",
    )

    try:
        # 调用 fit 方法，传入 y=None，期望捕获到 ValueError 异常
        estimator.fit(X, None)
    except ValueError as ve:
        # 如果异常消息不包含预期的任何一条信息，则重新抛出异常
        if not any(msg in str(ve) for msg in expected_err_msgs):
            raise ve


@ignore_warnings(category=FutureWarning)
# 确保在拟合之后检查 n_features_in_ 属性是否正确
def check_n_features_in_after_fitting(name, estimator_orig):
    # 获取评估器对象的安全标签
    tags = _safe_tags(estimator_orig)

    # 检查评估器是否支持的 X 类型是否在支持列表中，或者评估器是否标记为不进行验证
    is_supported_X_types = (
        "2darray" in tags["X_types"] or "categorical" in tags["X_types"]
    )

    # 如果评估器不支持的 X 类型或者标记为不进行验证，则直接返回
    if not is_supported_X_types or tags["no_validation"]:
        return

    # 创建随机数生成器对象，用于生成随机数据
    rng = np.random.RandomState(0)

    # 克隆原始的评估器对象，确保不修改原始对象
    estimator = clone(estimator_orig)
    # 设置评估器对象的随机状态
    set_random_state(estimator)
    # 如果评估器支持 'warm_start' 参数，则设置为 False
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    # 设定生成数据的样本数
    n_samples = 10
    # 生成服从正态分布的随机样本作为特征矩阵 X
    X = rng.normal(size=(n_samples, 4))
    # 将特征矩阵 X 根据评估器的要求进行适配处理
    X = _enforce_estimator_tags_X(estimator, X)

    # 如果评估器为回归器，则生成服从正态分布的随机目标值 y；否则生成二元随机目标值 y
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    # 将目标值 y 根据评估器的要求进行适配处理
    y = _enforce_estimator_tags_y(estimator, y)

    # 调用 fit 方法，训练评估器
    estimator.fit(X, y)
    # 断言评估器的 n_features_in_ 属性等于特征矩阵 X 的列数
    assert estimator.n_features_in_ == X.shape[1]

    # 检查评估器的其他方法是否正确检查了 n_features_in_ 属性
    check_methods = [
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
    ]
    # 创建一个错误的特征矩阵 X_bad，以验证方法对 n_features_in_ 的检查
    X_bad = X[:, [1]]

    # 设置错误消息模板，用于检查方法是否正确报告特征数不匹配的错误
    msg = f"X has 1 features, but \\w+ is expecting {X.shape[1]} features as input"
    # 遍历检查方法列表中的每一个方法名
    for method in check_methods:
        # 如果估计器对象没有该方法，跳过当前循环
        if not hasattr(estimator, method):
            continue
        
        # 获取方法的可调用对象
        callable_method = getattr(estimator, method)
        
        # 如果当前方法是"score"，则使用偏函数将其参数 y 固定为指定的 y 值
        if method == "score":
            callable_method = partial(callable_method, y=y)

        # 使用 pytest 的 raises 断言检查调用 callable_method(X_bad) 是否引发 ValueError，并匹配给定的错误消息 msg
        with raises(ValueError, match=msg):
            callable_method(X_bad)

    # 如果估计器对象没有 partial_fit 方法，直接返回
    if not hasattr(estimator, "partial_fit"):
        return
    
    # 克隆原始估计器对象
    estimator = clone(estimator_orig)
    
    # 如果估计器是分类器，使用 partial_fit 进行训练，并传递类别信息 classes=np.unique(y)
    if is_classifier(estimator):
        estimator.partial_fit(X, y, classes=np.unique(y))
    else:
        # 否则，仅使用 partial_fit 进行训练
        estimator.partial_fit(X, y)
    
    # 使用断言检查估计器的 n_features_in_ 属性是否等于输入数据 X 的特征数
    assert estimator.n_features_in_ == X.shape[1]

    # 使用 pytest 的 raises 断言检查调用 estimator.partial_fit(X_bad, y) 是否引发 ValueError，并匹配给定的错误消息 msg
    with raises(ValueError, match=msg):
        estimator.partial_fit(X_bad, y)
# 检查估计器的_get_tags_default_keys函数，确保实现了_get_tags函数，并且包含所有_DEFAULT_TAGS的键
def check_estimator_get_tags_default_keys(name, estimator_orig):
    # 克隆原始估计器对象
    estimator = clone(estimator_orig)
    # 如果估计器对象没有实现_get_tags函数，则直接返回
    if not hasattr(estimator, "_get_tags"):
        return
    
    # 获取实现的_get_tags函数返回的所有键，并转换为集合
    tags_keys = set(estimator._get_tags().keys())
    # 获取默认标签_DEFAULT_TAGS的所有键，并转换为集合
    default_tags_keys = set(_DEFAULT_TAGS.keys())
    # 断言实现的_get_tags函数返回的键集合与_DEFAULT_TAGS的键集合应该完全相同
    assert tags_keys.intersection(default_tags_keys) == default_tags_keys, (
        f"{name}._get_tags() is missing entries for the following default tags"
        f": {default_tags_keys - tags_keys.intersection(default_tags_keys)}"
    )


# 检查数据框列名的一致性，确保估计器能够处理数据框作为输入
def check_dataframe_column_names_consistency(name, estimator_orig):
    try:
        import pandas as pd
    except ImportError:
        # 如果导入pandas失败，则抛出跳过测试的异常
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )

    # 获取估计器的安全标签
    tags = _safe_tags(estimator_orig)
    # 判断估计器是否支持2darray或categorical类型的数据，或者是否禁用了验证
    is_supported_X_types = (
        "2darray" in tags["X_types"] or "categorical" in tags["X_types"]
    )

    # 如果不支持上述数据类型或禁用了验证，则直接返回
    if not is_supported_X_types or tags["no_validation"]:
        return

    # 创建一个随机数生成器
    rng = np.random.RandomState(0)

    # 克隆估计器对象并设置随机状态
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    # 创建一个原始数据矩阵，150行8列的正态分布随机数
    X_orig = rng.normal(size=(150, 8))

    # 根据估计器的标签要求调整X_orig
    X_orig = _enforce_estimator_tags_X(estimator, X_orig)
    n_samples, n_features = X_orig.shape

    # 生成列名为"col_i"的数据框X
    names = np.array([f"col_{i}" for i in range(n_features)])
    X = pd.DataFrame(X_orig, columns=names, copy=False)

    # 如果估计器是回归器，则生成相应形状的y数据，否则生成二元随机数y
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    # 根据估计器的标签要求调整y
    y = _enforce_estimator_tags_y(estimator, y)

    # 检查调用fit方法时是否会有关于特征名不合法的警告
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )
        estimator.fit(X, y)

    # 断言估计器对象在拟合后具有feature_names_in_属性
    if not hasattr(estimator, "feature_names_in_"):
        raise ValueError(
            "Estimator does not have a feature_names_in_ "
            "attribute after fitting with a dataframe"
        )
    assert isinstance(estimator.feature_names_in_, np.ndarray)
    assert estimator.feature_names_in_.dtype == object
    assert_array_equal(estimator.feature_names_in_, names)

    # 只检查sklearn估计器的feature_names_in_属性是否在文档字符串中
    module_name = estimator_orig.__module__
    if (
        module_name.startswith("sklearn.")
        and not ("test_" in module_name or module_name.endswith("_testing"))
        and ("feature_names_in_" not in (estimator_orig.__doc__))
    ):
        raise ValueError(
            f"Estimator {name} does not document its feature_names_in_ attribute"
        )

    # 需要检查的方法列表
    check_methods = []
    for method in (
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
        "score_samples",
        "predict_log_proba",
    ):
        # 对于模型估计器中的每个方法名称和对应的可调用方法，进行检查
        if not hasattr(estimator, method):
            continue  # 如果估计器对象不具有该方法，则跳过

        # 获取该方法的可调用对象
        callable_method = getattr(estimator, method)
        if method == "score":
            callable_method = partial(callable_method, y=y)  # 如果是 score 方法，则部分应用 y 参数
        check_methods.append((method, callable_method))  # 将方法名称和可调用方法组成元组加入检查列表

    for _, method in check_methods:
        # 捕获警告并过滤指定类型的警告
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="X does not have valid feature names",
                category=UserWarning,
                module="sklearn",
            )
            method(X)  # 在没有特征名无效警告的情况下执行方法，用于有效特征

    # 不合法的特征名列表
    invalid_names = [
        (names[::-1], "Feature names must be in the same order as they were in fit."),
        (
            [f"another_prefix_{i}" for i in range(n_features)],
            (
                "Feature names unseen at fit time:\n- another_prefix_0\n-"
                " another_prefix_1\n"
            ),
        ),
        (
            names[:3],
            f"Feature names seen at fit time, yet now missing:\n- {min(names[3:])}\n",
        ),
    ]
    # 获取估计器对象的参数，筛选包含 "early_stopping" 的参数
    params = {
        key: value
        for key, value in estimator.get_params().items()
        if "early_stopping" in key
    }
    # 检查是否启用了 early stopping
    early_stopping_enabled = any(value is True for value in params.values())

    # 对于每个无效的特征名和附加消息组合
    for invalid_name, additional_message in invalid_names:
        # 使用无效的特征名创建 DataFrame X_bad
        X_bad = pd.DataFrame(X, columns=invalid_name, copy=False)

        # 期望的错误消息，包含附加的消息
        expected_msg = re.escape(
            "The feature names should match those that were passed during fit.\n"
            f"{additional_message}"
        )
        # 对每个检查方法执行方法调用，并断言是否抛出了预期的 ValueError 异常
        for name, method in check_methods:
            with raises(
                ValueError, match=expected_msg, err_msg=f"{name} did not raise"
            ):
                method(X_bad)

        # 部分拟合在第二次调用时的检查
        # 如果启用了 early stopping 或者估计器没有 partial_fit 方法，则跳过
        if not hasattr(estimator, "partial_fit") or early_stopping_enabled:
            continue

        # 克隆原始估计器对象
        estimator = clone(estimator_orig)
        # 如果是分类器，获取类别信息并执行部分拟合
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)

        # 断言部分拟合是否抛出了预期的 ValueError 异常
        with raises(ValueError, match=expected_msg):
            estimator.partial_fit(X_bad, y)
def check_transformer_get_feature_names_out(name, transformer_orig):
    # 获取转换器的标签信息
    tags = transformer_orig._get_tags()
    # 如果输入类型不是二维数组或者标记为不需要验证，则返回
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    # 创建示例数据集 X 和 y
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    # 对数据集 X 进行标准化
    X = StandardScaler().fit_transform(X)

    # 克隆原始转换器
    transformer = clone(transformer_orig)
    # 应用标签强制函数 `_enforce_estimator_tags_X` 对转换器进行适配
    X = _enforce_estimator_tags_X(transformer, X)

    # 确定特征的数量
    n_features = X.shape[1]
    # 设置随机状态
    set_random_state(transformer)

    # 处理 y_，根据名称检查是否在 CROSS_DECOMPOSITION 中
    y_ = y
    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2

    # 使用转换器拟合和变换 X 和 y_
    X_transform = transformer.fit_transform(X, y=y_)

    # 创建输入特征的名称列表
    input_features = [f"feature{i}" for i in range(n_features)]

    # 当 input_features 的长度与 n_features_in_ 不一致时，引发 ValueError 异常
    with raises(ValueError, match="input_features should have length equal"):
        transformer.get_feature_names_out(input_features[::2])

    # 获取转换后的输出特征名称
    feature_names_out = transformer.get_feature_names_out(input_features)
    # 断言输出特征名称不为 None
    assert feature_names_out is not None
    # 断言输出特征名称的类型为 numpy 数组
    assert isinstance(feature_names_out, np.ndarray)
    # 断言输出特征名称的 dtype 是 object
    assert feature_names_out.dtype == object
    # 断言输出特征名称列表中的所有元素都是字符串类型
    assert all(isinstance(name, str) for name in feature_names_out)

    # 确定转换后的特征数量
    if isinstance(X_transform, tuple):
        n_features_out = X_transform[0].shape[1]
    else:
        n_features_out = X_transform.shape[1]

    # 断言输出特征名称列表的长度与实际输出特征数量相等
    assert (
        len(feature_names_out) == n_features_out
    ), f"Expected {n_features_out} feature names, got {len(feature_names_out)}"


def check_transformer_get_feature_names_out_pandas(name, transformer_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )

    # 获取转换器的标签信息
    tags = transformer_orig._get_tags()
    # 如果输入类型不是二维数组或者标记为不需要验证，则返回
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    # 创建示例数据集 X 和 y
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    # 对数据集 X 进行标准化
    X = StandardScaler().fit_transform(X)

    # 克隆原始转换器
    transformer = clone(transformer_orig)
    # 应用标签强制函数 `_enforce_estimator_tags_X` 对转换器进行适配
    X = _enforce_estimator_tags_X(transformer, X)

    # 确定特征的数量
    n_features = X.shape[1]
    # 设置随机状态
    set_random_state(transformer)

    # 处理 y_，根据名称检查是否在 CROSS_DECOMPOSITION 中
    y_ = y
    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2

    # 创建输入特征的名称列表
    feature_names_in = [f"col{i}" for i in range(n_features)]
    # 创建带有列名的 Pandas DataFrame
    df = pd.DataFrame(X, columns=feature_names_in, copy=False)
    # 使用转换器拟合和变换 DataFrame df 和 y_
    X_transform = transformer.fit_transform(df, y=y_)

    # 当 input_features 与 feature_names_in_ 不匹配时，引发 ValueError 异常
    invalid_feature_names = [f"bad{i}" for i in range(n_features)]
    with raises(ValueError, match="input_features is not equal to feature_names_in_"):
        transformer.get_feature_names_out(invalid_feature_names)

    # 获取默认的输出特征名称
    feature_names_out_default = transformer.get_feature_names_out()
    # 调用 transformer 对象的 get_feature_names_out 方法，传入 feature_names_in 参数，获取显式指定的输出特征名列表
    feature_names_in_explicit_names = transformer.get_feature_names_out(
        feature_names_in
    )
    
    # 使用 assert_array_equal 函数断言 feature_names_out_default 和 feature_names_in_explicit_names 相等
    assert_array_equal(feature_names_out_default, feature_names_in_explicit_names)
    
    # 如果 X_transform 是一个元组，则获取其第一个元素的列数作为 n_features_out
    if isinstance(X_transform, tuple):
        n_features_out = X_transform[0].shape[1]
    # 否则，直接获取 X_transform 的列数作为 n_features_out
    else:
        n_features_out = X_transform.shape[1]
    
    # 使用断言确保 feature_names_out_default 列表的长度与 n_features_out 相同
    assert (
        len(feature_names_out_default) == n_features_out
    ), f"Expected {n_features_out} feature names, got {len(feature_names_out_default)}"
# 检查参数验证的函数，确保当构造函数的参数值类型或值不合适时，会引发相关错误信息。
def check_param_validation(name, estimator_orig):
    # 使用种子值 0 创建一个随机数生成器对象
    rng = np.random.RandomState(0)
    # 创建一个形状为 (20, 5) 的均匀分布的随机数数组
    X = rng.uniform(size=(20, 5))
    # 创建一个形状为 (20,) 的随机整数数组，取值范围在 [0, 2) 之间
    y = rng.randint(0, 2, size=20)
    # 根据 estimator_orig 对象的类型要求对 y 进行标签强制执行
    y = _enforce_estimator_tags_y(estimator_orig, y)

    # 获取 estimator_orig 对象的参数名称集合
    estimator_params = estimator_orig.get_params(deep=False).keys()

    # 检查每个参数是否都有相应的约束条件
    if estimator_params:
        # 获取 estimator_orig 对象的参数约束条件名称集合
        validation_params = estimator_orig._parameter_constraints.keys()
        # 找出在参数约束条件中存在但在 estimator_params 中不存在的参数
        unexpected_params = set(validation_params) - set(estimator_params)
        # 找出在 estimator_params 中存在但在参数约束条件中不存在的参数
        missing_params = set(estimator_params) - set(validation_params)
        # 构造错误消息，指出参数约束条件与参数名称的不匹配情况
        err_msg = (
            f"Mismatch between _parameter_constraints and the parameters of {name}."
            f"\nConsider the unexpected parameters {unexpected_params} and expected but"
            f" missing parameters {missing_params}"
        )
        # 断言参数约束条件与参数名称应该一致，否则抛出错误消息 err_msg
        assert validation_params == estimator_params, err_msg

    # 创建一个确保所有参数都不是有效类型的对象
    param_with_bad_type = type("BadType", (), {})()

    # 定义适合方法的名称列表
    fit_methods = ["fit", "partial_fit", "fit_transform", "fit_predict"]


def check_set_output_transform(name, transformer_orig):
    # 检查默认配置下 transformer.set_output 不会改变变换的输出结果
    # 获取 transformer_orig 对象的标签
    tags = transformer_orig._get_tags()
    # 如果 "2darray" 不在 X_types 标签中，或者标签中有 no_validation，则直接返回
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    # 使用种子值 0 创建一个随机数生成器对象
    rng = np.random.RandomState(0)
    # 克隆 transformer_orig 对象
    transformer = clone(transformer_orig)

    # 创建一个形状为 (20, 5) 的均匀分布的随机数数组
    X = rng.uniform(size=(20, 5))
    # 根据 transformer_orig 对象的类型要求对 X 进行标签强制执行
    X = _enforce_estimator_tags_X(transformer_orig, X)
    # 创建一个形状为 (20,) 的随机整数数组，取值范围在 [0, 2) 之间
    y = rng.randint(0, 2, size=20)
    # 根据 transformer_orig 对象的类型要求对 y 进行标签强制执行
    y = _enforce_estimator_tags_y(transformer_orig, y)
    # 设置 transformer 对象的随机状态
    set_random_state(transformer)

    # 定义一个函数，先拟合再进行变换
    def fit_then_transform(est):
        # 如果 name 在 CROSS_DECOMPOSITION 中，则调用 fit(X, y).transform(X, y)
        if name in CROSS_DECOMPOSITION:
            return est.fit(X, y).transform(X, y)
        # 否则调用 fit(X, y).transform(X)
        return est.fit(X, y).transform(X)

    # 定义一个函数，执行拟合和变换操作
    def fit_transform(est):
        return est.fit_transform(X, y)

    # 定义一个字典，包含变换方法名称及其对应的方法
    transform_methods = {
        "transform": fit_then_transform,
        "fit_transform": fit_transform,
    }
    # 遍历 transform_methods 字典
    for name, transform_method in transform_methods.items():
        # 克隆 transformer 对象
        transformer = clone(transformer)
        # 如果 transformer 对象没有 name 对应的属性，则跳过该循环
        if not hasattr(transformer, name):
            continue
        # 执行变换方法，获取 X_trans_no_setting 变换后的结果
        X_trans_no_setting = transform_method(transformer)

        # 如果 name 在 CROSS_DECOMPOSITION 中，则只保留变换后结果的第一个数组
        if name in CROSS_DECOMPOSITION:
            X_trans_no_setting = X_trans_no_setting[0]

        # 设置 transformer 对象的输出为默认配置
        transformer.set_output(transform="default")
        # 再次执行变换方法，获取 X_trans_default 变换后的结果
        X_trans_default = transform_method(transformer)

        # 如果 name 在 CROSS_DECOMPOSITION 中，则只保留变换后结果的第一个数组
        if name in CROSS_DECOMPOSITION:
            X_trans_default = X_trans_default[0]

        # 断言默认配置和未设置输出的变换结果应该一致
        assert_allclose_dense_sparse(X_trans_no_setting, X_trans_default)


def _output_from_fit_transform(transformer, name, X, df, y):
    # This function is likely intended to handle the output from a fit_transform operation,
    # but its implementation details are not provided here.
    pass
    """生成用于测试 `set_output` 的输出，针对不同的配置：

    - 调用 `fit.transform` 或 `fit_transform` 中的任一方法；
    - 将 dataframe 或 numpy 数组传递给 fit 方法；
    - 将 dataframe 或 numpy 数组传递给 transform 方法。
    """
    # 初始化一个空字典用于存储输出结果
    outputs = {}

    # fit 然后 transform 的情况:
    # 定义不同的测试用例
    cases = [
        ("fit.transform/df/df", df, df),
        ("fit.transform/df/array", df, X),
        ("fit.transform/array/df", X, df),
        ("fit.transform/array/array", X, X),
    ]
    # 检查 transformer 对象是否同时具有 fit 和 transform 方法
    if all(hasattr(transformer, meth) for meth in ["fit", "transform"]):
        for case, data_fit, data_transform in cases:
            # 使用 data_fit 来拟合 transformer 对象
            transformer.fit(data_fit, y)
            # 根据是否在 CROSS_DECOMPOSITION 中，选择不同的 transform 方法
            if name in CROSS_DECOMPOSITION:
                X_trans, _ = transformer.transform(data_transform, y)
            else:
                X_trans = transformer.transform(data_transform)
            # 存储输出结果
            outputs[case] = (X_trans, transformer.get_feature_names_out())

    # fit_transform 的情况:
    cases = [
        ("fit_transform/df", df),
        ("fit_transform/array", X),
    ]
    # 检查 transformer 对象是否具有 fit_transform 方法
    if hasattr(transformer, "fit_transform"):
        for case, data in cases:
            # 根据是否在 CROSS_DECOMPOSITION 中，选择不同的 fit_transform 方法
            if name in CROSS_DECOMPOSITION:
                X_trans, _ = transformer.fit_transform(data, y)
            else:
                X_trans = transformer.fit_transform(data, y)
            # 存储输出结果
            outputs[case] = (X_trans, transformer.get_feature_names_out())

    # 返回所有的输出结果字典
    return outputs
# 检查由转换器生成的 DataFrame 是否有效。
# DataFrame 的实现通过此函数的参数指定。

def _check_generated_dataframe(
    name,
    case,
    index,
    outputs_default,
    outputs_dataframe_lib,
    is_supported_dataframe,
    create_dataframe,
    assert_frame_equal,
):
    """Check if the generated DataFrame by the transformer is valid.

    The DataFrame implementation is specified through the parameters of this function.

    Parameters
    ----------
    name : str
        The name of the transformer.
    case : str
        A single case from the cases generated by `_output_from_fit_transform`.
    index : index or None
        The index of the DataFrame. `None` if the library does not implement a DataFrame
        with an index.
    outputs_default : tuple
        A tuple containing the output data and feature names for the default output.
    outputs_dataframe_lib : tuple
        A tuple containing the output data and feature names for the pandas case.
    is_supported_dataframe : callable
        A callable that takes a DataFrame instance as input and return whether or
        E.g. `lambda X: isintance(X, pd.DataFrame)`.
    create_dataframe : callable
        A callable taking as parameters `data`, `columns`, and `index` and returns
        a callable. Be aware that `index` can be ignored. For example, polars dataframes
        would ignore the idnex.
    assert_frame_equal : callable
        A callable taking 2 dataframes to compare if they are equal.
    """
    
    X_trans, feature_names_default = outputs_default
    df_trans, feature_names_dataframe_lib = outputs_dataframe_lib

    # 断言生成的 DataFrame 是否被支持
    assert is_supported_dataframe(df_trans)

    # 我们总是依赖于用于生成 DataFrame 的转换器的 `get_feature_names_out` 输出作为列的真实情况。
    # 如果传入 transform 的是一个 DataFrame，则输出应具有相同的索引
    expected_index = index if case.endswith("df") else None

    # 创建期望的 DataFrame，以进行比较
    expected_dataframe = create_dataframe(
        X_trans, columns=feature_names_dataframe_lib, index=expected_index
    )

    try:
        # 断言生成的 DataFrame 与期望的 DataFrame 相等
        assert_frame_equal(df_trans, expected_dataframe)
    except AssertionError as e:
        # 如果不相等，抛出带有详细错误信息的 AssertionError
        raise AssertionError(
            f"{name} does not generate a valid dataframe in the {case} "
            "case. The generated dataframe is not equal to the expected "
            f"dataframe. The error message is: {e}"
        ) from e
    # 检查 transformer_orig 的标签，确定是否支持 2darray 类型输入或不需要验证
    tags = transformer_orig._get_tags()
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    # 设定随机数生成器，用于生成随机数据
    rng = np.random.RandomState(0)
    # 克隆 transformer_orig 以便进行后续的操作
    transformer = clone(transformer_orig)

    # 生成一个形状为 (20, 5) 的均匀分布的随机矩阵 X，并根据 transformer_orig 的要求进行类型强制转换
    X = rng.uniform(size=(20, 5))
    X = _enforce_estimator_tags_X(transformer_orig, X)
    
    # 生成一个长度为 20 的随机整数数组 y，并根据 transformer_orig 的要求进行类型强制转换
    y = rng.randint(0, 2, size=20)
    y = _enforce_estimator_tags_y(transformer_orig, y)
    
    # 设置 transformer 的随机状态
    set_random_state(transformer)

    # 生成特征列的名称列表，例如 ["col0", "col1", ..., "col4"]
    feature_names_in = [f"col{i}" for i in range(X.shape[1])]
    # 生成索引列表，例如 ["index0", "index1", ..., "index19"]
    index = [f"index{i}" for i in range(X.shape[0])]
    # 使用 create_dataframe 函数生成一个数据框 df，列名为 feature_names_in，索引为 index
    df = create_dataframe(X, columns=feature_names_in, index=index)

    # 克隆 transformer，并设置其输出为 "default"，生成默认输出
    transformer_default = clone(transformer).set_output(transform="default")
    # 使用 _output_from_fit_transform 函数获取默认输出
    outputs_default = _output_from_fit_transform(transformer_default, name, X, df, y)

    # 根据 context 的取值决定使用本地还是全局上下文
    if context == "local":
        # 如果使用本地上下文，则克隆 transformer 并设置其输出为 dataframe_lib
        transformer_df = clone(transformer).set_output(transform=dataframe_lib)
        # 使用 nullcontext() 作为上下文环境
        context_to_use = nullcontext()
    else:  # context == "global"
        # 如果使用全局上下文，则直接克隆 transformer
        transformer_df = clone(transformer)
        # 使用 config_context 设置 transform_output 为 dataframe_lib
        context_to_use = config_context(transform_output=dataframe_lib)

    try:
        # 使用 with 上下文 context_to_use 执行以下代码块
        with context_to_use:
            # 使用 _output_from_fit_transform 函数获取使用 transformer_df 的输出
            outputs_df = _output_from_fit_transform(transformer_df, name, X, df, y)
    except ValueError as e:
        # 如果出现 ValueError 异常，则检查错误消息中是否包含有关稀疏数据支持的信息
        capitalized_lib = dataframe_lib.capitalize()
        error_message = str(e)
        assert (
            f"{capitalized_lib} output does not support sparse data." in error_message
            or "The transformer outputs a scipy sparse matrix." in error_message
        ), e
        return

    # 对比默认输出和使用 dataframe_lib 输出的结果
    for case in outputs_default:
        _check_generated_dataframe(
            name,
            case,
            index,
            outputs_default[case],
            outputs_df[case],
            is_supported_dataframe,
            create_dataframe,
            assert_frame_equal,
        )
# 检查并设置输出转换函数，针对使用 Pandas 库的情境
def _check_set_output_transform_pandas_context(name, transformer_orig, context):
    try:
        import pandas as pd  # 尝试导入 Pandas 库
    except ImportError:  # 如果导入失败，说明 Pandas 库未安装，抛出跳过测试的异常
        raise SkipTest("pandas is not installed: not checking set output")

    # 调用通用的输出转换检查函数，指定使用 Pandas 库
    _check_set_output_transform_dataframe(
        name,
        transformer_orig,
        dataframe_lib="pandas",
        is_supported_dataframe=lambda X: isinstance(X, pd.DataFrame),  # 判断是否为 Pandas DataFrame
        create_dataframe=lambda X, columns, index: pd.DataFrame(  # 创建 Pandas DataFrame 的方法
            X, columns=columns, copy=False, index=index
        ),
        assert_frame_equal=pd.testing.assert_frame_equal,  # 使用 Pandas 的断言函数来比较 DataFrame
        context=context,  # 上下文环境参数
    )


# 检查并设置输出转换函数，使用 Pandas 库的特定情境
def check_set_output_transform_pandas(name, transformer_orig):
    _check_set_output_transform_pandas_context(name, transformer_orig, "local")  # 调用上述函数，指定本地上下文环境


# 检查并设置全局输出转换函数，使用 Pandas 库的特定情境
def check_global_output_transform_pandas(name, transformer_orig):
    _check_set_output_transform_pandas_context(name, transformer_orig, "global")  # 调用上述函数，指定全局上下文环境


# 检查并设置输出转换函数，针对使用 Polars 库的情境
def _check_set_output_transform_polars_context(name, transformer_orig, context):
    try:
        import polars as pl  # 尝试导入 Polars 库
        from polars.testing import assert_frame_equal  # 导入 Polars 的断言函数
    except ImportError:  # 如果导入失败，说明 Polars 库未安装，抛出跳过测试的异常
        raise SkipTest("polars is not installed: not checking set output")

    # 创建用于 Polars DataFrame 的特定方法
    def create_dataframe(X, columns, index):
        if isinstance(columns, np.ndarray):
            columns = columns.tolist()  # 如果列名是 NumPy 数组，转换为列表
        return pl.DataFrame(X, schema=columns, orient="row")  # 创建 Polars DataFrame 对象

    # 调用通用的输出转换检查函数，指定使用 Polars 库
    _check_set_output_transform_dataframe(
        name,
        transformer_orig,
        dataframe_lib="polars",
        is_supported_dataframe=lambda X: isinstance(X, pl.DataFrame),  # 判断是否为 Polars DataFrame
        create_dataframe=create_dataframe,  # 使用自定义的创建 DataFrame 方法
        assert_frame_equal=assert_frame_equal,  # 使用 Polars 的断言函数来比较 DataFrame
        context=context,  # 上下文环境参数
    )


# 检查并设置输出转换函数，使用 Polars 库的特定情境
def check_set_output_transform_polars(name, transformer_orig):
    _check_set_output_transform_polars_context(name, transformer_orig, "local")  # 调用上述函数，指定本地上下文环境


# 检查并设置全局输出转换函数，使用 Polars 库的特定情境
def check_global_set_output_transform_polars(name, transformer_orig):
    _check_set_output_transform_polars_context(name, transformer_orig, "global")  # 调用上述函数，指定全局上下文环境


# 忽略未来警告类别的装饰器函数，检查能够在只读输入数据上执行原地操作的估算器
@ignore_warnings(category=FutureWarning)
def check_inplace_ensure_writeable(name, estimator_orig):
    """检查估算器能够在只读输入数据上执行原地操作，即使用户没有显式请求复制数据。

    确保进行数据复制，并确保估算器不修改输入数组及其可写性。
    """
    rng = np.random.RandomState(0)  # 创建随机数生成器对象

    estimator = clone(estimator_orig)  # 克隆原始估算器对象
    set_random_state(estimator)  # 设置估算器对象的随机状态

    n_samples = 100  # 样本数量

    X, _ = make_blobs(n_samples=n_samples, n_features=3, random_state=rng)  # 生成示例数据集
    X = _enforce_estimator_tags_X(estimator, X)  # 根据估算器的标签要求处理数据

    # 对于特定的估算器，仅支持 Fortran 排序的输入进行原地操作
    if name in ("Lasso", "ElasticNet", "MultiTaskElasticNet", "MultiTaskLasso"):
        X = np.asfortranarray(X)  # 将数组转换为 Fortran 排序的数组

    # 添加一个缺失值，以便输入转换器必须执行某些操作
    # 检查 estimator 是否有 "missing_values" 属性，如果有，则将 X 的第一个元素设置为 NaN
    if hasattr(estimator, "missing_values"):
        X[0, 0] = np.nan

    # 检查 estimator 是否是回归器，如果是，则生成一个服从正态分布的随机数作为 y
    # 否则，生成一个在 [0, 2) 区间内的随机整数数组作为 y
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    
    # 根据 estimator 的类型和要求，对 y 进行必要的处理（例如标签约束）
    y = _enforce_estimator_tags_y(estimator, y)

    # 复制 X 的内容到 X_copy 中，以防止原始数据的修改
    X_copy = X.copy()

    # 将 X 设置为只读，防止在拟合过程中被修改
    X.setflags(write=False)

    # 使用 estimator 对 X 和 y 进行拟合
    estimator.fit(X, y)

    # 如果 estimator 具有 "transform" 方法，则对 X 进行变换
    if hasattr(estimator, "transform"):
        estimator.transform(X)

    # 断言 X 已经是不可写的状态
    assert not X.flags.writeable

    # 断言 X 和 X_copy 之间的数值近似相等
    assert_allclose(X, X_copy)
```