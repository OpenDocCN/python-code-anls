# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_docstring_parameters.py`

```
# 导入必要的库和模块
import importlib  # 用于动态导入模块
import inspect  # 提供对对象内部结构的反射支持
import os  # 提供对操作系统功能的访问
import warnings  # 控制警告的显示
from inspect import signature  # 获取函数的签名信息
from pkgutil import walk_packages  # 用于遍历包中的模块

import numpy as np  # 数值计算库
import pytest  # 测试框架

import sklearn  # 机器学习库
from sklearn.datasets import make_classification  # 生成分类数据集

# 使得调用 `all_estimators` 时能够发现实验性估计器
from sklearn.experimental import (
    enable_halving_search_cv,  # 启用半分搜索交叉验证
    enable_iterative_imputer,  # 启用迭代式填充器
)
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.preprocessing import FunctionTransformer  # 函数转换器
from sklearn.utils import all_estimators  # 获取所有估计器的工具函数
from sklearn.utils._testing import (
    _get_func_name,  # 获取函数名称
    check_docstring_parameters,  # 检查文档字符串中的参数
    ignore_warnings,  # 忽略警告的装饰器
)
from sklearn.utils.deprecation import _is_deprecated  # 判断是否已弃用的工具函数
from sklearn.utils.estimator_checks import (
    _construct_instance,  # 构造估计器实例的工具函数
    _enforce_estimator_tags_X,  # 强制估计器标签的工具函数（对 X）
    _enforce_estimator_tags_y,  # 强制估计器标签的工具函数（对 y）
)
from sklearn.utils.fixes import parse_version, sp_version  # 解析版本信息的工具函数

# walk_packages() 忽略 DeprecationWarnings，现在需要忽略 FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    # 获取 sklearn 包的路径
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    # 获取公共模块集合，排除以 "._" 开头或包含 ".tests." 的模块
    PUBLIC_MODULES = set(
        [
            pckg[1]
            for pckg in walk_packages(prefix="sklearn.", path=sklearn_path)
            if not ("._" in pckg[1] or ".tests." in pckg[1])
        ]
    )

# 需要忽略参数或文档字符串的函数列表
_DOCSTRING_IGNORES = [
    "sklearn.utils.deprecation.load_mlcomp",  # 加载 mlcomp 的函数
    "sklearn.pipeline.make_pipeline",  # 创建管道的函数
    "sklearn.pipeline.make_union",  # 创建联合的函数
    "sklearn.utils.extmath.safe_sparse_dot",  # 安全稀疏点积的函数
    "sklearn.utils._joblib",  # joblib 相关的工具函数
    "HalfBinomialLoss",  # 半二项损失函数（应该是类名或函数名）
]

# 如果默认情况下 y=None，则应该忽略 y 参数的方法列表
_METHODS_IGNORE_NONE_Y = [
    "fit",  # 拟合方法
    "score",  # 评分方法
    "fit_predict",  # 拟合并预测方法
    "fit_transform",  # 拟合并转换方法
    "partial_fit",  # 部分拟合方法
    "predict",  # 预测方法
]

# 忽略 numpydoc 0.8.0 版本下 collections.abc 在 Python 3.7 中的问题
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_docstring_parameters():
    # 测试模块文档字符串的格式

    # 如果没有找到 numpydoc，则跳过测试
    pytest.importorskip(
        "numpydoc", reason="numpydoc is required to test the docstrings"
    )

    # XXX unreached code as of v0.22
    from numpydoc import docscrape  # 导入文档解析工具

    incorrect = []  # 初始化错误列表
    # 遍历公共模块列表中的每个模块名
    for name in PUBLIC_MODULES:
        # 如果模块名以 ".conftest" 结尾，说明它是 pytest 工具相关的，不是 scikit-learn API 的一部分，跳过处理
        if name.endswith(".conftest"):
            # pytest 工具相关，不是 scikit-learn API 的一部分，跳过处理
            continue
        # 如果模块名是 "sklearn.utils.fixes"，这些文档字符串我们无法完全控制，跳过处理
        if name == "sklearn.utils.fixes":
            # 我们无法完全控制这些文档字符串，跳过处理
            continue
        # 使用警告捕获上下文，尝试导入模块
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        # 获取模块中的所有类
        classes = inspect.getmembers(module, inspect.isclass)
        # 过滤出属于 scikit-learn 的类
        classes = [cls for cls in classes if cls[1].__module__.startswith("sklearn")]
        # 遍历模块中的每个类及其类名
        for cname, cls in classes:
            # 初始化一个列表来收集当前类中存在的文档字符串错误
            this_incorrect = []
            # 如果类名在 _DOCSTRING_IGNORES 中，或者类名以下划线开头，则跳过处理
            if cname in _DOCSTRING_IGNORES or cname.startswith("_"):
                continue
            # 如果类是抽象类，则跳过处理
            if inspect.isabstract(cls):
                continue
            # 使用警告捕获上下文，尝试解析类的文档字符串
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            # 如果有警告产生，抛出运行时错误，显示类的 __init__ 方法在模块中的问题
            if len(w):
                raise RuntimeError(
                    "Error for __init__ of %s in %s:\n%s" % (cls, name, w[0])
                )

            # 如果类的 __new__ 方法被标记为过时，跳过处理
            if _is_deprecated(cls.__new__):
                continue

            # 检查类的 __init__ 方法的文档字符串参数
            this_incorrect += check_docstring_parameters(cls.__init__, cdoc)

            # 遍历类中的每个方法名
            for method_name in cdoc.methods:
                # 获取方法对象
                method = getattr(cls, method_name)
                # 如果方法被标记为过时，跳过处理
                if _is_deprecated(method):
                    continue
                param_ignore = None
                # 如果方法名在 _METHODS_IGNORE_NONE_Y 中，检查 y 参数
                if method_name in _METHODS_IGNORE_NONE_Y:
                    # 获取方法的签名信息
                    sig = signature(method)
                    # 如果方法的参数中有 "y"，并且它的默认值为 None，则忽略该参数（例如，对于 fit 和 score 方法）
                    if "y" in sig.parameters and sig.parameters["y"].default is None:
                        param_ignore = ["y"]  # 忽略 y 参数对于 fit 和 score 方法
                # 检查方法的文档字符串参数
                result = check_docstring_parameters(method, ignore=param_ignore)
                this_incorrect += result

            # 将当前类中的文档字符串错误添加到总体错误列表中
            incorrect += this_incorrect

        # 获取模块中的所有函数
        functions = inspect.getmembers(module, inspect.isfunction)
        # 过滤出模块内定义的函数，而非导入的函数
        functions = [fn for fn in functions if fn[1].__module__ == name]
        # 遍历模块中的每个函数及其函数名
        for fname, func in functions:
            # 不测试私有方法 / 函数
            if fname.startswith("_"):
                continue
            # 如果函数名是 "configuration" 且模块名以 "setup" 结尾，则跳过处理
            if fname == "configuration" and name.endswith("setup"):
                continue
            # 获取函数的名称
            name_ = _get_func_name(func)
            # 如果函数名中不包含 _DOCSTRING_IGNORES 中任何内容，并且函数未被标记为过时，则检查函数的文档字符串参数
            if not any(d in name_ for d in _DOCSTRING_IGNORES) and not _is_deprecated(
                func
            ):
                incorrect += check_docstring_parameters(func)

    # 将所有文档字符串错误消息连接为一个字符串
    msg = "\n".join(incorrect)
    # 如果存在文档字符串错误，则抛出断言错误，显示错误消息
    if len(incorrect) > 0:
        raise AssertionError("Docstring Error:\n" + msg)
# 定义一个函数，根据传入的 SearchCV 类型构造一个 LogisticRegression 实例和参数字典，返回构造的实例
def _construct_searchcv_instance(SearchCV):
    return SearchCV(LogisticRegression(), {"C": [0.1, 1]})


# 定义一个函数，根据传入的 Estimator 类型构造不同类型的实例
def _construct_compose_pipeline_instance(Estimator):
    # 如果 Estimator 是 ColumnTransformer 类型，则返回一个带有 passthrough 转换器的实例
    if Estimator.__name__ == "ColumnTransformer":
        return Estimator(transformers=[("transformer", "passthrough", [0, 1])])
    # 如果 Estimator 是 Pipeline 类型，则返回一个带有 LogisticRegression 分类器步骤的实例
    elif Estimator.__name__ == "Pipeline":
        return Estimator(steps=[("clf", LogisticRegression())])
    # 如果 Estimator 是 FeatureUnion 类型，则返回一个带有 FunctionTransformer 转换器的实例
    elif Estimator.__name__ == "FeatureUnion":
        return Estimator(transformer_list=[("transformer", FunctionTransformer())])


# 定义一个函数，根据传入的 Estimator 类型构造 SparseCoder 实例，并使用硬编码的字典作为参数
def _construct_sparse_coder(Estimator):
    # XXX: 假设特征数 n_features=3
    dictionary = np.array(
        [[0, 1, 0], [-1, -1, 2], [1, 1, 1], [0, 1, 1], [0, 2, 1]],
        dtype=np.float64,
    )
    return Estimator(dictionary=dictionary)


# 使用 ignore_warnings 装饰器忽略 sklearn 的收敛警告
@ignore_warnings(category=sklearn.exceptions.ConvergenceWarning)
# 根据 pytest 的参数化装饰器，在测试中使用所有的 Estimator 进行参数化
# TODO(1.6): 移除 "@pytest.mark.filterwarnings"，因为 SAMME.R 将被移除，并被 SAMME 算法替代
@pytest.mark.filterwarnings("ignore:The SAMME.R algorithm")
@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_fit_docstring_attributes(name, Estimator):
    # 导入 numpydoc 模块并跳过导入失败的情况
    pytest.importorskip("numpydoc")
    from numpydoc import docscrape

    # 使用 numpydoc 的 ClassDoc 类从 Estimator 中获取文档信息
    doc = docscrape.ClassDoc(Estimator)
    attributes = doc["Attributes"]

    # 根据 Estimator 类型构造对应的实例
    if Estimator.__name__ in (
        "HalvingRandomSearchCV",
        "RandomizedSearchCV",
        "HalvingGridSearchCV",
        "GridSearchCV",
    ):
        est = _construct_searchcv_instance(Estimator)
    elif Estimator.__name__ in (
        "ColumnTransformer",
        "Pipeline",
        "FeatureUnion",
    ):
        est = _construct_compose_pipeline_instance(Estimator)
    elif Estimator.__name__ == "SparseCoder":
        est = _construct_sparse_coder(Estimator)
    else:
        est = _construct_instance(Estimator)

    # 根据 Estimator 类型设置不同的参数
    if Estimator.__name__ == "SelectKBest":
        est.set_params(k=2)
    elif Estimator.__name__ == "DummyClassifier":
        est.set_params(strategy="stratified")
    elif Estimator.__name__ == "CCA" or Estimator.__name__.startswith("PLS"):
        # 对于 CCA 或以 PLS 开头的 Estimator，默认 n_components=2 对单一目标无效
        est.set_params(n_components=1)
    elif Estimator.__name__ in (
        "GaussianRandomProjection",
        "SparseRandomProjection",
    ):
        # 默认 n_components="auto" 可能会导致 X 的形状错误
        est.set_params(n_components=2)
    elif Estimator.__name__ == "TSNE":
        # 默认参数可能会引发错误，perplexity 必须小于样本数
        est.set_params(perplexity=2)

    # TODO(1.6): 移除（避免未来警告）
    if Estimator.__name__ in ("NMF", "MiniBatchNMF"):
        est.set_params(n_components="auto")

    # 对于 QuantileRegressor Estimator，根据 sp_version 设置 solver 参数
    if Estimator.__name__ == "QuantileRegressor":
        solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
        est.set_params(solver=solver)
    # 设置较低的最大迭代次数以加快测试速度：我们只关心已拟合属性的存在性。
    # 这应该与是否收敛无关。
    if "max_iter" in est.get_params():
        # 将估算器的最大迭代次数参数设置为2
        est.set_params(max_iter=2)
        # 如果估算器的类名为 "TSNE"，则将最大迭代次数设置为250
        if Estimator.__name__ == "TSNE":
            est.set_params(max_iter=250)

    # 如果估算器的参数中包含 "random_state"
    if "random_state" in est.get_params():
        # 将估算器的随机状态参数设置为0
        est.set_params(random_state=0)

    # 用于将来可能废弃的属性的字典
    skipped_attributes = {}

    # 如果估算器的类名以 "Vectorizer" 结尾
    if Estimator.__name__.endswith("Vectorizer"):
        # 对于某些特定输入数据要求的向量化器
        if Estimator.__name__ in (
            "CountVectorizer",
            "HashingVectorizer",
            "TfidfVectorizer",
        ):
            # 设置特定的输入数据 X
            X = [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ]
        elif Estimator.__name__ == "DictVectorizer":
            # 设置特定的输入数据 X
            X = [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}]
        y = None
    else:
        # 否则，使用 make_classification 生成数据 X 和标签 y
        X, y = make_classification(
            n_samples=20,
            n_features=3,
            n_redundant=0,
            n_classes=2,
            random_state=2,
        )

        # 对 y 进行标签强制处理
        y = _enforce_estimator_tags_y(est, y)
        # 对 X 进行特征强制处理
        X = _enforce_estimator_tags_X(est, X)

    # 根据估算器的 X_types 标签选择合适的数据拟合方法
    if "1dlabels" in est._get_tags()["X_types"]:
        est.fit(y)
    elif "2dlabels" in est._get_tags()["X_types"]:
        est.fit(np.c_[y, y])
    elif "3darray" in est._get_tags()["X_types"]:
        est.fit(X[np.newaxis, ...], y)
    else:
        est.fit(X, y)

    # 遍历所有属性进行检查
    for attr in attributes:
        # 如果属性名在跳过属性列表中，则跳过检查
        if attr.name in skipped_attributes:
            continue
        # 将属性描述转换为小写并连接为字符串
        desc = " ".join(attr.desc).lower()
        # 如果属性描述中包含 "only "，则跳过检查
        if "only " in desc:
            continue
        # 忽略将来可能废弃的警告
        with ignore_warnings(category=FutureWarning):
            # 确保估算器具有该属性
            assert hasattr(est, attr.name)

    # 获取所有已拟合的属性
    fit_attr = _get_all_fitted_attributes(est)
    # 获取所有属性的名称
    fit_attr_names = [attr.name for attr in attributes]
    # 找到未记录的属性
    undocumented_attrs = set(fit_attr).difference(fit_attr_names)
    # 从未记录的属性中去除跳过属性
    undocumented_attrs = set(undocumented_attrs).difference(skipped_attributes)
    # 如果存在未记录的属性，则引发断言错误
    if undocumented_attrs:
        raise AssertionError(
            f"Undocumented attributes for {Estimator.__name__}: {undocumented_attrs}"
        )
# 获取估算器（estimator）的所有已拟合属性，包括属性和方法
def _get_all_fitted_attributes(estimator):
    # 获取估算器实例的所有普通属性
    fit_attr = list(estimator.__dict__.keys())

    # 获取估算器实例的所有属性（包括属性和方法），包括会引发警告的情况
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)

        # 遍历估算器实例的类的所有属性和方法
        for name in dir(estimator.__class__):
            obj = getattr(estimator.__class__, name)
            # 如果属性不是一个属性（property），则继续下一个属性
            if not isinstance(obj, property):
                continue

            # 忽略可能引发 AttributeError 或已被弃用的属性
            try:
                # 尝试获取该属性的值
                getattr(estimator, name)
            except (AttributeError, FutureWarning):
                continue
            # 将符合条件的属性添加到已拟合属性列表中
            fit_attr.append(name)

    # 返回所有以 '_' 结尾且不以 '_' 开头的属性（通常这些是已拟合的属性）
    return [k for k in fit_attr if k.endswith("_") and not k.startswith("_")]
```