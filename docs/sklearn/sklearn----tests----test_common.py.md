# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_common.py`

```
"""
General tests for all estimators in sklearn.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os                                      # 导入操作系统功能模块
import pkgutil                                 # 导入包工具模块
import re                                      # 导入正则表达式模块
import sys                                     # 导入系统模块
import warnings                                # 导入警告处理模块
from functools import partial                  # 导入偏函数功能
from inspect import isgenerator, signature     # 导入检查器模块中的生成器和函数签名功能
from itertools import chain, product           # 导入迭代工具模块中的链和积功能
from pathlib import Path                       # 导入路径模块

import numpy as np                            # 导入NumPy科学计算库
import pytest                                  # 导入测试工具pytest
from scipy.linalg import LinAlgWarning         # 导入SciPy科学计算库线性代数警告类

import sklearn                                # 导入机器学习库scikit-learn
from sklearn.base import BaseEstimator         # 导入scikit-learn基础估计器类
from sklearn.cluster import (                  # 导入scikit-learn聚类模块中的多个聚类算法
    OPTICS,
    AffinityPropagation,
    Birch,
    MeanShift,
    SpectralClustering,
)
from sklearn.compose import ColumnTransformer  # 导入scikit-learn组合模块中的列转换器类
from sklearn.datasets import make_blobs        # 导入scikit-learn数据集模块中的生成聚类数据函数
from sklearn.decomposition import PCA          # 导入scikit-learn降维模块中的PCA算法类
from sklearn.exceptions import (               # 导入scikit-learn异常模块中的多个异常类
    ConvergenceWarning,
    FitFailedWarning
)

# make it possible to discover experimental estimators when calling `all_estimators`
from sklearn.experimental import (             # 导入scikit-learn实验性模块中的估计器启用函数
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.linear_model import (              # 导入scikit-learn线性模型模块中的多个线性模型类
    LogisticRegression,
    Ridge
)
from sklearn.linear_model._base import LinearClassifierMixin  # 导入scikit-learn线性模型基类的分类器混合类
from sklearn.manifold import (                  # 导入scikit-learn流形学习模块中的多个流形学习算法类
    TSNE,
    Isomap,
    LocallyLinearEmbedding
)
from sklearn.model_selection import (           # 导入scikit-learn模型选择模块中的多个交叉验证类
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV
)
from sklearn.neighbors import (                  # 导入scikit-learn邻居模块中的多个邻居算法类
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor
)
from sklearn.pipeline import Pipeline, make_pipeline  # 导入scikit-learn管道模块中的管道和制造管道函数
from sklearn.preprocessing import (                # 导入scikit-learn预处理模块中的多个数据预处理类
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler
)
from sklearn.semi_supervised import (              # 导入scikit-learn半监督学习模块中的多个半监督学习算法类
    LabelPropagation,
    LabelSpreading
)
from sklearn.utils import all_estimators           # 导入scikit-learn工具模块中的所有估计器函数
from sklearn.utils._tags import _DEFAULT_TAGS, _safe_tags  # 导入scikit-learn工具模块中的默认标签和安全标签
from sklearn.utils._testing import (                # 导入scikit-learn测试工具模块中的多个测试相关函数
    SkipTest,
    ignore_warnings,
    set_random_state
)
from sklearn.utils.estimator_checks import (        # 导入scikit-learn估计器检查模块中的多个估计器检查函数
    _construct_instance,
    _get_check_estimator_ids,
    _set_checking_parameters,
    check_class_weight_balanced_linear_classifier,
    check_dataframe_column_names_consistency,
    check_estimator,
    check_get_feature_names_out_error,
    check_global_output_transform_pandas,
    check_global_set_output_transform_polars,
    check_inplace_ensure_writeable,
    check_n_features_in_after_fitting,
    check_param_validation,
    check_set_output_transform,
    check_set_output_transform_pandas,
    check_set_output_transform_polars,
    check_transformer_get_feature_names_out,
    check_transformer_get_feature_names_out_pandas,
    parametrize_with_checks
)
from sklearn.utils.fixes import _IS_WASM            # 导入scikit-learn工具修复模块中的WASM修复变量


def test_all_estimator_no_base_class():
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = (
            "Base estimators such as {0} should not be included in all_estimators"
        ).format(name)
        assert not name.lower().startswith("base"), msg  # 断言基础估计器类名不应该以'base'开头
# 定义一个名为 _sample_func 的函数，接受参数 x 和可选参数 y，默认值为 1
def _sample_func(x, y=1):
    pass


# 定义一个名为 CallableEstimator 的类，继承自 BaseEstimator
class CallableEstimator(BaseEstimator):
    """Dummy development stub for an estimator.

    This is to make sure a callable estimator passes common tests.
    """

    # 定义类的调用方法，但不执行任何操作，使用 pragma: nocover 来避免测试覆盖此行
    def __call__(self):
        pass  # pragma: nocover


# 使用 pytest.mark.parametrize 装饰器为 test_get_check_estimator_ids 函数提供参数化测试数据
@pytest.mark.parametrize(
    "val, expected",
    [
        (partial(_sample_func, y=1), "_sample_func(y=1)"),
        (_sample_func, "_sample_func"),
        (partial(_sample_func, "world"), "_sample_func"),
        (LogisticRegression(C=2.0), "LogisticRegression(C=2.0)"),
        (
            LogisticRegression(
                random_state=1,
                solver="newton-cg",
                class_weight="balanced",
                warm_start=True,
            ),
            (
                "LogisticRegression(class_weight='balanced',random_state=1,"
                "solver='newton-cg',warm_start=True)"
            ),
        ),
        (CallableEstimator(), "CallableEstimator()"),
    ],
)
# 定义 test_get_check_estimator_ids 函数，验证 _get_check_estimator_ids 函数对输入的 val 是否返回了预期的 expected 值
def test_get_check_estimator_ids(val, expected):
    assert _get_check_estimator_ids(val) == expected


# 定义 _tested_estimators 函数，生成各种类型过滤条件下的所有估计器实例
def _tested_estimators(type_filter=None):
    for name, Estimator in all_estimators(type_filter=type_filter):
        try:
            estimator = _construct_instance(Estimator)
        except SkipTest:
            continue

        yield estimator


# 定义 _generate_pipeline 函数，生成包含 Ridge 和 LogisticRegression 作为最终估计器的 Pipeline 实例
def _generate_pipeline():
    for final_estimator in [Ridge(), LogisticRegression()]:
        yield Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("final_estimator", final_estimator),
            ]
        )


# 使用 @parametrize_with_checks 装饰器为 test_estimators 函数提供参数化测试数据
@parametrize_with_checks(list(chain(_tested_estimators(), _generate_pipeline())))
# 定义 test_estimators 函数，对给定的 estimator 执行通用的测试 check
def test_estimators(estimator, check, request):
    # Common tests for estimator instances
    with ignore_warnings(
        category=(FutureWarning, ConvergenceWarning, UserWarning, LinAlgWarning)
    ):
        _set_checking_parameters(estimator)
        check(estimator)


# 定义 test_check_estimator_generate_only 函数，验证 check_estimator 函数在仅生成模式下返回生成器对象
def test_check_estimator_generate_only():
    all_instance_gen_checks = check_estimator(LogisticRegression(), generate_only=True)
    assert isgenerator(all_instance_gen_checks)


# 定义 test_setup_py_check 函数，验证在 scikit-learn 源代码根目录下运行 `python setup.py check` 命令是否正常
def test_setup_py_check():
    pytest.importorskip("setuptools")
    # Smoke test `python setup.py check` command run at the root of the
    # scikit-learn source tree.
    cwd = os.getcwd()
    setup_path = Path(sklearn.__file__).parent.parent
    setup_filename = os.path.join(setup_path, "setup.py")
    if not os.path.exists(setup_filename):
        pytest.skip("setup.py not available")
    try:
        os.chdir(setup_path)
        old_argv = sys.argv
        sys.argv = ["setup.py", "check"]

        with warnings.catch_warnings():
            # The configuration spits out warnings when not finding
            # Blas/Atlas development headers
            warnings.simplefilter("ignore", UserWarning)
            with open("setup.py") as f:
                exec(f.read(), dict(__name__="__main__"))
    finally:
        sys.argv = old_argv
        os.chdir(cwd)
# 定义一个函数 `_tested_linear_classifiers`，用于获取所有分类器的迭代器
def _tested_linear_classifiers():
    # 获取所有的分类器，并过滤出类型为 "classifier" 的分类器
    classifiers = all_estimators(type_filter="classifier")

    # 忽略警告信息的上下文管理器
    with warnings.catch_warnings(record=True):
        # 遍历所有分类器的名称和类
        for name, clazz in classifiers:
            # 获取类的必需参数
            required_parameters = getattr(clazz, "_required_parameters", [])
            # 如果存在必需参数，则跳过当前分类器
            if len(required_parameters):
                # FIXME
                continue

            # 如果类定义了参数 "class_weight" 并且是 LinearClassifierMixin 的子类
            if "class_weight" in clazz().get_params().keys() and issubclass(
                clazz, LinearClassifierMixin
            ):
                # 生成器函数返回当前分类器的名称和类
                yield name, clazz


# 使用 parametrize 装饰器，为 `_tested_linear_classifiers` 函数的返回结果提供参数化测试
@pytest.mark.parametrize("name, Classifier", _tested_linear_classifiers())
def test_class_weight_balanced_linear_classifiers(name, Classifier):
    # 调用函数，检查带有平衡类权重的线性分类器的一致性
    check_class_weight_balanced_linear_classifier(name, Classifier)


# 使用 xfail 标记，针对特定条件（_IS_WASM 为真）的测试标记为预期失败
@pytest.mark.xfail(_IS_WASM, reason="importlib not supported for Pyodide packages")
# 忽略所有警告的装饰器
@ignore_warnings
def test_import_all_consistency():
    # 获取 sklearn 包的路径
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    # 遍历 sklearn 包路径下的所有包和模块
    pkgs = pkgutil.walk_packages(
        path=sklearn_path, prefix="sklearn.", onerror=lambda _: None
    )
    # 收集所有子模块的名称
    submods = [modname for _, modname, _ in pkgs]
    # 遍历每个子模块名称和 "sklearn" 自身
    for modname in submods + ["sklearn"]:
        # 如果模块名包含 ".tests." 则跳过
        if ".tests." in modname:
            continue
        # 如果模块名包含 "sklearn._build_utils" 则跳过
        if "sklearn._build_utils" in modname:
            continue
        # 动态导入模块
        package = __import__(modname, fromlist="dummy")
        # 遍历模块的 __all__ 属性
        for name in getattr(package, "__all__", ()):
            # 断言模块包含指定名称的属性
            assert hasattr(package, name), "Module '{0}' has no attribute '{1}'".format(
                modname, name
            )


# 测试函数，验证 sklearn 包的导入完整性
def test_root_import_all_completeness():
    # 获取 sklearn 包的路径
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    # 定义例外列表
    EXCEPTIONS = ("utils", "tests", "base", "setup", "conftest")
    # 遍历 sklearn 包路径下的所有包和模块
    for _, modname, _ in pkgutil.walk_packages(
        path=sklearn_path, onerror=lambda _: None
    ):
        # 如果模块名包含 "." 或者以 "_" 开头或者在例外列表中，则跳过
        if "." in modname or modname.startswith("_") or modname in EXCEPTIONS:
            continue
        # 断言模块名在 sklearn 包的 __all__ 属性中
        assert modname in sklearn.__all__


# 测试函数，验证所有测试是否可以导入
def test_all_tests_are_importable():
    # 定义正则表达式，用于匹配例外模块
    HAS_TESTS_EXCEPTIONS = re.compile(
        r"""(?x)
                                      \.externals(\.|$)|
                                      \.tests(\.|$)|
                                      \._
                                      """
    )
    # 定义资源模块集合
    resource_modules = {
        "sklearn.datasets.data",
        "sklearn.datasets.descr",
        "sklearn.datasets.images",
    }
    # 获取 sklearn 包的路径
    sklearn_path = [os.path.dirname(sklearn.__file__)]
    # 使用 walk_packages 遍历 sklearn 包路径下的所有包和模块，并生成模块名和是否为包的字典
    lookup = {
        name: ispkg
        for _, name, ispkg in pkgutil.walk_packages(sklearn_path, prefix="sklearn.")
    }
    # 查找缺失的测试模块列表，根据以下条件筛选：
    # - 模块名和其是否为包的布尔值组成的字典 lookup 中的每个条目
    # - 是包（ispkg 为 True）
    # - 模块名不在 resource_modules 列表中
    # - 模块名不匹配 HAS_TESTS_EXCEPTIONS 正则表达式
    # - 模块名加上 ".tests" 后不在 lookup 字典中
    missing_tests = [
        name
        for name, ispkg in lookup.items()
        if ispkg
        and name not in resource_modules
        and not HAS_TESTS_EXCEPTIONS.search(name)
        and name + ".tests" not in lookup
    ]
    
    # 使用断言确保 missing_tests 列表为空，否则触发 AssertionError
    assert missing_tests == [], (
        "{0} do not have `tests` subpackages. "
        "Perhaps they require "
        "__init__.py or an add_subpackage directive "
        "in the parent "
        "setup.py".format(missing_tests)
    )
# 确保向 check_estimator 或 parametrize_with_checks 传递类会引发错误

def test_class_support_removed():
    msg = "Passing a class was deprecated.* isn't supported anymore"
    
    # 使用 pytest 来检查是否会引发 TypeError 错误，并验证错误消息是否符合预期
    with pytest.raises(TypeError, match=msg):
        check_estimator(LogisticRegression)

    # 使用 pytest 来检查是否会引发 TypeError 错误，并验证错误消息是否符合预期
    with pytest.raises(TypeError, match=msg):
        parametrize_with_checks([LogisticRegression])


# 生成一个 ColumnTransformer 实例的生成器
def _generate_column_transformer_instances():
    yield ColumnTransformer(
        transformers=[
            # 创建一个名为 "trans1" 的转换器，使用 StandardScaler 对第 0 和第 1 列进行标准化
            ("trans1", StandardScaler(), [0, 1]),
        ]
    )


# 生成一个 SearchCV 实例的生成器，根据给定的参数组合
def _generate_search_cv_instances():
    # 使用 product 函数生成 SearchCV 实例和 Estimator 参数网格的所有组合
    for SearchCV, (Estimator, param_grid) in product(
        [
            GridSearchCV,
            HalvingGridSearchCV,
            RandomizedSearchCV,
            HalvingGridSearchCV,
        ],
        [
            (Ridge, {"alpha": [0.1, 1.0]}),
            (LogisticRegression, {"C": [0.1, 1.0]}),
        ],
    ):
        # 检查 SearchCV 类的初始化参数
        init_params = signature(SearchCV).parameters
        # 如果 init_params 中包含 "min_resources"，则设置额外参数为 {"min_resources": "smallest"}
        extra_params = (
            {"min_resources": "smallest"} if "min_resources" in init_params else {}
        )
        # 创建 SearchCV 实例，使用给定的 Estimator、param_grid 和其他额外参数
        search_cv = SearchCV(
            Estimator(), param_grid, cv=2, error_score="raise", **extra_params
        )
        # 设置搜索过程中的随机状态
        set_random_state(search_cv)
        # 生成 SearchCV 实例
        yield search_cv

    # 再次使用 product 函数生成 SearchCV 实例和 Estimator 参数网格的所有组合
    for SearchCV, (Estimator, param_grid) in product(
        [
            GridSearchCV,
            HalvingGridSearchCV,
            RandomizedSearchCV,
            HalvingRandomSearchCV,
        ],
        [
            (Ridge, {"ridge__alpha": [0.1, 1.0]}),
            (LogisticRegression, {"logisticregression__C": [0.1, 1.0]}),
        ],
    ):
        # 检查 SearchCV 类的初始化参数
        init_params = signature(SearchCV).parameters
        # 如果 init_params 中包含 "min_resources"，则设置额外参数为 {"min_resources": "smallest"}
        extra_params = (
            {"min_resources": "smallest"} if "min_resources" in init_params else {}
        )
        # 创建包含 PCA 转换器的管道，并使用给定的 Estimator、param_grid 和其他额外参数
        search_cv = SearchCV(
            make_pipeline(PCA(), Estimator()), param_grid, cv=2, **extra_params
        ).set_params(error_score="raise")
        # 设置搜索过程中的随机状态
        set_random_state(search_cv)
        # 生成 SearchCV 实例
        yield search_cv


# 使用 parametrize_with_checks 测试函数生成器，对 SearchCV 实例进行通用测试
@parametrize_with_checks(list(_generate_search_cv_instances()))
def test_search_cv(estimator, check, request):
    # Common tests for SearchCV instances
    # We have a separate test because those meta-estimators can accept a
    # wide range of base estimators (classifiers, regressors, pipelines)

    # 忽略指定的警告类型进行测试
    with ignore_warnings(
        category=(
            FutureWarning,
            ConvergenceWarning,
            UserWarning,
            FitFailedWarning,
        )
    ):
        # 调用 check 函数来测试 estimator
        check(estimator)


# 使用 pytest.mark.parametrize 来测试 _tested_estimators 函数返回的所有 estimator
@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_valid_tag_types(estimator):
    """Check that estimator tags are valid."""
    # 获取 estimator 的安全标签
    tags = _safe_tags(estimator)

    # 遍历标签字典，检查每个标签的类型是否正确
    for name, tag in tags.items():
        correct_tags = type(_DEFAULT_TAGS[name])
        if name == "_xfail_checks":
            # 对于 _xfail_checks，它可以是一个字典
            correct_tags = (correct_tags, dict)
        # 断言标签的类型是否与预期的类型相符
        assert isinstance(tag, correct_tags)
# 使用 pytest 的 mark.parametrize 装饰器，为函数 test_check_n_features_in_after_fitting 添加多个参数化的测试用例，
# 每个参数是一个机器学习估算器对象，由 _tested_estimators 函数返回的。
@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_check_n_features_in_after_fitting(estimator):
    # 设置检查参数，用于后续调用
    _set_checking_parameters(estimator)
    # 调用 check_n_features_in_after_fitting 函数，测试估算器在拟合后的特征数量检查
    check_n_features_in_after_fitting(estimator.__class__.__name__, estimator)


# 生成支持在拟合过程中预测的估算器的生成器函数
def _estimators_that_predict_in_fit():
    # 遍历所有通过 _tested_estimators 函数返回的估算器对象
    for estimator in _tested_estimators():
        # 获取估算器的参数集合
        est_params = set(estimator.get_params())
        # 如果估算器支持 "oob_score" 参数，则设置 oob_score=True 和 bootstrap=True
        if "oob_score" in est_params:
            yield estimator.set_params(oob_score=True, bootstrap=True)
        # 如果估算器支持 "early_stopping" 参数，则设置 early_stopping=True 和 n_iter_no_change=1
        elif "early_stopping" in est_params:
            est = estimator.set_params(early_stopping=True, n_iter_no_change=1)
            # 对于 MLPClassifier 和 MLPRegressor，设置特定标记以标记测试失败（xfail）
            if est.__class__.__name__ in {"MLPClassifier", "MLPRegressor"}:
                yield pytest.param(
                    est, marks=pytest.mark.xfail(msg="MLP still validates in fit")
                )
            else:
                yield est
        # 如果估算器支持 "n_iter_no_change" 参数，则设置 n_iter_no_change=1
        elif "n_iter_no_change" in est_params:
            yield estimator.set_params(n_iter_no_change=1)


# 在元估算器上运行 check_dataframe_column_names_consistency 测试时，
# 检查是否基础估算器在验证过程中检查列名的一致性
column_name_estimators = list(
    chain(
        _tested_estimators(),
        [make_pipeline(LogisticRegression(C=1))],
        list(_generate_search_cv_instances()),
        _estimators_that_predict_in_fit(),
    )
)


# 使用 pytest 的 mark.parametrize 装饰器，为函数 test_pandas_column_name_consistency 添加多个参数化的测试用例，
# 每个参数是一个元估算器对象，包括从 _tested_estimators 函数、make_pipeline 函数和 _generate_search_cv_instances 函数中生成的对象
@pytest.mark.parametrize(
    "estimator", column_name_estimators, ids=_get_check_estimator_ids
)
def test_pandas_column_name_consistency(estimator):
    # 设置检查参数，用于后续调用
    _set_checking_parameters(estimator)
    # 忽略 FutureWarning 类别的警告
    with ignore_warnings(category=(FutureWarning)):
        with warnings.catch_warnings(record=True) as record:
            # 调用 check_dataframe_column_names_consistency 函数，检查 Pandas DataFrame 列名的一致性
            check_dataframe_column_names_consistency(
                estimator.__class__.__name__, estimator
            )
        # 检查是否所有警告中都不包含 "was fitted without feature names" 的消息
        for warning in record:
            assert "was fitted without feature names" not in str(warning.message)


# 用于记录需要从测试中排除的模块列表，这些模块支持 get_feature_names_out 方法
GET_FEATURES_OUT_MODULES_TO_IGNORE = [
    "ensemble",
    "kernel_approximation",
]


# 检查是否应该包括在 get_feature_names_out 方法检查中的转换器函数
def _include_in_get_feature_names_out_check(transformer):
    # 如果转换器具有 get_feature_names_out 方法，则返回 True
    if hasattr(transformer, "get_feature_names_out"):
        return True
    # 否则，获取转换器的模块名，并检查是否在 GET_FEATURES_OUT_MODULES_TO_IGNORE 列表中
    module = transformer.__module__.split(".")[1]
    return module not in GET_FEATURES_OUT_MODULES_TO_IGNORE


# 从测试过程中生成支持 get_feature_names_out 方法的估算器列表
GET_FEATURES_OUT_ESTIMATORS = [
    est
    for est in _tested_estimators("transformer")
    if _include_in_get_feature_names_out_check(est)
]


# 使用 pytest 的 mark.parametrize 装饰器，为函数 test_transformers_get_feature_names_out 添加多个参数化的测试用例，
# 每个参数是一个支持 get_feature_names_out 方法的转换器对象
@pytest.mark.parametrize(
    "transformer", GET_FEATURES_OUT_ESTIMATORS, ids=_get_check_estimator_ids
)
def test_transformers_get_feature_names_out(transformer):
    # 设置检查参数，用于后续调用
    _set_checking_parameters(transformer)
    # 使用上下文管理器忽略未来警告类别的警告
    with ignore_warnings(category=(FutureWarning)):
        # 调用函数检查转换器的特征名称输出，传递转换器类名和转换器实例作为参数
        check_transformer_get_feature_names_out(
            transformer.__class__.__name__, transformer
        )
        # 调用函数检查转换器的 Pandas 数据框特征名称输出，传递转换器类名和转换器实例作为参数
        check_transformer_get_feature_names_out_pandas(
            transformer.__class__.__name__, transformer
        )
# 使用列表推导式筛选具有 "get_feature_names_out" 属性的测试过的估计器
ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT = [
    est for est in _tested_estimators() if hasattr(est, "get_feature_names_out")
]


# 参数化测试：测试具有 "get_feature_names_out" 属性的估计器的 get_feature_names_out 方法是否会引发错误
@pytest.mark.parametrize(
    "estimator", ESTIMATORS_WITH_GET_FEATURE_NAMES_OUT, ids=_get_check_estimator_ids
)
def test_estimators_get_feature_names_out_error(estimator):
    # 获取估计器的类名字符串
    estimator_name = estimator.__class__.__name__
    # 设置检查参数
    _set_checking_parameters(estimator)
    # 调用 check_get_feature_names_out_error 函数检查 get_feature_names_out 方法是否会引发错误
    check_get_feature_names_out_error(estimator_name, estimator)


# 参数化测试：测试所有 sklearn 提供的估计器是否在初始化或设置参数时不会引发错误
@pytest.mark.parametrize(
    "Estimator",
    [est for name, est in all_estimators()],
)
def test_estimators_do_not_raise_errors_in_init_or_set_params(Estimator):
    """Check that init or set_param does not raise errors."""
    # 获取估计器的参数签名
    params = signature(Estimator).parameters

    # 烟雾测试值，用于检测初始化或设置参数时是否引发错误
    smoke_test_values = [-1, 3.0, "helloworld", np.array([1.0, 4.0]), [1], {}, []]
    for value in smoke_test_values:
        # 创建一个新的参数字典，用烟雾测试值填充
        new_params = {key: value for key in params}

        # 初始化估计器，确保不会引发错误
        est = Estimator(**new_params)

        # 设置参数，确保不会引发错误
        est.set_params(**new_params)


# 参数化测试：测试所有被测试的估计器、生成的管道、列转换器实例以及搜索交叉验证实例的参数验证
@pytest.mark.parametrize(
    "estimator",
    chain(
        _tested_estimators(),
        _generate_pipeline(),
        _generate_column_transformer_instances(),
        _generate_search_cv_instances(),
    ),
    ids=_get_check_estimator_ids,
)
def test_check_param_validation(estimator):
    # 获取估计器的类名字符串
    name = estimator.__class__.__name__
    # 设置检查参数
    _set_checking_parameters(estimator)
    # 调用 check_param_validation 函数检查参数验证
    check_param_validation(name, estimator)


# 参数化测试：测试多个估计器是否支持 F 连续数组作为输入
@pytest.mark.parametrize(
    "Estimator",
    [
        AffinityPropagation,
        Birch,
        MeanShift,
        KNeighborsClassifier,
        KNeighborsRegressor,
        RadiusNeighborsClassifier,
        RadiusNeighborsRegressor,
        LabelPropagation,
        LabelSpreading,
        OPTICS,
        SpectralClustering,
        LocalOutlierFactor,
        LocallyLinearEmbedding,
        Isomap,
        TSNE,
    ],
)
def test_f_contiguous_array_estimator(Estimator):
    # 非回归测试，检查估计器是否支持 F 连续数组作为输入
    # 相关 GitHub 问题链接见函数注释
    X, _ = make_blobs(n_samples=80, n_features=4, random_state=0)
    X = np.asfortranarray(X)
    y = np.round(X[:, 0])

    # 初始化估计器
    est = Estimator()
    # 使用数据拟合估计器
    est.fit(X, y)

    # 如果估计器支持 transform 方法，则进行 transform 操作
    if hasattr(est, "transform"):
        est.transform(X)

    # 如果估计器支持 predict 方法，则进行 predict 操作
    if hasattr(est, "predict"):
        est.predict(X)


# 设置输出估计器列表，包含已测试的转换器、标准化器组合以及其它转换器
SET_OUTPUT_ESTIMATORS = list(
    chain(
        _tested_estimators("transformer"),
        [
            make_pipeline(StandardScaler(), MinMaxScaler()),
            OneHotEncoder(sparse_output=False),
            FunctionTransformer(feature_names_out="one-to-one"),
        ],
    )
)


# 参数化测试：测试设置输出转换的估计器
@pytest.mark.parametrize(
    "estimator", SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids
)
def test_set_output_transform(estimator):
    # 获取估计器的类名字符串
    name = estimator.__class__.__name__
    # 检查 estimator 是否具有 "set_output" 属性，如果没有则跳过测试
    if not hasattr(estimator, "set_output"):
        pytest.skip(
            f"Skipping check_set_output_transform for {name}: Does not support"
            " set_output API"
        )
    
    # 设置检查参数，用于后续的输出转换检查
    _set_checking_parameters(estimator)
    
    # 忽略未来警告，因为执行中可能会有一些未来的警告
    with ignore_warnings(category=(FutureWarning)):
        # 执行检查输出转换函数，传入估计器的类名和实例
        check_set_output_transform(estimator.__class__.__name__, estimator)
# 使用 pytest.mark.parametrize 装饰器来定义参数化测试函数，测试的参数为 SET_OUTPUT_ESTIMATORS 中的每个 estimator
# 和 _get_check_estimator_ids 函数返回的标识符
@pytest.mark.parametrize(
    "estimator", SET_OUTPUT_ESTIMATORS, ids=_get_check_estimator_ids
)
# 使用 pytest.mark.parametrize 装饰器来定义参数化测试函数，测试的参数为一组检查函数 check_func
# 包括 check_set_output_transform_pandas、check_global_output_transform_pandas、
# check_set_output_transform_polars、check_global_set_output_transform_polars
@pytest.mark.parametrize(
    "check_func",
    [
        check_set_output_transform_pandas,
        check_global_output_transform_pandas,
        check_set_output_transform_polars,
        check_global_set_output_transform_polars,
    ],
)
# 定义测试函数 test_set_output_transform_configured，接受 estimator 和 check_func 作为参数
def test_set_output_transform_configured(estimator, check_func):
    # 获取 estimator 的类名
    name = estimator.__class__.__name__
    # 如果 estimator 没有 set_output 属性，跳过当前测试用例
    if not hasattr(estimator, "set_output"):
        pytest.skip(
            f"Skipping {check_func.__name__} for {name}: Does not support"
            " set_output API yet"
        )
    # 设置检查参数
    _set_checking_parameters(estimator)
    # 忽略 FutureWarning 警告，执行 check_func
    with ignore_warnings(category=(FutureWarning)):
        check_func(estimator.__class__.__name__, estimator)


# 使用 pytest.mark.parametrize 装饰器来定义参数化测试函数，测试的参数为 _tested_estimators() 返回的每个 estimator
# 和 _get_check_estimator_ids 函数返回的标识符
@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
# 定义测试函数 test_check_inplace_ensure_writeable，接受 estimator 作为参数
def test_check_inplace_ensure_writeable(estimator):
    # 获取 estimator 的类名
    name = estimator.__class__.__name__

    # 如果 estimator 有 copy 属性，设置其 copy 参数为 False
    if hasattr(estimator, "copy"):
        estimator.set_params(copy=False)
    # 如果 estimator 有 copy_X 属性，设置其 copy_X 参数为 False
    elif hasattr(estimator, "copy_X"):
        estimator.set_params(copy_X=False)
    else:
        # 如果都没有对应属性，则抛出 SkipTest 异常
        raise SkipTest(f"{name} doesn't require writeable input.")

    # 设置检查参数
    _set_checking_parameters(estimator)

    # 根据 estimator 的类名进行特定设置
    if name == "HDBSCAN":
        estimator.set_params(metric="precomputed", algorithm="brute")

    if name == "PCA":
        estimator.set_params(svd_solver="full")

    if name == "KernelPCA":
        estimator.set_params(kernel="precomputed")

    # 执行检查函数 check_inplace_ensure_writeable
    check_inplace_ensure_writeable(name, estimator)
```