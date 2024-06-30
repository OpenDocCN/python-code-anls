# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\test_successive_halving.py`

```
# 导入 ceil 函数，用于向上取整
from math import ceil

# 导入所需的库
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵
import pytest  # 导入 Pytest 库，用于编写和运行测试
from scipy.stats import expon, norm, randint  # 导入 SciPy 库中的统计模块

# 导入 Scikit-Learn 中的相关模块和类
from sklearn.datasets import make_classification  # 用于生成分类数据集
from sklearn.dummy import DummyClassifier  # 用于创建虚拟分类器
from sklearn.experimental import enable_halving_search_cv  # 启用半数搜索交叉验证
from sklearn.model_selection import (
    GroupKFold,  # 分组 K 折交叉验证
    GroupShuffleSplit,  # 分组随机打乱划分
    HalvingGridSearchCV,  # 半数网格搜索交叉验证
    HalvingRandomSearchCV,  # 半数随机搜索交叉验证
    KFold,  # K 折交叉验证
    LeaveOneGroupOut,  # 留一分组交叉验证
    LeavePGroupsOut,  # 留 P 分组交叉验证
    ShuffleSplit,  # 随机打乱划分
    StratifiedKFold,  # 分层 K 折交叉验证
    StratifiedShuffleSplit,  # 分层随机打乱划分
)
from sklearn.model_selection._search_successive_halving import (
    _SubsampleMetaSplitter,  # 内部使用的子采样元分割器
    _top_k,  # 内部使用的 top-k 函数
)
from sklearn.model_selection.tests.test_search import (
    check_cv_results_array_types,  # 测试交叉验证结果数组类型
    check_cv_results_keys,  # 测试交叉验证结果键
)
from sklearn.svm import SVC, LinearSVC  # 导入 SVM 相关类

# 自定义虚拟分类器 FastClassifier，继承自 DummyClassifier
class FastClassifier(DummyClassifier):
    """Dummy classifier that accepts parameters a, b, ... z.

    These parameter don't affect the predictions and are useful for fast
    grid searching."""

    # 更新参数约束，接受从 a 到 z 的所有参数
    _parameter_constraints: dict = {
        **DummyClassifier._parameter_constraints,
        **{
            chr(key): "no_validation"  # 参数类型为 no_validation
            for key in range(ord("a"), ord("z") + 1)
        },
    }

    def __init__(
        self, strategy="stratified", random_state=None, constant=None, **kwargs
    ):
        super().__init__(
            strategy=strategy, random_state=random_state, constant=constant
        )

    def get_params(self, deep=False):
        # 获取参数，包括从 a 到 z 的所有参数
        params = super().get_params(deep=deep)
        for char in range(ord("a"), ord("z") + 1):
            params[chr(char)] = "whatever"  # 将参数设为 "whatever"
        return params


# 自定义虚拟分类器 SometimesFailClassifier，继承自 DummyClassifier
class SometimesFailClassifier(DummyClassifier):
    def __init__(
        self,
        strategy="stratified",
        random_state=None,
        constant=None,
        n_estimators=10,
        fail_fit=False,
        fail_predict=False,
        a=0,
    ):
        self.fail_fit = fail_fit  # 标记是否模拟拟合失败
        self.fail_predict = fail_predict  # 标记是否模拟预测失败
        self.n_estimators = n_estimators  # 设置估计器数量
        self.a = a  # 参数 a

        super().__init__(
            strategy=strategy, random_state=random_state, constant=constant
        )

    def fit(self, X, y):
        if self.fail_fit:
            raise Exception("fitting failed")  # 若模拟拟合失败，抛出异常
        return super().fit(X, y)  # 否则调用父类的拟合方法

    def predict(self, X):
        if self.fail_predict:
            raise Exception("predict failed")  # 若模拟预测失败，抛出异常
        return super().predict(X)  # 否则调用父类的预测方法


# 使用 Pytest 的参数化装饰器进行测试标记和参数化
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.FitFailedWarning")
@pytest.mark.filterwarnings("ignore:Scoring failed:UserWarning")
@pytest.mark.filterwarnings("ignore:One or more of the:UserWarning")
@pytest.mark.parametrize("HalvingSearch", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize("fail_at", ("fit", "predict"))
def test_nan_handling(HalvingSearch, fail_at):
    """Check the selection of the best scores in presence of failure represented by
    NaN values."""
    n_samples = 1_000  # 样本数量
    # 使用 make_classification 函数生成一个分类数据集 X 和对应的标签 y，用于后续的模型训练
    X, y = make_classification(n_samples=n_samples, random_state=0)
    
    # 使用 HalvingSearch 进行超参数搜索，基于 SometimesFailClassifier 模型
    # 搜索空间包括两个超参数：'fail_{fail_at}'（布尔值）和 'a'（整数范围 [0, 1, 2]）
    # 资源参数设置为 'n_estimators'，最大资源为 6，最小资源为 1，每次迭代资源减半
    search = HalvingSearch(
        SometimesFailClassifier(),
        {f"fail_{fail_at}": [False, True], "a": range(3)},
        resource="n_estimators",
        max_resources=6,
        min_resources=1,
        factor=2,
    )
    
    # 在生成的数据集 X, y 上进行搜索空间的拟合
    search.fit(X, y)
    
    # 确保在拟合/预测过程中失败的评估器排名始终低于成功拟合/预测的评估器
    assert not search.best_params_[f"fail_{fail_at}"]
    
    # 获取交叉验证结果中的平均测试分数
    scores = search.cv_results_["mean_test_score"]
    # 获取交叉验证结果中的排名
    ranks = search.cv_results_["rank_test_score"]
    
    # 确保有些分数为 NaN
    assert np.isnan(scores).any()
    
    # 获取所有 NaN 分数对应的唯一排名
    unique_nan_ranks = np.unique(ranks[np.isnan(scores)])
    # 确保所有 NaN 分数的排名相同
    assert unique_nan_ranks.shape[0] == 1
    # 确保 NaN 分数的排名最低
    assert (unique_nan_ranks[0] >= ranks).all()
@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    (
        "aggressive_elimination,"
        "max_resources,"
        "expected_n_iterations,"
        "expected_n_required_iterations,"
        "expected_n_possible_iterations,"
        "expected_n_remaining_candidates,"
        "expected_n_candidates,"
        "expected_n_resources,"
    ),
    [
        # 第一个参数化测试用例：启用激进淘汰，资源有限时的预期结果
        (True, "limited", 4, 4, 3, 1, [60, 20, 7, 3], [20, 20, 60, 180]),
        # 第二个参数化测试用例：禁用激进淘汰，资源有限时的预期结果
        (False, "limited", 3, 4, 3, 3, [60, 20, 7], [20, 60, 180]),
        # 第三个参数化测试用例：启用激进淘汰，资源无限时的预期结果
        (True, "unlimited", 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]),
        # 第四个参数化测试用例：禁用激进淘汰，资源无限时的预期结果
        (False, "unlimited", 4, 4, 4, 1, [60, 20, 7, 3], [37, 111, 333, 999]),
    ],
)
def test_aggressive_elimination(
    Est,
    aggressive_elimination,
    max_resources,
    expected_n_iterations,
    expected_n_required_iterations,
    expected_n_possible_iterations,
    expected_n_remaining_candidates,
    expected_n_candidates,
    expected_n_resources,
):
    # 测试 aggressive_elimination 参数的功能。

    # 创建一个虚拟数据集
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    # 定义参数网格
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    # 创建基础评估器
    base_estimator = FastClassifier()

    # 根据 max_resources 参数设置最大资源量
    if max_resources == "limited":
        max_resources = 180
    else:
        max_resources = n_samples

    # 初始化估计器对象
    sh = Est(
        base_estimator,
        param_grid,
        aggressive_elimination=aggressive_elimination,
        max_resources=max_resources,
        factor=3,
    )
    sh.set_params(verbose=True)  # 仅用于测试覆盖率

    # 对于 HalvingRandomSearchCV 类，设置候选数和最小资源
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    # 执行拟合操作
    sh.fit(X, y)

    # 断言各个属性的预期值
    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    assert sh.n_candidates_ == expected_n_candidates
    assert sh.n_remaining_candidates_ == expected_n_remaining_candidates
    assert ceil(sh.n_candidates_[-1] / sh.factor) == sh.n_remaining_candidates_
    [
        # 第一个元组：具有足够资源
        ("smallest", "auto", 2, 4, [20, 60]),
    
        # 第二个元组：具有足够资源，但手动设置了最小资源
        (50, "auto", 2, 3, [50, 150]),
    
        # 第三个元组：资源不足，只能执行一次迭代
        ("smallest", 30, 1, 1, [20]),
    
        # 第四个元组：使用 "exhaust" 模式，在最后一次迭代中尽可能使用所有资源
        ("exhaust", "auto", 2, 2, [333, 999]),
    
        # 后续的元组类似，依次展示不同的 "exhaust" 模式下的资源使用情况
        ("exhaust", 1000, 2, 2, [333, 999]),
        ("exhaust", 999, 2, 2, [333, 999]),
        ("exhaust", 600, 2, 2, [200, 600]),
        ("exhaust", 599, 2, 2, [199, 597]),
        ("exhaust", 300, 2, 2, [100, 300]),
        ("exhaust", 60, 2, 2, [20, 60]),
        ("exhaust", 50, 1, 1, [20]),
        ("exhaust", 20, 1, 1, [20]),
    ],
# 定义一个测试函数，用于测试最小资源和最大资源参数对每次迭代使用资源数量的影响
def test_min_max_resources(
    Est,
    min_resources,
    max_resources,
    expected_n_iterations,
    expected_n_possible_iterations,
    expected_n_resources,
):
    # 设置生成分类数据集的样本数
    n_samples = 1000
    # 生成分类数据集
    X, y = make_classification(n_samples=n_samples, random_state=0)
    # 参数网格定义
    param_grid = {"a": [1, 2], "b": [1, 2, 3]}
    # 创建一个快速分类器实例
    base_estimator = FastClassifier()

    # 使用给定的估计器类、参数网格、最小资源和最大资源创建半随机搜索对象
    sh = Est(
        base_estimator,
        param_grid,
        factor=3,
        min_resources=min_resources,
        max_resources=max_resources,
    )
    # 如果使用的估计器类是 HalvingRandomSearchCV，则设置候选参数组合数为6，与网格相同
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=6)

    # 对数据进行拟合
    sh.fit(X, y)

    # 预期所需迭代次数，根据给定的组合数和因子计算
    expected_n_required_iterations = 2  # given 6 combinations and factor = 3
    assert sh.n_iterations_ == expected_n_iterations
    assert sh.n_required_iterations_ == expected_n_required_iterations
    assert sh.n_possible_iterations_ == expected_n_possible_iterations
    assert sh.n_resources_ == expected_n_resources
    # 如果最小资源为 "exhaust"，则验证可能迭代次数、迭代次数与资源数量列表长度相等
    if min_resources == "exhaust":
        assert sh.n_possible_iterations_ == sh.n_iterations_ == len(sh.n_resources_)


# 使用参数化测试对两个估计器类（HalvingRandomSearchCV 和 HalvingGridSearchCV）进行测试
@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
@pytest.mark.parametrize(
    "max_resources, n_iterations, n_possible_iterations",
    [
        ("auto", 5, 9),  # all resources are used
        (1024, 5, 9),
        (700, 5, 8),
        (512, 5, 8),
        (511, 5, 7),
        (32, 4, 4),
        (31, 3, 3),
        (16, 3, 3),
        (4, 1, 1),  # max_resources == min_resources, only one iteration is possible
    ],
)
def test_n_iterations(Est, max_resources, n_iterations, n_possible_iterations):
    # 测试根据 max_resources 参数实际运行的迭代次数

    # 设置生成分类数据集的样本数
    n_samples = 1024
    # 生成分类数据集
    X, y = make_classification(n_samples=n_samples, random_state=1)
    # 参数网格定义
    param_grid = {"a": [1, 2], "b": list(range(10))}
    # 创建一个快速分类器实例
    base_estimator = FastClassifier()
    # 定义因子
    factor = 2

    # 使用给定的估计器类、参数网格、CV 折数、最大资源和最小资源创建半随机搜索对象
    sh = Est(
        base_estimator,
        param_grid,
        cv=2,
        factor=factor,
        max_resources=max_resources,
        min_resources=4,
    )
    # 如果使用的估计器类是 HalvingRandomSearchCV，则设置候选参数组合数为20
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=20)

    # 对数据进行拟合
    sh.fit(X, y)

    # 验证所需迭代次数与预期值相等
    assert sh.n_required_iterations_ == 5
    assert sh.n_iterations_ == n_iterations
    assert sh.n_possible_iterations_ == n_possible_iterations


# 使用参数化测试对两个估计器类（HalvingRandomSearchCV 和 HalvingGridSearchCV）进行测试
@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))
def test_resource_parameter(Est):
    # 测试资源参数的影响

    # 设置生成分类数据集的样本数
    n_samples = 1000
    # 生成分类数据集
    X, y = make_classification(n_samples=n_samples, random_state=0)
    # 参数网格定义
    param_grid = {"a": [1, 2], "b": list(range(10))}
    # 创建一个快速分类器实例
    base_estimator = FastClassifier()
    # 创建半随机搜索对象，使用给定的估计器类、参数网格、CV 折数、资源="c"、最大资源和因子
    sh = Est(base_estimator, param_grid, cv=2, resource="c", max_resources=10, factor=3)
    # 对数据进行拟合
    sh.fit(X, y)
    # 验证资源数量列表中的唯一性和预期值
    assert set(sh.n_resources_) == set([1, 3, 9])
    # 使用 zip 函数并行迭代 sh.cv_results_ 中的三个属性，分别是 n_resources, params, param_c
    # 这些属性来自于交叉验证结果的字典对象，用于验证它们是否相等
    for r_i, params, param_c in zip(
        sh.cv_results_["n_resources"],
        sh.cv_results_["params"],
        sh.cv_results_["param_c"],
    ):
        # 断言三个属性值必须相等，否则抛出 AssertionError
        assert r_i == params["c"] == param_c

    # 使用 pytest 模块中的 raises 函数捕获 ValueError 异常，并验证其错误消息
    with pytest.raises(
        ValueError, match="Cannot use resource=1234 which is not supported "
    ):
        # 创建 HalvingGridSearchCV 对象 sh，并设置 resource="1234"，max_resources=10
        sh = HalvingGridSearchCV(
            base_estimator, param_grid, cv=2, resource="1234", max_resources=10
        )
        # 对 sh 对象进行拟合操作，期望抛出 ValueError 异常
        sh.fit(X, y)

    # 使用 pytest 模块中的 raises 函数捕获 ValueError 异常，并验证其错误消息
    with pytest.raises(
        ValueError,
        match=(
            "Cannot use parameter c as the resource since it is part "
            "of the searched parameters."
        ),
    ):
        # 定义一个包含不支持的参数 c 的 param_grid 字典
        param_grid = {"a": [1, 2], "b": [1, 2], "c": [1, 3]}
        # 创建 HalvingGridSearchCV 对象 sh，并设置 resource="c"，max_resources=10
        sh = HalvingGridSearchCV(
            base_estimator, param_grid, cv=2, resource="c", max_resources=10
        )
        # 对 sh 对象进行拟合操作，期望抛出 ValueError 异常
        sh.fit(X, y)
@pytest.mark.parametrize(
    "max_resources, n_candidates, expected_n_candidates",
    [
        (512, "exhaust", 128),  # 设定最大资源、候选项数量及期望的候选项数量，使用 'exhaust' 生成所需数量的候选项
        (32, "exhaust", 8),    # 使用 'exhaust'，生成固定数量的候选项
        (32, 8, 8),             # 指定具体候选项数量为 8
        (32, 7, 7),             # 请求少于可能生成的候选项数量
        (32, 9, 9),             # 请求多于合理的候选项数量
    ],
)
def test_random_search(max_resources, n_candidates, expected_n_candidates):
    # 测试随机搜索，并验证生成的候选项数量是否符合预期

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {"a": norm, "b": norm}
    base_estimator = FastClassifier()
    # 创建 HalvingRandomSearchCV 对象
    sh = HalvingRandomSearchCV(
        base_estimator,
        param_grid,
        n_candidates=n_candidates,
        cv=2,
        max_resources=max_resources,
        factor=2,
        min_resources=4,
    )
    sh.fit(X, y)
    # 断言生成的候选项数量是否与预期相符
    assert sh.n_candidates_[0] == expected_n_candidates
    if n_candidates == "exhaust":
        # 确保 'exhaust' 会在最后一次迭代中尽可能使用更多的资源
        assert sh.n_resources_[-1] == max_resources


@pytest.mark.parametrize(
    "param_distributions, expected_n_candidates",
    [
        ({"a": [1, 2]}, 2),     # 所有参数都是列表，采样数量少于 n_candidates
        ({"a": randint(1, 3)}, 10),  # 不是所有参数都是列表，遵循 n_candidates
    ],
)
def test_random_search_discrete_distributions(
    param_distributions, expected_n_candidates
):
    # 确保随机搜索在请求超过可能的候选项数量时采样适当数量的候选项。参数的采样数量依赖于参数分布是否全为列表（参见 ParameterSampler 的详细说明）。
    # 这与 ParameterSampler 中的检查有些重复，但在 SH 的开发过程中发现了交互问题。

    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    base_estimator = FastClassifier()
    # 创建 HalvingRandomSearchCV 对象
    sh = HalvingRandomSearchCV(base_estimator, param_distributions, n_candidates=10)
    sh.fit(X, y)
    # 断言生成的候选项数量是否与预期相符
    assert sh.n_candidates_[0] == expected_n_candidates


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
@pytest.mark.parametrize(
    "params, expected_error_message",
    [
        (
            # 第一个字典：使用了不支持的参数 resource=not_a_parameter
            {"resource": "not_a_parameter"},
            "Cannot use resource=not_a_parameter which is not supported",
        ),
        (
            # 第二个字典：参数 a 不能作为资源使用，因为它是参数的一部分
            {"resource": "a", "max_resources": 100},
            "Cannot use parameter a as the resource since it is part of",
        ),
        (
            # 第三个字典：当 max_resources='auto' 时，resource 只能是 'n_samples'
            {"max_resources": "auto", "resource": "b"},
            "resource can only be 'n_samples' when max_resources='auto'",
        ),
        (
            # 第四个字典：min_resources_=15 大于 max_resources_=14
            {"min_resources": 15, "max_resources": 14},
            "min_resources_=15 is greater than max_resources_=14",
        ),
        (
            # 第五个字典：cv 参数必须生成一致的折叠
            {"cv": KFold(shuffle=True)},
            "must yield consistent folds",
        ),
        (
            # 第六个字典：cv 参数必须生成一致的折叠
            {"cv": ShuffleSplit()},
            "must yield consistent folds",
        ),
    ],
def test_input_errors(Est, params, expected_error_message):
    # 创建一个 FastClassifier 实例作为基础估计器
    base_estimator = FastClassifier()
    # 设置参数网格，这里只包含一个参数 'a' 的列表 [1]
    param_grid = {"a": [1]}
    # 生成一个包含100个样本的分类数据集 X, y
    X, y = make_classification(100)

    # 使用参数初始化 Est 对象（一般为某种学习器对象，如 HalvingRandomSearchCV）
    sh = Est(base_estimator, param_grid, **params)

    # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)


@pytest.mark.parametrize(
    "params, expected_error_message",
    [
        (
            {"n_candidates": "exhaust", "min_resources": "exhaust"},
            "cannot be both set to 'exhaust'",
        ),
    ],
)
def test_input_errors_randomized(params, expected_error_message):
    # 针对 HalvingRandomSearchCV 的特定测试

    # 创建一个 FastClassifier 实例作为基础估计器
    base_estimator = FastClassifier()
    # 设置参数网格，这里只包含一个参数 'a' 的列表 [1]
    param_grid = {"a": [1]}
    # 生成一个包含100个样本的分类数据集 X, y
    X, y = make_classification(100)

    # 使用参数初始化 HalvingRandomSearchCV 对象
    sh = HalvingRandomSearchCV(base_estimator, param_grid, **params)

    # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=expected_error_message):
        sh.fit(X, y)


@pytest.mark.parametrize(
    "fraction, subsample_test, expected_train_size, expected_test_size",
    [
        (0.5, True, 40, 10),
        (0.5, False, 40, 20),
        (0.2, True, 16, 4),
        (0.2, False, 16, 20),
    ],
)
def test_subsample_splitter_shapes(
    fraction, subsample_test, expected_train_size, expected_test_size
):
    # 确保由 SubsampleMetaSplitter 返回的拆分具有适当的大小

    n_samples = 100
    # 生成一个包含 n_samples 个样本的分类数据集 X, y
    X, y = make_classification(n_samples)
    # 初始化 _SubsampleMetaSplitter，使用 KFold(5) 作为基本交叉验证器
    cv = _SubsampleMetaSplitter(
        base_cv=KFold(5),
        fraction=fraction,
        subsample_test=subsample_test,
        random_state=None,
    )

    # 遍历所有的训练集和测试集拆分
    for train, test in cv.split(X, y):
        assert train.shape[0] == expected_train_size
        assert test.shape[0] == expected_test_size
        if subsample_test:
            assert train.shape[0] + test.shape[0] == int(n_samples * fraction)
        else:
            # 当不对测试集进行子采样时，确保测试集大小为 n_samples // cv.base_cv.get_n_splits()
            assert test.shape[0] == n_samples // cv.base_cv.get_n_splits()


@pytest.mark.parametrize("subsample_test", (True, False))
def test_subsample_splitter_determinism(subsample_test):
    # 确保 _SubsampleMetaSplitter 在调用 split() 时是一致的：
    # - 我们可以接受训练集不同（因为它们总是用不同的分数采样）
    # - 当我们不对测试集进行子采样时，希望它始终相同。这由 base_cv 的确定性来保证。

    # 注意：如果在 _SubsampleMetaSplitter.__init__ 中使用整数种子，可以强制使训练集和测试集始终相同。

    n_samples = 100
    # 生成一个包含 n_samples 个样本的分类数据集 X, y
    X, y = make_classification(n_samples)
    # 初始化 _SubsampleMetaSplitter，使用 KFold(5) 作为基本交叉验证器，fraction 设为 0.5
    cv = _SubsampleMetaSplitter(
        base_cv=KFold(5), fraction=0.5, subsample_test=subsample_test, random_state=None
    )

    # 检查多次调用 split() 的结果是否一致
    folds_a = list(cv.split(X, y, groups=None))
    folds_b = list(cv.split(X, y, groups=None))
    # 遍历两个迭代器 folds_a 和 folds_b，每次迭代得到一个元组 (train_a, test_a) 和 (train_b, test_b)
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        # 断言 train_a 和 train_b 不全相等
        assert not np.all(train_a == train_b)

        # 如果设置了 subsample_test，则断言 test_a 和 test_b 不全相等
        if subsample_test:
            assert not np.all(test_a == test_b)
        else:
            # 如果未设置 subsample_test，则断言 test_a 和 test_b 全相等
            assert np.all(test_a == test_b)
            # 断言 X 中索引为 test_a 和 test_b 的元素相等
            assert np.all(X[test_a] == X[test_b])
@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，用来多次运行同一个测试函数，参数化不同的输入组合
    "k, itr, expected",  # 参数化的参数：k, itr, expected
    [  # 参数化的输入值列表
        (1, 0, ["c"]),  # 第一组参数：k=1, itr=0, expected=["c"]
        (2, 0, ["a", "c"]),  # 第二组参数：k=2, itr=0, expected=["a", "c"]
        (4, 0, ["d", "b", "a", "c"]),  # 第三组参数：k=4, itr=0, expected=["d", "b", "a", "c"]
        (10, 0, ["d", "b", "a", "c"]),  # 第四组参数：k=10, itr=0, expected=["d", "b", "a", "c"]
        (1, 1, ["e"]),  # 第五组参数：k=1, itr=1, expected=["e"]
        (2, 1, ["f", "e"]),  # 第六组参数：k=2, itr=1, expected=["f", "e"]
        (10, 1, ["f", "e"]),  # 第七组参数：k=10, itr=1, expected=["f", "e"]
        (1, 2, ["i"]),  # 第八组参数：k=1, itr=2, expected=["i"]
        (10, 2, ["g", "h", "i"]),  # 第九组参数：k=10, itr=2, expected=["g", "h", "i"]
    ],
)
def test_top_k(k, itr, expected):
    results = {  # 模拟的测试结果字典
        "iter": [0, 0, 0, 0, 1, 1, 2, 2, 2],  # 迭代次数列表
        "mean_test_score": [4, 3, 5, 1, 11, 10, 5, 6, 9],  # 平均测试分数列表
        "params": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],  # 参数列表
    }
    got = _top_k(results, k=k, itr=itr)  # 调用被测试函数 _top_k，并获取结果
    assert np.all(got == expected)  # 断言结果符合预期


@pytest.mark.parametrize("Est", (HalvingRandomSearchCV, HalvingGridSearchCV))  # 参数化测试类 Est，可以是 HalvingRandomSearchCV 或 HalvingGridSearchCV
def test_cv_results(Est):
    # 测试 cv_results_ 是否正确匹配锦标赛逻辑，特别是每个连续迭代中继续的候选者是否是上一次迭代中的最佳候选者
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如未安装则跳过测试

    rng = np.random.RandomState(0)  # 使用随机种子初始化随机数生成器

    n_samples = 1000  # 样本数量
    X, y = make_classification(n_samples=n_samples, random_state=0)  # 生成分类数据集 X 和 y
    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}  # 参数网格定义
    base_estimator = FastClassifier()  # 使用 FastClassifier 作为基础估算器

    # 生成随机分数：确保没有平局，以免干扰排序并增加测试难度
    def scorer(est, X, y):
        return rng.rand()

    sh = Est(base_estimator, param_grid, factor=2, scoring=scorer)  # 初始化 Est 类实例 sh
    if Est is HalvingRandomSearchCV:
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")  # 设置参数为 HalvingRandomSearchCV 的特定值

    sh.fit(X, y)  # 对数据 X, y 进行拟合

    # 非回归检查 https://github.com/scikit-learn/scikit-learn/issues/19203
    assert isinstance(sh.cv_results_["iter"], np.ndarray)  # 断言 iter 列为 NumPy 数组类型
    assert isinstance(sh.cv_results_["n_resources"], np.ndarray)  # 断言 n_resources 列为 NumPy 数组类型

    cv_results_df = pd.DataFrame(sh.cv_results_)  # 将 cv_results_ 转换为 DataFrame 格式

    # 确保没有平局
    assert len(cv_results_df["mean_test_score"].unique()) == len(cv_results_df)  # 断言 mean_test_score 列的唯一值数量与 DataFrame 的行数相同

    cv_results_df["params_str"] = cv_results_df["params"].apply(str)  # 将 params 列转换为字符串格式
    table = cv_results_df.pivot(  # 使用 pivot 方法对 DataFrame 进行重塑
        index="params_str", columns="iter", values="mean_test_score"  # 设置索引、列和值
    )

    # 表格大致如下：
    # iter                    0      1       2        3   4   5
    # params_str
    # {'a': 'l2', 'b': 23} 0.75    NaN     NaN      NaN NaN NaN
    # {'a': 'l1', 'b': 30} 0.90  0.875     NaN      NaN NaN NaN
    # {'a': 'l1', 'b': 0}  0.75    NaN     NaN      NaN NaN NaN
    # {'a': 'l2', 'b': 3}  0.85  0.925  0.9125  0.90625 NaN NaN
    # {'a': 'l1', 'b': 5}  0.80    NaN     NaN      NaN NaN NaN
    # ...

    # NaN 表示在给定迭代中未评估该候选者，因为它在之前的某个迭代中不在前 K 名中
    # 确保确保在每个给定迭代中不在前 K 名中的候选者确实未在后续迭代中评估
    # 生成一个布尔数组，标记出table中的NaN值的位置
    nan_mask = pd.isna(table)
    # 获取搜索空间中的迭代次数
    n_iter = sh.n_iterations_
    # 对每一个迭代进行检查
    for it in range(n_iter - 1):
        # 获取当前迭代中已经被丢弃的候选者的掩码
        already_discarded_mask = nan_mask[it]

        # 确保如果一个候选者在当前迭代中已经被丢弃，那么在下一个迭代中它仍然保持被丢弃状态
        assert (
            already_discarded_mask & nan_mask[it + 1] == already_discarded_mask
        ).all()

        # 确保当前迭代中丢弃的候选者的数量是正确的
        discarded_now_mask = ~already_discarded_mask & nan_mask[it + 1]
        kept_mask = ~already_discarded_mask & ~discarded_now_mask
        assert kept_mask.sum() == sh.n_candidates_[it + 1]

        # 确保所有被丢弃的候选者的得分都低于保留的候选者的得分
        discarded_max_score = table[it].where(discarded_now_mask).max()
        kept_min_score = table[it].where(kept_mask).min()
        assert discarded_max_score < kept_min_score

    # 确保最佳候选者只从最后一个迭代中选择
    # 即使在之前的迭代中可能有更高的分数，也要确保最佳候选者只来自最后一轮迭代
    last_iter = cv_results_df["iter"].max()
    idx_best_last_iter = cv_results_df[cv_results_df["iter"] == last_iter][
        "mean_test_score"
    ].idxmax()
    idx_best_all_iters = cv_results_df["mean_test_score"].idxmax()

    # 确保GridSearchCV对象中记录的最佳参数与cv_results_df中最后一轮迭代的最佳参数相匹配
    assert sh.best_params_ == cv_results_df.iloc[idx_best_last_iter]["params"]
    # 确保最后一轮迭代的最佳平均测试分数比所有迭代中的最佳平均测试分数要低
    assert (
        cv_results_df.iloc[idx_best_last_iter]["mean_test_score"]
        < cv_results_df.iloc[idx_best_all_iters]["mean_test_score"]
    )
    # 确保最后一轮迭代的最佳参数与所有迭代中的最佳参数不同
    assert (
        cv_results_df.iloc[idx_best_last_iter]["params"]
        != cv_results_df.iloc[idx_best_all_iters]["params"]
    )
@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
# 定义测试函数，使用参数化装饰器，测试 HalvingGridSearchCV 和 HalvingRandomSearchCV 两个类
def test_base_estimator_inputs(Est):
    # 确保基础估计器在每次迭代时传递了正确的参数和样本数。
    pd = pytest.importorskip("pandas")
    # 导入 pandas 库，如果不存在则跳过测试

    passed_n_samples_fit = []
    # 用于记录每次 fit 方法传递的样本数

    passed_n_samples_predict = []
    # 用于记录每次 predict 方法传递的样本数

    passed_params = []
    # 用于记录每次 set_params 方法传递的参数

    class FastClassifierBookKeeping(FastClassifier):
        # 自定义的 FastClassifierBookKeeping 类，继承自 FastClassifier
        def fit(self, X, y):
            # 重写 fit 方法，记录传递的样本数并调用父类的 fit 方法
            passed_n_samples_fit.append(X.shape[0])
            return super().fit(X, y)

        def predict(self, X):
            # 重写 predict 方法，记录传递的样本数并调用父类的 predict 方法
            passed_n_samples_predict.append(X.shape[0])
            return super().predict(X)

        def set_params(self, **params):
            # 重写 set_params 方法，记录传递的参数并调用父类的 set_params 方法
            passed_params.append(params)
            return super().set_params(**params)

    n_samples = 1024
    # 样本数设置为 1024

    n_splits = 2
    # 分割数设置为 2

    X, y = make_classification(n_samples=n_samples, random_state=0)
    # 使用 make_classification 生成样本数据 X 和标签 y

    param_grid = {"a": ("l1", "l2"), "b": list(range(30))}
    # 参数网格设置，包括参数 'a' 可选值为 "l1" 和 "l2"，参数 'b' 可选值为 0 到 29 的整数

    base_estimator = FastClassifierBookKeeping()
    # 创建 FastClassifierBookKeeping 类的实例作为基础估计器

    sh = Est(
        base_estimator,
        param_grid,
        factor=2,
        cv=n_splits,
        return_train_score=False,
        refit=False,
    )
    # 创建 HalvingGridSearchCV 或 HalvingRandomSearchCV 的实例 sh，
    # 使用 base_estimator 作为基础估计器，param_grid 作为参数网格，
    # factor 设置为 2，cv 设置为 n_splits，return_train_score 设置为 False，refit 设置为 False

    if Est is HalvingRandomSearchCV:
        # 如果 Est 是 HalvingRandomSearchCV 类
        # 设置候选数为 2 * 30，并设置最小资源为 "exhaust"
        sh.set_params(n_candidates=2 * 30, min_resources="exhaust")

    sh.fit(X, y)
    # 使用 X, y 进行拟合

    assert len(passed_n_samples_fit) == len(passed_n_samples_predict)
    # 断言 fit 方法和 predict 方法调用次数相同

    passed_n_samples = [
        x + y for (x, y) in zip(passed_n_samples_fit, passed_n_samples_predict)
    ]
    # 计算每次调用的总样本数，列表推导式

    # 列表长度为 n_splits * n_iter * n_candidates_at_i。
    # 每个大小为 n_splits 的块对应于相同迭代的 n_splits 折叠，
    # 因此它们包含相同的值。我们对其进行子采样，使列表的长度为 n_iter * n_candidates_at_it
    passed_n_samples = passed_n_samples[::n_splits]
    # 使用步长 n_splits 进行子采样，以符合预期长度

    passed_params = passed_params[::n_splits]
    # 使用步长 n_splits 进行子采样，以符合预期长度

    cv_results_df = pd.DataFrame(sh.cv_results_)
    # 将交叉验证结果转换为 pandas DataFrame 格式

    assert len(passed_params) == len(passed_n_samples) == len(cv_results_df)
    # 断言传递参数、样本数和交叉验证结果的长度一致

    uniques, counts = np.unique(passed_n_samples, return_counts=True)
    # 统计传递样本数的唯一值和频数

    assert (sh.n_resources_ == uniques).all()
    # 断言 HalvingGridSearchCV 或 HalvingRandomSearchCV 的 n_resources_ 属性与唯一值相同

    assert (sh.n_candidates_ == counts).all()
    # 断言 HalvingGridSearchCV 或 HalvingRandomSearchCV 的 n_candidates_ 属性与频数相同

    assert (cv_results_df["params"] == passed_params).all()
    # 断言交叉验证结果中的参数列与传递参数列表相同

    assert (cv_results_df["n_resources"] == passed_n_samples).all()
    # 断言交叉验证结果中的 n_resources 列与传递样本数列表相同


@pytest.mark.parametrize("Est", (HalvingGridSearchCV, HalvingRandomSearchCV))
# 定义测试函数，使用参数化装饰器，测试 HalvingGridSearchCV 和 HalvingRandomSearchCV 两个类
def test_groups_support(Est):
    # 检查如果 groups 为 None，则 ValueError 是否传播到 HalvingGridSearchCV 和 HalvingRandomSearchCV
    # 并且检查 groups 是否正确传递给 cv 对象
    rng = np.random.RandomState(0)
    # 使用种子 0 初始化随机数生成器 rng

    X, y = make_classification(n_samples=50, n_classes=2, random_state=0)
    # 使用 make_classification 生成样本数据 X 和标签 y，共 50 个样本，2 类

    groups = rng.randint(0, 3, 50)
    # 生成长度为 50 的随机分组数据，取值范围在 0 到 2 之间

    clf = LinearSVC(random_state=0)
    # 创建 LinearSVC 的实例 clf，用于后续测试

    grid = {"C": [1]}
    # 网格搜索的参数设置，包含参数 'C' 可选值为 1

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(random_state=0),
    ]
    # 定义不同的分组交叉验证策略，用于测试
    # 错误消息，指示 'groups' 参数不应为 None
    error_msg = "The 'groups' parameter should not be None."
    
    # 对于每个交叉验证策略 cv 在 group_cvs 列表中
    for cv in group_cvs:
        # 使用给定的分类器 clf、参数网格 grid、交叉验证策略 cv 和随机种子 0，创建 GridSearchCV 对象 gs
        gs = Est(clf, grid, cv=cv, random_state=0)
        # 使用 pytest 断言检查是否会抛出 ValueError 异常，并匹配预定义的错误消息
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        # 使用给定的分组信息 groups，执行数据拟合
        gs.fit(X, y, groups=groups)
    
    # 对于每个非分组的交叉验证策略 cv 在 non_group_cvs 列表中
    for cv in non_group_cvs:
        # 使用给定的分类器 clf、参数网格 grid 和交叉验证策略 cv，创建 GridSearchCV 对象 gs
        gs = Est(clf, grid, cv=cv)
        # 应当不会引发错误，直接进行数据拟合
        gs.fit(X, y)
@pytest.mark.parametrize("SearchCV", [HalvingRandomSearchCV, HalvingGridSearchCV])
def test_min_resources_null(SearchCV):
    """检查如果最小资源设置为0，是否会引发错误。"""

    # 创建一个快速分类器的实例作为基础估计器
    base_estimator = FastClassifier()
    
    # 定义一个简单的参数网格
    param_grid = {"a": [1]}
    
    # 创建一个空的 numpy 数组作为输入特征矩阵 X
    X = np.empty(0).reshape(0, 3)

    # 使用指定的搜索策略类创建搜索对象
    search = SearchCV(base_estimator, param_grid, min_resources="smallest")

    # 准备一个错误消息，用于检查是否引发 ValueError 异常
    err_msg = "min_resources_=0: you might have passed an empty dataset X."
    
    # 使用 pytest 的断言检查是否引发了 ValueError 异常，并且错误消息匹配预期
    with pytest.raises(ValueError, match=err_msg):
        search.fit(X, [])


@pytest.mark.parametrize("SearchCV", [HalvingGridSearchCV, HalvingRandomSearchCV])
def test_select_best_index(SearchCV):
    """检查 Halving 搜索的最佳索引选择策略。"""

    # 创建一个虚拟的结果字典，用于模拟搜索结果
    results = {
        "iter": np.array([0, 0, 0, 0, 1, 1, 2, 2, 2]),
        "mean_test_score": np.array([4, 3, 5, 1, 11, 10, 5, 6, 9]),
        "params": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
    }

    # 调用搜索策略类的方法，获取最佳结果的索引
    best_index = SearchCV._select_best_index(None, None, results)
    
    # 使用断言验证最佳索引的预期值
    assert best_index == 8


def test_halving_random_search_list_of_dicts():
    """检查 HalvingRandomSearchCV 使用 param_distribution 为字典列表的行为。"""

    # 创建一个简单的分类数据集 X 和对应的标签 y
    X, y = make_classification(n_samples=150, n_features=4, random_state=42)

    # 定义一个参数分布的列表，包含不同的参数组合
    params = [
        {"kernel": ["rbf"], "C": expon(scale=10), "gamma": expon(scale=0.1)},
        {"kernel": ["poly"], "degree": [2, 3]},
    ]
    
    # 定义用于检查 CV 结果的键列表
    param_keys = (
        "param_C",
        "param_degree",
        "param_gamma",
        "param_kernel",
    )
    
    # 定义 CV 结果中分数相关的键列表
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    
    # 定义额外的 CV 结果中的键列表
    extra_keys = ("n_resources", "iter")

    # 使用 HalvingRandomSearchCV 创建一个搜索对象
    search = HalvingRandomSearchCV(
        SVC(), cv=3, param_distributions=params, return_train_score=True, random_state=0
    )
    
    # 对数据集进行拟合
    search.fit(X, y)
    
    # 计算候选参数的总数
    n_candidates = sum(search.n_candidates_)
    
    # 获取 CV 结果字典
    cv_results = search.cv_results_
    
    # 检查 CV 结果的结构是否符合预期
    check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates, extra_keys)
    
    # 定义预期的 CV 结果数组类型
    expected_cv_results_kinds = {
        "param_C": "f",
        "param_degree": "i",
        "param_gamma": "f",
        "param_kernel": "O",
    }
    
    # 检查 CV 结果数组的类型是否符合预期
    check_cv_results_array_types(
        search, param_keys, score_keys, expected_cv_results_kinds
    )
    
    # 使用断言验证特定条件下的参数组合是否符合预期
    assert all(
        (
            cv_results["param_C"].mask[i]
            and cv_results["param_gamma"].mask[i]
            and not cv_results["param_degree"].mask[i]
        )
        for i in range(n_candidates)
        if cv_results["param_kernel"][i] == "poly"
    )
    # 断言语句，用于验证条件是否为真，否则抛出 AssertionError
    assert all(
        (
            not cv_results["param_C"].mask[i]  # 检查 param_C 的掩码是否为 False
            and not cv_results["param_gamma"].mask[i]  # 检查 param_gamma 的掩码是否为 False
            and cv_results["param_degree"].mask[i]  # 检查 param_degree 的掩码是否为 True
        )
        # 针对每一个候选项的索引 i，确保以下条件成立
        for i in range(n_candidates)
        # 仅当 param_kernel 为 "rbf" 时才进行条件检查
        if cv_results["param_kernel"][i] == "rbf"
    )
```