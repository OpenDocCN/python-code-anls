# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\test_split.py`

```
"""Test the split module"""

import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from itertools import combinations, combinations_with_replacement, permutations  # 导入组合和排列相关函数

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架
from scipy import stats  # 导入SciPy统计模块
from scipy.sparse import issparse  # 导入SciPy稀疏矩阵判断函数
from scipy.special import comb  # 导入SciPy组合函数

from sklearn import config_context  # 导入scikit-learn配置上下文
from sklearn.datasets import load_digits, make_classification  # 导入数据集加载函数
from sklearn.dummy import DummyClassifier  # 导入虚拟分类器
from sklearn.model_selection import (  # 导入交叉验证和数据集划分函数
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    check_cv,
    cross_val_score,
    train_test_split,
)
from sklearn.model_selection._split import (  # 导入内部交叉验证划分函数
    _build_repr,
    _validate_shuffle_split,
    _yields_constant_splits,
)
from sklearn.svm import SVC  # 导入支持向量机分类器
from sklearn.tests.metadata_routing_common import assert_request_is_empty  # 导入测试用例函数
from sklearn.utils._array_api import (  # 导入数组API相关函数
    _convert_to_numpy,
    get_namespace,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._array_api import (  # 导入数组API设备支持
    device as array_api_device,
)
from sklearn.utils._mocking import MockDataFrame  # 导入模拟数据框类
from sklearn.utils._testing import (  # 导入测试工具函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.estimator_checks import (  # 导入估计器检查函数
    _array_api_for_tests,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS  # 导入SciPy容器修复
from sklearn.utils.validation import _num_samples  # 导入样本数验证函数

NO_GROUP_SPLITTERS = [  # 不需要组信息的交叉验证分离器列表
    KFold(),
    StratifiedKFold(),
    TimeSeriesSplit(),
    LeaveOneOut(),
    LeavePOut(p=2),
    ShuffleSplit(),
    StratifiedShuffleSplit(test_size=0.5),
    PredefinedSplit([1, 1, 2, 2]),
    RepeatedKFold(),
    RepeatedStratifiedKFold(),
]

GROUP_SPLITTERS = [  # 需要组信息的交叉验证分离器列表
    GroupKFold(),
    LeavePGroupsOut(n_groups=1),
    StratifiedGroupKFold(),
    LeaveOneGroupOut(),
    GroupShuffleSplit(),
]
GROUP_SPLITTER_NAMES = set(splitter.__class__.__name__ for splitter in GROUP_SPLITTERS)  # 提取组信息分离器的类名集合

ALL_SPLITTERS = NO_GROUP_SPLITTERS + GROUP_SPLITTERS  # 合并所有分离器列表，类型标记忽略

X = np.ones(10)  # 创建包含10个元素的全一数组X
y = np.arange(10) // 2  # 创建0到4的数组y
test_groups = (  # 测试用组数组
    np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),  # 一维组
    np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),  # 一维组
    np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]),  # 一维组
    np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),  # 一维组
    [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],  # 一维组
    ["1", "1", "1", "1", "2", "2", "2", "3", "3", "3", "3", "3"],  # 一维组
)
digits = load_digits()  # 加载手写数字数据集

pytestmark = pytest.mark.filterwarnings(  # 设置pytest标记过滤警告
    "error:The groups parameter:UserWarning:sklearn.*"
)


def _split(splitter, X, y, groups):  # 定义内部函数_split，用于选择合适的分离器进行数据分割
    if splitter.__class__.__name__ in GROUP_SPLITTER_NAMES:  # 如果分离器需要组信息
        return splitter.split(X, y, groups=groups)  # 调用分离器的split方法进行分割
    else:
        return splitter.split(X, y)  # 否则调用分离器的split方法进行分割


@ignore_warnings  # 忽略警告装饰器
def test_cross_validator_with_default_params():  # 定义测试函数test_cross_validator_with_default_params
    # 定义样本数
    n_samples = 4
    # 定义唯一分组数
    n_unique_groups = 4
    # 定义拆分数
    n_splits = 2
    # 定义参数 p
    p = 2
    # 定义洗牌拆分数，默认为 10
    n_shuffle_splits = 10  # (默认值)

    # 创建二维数组 X
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    # 创建一维数组 X_1d
    X_1d = np.array([1, 2, 3, 4])
    # 创建标签数组 y
    y = np.array([1, 1, 2, 2])
    # 创建分组数组 groups
    groups = np.array([1, 2, 3, 4])

    # 创建 LeaveOneOut 交叉验证器对象
    loo = LeaveOneOut()
    # 创建 LeavePOut 交叉验证器对象，参数为 p
    lpo = LeavePOut(p)
    # 创建 KFold 交叉验证器对象，拆分数为 n_splits
    kf = KFold(n_splits)
    # 创建 StratifiedKFold 交叉验证器对象，拆分数为 n_splits
    skf = StratifiedKFold(n_splits)
    # 创建 LeaveOneGroupOut 交叉验证器对象
    lolo = LeaveOneGroupOut()
    # 创建 LeavePGroupsOut 交叉验证器对象，分组数为 p
    lopo = LeavePGroupsOut(p)
    # 创建 ShuffleSplit 交叉验证器对象，洗牌拆分数为 n_shuffle_splits
    ss = ShuffleSplit(random_state=0)
    # 创建 PredefinedSplit 交叉验证器对象，使用预定义的测试折叠
    ps = PredefinedSplit([1, 1, 2, 2])  # n_splits = np of unique folds = 2
    # 创建 StratifiedGroupKFold 交叉验证器对象，拆分数为 n_splits
    sgkf = StratifiedGroupKFold(n_splits)

    # 各交叉验证器对象的字符串表示
    loo_repr = "LeaveOneOut()"
    lpo_repr = "LeavePOut(p=2)"
    kf_repr = "KFold(n_splits=2, random_state=None, shuffle=False)"
    skf_repr = "StratifiedKFold(n_splits=2, random_state=None, shuffle=False)"
    lolo_repr = "LeaveOneGroupOut()"
    lopo_repr = "LeavePGroupsOut(n_groups=2)"
    ss_repr = (
        "ShuffleSplit(n_splits=10, random_state=0, test_size=None, train_size=None)"
    )
    ps_repr = "PredefinedSplit(test_fold=array([1, 1, 2, 2]))"
    sgkf_repr = "StratifiedGroupKFold(n_splits=2, random_state=None, shuffle=False)"

    # 预期的拆分数列表，对应每个交叉验证器对象
    n_splits_expected = [
        n_samples,
        comb(n_samples, p),
        n_splits,
        n_splits,
        n_unique_groups,
        comb(n_unique_groups, p),
        n_shuffle_splits,
        2,
        n_splits,
    ]

    # 遍历每个交叉验证器对象和其字符串表示
    for i, (cv, cv_repr) in enumerate(
        zip(
            [loo, lpo, kf, skf, lolo, lopo, ss, ps, sgkf],
            [
                loo_repr,
                lpo_repr,
                kf_repr,
                skf_repr,
                lolo_repr,
                lopo_repr,
                ss_repr,
                ps_repr,
                sgkf_repr,
            ],
        )
    ):
        # 测试 get_n_splits 方法是否正确
        assert n_splits_expected[i] == cv.get_n_splits(X, y, groups)

        # 测试即使数据是一维的情况下，交叉验证器是否按预期工作
        np.testing.assert_equal(
            list(cv.split(X, y, groups)), list(cv.split(X_1d, y, groups))
        )
        
        # 测试返回的训练、测试索引是否为整数
        for train, test in cv.split(X, y, groups):
            assert np.asarray(train).dtype.kind == "i"
            assert np.asarray(test).dtype.kind == "i"

        # 测试交叉验证器的字符串表示是否正常工作
        assert cv_repr == repr(cv)

    # 测试 get_n_splits 方法抛出 ValueError 的情况
    msg = "The 'X' parameter should not be None."
    with pytest.raises(ValueError, match=msg):
        loo.get_n_splits(None, y, groups)
    with pytest.raises(ValueError, match=msg):
        lpo.get_n_splits(None, y, groups)
# 定义一个函数用于测试二维标签（y）和多标签情况下的功能性
def test_2d_y():
    # 设置样本数量
    n_samples = 30
    # 创建一个随机数生成器对象，种子为1
    rng = np.random.RandomState(1)
    # 生成一个形状为(n_samples, 2)的随机整数数组，取值范围为0到2，表示特征X
    X = rng.randint(0, 3, size=(n_samples, 2))
    # 生成一个形状为(n_samples,)的随机整数数组，取值范围为0到2，表示目标y
    y = rng.randint(0, 3, size=(n_samples,))
    # 将y变形为二维数组，形状为(n_samples, 1)
    y_2d = y.reshape(-1, 1)
    # 生成一个形状为(n_samples, 3)的随机整数数组，取值范围为0到1，表示多标签y
    y_multilabel = rng.randint(0, 2, size=(n_samples, 3))
    # 生成一个形状为(n_samples,)的随机整数数组，取值范围为0到2，表示组信息
    groups = rng.randint(0, 3, size=(n_samples,))
    # 定义不同的交叉验证分离器
    splitters = [
        LeaveOneOut(),
        LeavePOut(p=2),
        KFold(),
        StratifiedKFold(),
        RepeatedKFold(),
        RepeatedStratifiedKFold(),
        StratifiedGroupKFold(),
        ShuffleSplit(),
        StratifiedShuffleSplit(test_size=0.5),
        GroupShuffleSplit(),
        LeaveOneGroupOut(),
        LeavePGroupsOut(n_groups=2),
        GroupKFold(n_splits=3),
        TimeSeriesSplit(),
        PredefinedSplit(test_fold=groups),
    ]
    # 遍历所有的分离器
    for splitter in splitters:
        # 对X和y执行拆分操作，并将结果转换为列表（但不使用结果）
        list(_split(splitter, X, y, groups=groups))
        # 对X和y_2d执行拆分操作，并将结果转换为列表（但不使用结果）
        list(_split(splitter, X, y_2d, groups=groups))
        try:
            # 对X和y_multilabel执行拆分操作，并将结果转换为列表（但不使用结果）
            list(_split(splitter, X, y_multilabel, groups=groups))
        except ValueError as e:
            # 如果拆分多标签y时抛出值错误，则断言特定的错误消息
            allowed_target_types = ("binary", "multiclass")
            msg = "Supported target types are: {}. Got 'multilabel".format(
                allowed_target_types
            )
            assert msg in str(e)


# 定义一个函数，用于检查训练集和测试集的有效性分割
def check_valid_split(train, test, n_samples=None):
    # 使用Python集合来获取更详细的断言失败消息
    train, test = set(train), set(test)

    # 训练集和测试集应该没有重叠部分
    assert train.intersection(test) == set()

    if n_samples is not None:
        # 检查训练集和测试集的并集是否覆盖了所有的索引
        assert train.union(test) == set(range(n_samples))


# 定义一个函数，用于检查交叉验证对象覆盖的样本集合情况
def check_cv_coverage(cv, X, y, groups, expected_n_splits):
    # 获取样本数量
    n_samples = _num_samples(X)
    # 断言交叉验证的拆分数是否等于预期的拆分数
    assert cv.get_n_splits(X, y, groups) == expected_n_splits

    collected_test_samples = set()
    iterations = 0
    # 遍历交叉验证的每一个拆分
    for train, test in cv.split(X, y, groups):
        # 检查当前拆分的训练集和测试集的有效性
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        # 更新收集到的测试样本集合
        collected_test_samples.update(test)

    # 断言累积的测试样本集合是否覆盖了整个数据集
    assert iterations == expected_n_splits
    if n_samples is not None:
        assert collected_test_samples == set(range(n_samples))


# 定义一个函数，用于测试KFold的值错误情况
def test_kfold_valueerrors():
    # 创建两个示例数组X1和X2
    X1 = np.array([[1, 2], [3, 4], [5, 6]])
    X2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    # 检查如果样本数不足时是否会引发错误
    (ValueError, next, KFold(4).split(X1))

    # 检查如果最少成员类别的数量太少时是否会引发警告
    y = np.array([3, 3, -1, -1, 3])

    skf_3 = StratifiedKFold(3)
    with pytest.warns(Warning, match="The least populated class"):
        next(skf_3.split(X2, y))

    sgkf_3 = StratifiedGroupKFold(3)
    naive_groups = np.arange(len(y))
    # 使用 pytest 检查是否产生特定类型的警告（Warning），并检查警告消息是否包含特定文本
    with pytest.warns(Warning, match="The least populated class"):
        # 调用 split 方法执行数据集拆分操作，使用给定的分组信息 naive_groups
        next(sgkf_3.split(X2, y, naive_groups))

    # 忽略所有警告，验证即使有警告，拆分仍然能够成功完成
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 调用 check_cv_coverage 函数，验证交叉验证的覆盖情况，不指定分组信息
        check_cv_coverage(skf_3, X2, y, groups=None, expected_n_splits=3)

    # 忽略所有警告，验证即使有警告，拆分仍然能够成功完成，使用 naive_groups 作为分组信息
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 调用 check_cv_coverage 函数，验证交叉验证的覆盖情况，使用 naive_groups 作为分组信息
        check_cv_coverage(sgkf_3, X2, y, groups=naive_groups, expected_n_splits=3)

    # 检查是否会因为某些类别的样本数量不足而引发错误
    y = np.array([3, 3, -1, -1, 2])

    # 确保调用 split 方法时，如果某些类别的样本数量过少，会抛出 ValueError 异常
    with pytest.raises(ValueError):
        next(skf_3.split(X2, y))
    with pytest.raises(ValueError):
        next(sgkf_3.split(X2, y))

    # 当折数（folds）小于等于 1 时，应该引发 ValueError 异常
    with pytest.raises(ValueError):
        KFold(0)
    with pytest.raises(ValueError):
        KFold(1)
    # 检查特定错误消息是否与预期的错误消息匹配
    error_string = "k-fold cross-validation requires at least one train/test split"
    with pytest.raises(ValueError, match=error_string):
        StratifiedKFold(0)
    with pytest.raises(ValueError, match=error_string):
        StratifiedKFold(1)
    with pytest.raises(ValueError, match=error_string):
        StratifiedGroupKFold(0)
    with pytest.raises(ValueError, match=error_string):
        StratifiedGroupKFold(1)

    # 当 n_splits 参数不是整数时，应该引发 ValueError 异常
    with pytest.raises(ValueError):
        KFold(1.5)
    with pytest.raises(ValueError):
        KFold(2.0)
    with pytest.raises(ValueError):
        StratifiedKFold(1.5)
    with pytest.raises(ValueError):
        StratifiedKFold(2.0)
    with pytest.raises(ValueError):
        StratifiedGroupKFold(1.5)
    with pytest.raises(ValueError):
        StratifiedGroupKFold(2.0)

    # 当 shuffle 参数不是布尔值时，应该引发 TypeError 异常
    with pytest.raises(TypeError):
        KFold(n_splits=4, shuffle=None)
# 定义测试函数，用于验证 KFold 的索引生成是否正确
def test_kfold_indices():
    # 创建一个长度为 18 的全为 1 的数组
    X1 = np.ones(18)
    # 使用 KFold 将数据分为 3 折
    kf = KFold(3)
    # 调用函数检查是否所有的索引都在测试折中返回
    check_cv_coverage(kf, X1, y=None, groups=None, expected_n_splits=3)

    # 创建一个长度为 17 的全为 1 的数组
    X2 = np.ones(17)
    # 使用 KFold 将数据分为 3 折
    kf = KFold(3)
    # 再次调用函数检查是否所有的索引都在测试折中返回，即使不可能产生相等大小的折叠
    check_cv_coverage(kf, X2, y=None, groups=None, expected_n_splits=3)

    # 断言语句，验证 get_n_splits 返回的折数是否与预期的 5 相符
    assert 5 == KFold(5).get_n_splits(X2)


# 定义测试函数，验证 KFold 在不进行洗牌时是否能保持数据的顺序
def test_kfold_no_shuffle():
    # 创建一个包含子数组的列表
    X2 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

    # 使用 KFold 将数据分为 2 折，对前四个子数组进行分割
    splits = KFold(2).split(X2[:-1])
    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [0, 1]
    train, test = next(splits)
    assert_array_equal(test, [0, 1])
    assert_array_equal(train, [2, 3])

    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [2, 3]
    train, test = next(splits)
    assert_array_equal(test, [2, 3])
    assert_array_equal(train, [0, 1])

    # 使用 KFold 将数据分为 2 折，对所有子数组进行分割
    splits = KFold(2).split(X2)
    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [0, 1, 2]
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 2])
    assert_array_equal(train, [3, 4])

    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [3, 4]
    train, test = next(splits)
    assert_array_equal(test, [3, 4])
    assert_array_equal(train, [0, 1, 2])


# 定义测试函数，验证 StratifiedKFold 在不进行洗牌时是否能保持数据的顺序
def test_stratified_kfold_no_shuffle():
    # 创建一个包含 4 个元素的全为 1 的数组和一个类标签数组
    X, y = np.ones(4), [1, 1, 0, 0]
    # 使用 StratifiedKFold 将数据分为 2 折
    splits = StratifiedKFold(2).split(X, y)
    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [0, 2]
    train, test = next(splits)
    assert_array_equal(test, [0, 2])
    assert_array_equal(train, [1, 3])

    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [1, 3]
    train, test = next(splits)
    assert_array_equal(test, [1, 3])
    assert_array_equal(train, [0, 2])

    # 创建一个包含 7 个元素的全为 1 的数组和一个类标签数组
    X, y = np.ones(7), [1, 1, 1, 0, 0, 0, 0]
    # 使用 StratifiedKFold 将数据分为 2 折
    splits = StratifiedKFold(2).split(X, y)
    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [0, 1, 3, 4]
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 3, 4])
    assert_array_equal(train, [2, 5, 6])

    # 获取下一个分割的训练集和测试集，验证测试集中的索引是否为 [2, 5, 6]
    train, test = next(splits)
    assert_array_equal(test, [2, 5, 6])
    assert_array_equal(train, [0, 1, 3, 4])

    # 断言语句，验证 get_n_splits 返回的折数是否与预期的 5 相符
    assert 5 == StratifiedKFold(5).get_n_splits(X, y)

    # 创建一个包含 7 个元素的全为 1 的数组和两个字符串标签数组
    X = np.ones(7)
    y1 = ["1", "1", "1", "0", "0", "0", "0"]
    y2 = [1, 1, 1, 0, 0, 0, 0]
    # 使用 StratifiedKFold 分别对两个标签数组进行分割，并验证结果是否相等
    np.testing.assert_equal(
        list(StratifiedKFold(2).split(X, y1)), list(StratifiedKFold(2).split(X, y2))
    )

    # 创建一个与标签数组相同长度的全为 1 的数组
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    X = np.ones_like(y)
    # 使用 StratifiedKFold 和 KFold 分别对数据进行分割，并验证结果是否相等
    np.testing.assert_equal(
        list(StratifiedKFold(3).split(X, y)), list(KFold(3).split(X, y))
    )


# 使用 pytest 的参数化功能，定义测试函数，验证 StratifiedKFold 和 StratifiedGroupKFold 是否能在单独分割中保持类比例
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("k", [4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("kfold", [StratifiedKFold, StratifiedGroupKFold])
def test_stratified_kfold_ratios(k, shuffle, kfold):
    # 检查 StratifiedKFold 在各个单独分割中是否能保持类比例
    # 重复进行洗牌开关关闭和打开的测试
    n_samples = 1000
    X = np.ones(n_samples)
    # 创建一个数组 y，用于存储样本标签，实现数据分层
    y = np.array(
        [4] * int(0.10 * n_samples)        # 创建10%样本标签为4的数据
        + [0] * int(0.89 * n_samples)      # 创建89%样本标签为0的数据
        + [1] * int(0.01 * n_samples)      # 创建1%样本标签为1的数据
    )
    # 使用 np.arange() 创建一个与 y 长度相同的数组作为分组标识
    groups = np.arange(len(y))
    # 计算 y 中各标签出现频率的比例
    distr = np.bincount(y) / len(y)

    # 初始化空列表，用于存储每次划分的测试集大小
    test_sizes = []
    # 如果 shuffle 为 True，则设置 random_state 为 0，否则为 None
    random_state = None if not shuffle else 0
    # 使用 kfold 函数创建 K 折交叉验证对象 skf
    skf = kfold(k, random_state=random_state, shuffle=shuffle)
    # 对每一次 K 折交叉验证进行循环
    for train, test in _split(skf, X, y, groups=groups):
        # 检查训练集中各类别标签的频率是否与总体一致，容忍度为 0.02
        assert_allclose(np.bincount(y[train]) / len(train), distr, atol=0.02)
        # 检查测试集中各类别标签的频率是否与总体一致，容忍度为 0.02
        assert_allclose(np.bincount(y[test]) / len(test), distr, atol=0.02)
        # 将当前测试集的大小添加到 test_sizes 列表中
        test_sizes.append(len(test))
    # 断言所有测试集的大小差距不超过 1
    assert np.ptp(test_sizes) <= 1
@pytest.mark.parametrize("shuffle", [False, True])
# 使用 pytest.mark.parametrize 来指定参数化测试，shuffle 参数分别为 False 和 True
@pytest.mark.parametrize("k", [4, 6, 7])
# 参数化 k 参数，测试值为 4, 6, 7
@pytest.mark.parametrize("kfold", [StratifiedKFold, StratifiedGroupKFold])
# 参数化 kfold 参数，测试 StratifiedKFold 和 StratifiedGroupKFold 两种类

def test_stratified_kfold_label_invariance(k, shuffle, kfold):
    # 检查分层 k 折交叉验证在标签不变的情况下是否给出相同的索引

    n_samples = 100
    y = np.array(
        [2] * int(0.10 * n_samples)
        + [0] * int(0.89 * n_samples)
        + [1] * int(0.01 * n_samples)
    )
    # 创建一个包含 2, 0, 1 的数组 y，确保完美的分层是使用 StratifiedGroupKFold
    X = np.ones(len(y))
    # 创建一个全为 1 的数组 X，长度与 y 相同
    groups = np.arange(len(y))

    def get_splits(y):
        # 定义一个内部函数 get_splits，用于获取交叉验证的分割
        random_state = None if not shuffle else 0
        return [
            (list(train), list(test))
            for train, test in _split(
                kfold(k, random_state=random_state, shuffle=shuffle),
                X,
                y,
                groups=groups,
            )
        ]

    splits_base = get_splits(y)
    # 获取基准分割
    for perm in permutations([0, 1, 2]):
        y_perm = np.take(perm, y)
        # 对 y 进行排列组合得到新的排列 y_perm
        splits_perm = get_splits(y_perm)
        # 获取排列后的分割结果 splits_perm
        assert splits_perm == splits_base
        # 断言排列后的分割结果与基准分割结果相同


def test_kfold_balance():
    # 检查 KFold 返回的折叠是否具有平衡的大小
    for i in range(11, 17):
        kf = KFold(5).split(X=np.ones(i))
        # 使用 KFold 将全为 1 的数组 X 划分成 5 折
        sizes = [len(test) for _, test in kf]

        assert (np.max(sizes) - np.min(sizes)) <= 1
        # 断言每折的大小差异不超过 1
        assert np.sum(sizes) == i
        # 断言所有折的大小之和等于 i


@pytest.mark.parametrize("kfold", [StratifiedKFold, StratifiedGroupKFold])
# 参数化 kfold 参数，测试 StratifiedKFold 和 StratifiedGroupKFold 两种类
def test_stratifiedkfold_balance(kfold):
    # 检查 KFold 返回的折叠是否具有平衡的大小（仅当分层可能时）
    # 分别测试打开和关闭随机打乱的情况
    X = np.ones(17)
    # 创建一个长度为 17 的全为 1 的数组 X
    y = [0] * 3 + [1] * 14
    # 创建一个包含 0 和 1 的数组 y，比例为 3:14
    groups = np.arange(len(y))

    for shuffle in (True, False):
        # 遍历随机打乱的情况（True 和 False）
        cv = kfold(3, shuffle=shuffle)
        # 使用给定的 kfold 和 shuffle 创建交叉验证对象 cv
        for i in range(11, 17):
            skf = _split(cv, X[:i], y[:i], groups[:i])
            # 使用 _split 函数进行交叉验证划分
            sizes = [len(test) for _, test in skf]

            assert (np.max(sizes) - np.min(sizes)) <= 1
            # 断言每折的大小差异不超过 1
            assert np.sum(sizes) == i
            # 断言所有折的大小之和等于 i


def test_shuffle_kfold():
    # 检查索引是否被正确地随机打乱
    kf = KFold(3)
    kf2 = KFold(3, shuffle=True, random_state=0)
    kf3 = KFold(3, shuffle=True, random_state=1)

    X = np.ones(300)
    # 创建一个长度为 300 的全为 1 的数组 X

    all_folds = np.zeros(300)
    # 创建一个长度为 300 的全为 0 的数组 all_folds
    for (tr1, te1), (tr2, te2), (tr3, te3) in zip(
        kf.split(X), kf2.split(X), kf3.split(X)
    ):
        # 使用三个不同的随机种子创建三个 KFold 对象进行循环
        for tr_a, tr_b in combinations((tr1, tr2, tr3), 2):
            # 断言没有完全重叠的训练集索引
            assert len(np.intersect1d(tr_a, tr_b)) != len(tr1)

        # 将 kf2 的所有测试索引设置为 1
        all_folds[te2] = 1

    # 断言所有索引都在不同的测试折中返回
    assert sum(all_folds) == 300
    # 断言所有折的测试集索引之和等于 300


@pytest.mark.parametrize("kfold", [KFold, StratifiedKFold, StratifiedGroupKFold])
# 参数化 kfold 参数，测试 KFold、StratifiedKFold 和 StratifiedGroupKFold 三种类
def test_shuffle_kfold_stratifiedkfold_reproducibility(kfold):
    # 检查随机打乱的 KFold 和 StratifiedKFold 的可重现性
    X = np.ones(15)  # 创建一个包含15个1的NumPy数组 X，用于第一个数据集
    y = [0] * 7 + [1] * 8  # 创建一个长度为15的标签列表 y，其中前7个元素为0，后8个元素为1，对应于第一个数据集的标签
    groups_1 = np.arange(len(y))  # 创建一个与数据集长度相同的NumPy数组 groups_1，用于分组信息

    X2 = np.ones(16)  # 创建一个包含16个1的NumPy数组 X2，用于第二个数据集，长度不为3的倍数
    y2 = [0] * 8 + [1] * 8  # 创建一个长度为16的标签列表 y2，其中前8个元素为0，后8个元素为1，对应于第二个数据集的标签
    groups_2 = np.arange(len(y2))  # 创建一个与第二个数据集长度相同的NumPy数组 groups_2，用于分组信息

    # 检查当 shuffle 参数为 True 时，使用相同的 random_state 整数时，多次调用 split 方法应产生相同的分割
    kf = kfold(3, shuffle=True, random_state=0)

    np.testing.assert_equal(
        list(_split(kf, X, y, groups_1)), list(_split(kf, X, y, groups_1))
    )

    # 检查当 shuffle 参数为 True 时，使用随机状态为 RandomState 实例或 None 时，多次调用 split 方法通常（但不总是）会产生不同的分割
    kf = kfold(3, shuffle=True, random_state=np.random.RandomState(0))
    for data in zip((X, X2), (y, y2), (groups_1, groups_2)):
        # 测试两个分割是否不同的交叉验证 (cv)
        for (_, test_a), (_, test_b) in zip(_split(kf, *data), _split(kf, *data)):
            # cv.split(...) 返回一个元组数组，每个元组包含一个训练索引数组和一个测试索引数组
            # 确保在未设置随机状态时，数据的分割不相同
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(test_a, test_b)
# 定义测试函数，验证 StratifiedKFold 类在进行分层折叠交叉验证时的功能
def test_shuffle_stratifiedkfold():
    # 创建一个包含 40 个值为 1 的数组
    X_40 = np.ones(40)
    # 创建类别标签，前 20 个为 0，后 20 个为 1
    y = [0] * 20 + [1] * 20
    # 使用 shuffle=True 和不同的随机种子创建两个 StratifiedKFold 对象
    kf0 = StratifiedKFold(5, shuffle=True, random_state=0)
    kf1 = StratifiedKFold(5, shuffle=True, random_state=1)
    # 遍历每一折的测试集，确保两个对象生成的测试集不相同
    for (_, test0), (_, test1) in zip(kf0.split(X_40, y), kf1.split(X_40, y)):
        assert set(test0) != set(test1)
    # 检查交叉验证的覆盖率
    check_cv_coverage(kf0, X_40, y, groups=None, expected_n_splits=5)

    # 确保在 StratifiedKFold 中，使用不同的 random_state 对每个类别的样本进行混洗
    # 参见 https://github.com/scikit-learn/scikit-learn/pull/13124
    # 创建包含 10 个值的数组 X 和类别标签 y（前 5 个为 0，后 5 个为 1）
    X = np.arange(10)
    y = [0] * 5 + [1] * 5
    # 使用 shuffle=True 和不同的随机种子创建两个 StratifiedKFold 对象
    kf1 = StratifiedKFold(5, shuffle=True, random_state=0)
    kf2 = StratifiedKFold(5, shuffle=True, random_state=1)
    # 获取每个分折的测试集，并进行排序
    test_set1 = sorted([tuple(s[1]) for s in kf1.split(X, y)])
    test_set2 = sorted([tuple(s[1]) for s in kf2.split(X, y)])
    # 确保两个对象生成的测试集不同
    assert test_set1 != test_set2


# 定义测试函数，验证 KFold 类在检测依赖性样本（digits 数据集）时的功能
def test_kfold_can_detect_dependent_samples_on_digits():  # see #2372
    # digits 数据集中的样本存在依赖性：它们显然是按作者分组的，
    # 虽然我们没有关于分组段位置的任何信息。我们可以通过计算 k-fold 交叉验证
    # 进行高亮显示，并与不混洗的情况进行比较：我们观察到混洗的情况错误地假设了 IID，
    # 因此过于乐观地估计了更高的准确度（大约为 0.93），而非混洗的情况则估计了较低的准确度
    # （大约为 0.81）。

    # 从 digits 数据集中取前 600 个样本和对应标签
    X, y = digits.data[:600], digits.target[:600]
    # 创建 SVM 分类器模型
    model = SVC(C=10, gamma=0.005)

    n_splits = 3

    # 创建一个不混洗的 KFold 对象，进行交叉验证并计算平均得分
    cv = KFold(n_splits=n_splits, shuffle=False)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert 0.92 > mean_score
    assert mean_score > 0.80

    # 混洗数据人为地打破了依赖性，并隐藏了模型对作者写作风格过拟合的情况，
    # 从而产生了严重高估的得分：

    # 使用混洗数据创建 KFold 对象，并进行交叉验证并计算平均得分
    cv = KFold(n_splits, shuffle=True, random_state=0)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert mean_score > 0.92

    cv = KFold(n_splits, shuffle=True, random_state=1)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert mean_score > 0.92

    # 类似地，StratifiedKFold 应尽可能少地混洗数据
    # （同时尊重平衡的类别约束），因此也能够通过不过度估计 CV 得分来检测依赖性。
    # 由于 digits 数据集大致平衡，因此估计的平均得分接近于非混洗 KFold 的得分

    # 创建 StratifiedKFold 对象进行交叉验证并计算平均得分
    cv = StratifiedKFold(n_splits)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert 0.94 > mean_score
    assert mean_score > 0.80


# 定义测试函数，验证 StratifiedGroupKFold 类在简单示例中的功能
def test_stratified_group_kfold_trivial():
    # 创建 StratifiedGroupKFold 对象，将分为 3 折
    sgkf = StratifiedGroupKFold(n_splits=3)
    # 简单示例 - 分组具有相同的分布
    # 创建一个包含6个1和12个0的 NumPy 数组
    y = np.array([1] * 6 + [0] * 12)
    # 创建一个与 y 具有相同形状的全1数组，并将其形状改变为列向量形式
    X = np.ones_like(y).reshape(-1, 1)
    # 定义数据分组，共18个数据点，按照给定的分组方式进行分组
    groups = np.asarray((1, 2, 3, 4, 5, 6, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6))
    # 计算 y 中每个类别的分布比例
    distr = np.bincount(y) / len(y)
    # 初始化一个空列表，用于存储每次测试集的大小
    test_sizes = []
    # 使用分组的 StratifiedGroupKFold 进行数据集的分割和交叉验证
    for train, test in sgkf.split(X, y, groups):
        # 检查分组约束条件，确保训练集和测试集的分组没有交集
        assert np.intersect1d(groups[train], groups[test]).size == 0
        # 检查训练集 y 的分布是否与整体数据集一致
        assert_allclose(np.bincount(y[train]) / len(train), distr, atol=0.02)
        # 检查测试集 y 的分布是否与整体数据集一致
        assert_allclose(np.bincount(y[test]) / len(test), distr, atol=0.02)
        # 将当前测试集的大小加入到 test_sizes 列表中
        test_sizes.append(len(test))
    # 最终检查所有测试集大小的范围是否小于等于1
    assert np.ptp(test_sizes) <= 1
def test_stratified_group_kfold_approximate():
    # 创建 StratifiedGroupKFold 对象，将数据分成 3 折
    sgkf = StratifiedGroupKFold(n_splits=3)
    
    # 创建一个包含6个1和12个0的数组作为目标变量 y
    y = np.array([1] * 6 + [0] * 12)
    
    # 创建一个形状为 (18, 1) 的全1数组作为特征变量 X
    X = np.ones_like(y).reshape(-1, 1)
    
    # 创建一个表示样本所属分组的数组
    groups = np.array([1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6])
    
    # 预期的类别分布
    expected = np.asarray([[0.833, 0.166], [0.666, 0.333], [0.5, 0.5]])
    
    # 存储每次测试集大小的列表
    test_sizes = []
    
    # 对每个分折进行迭代，分别从 sgkf.split 返回的结果和 expected 中取值
    for (train, test), expect_dist in zip(sgkf.split(X, y, groups), expected):
        # 检查分组约束条件
        assert np.intersect1d(groups[train], groups[test]).size == 0
        
        # 计算测试集上的类别分布
        split_dist = np.bincount(y[test]) / len(test)
        
        # 检查测试集上的类别分布是否接近预期分布
        assert_allclose(split_dist, expect_dist, atol=0.001)
        
        # 记录当前测试集的大小
        test_sizes.append(len(test))
    
    # 检查测试集大小的极差是否小于等于1
    assert np.ptp(test_sizes) <= 1


@pytest.mark.parametrize(
    "y, groups, expected",
    [
        (
            np.array([0] * 6 + [1] * 6),
            np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
            np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
        ),
        (
            np.array([0] * 9 + [1] * 3),
            np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6]),
            np.asarray([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]),
        ),
    ],
)
def test_stratified_group_kfold_homogeneous_groups(y, groups, expected):
    # 创建 StratifiedGroupKFold 对象，将数据分成 3 折
    sgkf = StratifiedGroupKFold(n_splits=3)
    
    # 创建一个形状为 (12, 1) 的全1数组作为特征变量 X
    X = np.ones_like(y).reshape(-1, 1)
    
    # 对每个分折进行迭代，分别从 sgkf.split 返回的结果和 expected 中取值
    for (train, test), expect_dist in zip(sgkf.split(X, y, groups), expected):
        # 检查分组约束条件
        assert np.intersect1d(groups[train], groups[test]).size == 0
        
        # 计算测试集上的类别分布
        split_dist = np.bincount(y[test]) / len(test)
        
        # 检查测试集上的类别分布是否接近预期分布
        assert_allclose(split_dist, expect_dist, atol=0.001)


@pytest.mark.parametrize("cls_distr", [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.8, 0.2)])
@pytest.mark.parametrize("n_groups", [5, 30, 70])
def test_stratified_group_kfold_against_group_kfold(cls_distr, n_groups):
    # 检查当样本量足够时，StratifiedGroupKFold 产生的分折比普通 GroupKFold 更好地满足分层的要求
    n_splits = 5
    
    # 创建 StratifiedGroupKFold 和 GroupKFold 对象，将数据分成 n_splits 折
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    gkf = GroupKFold(n_splits=n_splits)
    
    # 创建一个随机数生成器
    rng = np.random.RandomState(0)
    
    # 样本点数目
    n_points = 1000
    
    # 从类别分布 cls_distr 中随机选择样本类别
    y = rng.choice(2, size=n_points, p=cls_distr)
    
    # 创建一个形状为 (1000, 1) 的全1数组作为特征变量 X
    X = np.ones_like(y).reshape(-1, 1)
    
    # 从 n_groups 中随机选择样本所属的分组
    g = rng.choice(n_groups, n_points)
    
    # 使用 StratifiedGroupKFold 和 GroupKFold 分别对数据进行分折
    sgkf_folds = sgkf.split(X, y, groups=g)
    gkf_folds = gkf.split(X, y, groups=g)
    
    # 初始化 StratifiedGroupKFold 和 GroupKFold 的熵值
    sgkf_entr = 0
    gkf_entr = 0
    
    # 对每个分折进行迭代，分别从 sgkf_folds 和 gkf_folds 返回的结果中取值
    for (sgkf_train, sgkf_test), (_, gkf_test) in zip(sgkf_folds, gkf_folds):
        # 检查分组约束条件
        assert np.intersect1d(g[sgkf_train], g[sgkf_test]).size == 0
        
        # 计算 StratifiedGroupKFold 测试集上的类别分布
        sgkf_distr = np.bincount(y[sgkf_test]) / len(sgkf_test)
        
        # 计算 GroupKFold 测试集上的类别分布
        gkf_distr = np.bincount(y[gkf_test]) / len(gkf_test)
        
        # 计算 StratifiedGroupKFold 的熵值
        sgkf_entr += stats.entropy(sgkf_distr, qk=cls_distr)
        
        # 计算 GroupKFold 的熵值
        gkf_entr += stats.entropy(gkf_distr, qk=cls_distr)
    
    # 计算平均的 StratifiedGroupKFold 和 GroupKFold 的熵值
    sgkf_entr /= n_splits
    gkf_entr /= n_splits
    
    # 检查 StratifiedGroupKFold 的熵值是否小于等于 GroupKFold 的熵值
    assert sgkf_entr <= gkf_entr
# 定义一个测试函数，用于测试 ShuffleSplit 的不同参数设置下的行为
def test_shuffle_split():
    # 创建 ShuffleSplit 对象 ss1，使用 20% 的数据作为测试集，随机种子为 0，返回索引生成器
    ss1 = ShuffleSplit(test_size=0.2, random_state=0).split(X)
    # 创建 ShuffleSplit 对象 ss2，使用 2 个数据点作为测试集，随机种子为 0，返回索引生成器
    ss2 = ShuffleSplit(test_size=2, random_state=0).split(X)
    # 创建 ShuffleSplit 对象 ss3，使用 2 个数据点作为测试集（以 np.int32 类型指定），随机种子为 0，返回索引生成器
    ss3 = ShuffleSplit(test_size=np.int32(2), random_state=0).split(X)
    # 创建 ShuffleSplit 对象 ss4，使用 2 个数据点作为测试集（以 int 类型指定），随机种子为 0，返回索引生成器
    ss4 = ShuffleSplit(test_size=int(2), random_state=0).split(X)
    
    # 遍历生成的索引生成器，对比各个 ShuffleSplit 对象生成的训练集和测试集是否一致
    for t1, t2, t3, t4 in zip(ss1, ss2, ss3, ss4):
        assert_array_equal(t1[0], t2[0])  # 检查训练集的第一个返回是否一致
        assert_array_equal(t2[0], t3[0])  # 检查训练集的第一个返回是否一致
        assert_array_equal(t3[0], t4[0])  # 检查训练集的第一个返回是否一致
        assert_array_equal(t1[1], t2[1])  # 检查测试集的第一个返回是否一致
        assert_array_equal(t2[1], t3[1])  # 检查测试集的第一个返回是否一致
        assert_array_equal(t3[1], t4[1])  # 检查测试集的第一个返回是否一致


# 使用 pytest.mark.parametrize 运行参数化测试，测试 ShuffleSplit 和 StratifiedShuffleSplit 的默认测试集大小行为
@pytest.mark.parametrize("split_class", [ShuffleSplit, StratifiedShuffleSplit])
@pytest.mark.parametrize(
    "train_size, exp_train, exp_test", [(None, 9, 1), (8, 8, 2), (0.8, 8, 2)]
)
def test_shuffle_split_default_test_size(split_class, train_size, exp_train, exp_test):
    # 检查默认测试集大小的预期行为，即 train_size 为 None 时为 9:1 的比例，8 时为 8:2，0.8 时也为 8:2
    X = np.ones(10)
    y = np.ones(10)

    # 获得训练集和测试集的索引
    X_train, X_test = next(split_class(train_size=train_size).split(X, y))

    # 检查训练集和测试集的长度是否符合预期
    assert len(X_train) == exp_train
    assert len(X_test) == exp_test


# 使用 pytest.mark.parametrize 运行参数化测试，测试 GroupShuffleSplit 的默认测试集大小行为
@pytest.mark.parametrize(
    "train_size, exp_train, exp_test", [(None, 8, 2), (7, 7, 3), (0.7, 7, 3)]
)
def test_group_shuffle_split_default_test_size(train_size, exp_train, exp_test):
    # 检查默认测试集大小的预期行为，即 train_size 为 None 时为 8:2 的比例，7 时为 7:3，0.7 时也为 7:3
    X = np.ones(10)
    y = np.ones(10)
    groups = range(10)

    # 获得训练集和测试集的索引
    X_train, X_test = next(GroupShuffleSplit(train_size=train_size).split(X, y, groups))

    # 检查训练集和测试集的长度是否符合预期
    assert len(X_train) == exp_train
    assert len(X_test) == exp_test


# 测试 StratifiedShuffleSplit 对象的初始化行为
@ignore_warnings
def test_stratified_shuffle_split_init():
    X = np.arange(7)
    y = np.asarray([0, 1, 1, 1, 2, 2, 2])

    # 检查当一个类别只有一个样本时是否会引发 ValueError
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(3, test_size=0.2).split(X, y))

    # 检查当测试集大小小于类别数时是否会引发 ValueError
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(3, test_size=2).split(X, y))

    # 检查当训练集大小小于类别数时是否会引发 ValueError
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(3, test_size=3, train_size=2).split(X, y))

    X = np.arange(9)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # 检查训练集或测试集过小时是否会引发 ValueError
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(train_size=2).split(X, y))
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(test_size=2).split(X, y))


def test_stratified_shuffle_split_respects_test_size():
    # 检查 StratifiedShuffleSplit 对象是否按照指定的测试集大小进行分割
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2])
    test_size = 5
    train_size = 10
    # 使用 StratifiedShuffleSplit 进行分层随机划分数据集
    sss = StratifiedShuffleSplit(
        6, test_size=test_size, train_size=train_size, random_state=0
    ).split(np.ones(len(y)), y)
    # 遍历分层随机划分得到的训练集和测试集
    for train, test in sss:
        # 断言训练集的大小符合预期
        assert len(train) == train_size
        # 断言测试集的大小符合预期
        assert len(test) == test_size
# 定义函数用于测试分层随机分割迭代器的行为
def test_stratified_shuffle_split_iter():
    # 准备不同的测试数据集合
    ys = [
        np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),  # 第一个数据集
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),  # 第二个数据集
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2] * 2),  # 第三个数据集
        np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),  # 第四个数据集
        np.array([-1] * 800 + [1] * 50),  # 第五个数据集
        np.concatenate([[i] * (100 + i) for i in range(11)]),  # 第六个数据集
        [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],  # 第七个数据集
        ["1", "1", "1", "1", "2", "2", "2", "3", "3", "3", "3", "3"],  # 第八个数据集
    ]

    # 遍历每个数据集进行测试
    for y in ys:
        # 使用 StratifiedShuffleSplit 创建对象 sss，并进行分割
        sss = StratifiedShuffleSplit(6, test_size=0.33, random_state=0).split(
            np.ones(len(y)), y
        )
        y = np.asanyarray(y)  # 将 y 转换为可索引的 numpy 数组，用于 y[train]
        # 计算测试集的大小，ceil 函数用于向上取整
        test_size = np.ceil(0.33 * len(y))
        train_size = len(y) - test_size
        # 遍历每个分割得到的训练集和测试集
        for train, test in sss:
            # 断言训练集和测试集的类别是相同的
            assert_array_equal(np.unique(y[train]), np.unique(y[test]))
            # 检查分割后各类别的比例是否保持一致
            p_train = np.bincount(np.unique(y[train], return_inverse=True)[1]) / float(
                len(y[train])
            )
            p_test = np.bincount(np.unique(y[test], return_inverse=True)[1]) / float(
                len(y[test])
            )
            assert_array_almost_equal(p_train, p_test, 1)
            # 断言训练集和测试集的大小之和等于总数据集大小
            assert len(train) + len(test) == y.size
            # 断言训练集和测试集的大小分别为预期的大小
            assert len(train) == train_size
            assert len(test) == test_size
            # 断言训练集和测试集没有交集
            assert_array_equal(np.intersect1d(train, test), [])


# 定义函数用于测试分层随机分割的均匀性
def test_stratified_shuffle_split_even():
    # 测试 StratifiedShuffleSplit，确保索引被均匀地抽取
    n_folds = 5
    n_splits = 1000

    def assert_counts_are_ok(idx_counts, p):
        # 在这里测试索引计数的分布是否接近二项分布
        threshold = 0.05 / n_splits
        bf = stats.binom(n_splits, p)
        for count in idx_counts:
            prob = bf.pmf(count)
            assert (
                prob > threshold
            ), "An index is not drawn with chance corresponding to even draws"
    # 对于两种不同的样本数量进行迭代，分别为6和22
    for n_samples in (6, 22):
        # 创建一个包含相等数量0和1的数组，用于分组（每组中0和1的数量相同）
        groups = np.array((n_samples // 2) * [0, 1])
        # 使用分层随机拆分，指定拆分次数、测试集大小和随机种子
        splits = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=1.0 / n_folds, random_state=0
        )

        # 初始化训练集和测试集的计数器列表，每个计数器初始为0
        train_counts = [0] * n_samples
        test_counts = [0] * n_samples
        # 实际执行的拆分次数初始化为0
        n_splits_actual = 0
        # 对于每个拆分，分别获取训练集和测试集的索引
        for train, test in splits.split(X=np.ones(n_samples), y=groups):
            n_splits_actual += 1
            # 分别对训练集和测试集的索引进行计数
            for counter, ids in [(train_counts, train), (test_counts, test)]:
                for id in ids:
                    counter[id] += 1
        # 断言实际拆分次数与指定的拆分次数相同
        assert n_splits_actual == n_splits

        # 根据给定的样本数量和拆分比例，计算预期的训练集和测试集大小
        n_train, n_test = _validate_shuffle_split(
            n_samples, test_size=1.0 / n_folds, train_size=1.0 - (1.0 / n_folds)
        )

        # 断言训练集和测试集的长度与预期相符
        assert len(train) == n_train
        assert len(test) == n_test
        # 断言训练集和测试集没有重叠的样本
        assert len(set(train).intersection(test)) == 0

        # 计算分组中各个类别的数量
        group_counts = np.unique(groups)
        # 断言拆分对象的测试集大小与预期的测试集大小相同
        assert splits.test_size == 1.0 / n_folds
        # 断言训练集和测试集的总样本数等于样本总数
        assert n_train + n_test == len(groups)
        # 断言分组中的类别数量为2
        assert len(group_counts) == 2
        # 计算预期的测试集和训练集所占比例
        ex_test_p = float(n_test) / n_samples
        ex_train_p = float(n_train) / n_samples

        # 断言训练集和测试集的实际比例与预期比例相符
        assert_counts_are_ok(train_counts, ex_train_p)
        assert_counts_are_ok(test_counts, ex_test_p)
def test_stratified_shuffle_split_overlap_train_test_bug():
    # 见 https://github.com/scikit-learn/scikit-learn/issues/6121 原始 bug 报告
    y = [0, 1, 2, 3] * 3 + [4, 5] * 5
    X = np.ones_like(y)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

    train, test = next(sss.split(X=X, y=y))

    # 没有重叠
    assert_array_equal(np.intersect1d(train, test), [])

    # 完全分区
    assert_array_equal(np.union1d(train, test), np.arange(len(y)))


def test_stratified_shuffle_split_multilabel():
    # 修复问题 9037
    for y in [
        np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
        np.array([[0, 1], [1, 1], [1, 1], [0, 1]]),
    ]:
        X = np.ones_like(y)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        train, test = next(sss.split(X=X, y=y))
        y_train = y[train]
        y_test = y[test]

        # 没有重叠
        assert_array_equal(np.intersect1d(train, test), [])

        # 完全分区
        assert_array_equal(np.union1d(train, test), np.arange(len(y)))

        # 正确分层整行
        # （设计上，这里 y[:, 0] 唯一确定整行 y）
        expected_ratio = np.mean(y[:, 0])
        assert expected_ratio == np.mean(y_train[:, 0])
        assert expected_ratio == np.mean(y_test[:, 0])


def test_stratified_shuffle_split_multilabel_many_labels():
    # 修复 PR #9922：对于具有 > 1000 个标签的多标签数据，str(row) 在位置 4 到 len(row) - 4 的元素会用省略号截断，
    # 因此不能正确使用幂集方法将多标签问题转换为多类问题；此测试检查修复了这个问题。
    row_with_many_zeros = [1, 0, 1] + [0] * 1000 + [1, 0, 1]
    row_with_many_ones = [1, 0, 1] + [1] * 1000 + [1, 0, 1]
    y = np.array([row_with_many_zeros] * 10 + [row_with_many_ones] * 100)
    X = np.ones_like(y)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    train, test = next(sss.split(X=X, y=y))
    y_train = y[train]
    y_test = y[test]

    # 正确分层整行
    # （设计上，这里 y[:, 4] 唯一确定整行 y）
    expected_ratio = np.mean(y[:, 4])
    assert expected_ratio == np.mean(y_train[:, 4])
    assert expected_ratio == np.mean(y_test[:, 4])


def test_predefinedsplit_with_kfold_split():
    # 检查 PredefinedSplit 是否能够复现由 KFold 生成的分割。
    folds = np.full(10, -1.0)
    kf_train = []
    kf_test = []
    for i, (train_ind, test_ind) in enumerate(KFold(5, shuffle=True).split(X)):
        kf_train.append(train_ind)
        kf_test.append(test_ind)
        folds[test_ind] = i
    ps = PredefinedSplit(folds)
    # n_splits 简单地是唯一折数
    assert len(np.unique(folds)) == ps.get_n_splits()
    ps_train, ps_test = zip(*ps.split())
    # 检查两个数组 ps_train 和 kf_train 是否完全相等，如果不相等将引发 AssertionError
    assert_array_equal(ps_train, kf_train)
    # 检查两个数组 ps_test 和 kf_test 是否完全相等，如果不相等将引发 AssertionError
    assert_array_equal(ps_test, kf_test)
# 定义测试函数，用于测试GroupShuffleSplit类的功能
def test_group_shuffle_split():
    # 遍历测试组列表中的每一个组
    for groups_i in test_groups:
        # 创建长度为当前组长度的全1数组作为输入特征X和输出标签y
        X = y = np.ones(len(groups_i))
        # 定义分割的次数为6
        n_splits = 6
        # 测试集大小为总数据的1/3
        test_size = 1.0 / 3
        # 创建GroupShuffleSplit对象，设置分割次数、测试集大小和随机种子
        slo = GroupShuffleSplit(n_splits, test_size=test_size, random_state=0)

        # 确保对象的字符串表示正常工作
        repr(slo)

        # 测试分割的长度是否正确
        assert slo.get_n_splits(X, y, groups=groups_i) == n_splits

        # 获取组i中唯一的值的数组
        l_unique = np.unique(groups_i)
        # 将组i转换为numpy数组l
        l = np.asarray(groups_i)

        # 使用slo对象进行数据分割，遍历返回的训练集索引和测试集索引
        for train, test in slo.split(X, y, groups=groups_i):
            # 第一个测试：确保训练集中没有测试集中的组，反之亦然
            l_train_unique = np.unique(l[train])
            l_test_unique = np.unique(l[test])
            assert not np.any(np.isin(l[train], l_test_unique))
            assert not np.any(np.isin(l[test], l_train_unique))

            # 第二个测试：训练集和测试集的数据总和应等于原始数据总数
            assert l[train].size + l[test].size == l.size

            # 第三个测试：训练集和测试集应该是不相交的
            assert_array_equal(np.intersect1d(train, test), [])

            # 第四个测试：
            # 检查唯一的训练集和测试集组是否正确，允许舍入误差为1
            assert abs(len(l_test_unique) - round(test_size * len(l_unique))) <= 1
            assert abs(len(l_train_unique) - round((1.0 - test_size) * len(l_unique))) <= 1


# 定义测试函数，测试LeaveOneGroupOut和LeavePGroupsOut类的功能
def test_leave_one_p_group_out():
    # 创建LeaveOneGroupOut对象
    logo = LeaveOneGroupOut()
    # 创建LeavePGroupsOut对象，设置排除组的数量为1和2
    lpgo_1 = LeavePGroupsOut(n_groups=1)
    lpgo_2 = LeavePGroupsOut(n_groups=2)

    # 确保对象的字符串表示正常工作
    assert repr(logo) == "LeaveOneGroupOut()"
    assert repr(lpgo_1) == "LeavePGroupsOut(n_groups=1)"
    assert repr(lpgo_2) == "LeavePGroupsOut(n_groups=2)"
    assert repr(LeavePGroupsOut(n_groups=3)) == "LeavePGroupsOut(n_groups=3)"

    # 枚举遍历每一个(cv, p_groups_out)元组，其中cv为LeaveOneGroupOut或LeavePGroupsOut对象，p_groups_out为排除组的数量
    for j, (cv, p_groups_out) in enumerate(((logo, 1), (lpgo_1, 1), (lpgo_2, 2))):
        # 枚举遍历测试组列表中的每一个组
        for i, groups_i in enumerate(test_groups):
            # 计算组i中唯一值的数量
            n_groups = len(np.unique(groups_i))
            # 计算分割的次数，如果p_groups_out为1，则为组数；否则为组数乘以组数减一再除以2
            n_splits = n_groups if p_groups_out == 1 else n_groups * (n_groups - 1) / 2
            # 创建长度为组i长度的全1数组作为输入特征X和输出标签y
            X = y = np.ones(len(groups_i))

            # 测试分割的长度是否正确
            assert cv.get_n_splits(X, y, groups=groups_i) == n_splits

            # 将组i转换为numpy数组groups_arr
            groups_arr = np.asarray(groups_i)

            # 使用cv对象进行数据分割，遍历返回的训练集索引和测试集索引
            for train, test in cv.split(X, y, groups=groups_i):
                # 第一个测试：确保训练集中没有测试集中的组
                assert_array_equal(
                    np.intersect1d(groups_arr[train], groups_arr[test]).tolist(), []
                )

                # 第二个测试：训练集和测试集的数据总和应等于原始数据总数
                assert len(train) + len(test) == len(groups_i)

                # 第三个测试：测试集中的组数量应等于p_groups_out
                assert np.unique(groups_arr[test]).shape[0], p_groups_out
    # 使用虚拟参数检查 get_n_splits() 的返回值
    assert logo.get_n_splits(None, None, ["a", "b", "c", "b", "c"]) == 3
    assert logo.get_n_splits(groups=[1.0, 1.1, 1.0, 1.2]) == 3
    assert lpgo_2.get_n_splits(None, None, np.arange(4)) == 6
    assert lpgo_1.get_n_splits(groups=np.arange(4)) == 4

    # 如果 `groups` 参数非法，抛出 ValueError 异常
    with pytest.raises(ValueError):
        logo.get_n_splits(None, None, [0.0, np.nan, 0.0])
    with pytest.raises(ValueError):
        lpgo_2.get_n_splits(None, None, [0.0, np.inf, 0.0])

    # 使用 pytest 检查是否抛出特定的 ValueError 异常，匹配指定的错误消息
    msg = "The 'groups' parameter should not be None."
    with pytest.raises(ValueError, match=msg):
        logo.get_n_splits(None, None, None)
    with pytest.raises(ValueError, match=msg):
        lpgo_1.get_n_splits(None, None, None)
def test_leave_group_out_changing_groups():
    # 检查 LeaveOneGroupOut 和 LeavePGroupsOut 在更改 groups 变量后仍正常工作
    groups = np.array([0, 1, 2, 1, 1, 2, 0, 0])
    X = np.ones(len(groups))
    # 复制一份 groups 数组，确保不改变原始数组
    groups_changing = np.array(groups, copy=True)
    # 使用 LeaveOneGroupOut 分割数据，传入原始 groups 数组
    lolo = LeaveOneGroupOut().split(X, groups=groups)
    # 使用 LeaveOneGroupOut 分割数据，传入复制后的 groups_changing 数组
    lolo_changing = LeaveOneGroupOut().split(X, groups=groups)
    # 使用 LeavePGroupsOut 分割数据，传入原始 groups 数组
    lplo = LeavePGroupsOut(n_groups=2).split(X, groups=groups)
    # 使用 LeavePGroupsOut 分割数据，传入原始 groups 数组
    lplo_changing = LeavePGroupsOut(n_groups=2).split(X, groups=groups)
    # 将 groups_changing 数组全部设置为 0
    groups_changing[:] = 0
    # 遍历 lolo 和 lolo_changing 结果对，并比较训练集和测试集是否相同
    for llo, llo_changing in [(lolo, lolo_changing), (lplo, lplo_changing)]:
        for (train, test), (train_chan, test_chan) in zip(llo, llo_changing):
            assert_array_equal(train, train_chan)
            assert_array_equal(test, test_chan)

    # 检查 LeavePGroupsOut(n_groups=2) 的拆分数是否为 3
    assert 3 == LeavePGroupsOut(n_groups=2).get_n_splits(X, y=X, groups=groups)
    # 检查 LeaveOneGroupOut 的拆分数是否为 3
    assert 3 == LeaveOneGroupOut().get_n_splits(X, y=X, groups=groups)


def test_leave_group_out_order_dependence():
    # 检查 LeaveOneGroupOut 是否按照被排除的组的索引进行拆分排序
    groups = np.array([2, 2, 0, 0, 1, 1])
    X = np.ones(len(groups))

    # 使用 LeaveOneGroupOut 对象拆分数据，并返回迭代器 splits
    splits = iter(LeaveOneGroupOut().split(X, groups=groups))

    # 预期的训练集和测试集的索引列表
    expected_indices = [
        ([0, 1, 4, 5], [2, 3]),
        ([0, 1, 2, 3], [4, 5]),
        ([2, 3, 4, 5], [0, 1]),
    ]

    # 遍历预期的索引列表，逐一比较拆分结果
    for expected_train, expected_test in expected_indices:
        train, test = next(splits)
        assert_array_equal(train, expected_train)
        assert_array_equal(test, expected_test)


def test_leave_one_p_group_out_error_on_fewer_number_of_groups():
    X = y = groups = np.ones(0)
    # 检查空数组时是否抛出 ValueError 异常
    msg = re.escape("Found array with 0 sample(s)")
    with pytest.raises(ValueError, match=msg):
        next(LeaveOneGroupOut().split(X, y, groups))

    X = y = groups = np.ones(1)
    # 检查少于两个唯一组时是否抛出 ValueError 异常
    msg = re.escape(
        f"The groups parameter contains fewer than 2 unique groups ({groups})."
        " LeaveOneGroupOut expects at least 2."
    )
    with pytest.raises(ValueError, match=msg):
        next(LeaveOneGroupOut().split(X, y, groups))

    X = y = groups = np.ones(1)
    # 检查少于四个唯一组时是否抛出 ValueError 异常
    msg = re.escape(
        "The groups parameter contains fewer than (or equal to) n_groups "
        f"(3) numbers of unique groups ({groups}). LeavePGroupsOut expects "
        "that at least n_groups + 1 (4) unique groups "
        "be present"
    )
    with pytest.raises(ValueError, match=msg):
        next(LeavePGroupsOut(n_groups=3).split(X, y, groups))

    X = y = groups = np.arange(3)
    # 检查少于四个唯一组时是否抛出 ValueError 异常
    msg = re.escape(
        "The groups parameter contains fewer than (or equal to) n_groups "
        f"(3) numbers of unique groups ({groups}). LeavePGroupsOut expects "
        "that at least n_groups + 1 (4) unique groups "
        "be present"
    )
    # 使用 pytest 来测试代码中的异常情况，期望捕获到 ValueError，并匹配给定的错误消息 msg
    with pytest.raises(ValueError, match=msg):
        # 调用 LeavePGroupsOut 分割方法 split()，设置分组数为 3，尝试对数据集 X, y 进行分割操作，并传入分组信息 groups
        next(LeavePGroupsOut(n_groups=3).split(X, y, groups))
# 忽略警告装饰器，用于测试重复的交叉验证值错误
@ignore_warnings
def test_repeated_cv_value_errors():
    # 对于每个交叉验证类（RepeatedKFold, RepeatedStratifiedKFold）
    for cv in (RepeatedKFold, RepeatedStratifiedKFold):
        # 期望抛出值错误异常，因为 n_repeats 不是整数或小于等于 0
        with pytest.raises(ValueError):
            cv(n_repeats=0)
        with pytest.raises(ValueError):
            cv(n_repeats=1.5)


# 使用参数化测试函数标记，测试重复交叉验证的表示形式
@pytest.mark.parametrize("RepeatedCV", [RepeatedKFold, RepeatedStratifiedKFold])
def test_repeated_cv_repr(RepeatedCV):
    n_splits, n_repeats = 2, 6
    # 创建指定类的重复交叉验证对象
    repeated_cv = RepeatedCV(n_splits=n_splits, n_repeats=n_repeats)
    # 构建预期的表示形式字符串
    repeated_cv_repr = "{}(n_repeats=6, n_splits=2, random_state=None)".format(
        repeated_cv.__class__.__name__
    )
    # 断言对象的 repr 是否等于预期的表示形式字符串
    assert repeated_cv_repr == repr(repeated_cv)


# 测试重复 K 折交叉验证的确定性分割
def test_repeated_kfold_determinstic_split():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    random_state = 258173307
    rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)

    # 在每次循环中，分割应产生相同且确定性的结果
    for _ in range(3):
        splits = rkf.split(X)
        train, test = next(splits)
        assert_array_equal(train, [2, 4])
        assert_array_equal(test, [0, 1, 3])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 3])
        assert_array_equal(test, [2, 4])

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [2, 3, 4])

        train, test = next(splits)
        assert_array_equal(train, [2, 3, 4])
        assert_array_equal(test, [0, 1])

        # 检查是否触发了 StopIteration 异常，表示没有更多的分割
        with pytest.raises(StopIteration):
            next(splits)


# 测试获取重复 K 折交叉验证的总分割数
def test_get_n_splits_for_repeated_kfold():
    n_splits = 3
    n_repeats = 4
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    expected_n_splits = n_splits * n_repeats
    # 断言获取的总分割数是否等于预期的总分割数
    assert expected_n_splits == rkf.get_n_splits()


# 测试获取重复分层 K 折交叉验证的总分割数
def test_get_n_splits_for_repeated_stratified_kfold():
    n_splits = 3
    n_repeats = 4
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    expected_n_splits = n_splits * n_repeats
    # 断言获取的总分割数是否等于预期的总分割数
    assert expected_n_splits == rskf.get_n_splits()


# 测试重复分层 K 折交叉验证的确定性分割
def test_repeated_stratified_kfold_determinstic_split():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [1, 1, 1, 0, 0]
    random_state = 1944695409
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=random_state)

    # 在每次循环中，分割应产生相同且确定性的结果
    # 进行三次循环，每次执行以下操作：
    for _ in range(3):
        # 使用随机分割器将数据集 X 和标签 y 进行分割
        splits = rskf.split(X, y)
        # 获取下一个分割后的训练集和测试集
        train, test = next(splits)
        # 断言训练集与预期的数组相等
        assert_array_equal(train, [1, 4])
        # 断言测试集与预期的数组相等
        assert_array_equal(test, [0, 2, 3])

        # 获取下一个分割后的训练集和测试集
        train, test = next(splits)
        # 断言训练集与预期的数组相等
        assert_array_equal(train, [0, 2, 3])
        # 断言测试集与预期的数组相等
        assert_array_equal(test, [1, 4])

        # 获取下一个分割后的训练集和测试集
        train, test = next(splits)
        # 断言训练集与预期的数组相等
        assert_array_equal(train, [2, 3])
        # 断言测试集与预期的数组相等
        assert_array_equal(test, [0, 1, 4])

        # 获取下一个分割后的训练集和测试集
        train, test = next(splits)
        # 断言训练集与预期的数组相等
        assert_array_equal(train, [0, 1, 4])
        # 断言测试集与预期的数组相等
        assert_array_equal(test, [2, 3])

        # 使用 pytest 检查是否抛出 StopIteration 异常
        with pytest.raises(StopIteration):
            next(splits)
# 定义测试函数，用于测试 train_test_split 函数的异常情况
def test_train_test_split_errors():
    # 检查在没有传递任何参数时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split)

    # 检查当 train_size 超出范围时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split, range(3), train_size=1.1)

    # 检查当 test_size 和 train_size 之和超出 1 时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split, range(3), test_size=0.6, train_size=0.6)
    
    # 检查当使用 np.float32 类型的 test_size 和 train_size 且之和超出 1 时是否会引发 ValueError 异常
    pytest.raises(
        ValueError,
        train_test_split,
        range(3),
        test_size=np.float32(0.6),
        train_size=np.float32(0.6),
    )
    
    # 检查当 test_size 参数类型错误时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split, range(3), test_size="wrong_type")
    
    # 检查当 test_size 大于样本数或 train_size 大于样本数时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split, range(3), test_size=2, train_size=4)
    
    # 检查当传递未知参数时是否会引发 TypeError 异常
    pytest.raises(TypeError, train_test_split, range(3), some_argument=1.1)
    
    # 检查当传递不合法的参数组合时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split, range(3), range(42))
    
    # 检查当设置 shuffle=False 且 stratify=True 时是否会引发 ValueError 异常
    pytest.raises(ValueError, train_test_split, range(10), shuffle=False, stratify=True)

    # 使用 pytest 来检查 train_size 参数超出范围时是否会得到预期的错误信息
    with pytest.raises(
        ValueError,
        match=r"train_size=11 should be either positive and "
        r"smaller than the number of samples 10 or a "
        r"float in the \(0, 1\) range",
    ):
        train_test_split(range(10), train_size=11, test_size=1)


# 参数化测试函数，测试 train_test_split 函数的默认 test_size 行为
@pytest.mark.parametrize(
    "train_size, exp_train, exp_test", [(None, 7, 3), (8, 8, 2), (0.8, 8, 2)]
)
def test_train_test_split_default_test_size(train_size, exp_train, exp_test):
    # 检查默认 test_size 参数是否有预期的行为，即根据 train_size 补充 test_size
    X_train, X_test = train_test_split(X, train_size=train_size)

    # 断言返回的训练集大小是否符合预期
    assert len(X_train) == exp_train
    
    # 断言返回的测试集大小是否符合预期
    assert len(X_test) == exp_test


# 参数化测试函数，测试 train_test_split 函数在不同参数组合下的行为
@pytest.mark.parametrize(
    "array_namespace, device, dtype_name", yield_namespace_device_dtype_combinations()
)
@pytest.mark.parametrize(
    "shuffle,stratify",
    (
        (True, None),
        (True, np.hstack((np.ones(6), np.zeros(4)))),
        # 当 shuffle=False 时，stratify 参数应为 None，否则应引发 ValueError 异常
        (False, None),
    ),
)
def test_array_api_train_test_split(
    shuffle, stratify, array_namespace, device, dtype_name
):
    # 根据 array_namespace 和 device 选择合适的数组 API
    xp = _array_api_for_tests(array_namespace, device)

    # 创建 NumPy 数组作为测试数据
    X = np.arange(100).reshape((10, 10))
    y = np.arange(10)

    # 将 NumPy 数组转换为指定 dtype 的数组
    X_np = X.astype(dtype_name)
    X_xp = xp.asarray(X_np, device=device)

    y_np = y.astype(dtype_name)
    y_xp = xp.asarray(y_np, device=device)

    # 使用 train_test_split 函数分割数据集
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y, random_state=0, shuffle=shuffle, stratify=stratify
    )
    # 在设置上下文配置时，确保 array_api_dispatch 参数为 True
    with config_context(array_api_dispatch=True):
        # 如果有指定分层信息，则将其转换为对应的数组表示（如果为 None 则保持原样）
        if stratify is not None:
            stratify_xp = xp.asarray(stratify)
        else:
            stratify_xp = stratify
        
        # 使用 train_test_split 函数将数据集分割为训练集和测试集
        X_train_xp, X_test_xp, y_train_xp, y_test_xp = train_test_split(
            X_xp, y_xp, shuffle=shuffle, stratify=stratify_xp, random_state=0
        )

        # 检查命名空间是否在 array_api_dispatch 启用时保持不变
        assert get_namespace(X_train_xp)[0] == get_namespace(X_xp)[0]
        assert get_namespace(X_test_xp)[0] == get_namespace(X_xp)[0]
        assert get_namespace(y_train_xp)[0] == get_namespace(y_xp)[0]
        assert get_namespace(y_test_xp)[0] == get_namespace(y_xp)[0]

    # 检查输出的设备和数据类型是否与输入保持一致
    assert array_api_device(X_train_xp) == array_api_device(X_xp)
    assert array_api_device(y_train_xp) == array_api_device(y_xp)
    assert array_api_device(X_test_xp) == array_api_device(X_xp)
    assert array_api_device(y_test_xp) == array_api_device(y_xp)

    # 检查训练集和测试集的数据类型是否与输入保持一致
    assert X_train_xp.dtype == X_xp.dtype
    assert y_train_xp.dtype == y_xp.dtype
    assert X_test_xp.dtype == X_xp.dtype
    assert y_test_xp.dtype == y_xp.dtype

    # 使用 assert_allclose 检查转换为 numpy 后的数据是否与预期的 numpy 数组接近
    assert_allclose(
        _convert_to_numpy(X_train_xp, xp=xp),
        X_train_np,
    )
    assert_allclose(
        _convert_to_numpy(X_test_xp, xp=xp),
        X_test_np,
    )
@pytest.mark.parametrize("coo_container", COO_CONTAINERS)
def test_train_test_split(coo_container):
    # 创建一个 10x10 的 NumPy 数组 X
    X = np.arange(100).reshape((10, 10))
    # 使用给定的容器函数 coo_container 将 X 转换为稀疏表示
    X_s = coo_container(X)
    # 创建一个长度为 10 的 NumPy 数组 y
    y = np.arange(10)

    # 简单测试，将数据集 X 和 y 按照默认参数进行分割
    split = train_test_split(X, y, test_size=None, train_size=0.5)
    X_train, X_test, y_train, y_test = split
    assert len(y_test) == len(y_train)
    # 检验 X 和 y 的对应关系是否保持
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)

    # 测试不将列表默认转换为其他数据类型
    split = train_test_split(X, X_s, y.tolist())
    X_train, X_test, X_s_train, X_s_test, y_train, y_test = split
    assert isinstance(y_train, list)
    assert isinstance(y_test, list)

    # 允许处理多维数组
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
    split = train_test_split(X_4d, y_3d)
    assert split[0].shape == (7, 5, 3, 2)
    assert split[1].shape == (3, 5, 3, 2)
    assert split[2].shape == (7, 7, 11)
    assert split[3].shape == (3, 7, 11)

    # 测试分层选项
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    for test_size, exp_test_size in zip([2, 4, 0.25, 0.5, 0.75], [2, 4, 2, 4, 6]):
        train, test = train_test_split(
            y, test_size=test_size, stratify=y, random_state=0
        )
        assert len(test) == exp_test_size
        assert len(test) + len(train) == len(y)
        # 检查数据中类别 1 和 2 的比例是否被保持
        assert np.sum(train == 1) == np.sum(train == 2)

    # 测试不洗牌的分割
    y = np.arange(10)
    for test_size in [2, 0.2]:
        train, test = train_test_split(y, shuffle=False, test_size=test_size)
        assert_array_equal(test, [8, 9])
        assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 7])


def test_train_test_split_32bit_overflow():
    """Check for integer overflow on 32-bit platforms.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20774
    """

    # 定义一个足够大的数字，使得在 32 位平台上 'n * n * train_size' 表达式会发生整数溢出
    big_number = 100000

    # 'y' 的定义是为了复现，至少一个类的人口数量应该与 X 的大小在同一个数量级
    X = np.arange(big_number)
    y = X > (0.99 * big_number)

    # 使用 train_test_split 进行数据集分割，测试整数溢出情况
    split = train_test_split(X, y, stratify=y, train_size=0.25)
    X_train, X_test, y_train, y_test = split

    assert X_train.size + X_test.size == big_number
    assert y_train.size + y_test.size == big_number


@ignore_warnings
def test_train_test_split_pandas():
    # 检查 train_test_split 不会破坏 pandas dataframe
    types = [MockDataFrame]
    try:
        from pandas import DataFrame

        types.append(DataFrame)
    except ImportError:
        pass
    # 遍历类型列表 types 中的每个元素，每个元素赋值给变量 InputFeatureType
    for InputFeatureType in types:
        # 使用当前类型 InputFeatureType 创建一个名为 X_df 的数据框（DataFrame）
        X_df = InputFeatureType(X)
        # 将 X_df 数据框进行训练集和测试集的划分，得到训练集 X_train 和测试集 X_test
        X_train, X_test = train_test_split(X_df)
        # 断言 X_train 是 InputFeatureType 类型的对象
        assert isinstance(X_train, InputFeatureType)
        # 断言 X_test 是 InputFeatureType 类型的对象
        assert isinstance(X_test, InputFeatureType)
# 使用 pytest.mark.parametrize 装饰器为 test_train_test_split_sparse 函数添加参数化测试
@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
# 定义测试函数 test_train_test_split_sparse，验证 train_test_split 是否能够将稀疏矩阵转换为 CSR 格式
def test_train_test_split_sparse(sparse_container):
    # 创建一个 10x10 的 numpy 数组 X
    X = np.arange(100).reshape((10, 10))
    # 使用给定的稀疏容器将 X 转换为稀疏矩阵 X_s
    X_s = sparse_container(X)
    # 调用 train_test_split 函数将 X_s 分割为训练集和测试集 X_train, X_test
    X_train, X_test = train_test_split(X_s)
    # 断言 X_train 是稀疏矩阵且格式为 "csr"
    assert issparse(X_train) and X_train.format == "csr"
    # 断言 X_test 是稀疏矩阵且格式为 "csr"
    assert issparse(X_test) and X_test.format == "csr"


# 定义测试函数 test_train_test_split_mock_pandas，验证 train_test_split 是否能够正确处理 MockDataFrame 对象
def test_train_test_split_mock_pandas():
    # 创建一个 MockDataFrame 对象 X_df
    X_df = MockDataFrame(X)
    # 调用 train_test_split 函数将 X_df 分割为训练集和测试集 X_train, X_test
    X_train, X_test = train_test_split(X_df)
    # 断言 X_train 是 MockDataFrame 类型
    assert isinstance(X_train, MockDataFrame)
    # 断言 X_test 是 MockDataFrame 类型
    assert isinstance(X_test, MockDataFrame)
    # 再次调用 train_test_split 函数，但这次没有对返回值进行使用


# 定义测试函数 test_train_test_split_list_input，验证 train_test_split 对于列表类型的输入的处理
def test_train_test_split_list_input():
    # 创建一个包含 7 个元素的 numpy 数组 X，值全为 1
    X = np.ones(7)
    # 创建三种不同的列表 y1, y2, y3 作为分类标签
    y1 = ["1"] * 4 + ["0"] * 3
    y2 = np.hstack((np.ones(4), np.zeros(3)))
    y3 = y2.tolist()

    # 在 stratify=True 和 stratify=False 的情况下分别调用 train_test_split 函数三次，验证其处理逻辑
    for stratify in (True, False):
        # 对于 y1 的测试
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y1, stratify=y1 if stratify else None, random_state=0
        )
        # 对于 y2 的测试
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y2, stratify=y2 if stratify else None, random_state=0
        )
        # 对于 y3 的测试
        X_train3, X_test3, y_train3, y_test3 = train_test_split(
            X, y3, stratify=y3 if stratify else None, random_state=0
        )

        # 使用 np.testing.assert_equal 断言不同参数下的训练集和测试集的一致性
        np.testing.assert_equal(X_train1, X_train2)
        np.testing.assert_equal(y_train2, y_train3)
        np.testing.assert_equal(X_test1, X_test3)
        np.testing.assert_equal(y_test3, y_test2)


# 使用 pytest.mark.parametrize 装饰器为 test_shufflesplit_errors 函数添加参数化测试
@pytest.mark.parametrize(
    "test_size, train_size",
    [(2.0, None), (1.0, None), (0.1, 0.95), (None, 1j), (11, None), (10, None), (8, 3)],
)
# 定义测试函数 test_shufflesplit_errors，验证 ShuffleSplit 在不合法参数情况下是否能正确抛出 ValueError
def test_shufflesplit_errors(test_size, train_size):
    # 使用 pytest.raises 检测是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 创建 ShuffleSplit 对象并尝试调用其 split 方法，验证其在给定参数下是否抛出异常
        next(ShuffleSplit(test_size=test_size, train_size=train_size).split(X))


# 定义测试函数 test_shufflesplit_reproducible，验证 ShuffleSplit 在给定 random_state 下是否可重现
def test_shufflesplit_reproducible():
    # 创建 ShuffleSplit 对象 ss，指定 random_state 为 21
    ss = ShuffleSplit(random_state=21)
    # 使用 assert_array_equal 断言两次迭代结果是否一致
    assert_array_equal([a for a, b in ss.split(X)], [a for a, b in ss.split(X)])


# 定义测试函数 test_stratifiedshufflesplit_list_input，验证 StratifiedShuffleSplit 对列表类型的输入的处理
def test_stratifiedshufflesplit_list_input():
    # 创建 StratifiedShuffleSplit 对象 sss，指定 test_size 为 2，random_state 为 42
    sss = StratifiedShuffleSplit(test_size=2, random_state=42)
    # 创建包含 7 个元素的 numpy 数组 X，值全为 1
    X = np.ones(7)
    # 创建三种不同的列表 y1, y2, y3 作为分类标签
    y1 = ["1"] * 4 + ["0"] * 3
    y2 = np.hstack((np.ones(4), np.zeros(3)))
    y3 = y2.tolist()

    # 使用 np.testing.assert_equal 断言 StratifiedShuffleSplit 对不同标签列表的处理结果一致性
    np.testing.assert_equal(list(sss.split(X, y1)), list(sss.split(X, y2)))
    np.testing.assert_equal(list(sss.split(X, y3)), list(sss.split(X, y2)))


# 定义测试函数 test_train_test_split_allow_nans，验证 train_test_split 是否允许包含 NaN 值的输入数据
def test_train_test_split_allow_nans():
    # 创建一个包含 NaN 值的 10x20 的浮点数 numpy 数组 X
    X = np.arange(200, dtype=np.float64).reshape(10, -1)
    X[2, :] = np.nan
    # 创建 y 数组，包含 [0, 1] 的重复值，共 10 个
    y = np.repeat([0, 1], X.shape[0] / 2)
    # 使用train_test_split函数将数据集X和标签y划分为训练集和测试集
    train_test_split(X, y, test_size=0.2, random_state=42)
# 定义测试函数 `test_check_cv`，用于验证交叉验证函数 `check_cv` 的正确性
def test_check_cv():
    # 创建长度为 9 的全 1 数组 X
    X = np.ones(9)
    # 调用 `check_cv` 函数创建交叉验证对象 cv，参数 classifier=False 表示不考虑分类器
    cv = check_cv(3, classifier=False)
    # 使用 np.testing.assert_equal 进行递归比较 KFold(3).split(X) 和 cv.split(X) 的结果
    np.testing.assert_equal(list(KFold(3).split(X)), list(cv.split(X)))

    # 创建二元分类的标签数组 y_binary
    y_binary = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1])
    # 调用 `check_cv` 函数创建 StratifiedKFold 对象 cv，用于处理二元分类情况
    cv = check_cv(3, y_binary, classifier=True)
    # 使用 np.testing.assert_equal 比较 StratifiedKFold(3).split(X, y_binary) 和 cv.split(X, y_binary) 的结果
    np.testing.assert_equal(
        list(StratifiedKFold(3).split(X, y_binary)), list(cv.split(X, y_binary))
    )

    # 创建多类别分类的标签数组 y_multiclass
    y_multiclass = np.array([0, 1, 0, 1, 2, 1, 2, 0, 2])
    # 调用 `check_cv` 函数创建 StratifiedKFold 对象 cv，用于处理多类别分类情况
    cv = check_cv(3, y_multiclass, classifier=True)
    # 使用 np.testing.assert_equal 比较 StratifiedKFold(3).split(X, y_multiclass) 和 cv.split(X, y_multiclass) 的结果
    np.testing.assert_equal(
        list(StratifiedKFold(3).split(X, y_multiclass)), list(cv.split(X, y_multiclass))
    )

    # 将 y_multiclass 转换为二维数组 y_multiclass_2d
    y_multiclass_2d = y_multiclass.reshape(-1, 1)
    # 调用 `check_cv` 函数创建 StratifiedKFold 对象 cv，用于处理二维多类别分类情况
    cv = check_cv(3, y_multiclass_2d, classifier=True)
    # 使用 np.testing.assert_equal 比较 StratifiedKFold(3).split(X, y_multiclass_2d) 和 cv.split(X, y_multiclass_2d) 的结果
    np.testing.assert_equal(
        list(StratifiedKFold(3).split(X, y_multiclass_2d)),
        list(cv.split(X, y_multiclass_2d)),
    )

    # 使用断言验证 StratifiedKFold(3).split(X, y_multiclass_2d) 的第一个 fold 的训练集与 KFold(3).split(X, y_multiclass_2d) 的第一个 fold 的训练集不全相等
    assert not np.all(
        next(StratifiedKFold(3).split(X, y_multiclass_2d))[0]
        == next(KFold(3).split(X, y_multiclass_2d))[0]
    )

    # 更新 X 为长度为 5 的全 1 数组
    X = np.ones(5)
    # 创建多标签多输出的标签数组 y_multilabel
    y_multilabel = np.array(
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0]]
    )
    # 调用 `check_cv` 函数创建 KFold 对象 cv，用于处理多标签多输出情况
    cv = check_cv(3, y_multilabel, classifier=True)
    # 使用 np.testing.assert_equal 比较 KFold(3).split(X) 和 cv.split(X) 的结果
    np.testing.assert_equal(list(KFold(3).split(X)), list(cv.split(X)))

    # 创建多输出的标签数组 y_multioutput
    y_multioutput = np.array([[1, 2], [0, 3], [0, 0], [3, 1], [2, 0]])
    # 调用 `check_cv` 函数创建 KFold 对象 cv，用于处理多输出情况
    cv = check_cv(3, y_multioutput, classifier=True)
    # 使用 np.testing.assert_equal 比较 KFold(3).split(X) 和 cv.split(X) 的结果
    np.testing.assert_equal(list(KFold(3).split(X)), list(cv.split(X)))

    # 使用 pytest.raises 检测 ValueError 异常
    with pytest.raises(ValueError):
        check_cv(cv="lolo")


# 定义测试函数 `test_cv_iterable_wrapper`，用于验证交叉验证对象的包装器
def test_cv_iterable_wrapper():
    # 创建 KFold 迭代器 kf_iter，并传入数据 X 和 y
    kf_iter = KFold().split(X, y)
    # 调用 `check_cv` 函数创建交叉验证对象 kf_iter_wrapped，用于包装 kf_iter
    kf_iter_wrapped = check_cv(kf_iter)
    # 使用 np.testing.assert_equal 比较 kf_iter_wrapped.split(X, y) 的结果两次调用是否一致
    np.testing.assert_equal(
        list(kf_iter_wrapped.split(X, y)), list(kf_iter_wrapped.split(X, y))
    )

    # 创建随机化的 KFold 迭代器 kf_randomized_iter，并传入数据 X 和 y
    kf_randomized_iter = KFold(shuffle=True, random_state=0).split(X, y)
    # 调用 `check_cv` 函数创建交叉验证对象 kf_randomized_iter_wrapped，用于包装 kf_randomized_iter
    kf_randomized_iter_wrapped = check_cv(kf_randomized_iter)
    # 使用 np.testing.assert_equal 比较 kf_randomized_iter_wrapped.split(X, y) 的结果两次调用是否一致
    np.testing.assert_equal(
        list(kf_randomized_iter_wrapped.split(X, y)),
        list(kf_randomized_iter_wrapped.split(X, y)),
    )

    # 使用 try-except 块验证 kf_iter_wrapped.split(X, y) 和 kf_randomized_iter_wrapped.split(X, y) 的结果是否相等
    try:
        splits_are_equal = True
        np.testing.assert_equal(
            list(kf_iter_wrapped.split(X, y)),
            list(kf_randomized_iter_wrapped.split(X, y)),
        )
    except AssertionError:
        splits_are_equal = False
    # 使用断言验证 splits_are_equal 应为 False，即如果交叉验证是随机的，多次调用 split 应该产生不同的结果
    assert not splits_are_equal, (
        "If the splits are randomized, "
        "successive calls to split should yield different results"
    )


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数 `test_group_kfold`，测试 GroupKFold 和 StratifiedGroupKFold 类
@pytest.mark.parametrize("kfold", [GroupKFold, StratifiedGroupKFold])
def test_group_kfold(kfold):
    # 使用随机种子为 0 的随机数生成器创建 rng 对象
    rng = np.random.RandomState(0)

    # 测试参数
    n_groups = 15
    n_samples = 1000
    n_splits = 5

    X = y = np.ones(n_samples)

    # Construct the test data
    tolerance = 0.05 * n_samples  # 允许误差为总样本数的5%
    groups = rng.randint(0, n_groups, n_samples)  # 随机生成长度为 n_samples 的组标签

    ideal_n_groups_per_fold = n_samples // n_splits  # 每折理想的组数

    len(np.unique(groups))
    # 获取每个折的测试集索引
    folds = np.zeros(n_samples)
    lkf = kfold(n_splits=n_splits)
    for i, (_, test) in enumerate(lkf.split(X, y, groups)):
        folds[test] = i

    # 检查每个折的大小是否接近
    assert len(folds) == len(groups)
    for i in np.unique(folds):
        assert tolerance >= abs(sum(folds == i) - ideal_n_groups_per_fold)

    # 检查每个组是否只出现在一个折中
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1

    # 检查没有组同时出现在训练集和测试集
    groups = np.asarray(groups, dtype=object)
    for train, test in lkf.split(X, y, groups):
        assert len(np.intersect1d(groups[train], groups[test])) == 0

    # 构建测试数据
    groups = np.array(
        [
            "Albert",
            "Jean",
            "Bertrand",
            "Michel",
            "Jean",
            "Francis",
            "Robert",
            "Michel",
            "Rachel",
            "Lois",
            "Michelle",
            "Bernard",
            "Marion",
            "Laura",
            "Jean",
            "Rachel",
            "Franck",
            "John",
            "Gael",
            "Anna",
            "Alix",
            "Robert",
            "Marion",
            "David",
            "Tony",
            "Abel",
            "Becky",
            "Madmood",
            "Cary",
            "Mary",
            "Alexandre",
            "David",
            "Francis",
            "Barack",
            "Abdoul",
            "Rasha",
            "Xi",
            "Silvia",
        ]
    )

    n_groups = len(np.unique(groups))
    n_samples = len(groups)
    n_splits = 5
    tolerance = 0.05 * n_samples  # 允许误差为总样本数的5%
    ideal_n_groups_per_fold = n_samples // n_splits  # 每折理想的组数

    X = y = np.ones(n_samples)

    # 获取每个折的测试集索引
    folds = np.zeros(n_samples)
    for i, (_, test) in enumerate(lkf.split(X, y, groups)):
        folds[test] = i

    # 检查每个折的大小是否接近
    assert len(folds) == len(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for group in np.unique(groups):
            assert len(np.unique(folds[groups == group])) == 1

    # 检查没有组同时出现在训练集和测试集
    groups = np.asarray(groups, dtype=object)
    # 使用 LeaveOneGroupOut 分割器分割数据集 X 和标签 y，同时考虑组信息 groups
    for train, test in lkf.split(X, y, groups):
        # 断言每个训练集和测试集的组没有交集
        assert len(np.intersect1d(groups[train], groups[test])) == 0

    # 将 groups 转换成列表，并使用列表作为组信息进行 LeaveOneGroupOut 分割
    cv_iter = list(lkf.split(X, y, groups.tolist()))
    # 使用 zip 函数同时迭代两个 LeaveOneGroupOut 分割的结果
    for (train1, test1), (train2, test2) in zip(lkf.split(X, y, groups), cv_iter):
        # 断言两个分割方法得到的训练集相同
        assert_array_equal(train1, train2)
        # 断言两个分割方法得到的测试集相同
        assert_array_equal(test1, test2)

    # 创建一个包含重复组的 groups 数组，以及与之相同长度的全为1的特征 X 和标签 y 数组
    groups = np.array([1, 1, 1, 2, 2])
    X = y = np.ones(len(groups))
    # 使用 GroupKFold 分割器尝试将数据集分为3折，应该抛出 ValueError 异常，匹配指定的错误信息
    with pytest.raises(ValueError, match="Cannot have number of splits.*greater"):
        next(GroupKFold(n_splits=3).split(X, y, groups))
def test_time_series_cv():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]

    # 如果折叠数大于样本数，应该失败
    with pytest.raises(ValueError, match="Cannot have number of folds.*greater"):
        next(TimeSeriesSplit(n_splits=7).split(X))

    # 创建一个时间序列交叉验证对象，指定折叠数为2
    tscv = TimeSeriesSplit(2)

    # 手动检查时间序列交叉验证在玩具数据集上是否保留了数据的顺序
    splits = tscv.split(X[:-1])
    train, test = next(splits)
    assert_array_equal(train, [0, 1])  # 断言：训练集应为 [0, 1]
    assert_array_equal(test, [2, 3])   # 断言：测试集应为 [2, 3]

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])  # 断言：训练集应为 [0, 1, 2, 3]
    assert_array_equal(test, [4, 5])          # 断言：测试集应为 [4, 5]

    splits = TimeSeriesSplit(2).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2])  # 断言：训练集应为 [0, 1, 2]
    assert_array_equal(test, [3, 4])      # 断言：测试集应为 [3, 4]

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])  # 断言：训练集应为 [0, 1, 2, 3, 4]
    assert_array_equal(test, [5, 6])              # 断言：测试集应为 [5, 6]

    # 检查 get_n_splits 方法返回正确的折叠数
    splits = TimeSeriesSplit(2).split(X)
    n_splits_actual = len(list(splits))
    assert n_splits_actual == tscv.get_n_splits()  # 断言：折叠数应与 tscv 对象的折叠数相同
    assert n_splits_actual == 2


def _check_time_series_max_train_size(splits, check_splits, max_train_size):
    for (train, test), (check_train, check_test) in zip(splits, check_splits):
        assert_array_equal(test, check_test)  # 断言：测试集应与预期的测试集相同
        assert len(check_train) <= max_train_size  # 断言：训练集的长度应小于等于 max_train_size
        suffix_start = max(len(train) - max_train_size, 0)
        assert_array_equal(check_train, train[suffix_start:])  # 断言：训练集应与预期的训练集相同


def test_time_series_max_train_size():
    X = np.zeros((6, 1))
    splits = TimeSeriesSplit(n_splits=3).split(X)
    check_splits = TimeSeriesSplit(n_splits=3, max_train_size=3).split(X)
    _check_time_series_max_train_size(splits, check_splits, max_train_size=3)

    # 测试当一个折叠的大小大于 max_train_size 的情况
    check_splits = TimeSeriesSplit(n_splits=3, max_train_size=2).split(X)
    _check_time_series_max_train_size(splits, check_splits, max_train_size=2)

    # 测试每个折叠的大小都小于 max_train_size 的情况
    check_splits = TimeSeriesSplit(n_splits=3, max_train_size=5).split(X)
    _check_time_series_max_train_size(splits, check_splits, max_train_size=2)


def test_time_series_test_size():
    X = np.zeros((10, 1))

    # 单独测试 test_size 参数
    splits = TimeSeriesSplit(n_splits=3, test_size=3).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0])       # 断言：第一个训练集应为 [0]
    assert_array_equal(test, [1, 2, 3])  # 断言：第一个测试集应为 [1, 2, 3]

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])  # 断言：第二个训练集应为 [0, 1, 2, 3]
    assert_array_equal(test, [4, 5, 6])       # 断言：第二个测试集应为 [4, 5, 6]

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6])  # 断言：第三个训练集应为 [0, 1, 2, 3, 4, 5, 6]
    assert_array_equal(test, [7, 8, 9])              # 断言：第三个测试集应为 [7, 8, 9]

    # 测试同时设置 test_size 和 max_train_size 参数
    splits = TimeSeriesSplit(n_splits=2, test_size=2, max_train_size=4).split(X)

    train, test = next(splits)
    assert_array_equal(train, [2, 3, 4, 5])  # 断言：第一个训练集应为 [2, 3, 4, 5]
    assert_array_equal(test, [6, 7])          # 断言：第一个测试集应为 [6, 7]
    # 从生成器 splits 中获取下一个训练集和测试集
    train, test = next(splits)
    # 使用 assert_array_equal 函数断言 train 应为 [4, 5, 6, 7]
    assert_array_equal(train, [4, 5, 6, 7])
    # 使用 assert_array_equal 函数断言 test 应为 [8, 9]
    assert_array_equal(test, [8, 9])

    # 应当由于数据点不足以满足配置而失败
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常，并验证异常消息包含指定的字符串
    with pytest.raises(ValueError, match="Too many splits.*with test_size"):
        # 使用 TimeSeriesSplit 进行时间序列分割，设置 n_splits=5 和 test_size=2
        splits = TimeSeriesSplit(n_splits=5, test_size=2).split(X)
        # 获取 splits 生成器的下一个值，预期会触发异常
        next(splits)
def test_time_series_gap():
    # 创建一个形状为 (10, 1) 的全零数组 X
    X = np.zeros((10, 1))

    # 使用 TimeSeriesSplit 进行时间序列分割，设置分割数为 2，间隔为 2
    splits = TimeSeriesSplit(n_splits=2, gap=2).split(X)

    # 获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [0, 1]
    assert_array_equal(train, [0, 1])
    # 断言测试集是否为 [4, 5, 6]
    assert_array_equal(test, [4, 5, 6])

    # 再次获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [0, 1, 2, 3, 4]
    assert_array_equal(train, [0, 1, 2, 3, 4])
    # 断言测试集是否为 [7, 8, 9]
    assert_array_equal(test, [7, 8, 9])

    # 使用 TimeSeriesSplit 进行时间序列分割，设置分割数为 3，间隔为 2，最大训练集大小为 2
    splits = TimeSeriesSplit(n_splits=3, gap=2, max_train_size=2).split(X)

    # 获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [0, 1]
    assert_array_equal(train, [0, 1])
    # 断言测试集是否为 [4, 5]
    assert_array_equal(test, [4, 5])

    # 再次获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [2, 3]
    assert_array_equal(train, [2, 3])
    # 断言测试集是否为 [6, 7]
    assert_array_equal(test, [6, 7])

    # 再次获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [4, 5]
    assert_array_equal(train, [4, 5])
    # 断言测试集是否为 [8, 9]
    assert_array_equal(test, [8, 9])

    # 使用 TimeSeriesSplit 进行时间序列分割，设置分割数为 2，间隔为 2，最大训练集大小为 4，测试集大小为 2
    splits = TimeSeriesSplit(n_splits=2, gap=2, max_train_size=4, test_size=2).split(X)

    # 获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [0, 1, 2, 3]
    assert_array_equal(train, [0, 1, 2, 3])
    # 断言测试集是否为 [6, 7]
    assert_array_equal(test, [6, 7])

    # 再次获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [2, 3, 4, 5]
    assert_array_equal(train, [2, 3, 4, 5])
    # 断言测试集是否为 [8, 9]
    assert_array_equal(test, [8, 9])

    # 使用 TimeSeriesSplit 进行时间序列分割，设置分割数为 2，间隔为 2，测试集大小为 3
    splits = TimeSeriesSplit(n_splits=2, gap=2, test_size=3).split(X)

    # 获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [0, 1]
    assert_array_equal(train, [0, 1])
    # 断言测试集是否为 [4, 5, 6]
    assert_array_equal(test, [4, 5, 6])

    # 再次获取下一个分割的训练集和测试集
    train, test = next(splits)
    # 断言训练集是否为 [0, 1, 2, 3, 4]
    assert_array_equal(train, [0, 1, 2, 3, 4])
    # 断言测试集是否为 [7, 8, 9]
    assert_array_equal(test, [7, 8, 9])

    # 验证是否会抛出正确的错误
    with pytest.raises(ValueError, match="Too many splits.*and gap"):
        # 尝试使用 TimeSeriesSplit 进行过多的分割，预期会抛出 ValueError
        splits = TimeSeriesSplit(n_splits=4, gap=2).split(X)
        next(splits)
def test_shuffle_split_empty_trainset(CVSplitter):
    # 创建一个 CVSplitter 对象，设置测试集大小为 0.99
    cv = CVSplitter(test_size=0.99)
    X, y = [[1]], [0]  # 1 个样本
    # 使用 pytest 断言检测是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(
        ValueError,
        match=(
            "With n_samples=1, test_size=0.99 and train_size=None, "
            "the resulting train set will be empty"
        ),
    ):
        # 调用 _split 函数进行分割，期望抛出异常
        next(_split(cv, X, y, groups=[1]))


def test_train_test_split_empty_trainset():
    (X,) = [[1]]  # 1 个样本
    # 使用 pytest 断言检测是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(
        ValueError,
        match=(
            "With n_samples=1, test_size=0.99 and train_size=None, "
            "the resulting train set will be empty"
        ),
    ):
        # 调用 train_test_split 函数进行分割，期望抛出异常
        train_test_split(X, test_size=0.99)

    X = [[1], [1], [1]]  # 3 个样本，要求超过 2/3 的测试集
    # 使用 pytest 断言检测是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(
        ValueError,
        match=(
            "With n_samples=3, test_size=0.67 and train_size=None, "
            "the resulting train set will be empty"
        ),
    ):
        # 再次调用 train_test_split 函数进行分割，期望抛出异常
        train_test_split(X, test_size=0.67)


def test_leave_one_out_empty_trainset():
    # 创建一个 LeaveOneOut 的交叉验证对象
    cv = LeaveOneOut()
    X, y = [[1]], [0]  # 1 个样本
    # 使用 pytest 断言检测是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(ValueError, match="Cannot perform LeaveOneOut with n_samples=1"):
        # 调用 split 函数进行分割，期望抛出异常
        next(cv.split(X, y))


def test_leave_p_out_empty_trainset():
    # 创建一个 LeavePOut 的交叉验证对象，设置 p=2
    cv = LeavePOut(p=2)
    X, y = [[1], [2]], [0, 3]  # 2 个样本
    # 使用 pytest 断言检测是否会抛出 ValueError 异常，并匹配特定错误消息
    with pytest.raises(
        ValueError, match="p=2 must be strictly less than the number of samples=2"
    ):
        # 调用 split 函数进行分割，期望抛出异常
        next(cv.split(X, y))


@pytest.mark.parametrize("Klass", (KFold, StratifiedKFold, StratifiedGroupKFold))
def test_random_state_shuffle_false(Klass):
    # 当 shuffle=False 时传入非默认的 random_state 没有意义
    with pytest.raises(ValueError, match="has no effect since shuffle is False"):
        # 使用参数化测试，检测是否会抛出 ValueError 异常，并匹配特定错误消息
        Klass(3, shuffle=False, random_state=0)


@pytest.mark.parametrize(
    "cv, expected",
    # 创建一个包含不同交叉验证策略及其参数的列表
    [
        # 使用默认参数创建 KFold 交叉验证对象，并指定需要打乱数据
        (KFold(), True),
        # 创建具有指定参数（打乱数据和随机种子）的 KFold 交叉验证对象
        (KFold(shuffle=True, random_state=123), True),
        # 使用默认参数创建 StratifiedKFold 交叉验证对象，并指定需要打乱数据
        (StratifiedKFold(), True),
        # 创建具有指定参数（打乱数据和随机种子）的 StratifiedKFold 交叉验证对象
        (StratifiedKFold(shuffle=True, random_state=123), True),
        # 创建具有指定参数（打乱数据和随机种子）的 StratifiedGroupKFold 交叉验证对象
        (StratifiedGroupKFold(shuffle=True, random_state=123), True),
        # 使用默认参数创建 StratifiedGroupKFold 交叉验证对象，并指定需要打乱数据
        (StratifiedGroupKFold(), True),
        # 创建具有指定参数（随机种子）的 RepeatedKFold 交叉验证对象
        (RepeatedKFold(random_state=123), True),
        # 创建具有指定参数（随机种子）的 RepeatedStratifiedKFold 交叉验证对象
        (RepeatedStratifiedKFold(random_state=123), True),
        # 创建具有指定参数（随机种子）的 ShuffleSplit 交叉验证对象
        (ShuffleSplit(random_state=123), True),
        # 创建具有指定参数（随机种子）的 GroupShuffleSplit 交叉验证对象
        (GroupShuffleSplit(random_state=123), True),
        # 创建具有指定参数（随机种子）的 StratifiedShuffleSplit 交叉验证对象
        (StratifiedShuffleSplit(random_state=123), True),
        # 使用默认参数创建 GroupKFold 交叉验证对象，并指定需要打乱数据
        (GroupKFold(), True),
        # 使用默认参数创建 TimeSeriesSplit 交叉验证对象，并指定需要打乱数据
        (TimeSeriesSplit(), True),
        # 使用默认参数创建 LeaveOneOut 交叉验证对象，并指定需要打乱数据
        (LeaveOneOut(), True),
        # 使用默认参数创建 LeaveOneGroupOut 交叉验证对象，并指定需要打乱数据
        (LeaveOneGroupOut(), True),
        # 创建具有指定参数（分组数量）的 LeavePGroupsOut 交叉验证对象
        (LeavePGroupsOut(n_groups=2), True),
        # 创建具有指定参数（留出样本数量）的 LeavePOut 交叉验证对象
        (LeavePOut(p=2), True),
        # 创建具有指定参数（打乱数据但不使用随机种子）的 KFold 交叉验证对象
        (KFold(shuffle=True, random_state=None), False),
        # 创建具有指定参数（打乱数据但不使用随机种子）的 KFold 交叉验证对象
        (KFold(shuffle=True, random_state=None), False),
        # 创建具有指定参数（打乱数据和随机种子对象）的 StratifiedKFold 交叉验证对象
        (StratifiedKFold(shuffle=True, random_state=np.random.RandomState(0)), False),
        # 创建具有指定参数（打乱数据和随机种子对象）的 StratifiedKFold 交叉验证对象
        (StratifiedKFold(shuffle=True, random_state=np.random.RandomState(0)), False),
        # 创建具有指定参数（随机种子对象）的 RepeatedKFold 交叉验证对象
        (RepeatedKFold(random_state=None), False),
        # 创建具有指定参数（随机种子对象）的 RepeatedKFold 交叉验证对象
        (RepeatedKFold(random_state=np.random.RandomState(0)), False),
        # 创建具有指定参数（随机种子对象）的 RepeatedStratifiedKFold 交叉验证对象
        (RepeatedStratifiedKFold(random_state=None), False),
        # 创建具有指定参数（随机种子对象）的 RepeatedStratifiedKFold 交叉验证对象
        (RepeatedStratifiedKFold(random_state=np.random.RandomState(0)), False),
        # 创建具有指定参数（随机种子对象）的 ShuffleSplit 交叉验证对象
        (ShuffleSplit(random_state=None), False),
        # 创建具有指定参数（随机种子对象）的 ShuffleSplit 交叉验证对象
        (ShuffleSplit(random_state=np.random.RandomState(0)), False),
        # 创建具有指定参数（随机种子对象）的 GroupShuffleSplit 交叉验证对象
        (GroupShuffleSplit(random_state=None), False),
        # 创建具有指定参数（随机种子对象）的 GroupShuffleSplit 交叉验证对象
        (GroupShuffleSplit(random_state=np.random.RandomState(0)), False),
        # 创建具有指定参数（随机种子对象）的 StratifiedShuffleSplit 交叉验证对象
        (StratifiedShuffleSplit(random_state=None), False),
        # 创建具有指定参数（随机种子对象）的 StratifiedShuffleSplit 交叉验证对象
        (StratifiedShuffleSplit(random_state=np.random.RandomState(0)), False),
    ],
# 测试确保 _yields_constant_splits 函数返回预期的常数分割结果
def test_yields_constant_splits(cv, expected):
    assert _yields_constant_splits(cv) == expected


# 使用参数化测试，对所有的分割器进行测试，并使用分割器的字符串表示作为标识符
@pytest.mark.parametrize("cv", ALL_SPLITTERS, ids=[str(cv) for cv in ALL_SPLITTERS])
def test_splitter_get_metadata_routing(cv):
    """检查 get_metadata_routing 方法返回正确的 MetadataRouter。"""
    assert hasattr(cv, "get_metadata_routing")
    # 获取元数据路由对象
    metadata = cv.get_metadata_routing()
    if cv in GROUP_SPLITTERS:
        # 对于支持分组的分割器，确保分割请求中包含 "groups" 字段
        assert metadata.split.requests["groups"] is True
    elif cv in NO_GROUP_SPLITTERS:
        # 对于不支持分组的分割器，确保分割请求为空
        assert not metadata.split.requests

    # 确保除了 "split" 外的其他请求为空
    assert_request_is_empty(metadata, exclude=["split"])


# 使用参数化测试，对所有的分割器进行测试，并使用分割器的字符串表示作为标识符
@pytest.mark.parametrize("cv", ALL_SPLITTERS, ids=[str(cv) for cv in ALL_SPLITTERS])
def test_splitter_set_split_request(cv):
    """检查 set_split_request 方法在组分割器中定义，在其他分割器中未定义。"""
    if cv in GROUP_SPLITTERS:
        assert hasattr(cv, "set_split_request")
    elif cv in NO_GROUP_SPLITTERS:
        assert not hasattr(cv, "set_split_request")


# 使用参数化测试，对所有不支持分组的分割器进行测试
@pytest.mark.parametrize("cv", NO_GROUP_SPLITTERS, ids=str)
def test_no_group_splitters_warns_with_groups(cv):
    # 准备警告消息，说明分组参数在特定的分割器中被忽略
    msg = f"The groups parameter is ignored by {cv.__class__.__name__}"

    # 准备测试数据
    n_samples = 30
    rng = np.random.RandomState(1)
    X = rng.randint(0, 3, size=(n_samples, 2))
    y = rng.randint(0, 3, size=(n_samples,))
    groups = rng.randint(0, 3, size=(n_samples,))

    # 使用 pytest 的 warn 断言，确保特定分割器在使用 groups 参数时发出用户警告
    with pytest.warns(UserWarning, match=msg):
        cv.split(X, y, groups=groups)
```