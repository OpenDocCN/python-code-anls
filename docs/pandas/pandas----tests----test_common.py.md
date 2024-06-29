# `D:\src\scipysrc\pandas\pandas\tests\test_common.py`

```
import collections
from functools import partial  # 导入 partial 函数，用于创建 partial 对象
import string  # 导入 string 模块，提供字符串相关的常量和函数
import subprocess  # 导入 subprocess 模块，用于执行外部命令
import sys  # 导入 sys 模块，提供对解释器的访问和控制

import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

import pandas as pd  # 导入 pandas 库，用于数据分析
from pandas import Series  # 从 pandas 中导入 Series 类
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块
from pandas.core import ops  # 导入 pandas 核心模块中的 ops 子模块，用于操作
import pandas.core.common as com  # 导入 pandas 核心模块中的 common 子模块，提供通用函数
from pandas.util.version import Version  # 从 pandas 的 util.version 模块导入 Version 类


def test_get_callable_name():
    getname = com.get_callable_name  # 获取 com 模块中的 get_callable_name 函数引用

    def fn(x):
        return x

    lambda_ = lambda x: x  # 创建一个匿名函数 lambda_
    part1 = partial(fn)  # 使用 partial 创建一个部分应用函数 part1
    part2 = partial(part1)  # 使用 partial 创建另一个部分应用函数 part2

    class somecall:
        def __call__(self):
            # This shouldn't actually get called below; somecall.__init__
            #  should.
            raise NotImplementedError

    assert getname(fn) == "fn"  # 测试 get_callable_name 函数对 fn 的命名
    assert getname(lambda_)  # 测试 get_callable_name 函数对 lambda_ 的命名
    assert getname(part1) == "fn"  # 测试 get_callable_name 函数对 part1 的命名
    assert getname(part2) == "fn"  # 测试 get_callable_name 函数对 part2 的命名
    assert getname(somecall()) == "somecall"  # 测试 get_callable_name 函数对 somecall 的命名
    assert getname(1) is None  # 测试 get_callable_name 函数对整数 1 的命名


def test_any_none():
    assert com.any_none(1, 2, 3, None)  # 测试 any_none 函数对参数包含 None 的情况
    assert not com.any_none(1, 2, 3, 4)  # 测试 any_none 函数对参数不包含 None 的情况


def test_all_not_none():
    assert com.all_not_none(1, 2, 3, 4)  # 测试 all_not_none 函数对所有参数都不为 None 的情况
    assert not com.all_not_none(1, 2, 3, None)  # 测试 all_not_none 函数对存在参数为 None 的情况
    assert not com.all_not_none(None, None, None, None)  # 测试 all_not_none 函数对所有参数都为 None 的情况


def test_random_state():
    # Check with seed
    state = com.random_state(5)  # 使用 com 模块中的 random_state 函数生成随机状态对象 state
    assert state.uniform() == np.random.RandomState(5).uniform()  # 验证 state 的 uniform 方法的结果与 np.random.RandomState(5) 一致

    # Check with random state object
    state2 = np.random.RandomState(10)  # 创建一个 numpy 的随机状态对象 state2
    assert com.random_state(state2).uniform() == np.random.RandomState(10).uniform()  # 验证 com.random_state 对 state2 的处理结果与 np.random.RandomState(10) 一致

    # check with no arg random state
    assert com.random_state() is np.random  # 验证未提供参数时，com.random_state 返回 np.random

    # check array-like
    # GH32503
    state_arr_like = np.random.default_rng(None).integers(
        0, 2**31, size=624, dtype="uint32"
    )  # 创建一个类似数组的随机状态 state_arr_like
    assert (
        com.random_state(state_arr_like).uniform()
        == np.random.RandomState(state_arr_like).uniform()
    )  # 验证 com.random_state 对 state_arr_like 的处理结果与 np.random.RandomState(state_arr_like) 一致

    # Check BitGenerators
    # GH32503
    assert (
        com.random_state(np.random.MT19937(3)).uniform()
        == np.random.RandomState(np.random.MT19937(3)).uniform()
    )  # 验证 com.random_state 对 numpy 的 BitGenerators 的处理结果与 np.random.RandomState 的结果一致
    assert (
        com.random_state(np.random.PCG64(11)).uniform()
        == np.random.RandomState(np.random.PCG64(11)).uniform()
    )  # 验证 com.random_state 对 numpy 的 BitGenerators 的处理结果与 np.random.RandomState 的结果一致

    # Error for floats or strings
    msg = (
        "random_state must be an integer, array-like, a BitGenerator, Generator, "
        "a numpy RandomState, or None"
    )  # 定义错误消息字符串
    with pytest.raises(ValueError, match=msg):  # 使用 pytest 检查 ValueError 异常并匹配特定消息
        com.random_state("test")  # 测试 com.random_state 对字符串参数的处理

    with pytest.raises(ValueError, match=msg):  # 使用 pytest 检查 ValueError 异常并匹配特定消息
        com.random_state(5.5)  # 测试 com.random_state 对浮点数参数的处理

@pytest.mark.parametrize(
    "left, right, expected",
    [
        # Case 1: Both series have the same name "x", expected to match on name.
        (Series([1], name="x"), Series([2], name="x"), "x"),
        # Case 2: Series have different names "x" and "y", expected not to match.
        (Series([1], name="x"), Series([2], name="y"), None),
        # Case 3: One series has no name, expected not to match.
        (Series([1]), Series([2], name="x"), None),
        # Case 4: One series has a name "x" and the other has no name, expected not to match.
        (Series([1], name="x"), Series([2]), None),
        # Case 5: One series has name "x" and the other is a list, expected to match on name.
        (Series([1], name="x"), [2], "x"),
        # Case 6: One element is a list, the other is a series with name "y", expected not to match.
        ([1], Series([2], name="y"), "y"),
        # matching NAs: Series with name np.nan and empty index with name np.nan, expected to match on name.
        (Series([1], name=np.nan), pd.Index([], name=np.nan), np.nan),
        # matching NAs: Series with name np.nan and empty index with name pd.NaT, expected not to match.
        (Series([1], name=np.nan), pd.Index([], name=pd.NaT), None),
        # matching NAs: Series with name pd.NA and empty index with name pd.NA, expected to match on name.
        (Series([1], name=pd.NA), pd.Index([], name=pd.NA), pd.NA),
        # tuple name GH#39757: Series with name np.int64(1) and empty index with tuple name (np.int64(1), np.int64(2)), expected not to match.
        (
            Series([1], name=np.int64(1)),
            pd.Index([], name=(np.int64(1), np.int64(2))),
            None,
        ),
        # tuple name GH#39757: Series and index both with tuple name (np.int64(1), np.int64(2)), expected to match on name.
        (
            Series([1], name=(np.int64(1), np.int64(2))),
            pd.Index([], name=(np.int64(1), np.int64(2))),
            (np.int64(1), np.int64(2)),
        ),
        # pytest.param: Series and index both with tuple name (np.float64("nan"), np.int64(2)), expected to match on name but marked as expected failure due to NA handling inside tuples not being checked.
        pytest.param(
            Series([1], name=(np.float64("nan"), np.int64(2))),
            pd.Index([], name=(np.float64("nan"), np.int64(2))),
            (np.float64("nan"), np.int64(2)),
            marks=pytest.mark.xfail(
                reason="Not checking for matching NAs inside tuples."
            ),
        ),
    ],
)

def test_maybe_match_name(left, right, expected):
    # 调用 _maybe_match_name 函数，比较 left 和 right 的匹配情况
    res = ops.common._maybe_match_name(left, right)
    # 断言结果要么是 expected，要么与 expected 相等
    assert res is expected or res == expected


def test_standardize_mapping():
    # 测试不允许未初始化的 defaultdict
    msg = r"to_dict\(\) only accepts initialized defaultdicts"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(collections.defaultdict)

    # 测试不允许非映射类型的子类型，这里使用空列表作为例子
    msg = "unsupported type: <class 'list'>"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping([])

    # 再次测试不允许非映射类型的子类型，这里使用列表类本身作为例子
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(list)

    fill = {"bad": "data"}
    # 测试标准化映射，传入一个字典
    assert com.standardize_mapping(fill) == dict

    # 将空字典作为实例传入测试标准化映射
    assert com.standardize_mapping({}) == dict

    dd = collections.defaultdict(list)
    # 测试标准化映射后返回的类型是否为 partial
    assert isinstance(com.standardize_mapping(dd), partial)


def test_git_version():
    # 测试 pandas 的 git 版本字符串长度是否为 40
    git_version = pd.__git_version__
    assert len(git_version) == 40
    # 测试 git 版本字符串是否只包含十六进制字符
    assert all(c in string.hexdigits for c in git_version)


def test_version_tag():
    # GH 21295
    version = Version(pd.__version__)
    try:
        # 测试版本号是否大于 "0.0.1"
        version > Version("0.0.1")
    except TypeError as err:
        raise ValueError(
            "No git tags exist, please sync tags between upstream and your repo"
        ) from err


@pytest.mark.parametrize(
    "obj", [(obj,) for obj in pd.__dict__.values() if callable(obj)]
)
def test_serializable(obj):
    # GH 35611
    # 对象序列化测试，使用 round_trip_pickle 函数进行序列化和反序列化
    unpickled = tm.round_trip_pickle(obj)
    # 断言序列化前后对象的类型是否一致
    assert type(obj) == type(unpickled)


class TestIsBoolIndexer:
    def test_non_bool_array_with_na(self):
        # 测试带有 NaN 值的非布尔数组，确保不触发异常
        arr = np.array(["A", "B", np.nan], dtype=object)
        assert not com.is_bool_indexer(arr)

    def test_list_subclass(self):
        # GH#42433
        # 测试列表的子类是否作为布尔索引器的正常情况
        class MyList(list):
            pass

        val = MyList(["a"])
        assert not com.is_bool_indexer(val)

        val = MyList([True])
        assert com.is_bool_indexer(val)

    def test_frozenlist(self):
        # GH#42461
        # 测试冻结列表是否作为布尔索引器的正常情况
        data = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data=data)

        frozen = df.index.names[1:]
        assert not com.is_bool_indexer(frozen)

        result = df[frozen]
        expected = df[[]]
        # 断言结果与期望的空 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("with_exception", [True, False])
def test_temp_setattr(with_exception):
    # GH#45954
    ser = Series(dtype=object)
    ser.name = "first"
    # 在任一情况下引发 ValueError，以满足 pytest.raises 的要求
    match = "Inside exception raised" if with_exception else "Outside exception raised"
    with pytest.raises(ValueError, match=match):
        with com.temp_setattr(ser, "name", "second"):
            assert ser.name == "second"
            if with_exception:
                raise ValueError("Inside exception raised")
        raise ValueError("Outside exception raised")
    # 断言：确保 ser 对象的名称为 "first"
    assert ser.name == "first"
@pytest.mark.single_cpu
def test_str_size():
    # 用于标识测试案例 GH#21758
    # 设置字符串变量 a
    a = "a"
    # 获取变量 a 的内存大小
    expected = sys.getsizeof(a)
    # 获取 Python 解释器的可执行文件路径，并将反斜杠替换为斜杠
    pyexe = sys.executable.replace("\\", "/")
    # 准备要执行的命令列表
    call = [
        pyexe,
        "-c",
        # 在子进程中执行的 Python 代码，测试字符串 a 的内存大小
        "a='a';import sys;sys.getsizeof(a);import pandas;print(sys.getsizeof(a));",
    ]
    # 执行命令，并获取其输出，截取最后的数值并去除换行符
    result = subprocess.check_output(call).decode()[-4:-1].strip("\n")
    # 断言子进程输出的结果与预期的内存大小相等
    assert int(result) == int(expected)
```