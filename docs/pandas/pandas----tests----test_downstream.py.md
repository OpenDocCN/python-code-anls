# `D:\src\scipysrc\pandas\pandas\tests\test_downstream.py`

```
"""
Testing that we work in the downstream packages
"""

# 导入所需的模块和库
import array  # 导入array模块
from functools import partial  # 导入functools模块中的partial函数
import subprocess  # 导入subprocess模块
import sys  # 导入sys模块

import numpy as np  # 导入numpy库并使用np作为别名
import pytest  # 导入pytest库

from pandas.errors import IntCastingNaNError  # 从pandas.errors中导入IntCastingNaNError异常类

import pandas as pd  # 导入pandas库并使用pd作为别名
from pandas import (  # 从pandas中导入多个类
    DataFrame,
    DatetimeIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm  # 导入pandas._testing模块并使用tm作为别名


@pytest.fixture
def df():
    return DataFrame({"A": [1, 2, 3]})


def test_dask(df):
    # dask设置"compute.use_numexpr"为False，因此捕获当前值
    # 并确保在测试结束后重置它，以避免影响其他测试
    olduse = pd.get_option("compute.use_numexpr")

    try:
        pytest.importorskip("toolz")  # 导入toolz模块，如果导入失败则跳过该测试
        dd = pytest.importorskip("dask.dataframe")  # 导入dask.dataframe模块，如果导入失败则跳过该测试

        ddf = dd.from_pandas(df, npartitions=3)  # 将DataFrame转换为Dask DataFrame
        assert ddf.A is not None  # 断言确保ddf的A列不为None
        assert ddf.compute() is not None  # 断言确保计算ddf后结果不为None
    finally:
        pd.set_option("compute.use_numexpr", olduse)  # 恢复"compute.use_numexpr"的原始设置


# TODO(CoW) see https://github.com/pandas-dev/pandas/pull/51082
@pytest.mark.skip(reason="not implemented with CoW")
def test_dask_ufunc():
    # dask设置"compute.use_numexpr"为False，因此捕获当前值
    # 并确保在测试结束后重置它，以避免影响其他测试
    olduse = pd.get_option("compute.use_numexpr")

    try:
        da = pytest.importorskip("dask.array")  # 导入dask.array模块，如果导入失败则跳过该测试
        dd = pytest.importorskip("dask.dataframe")  # 导入dask.dataframe模块，如果导入失败则跳过该测试

        s = Series([1.5, 2.3, 3.7, 4.0])  # 创建一个Series对象
        ds = dd.from_pandas(s, npartitions=2)  # 将Series转换为Dask DataFrame

        result = da.log(ds).compute()  # 计算Dask数组的对数并获取结果
        expected = np.log(s)  # 获取NumPy数组的对数作为期望结果
        tm.assert_series_equal(result, expected)  # 使用测试工具断言结果与期望值相等
    finally:
        pd.set_option("compute.use_numexpr", olduse)  # 恢复"compute.use_numexpr"的原始设置


def test_construct_dask_float_array_int_dtype_match_ndarray():
    # GH#40110 确保我们像对待ndarray一样对待float-dtype dask数组
    dd = pytest.importorskip("dask.dataframe")  # 导入dask.dataframe模块，如果导入失败则跳过该测试

    arr = np.array([1, 2.5, 3])  # 创建一个NumPy数组
    darr = dd.from_array(arr)  # 将NumPy数组转换为Dask数组

    res = Series(darr)  # 使用Dask数组创建一个Series对象
    expected = Series(arr)  # 使用NumPy数组创建一个Series对象作为期望值
    tm.assert_series_equal(res, expected)  # 使用测试工具断言结果与期望值相等

    # GH#49599 在2.0版本中，我们抛出异常而不是默默地忽略dtype不匹配的情况
    msg = "Trying to coerce float values to integers"
    with pytest.raises(ValueError, match=msg):
        Series(darr, dtype="i8")  # 尝试使用dtype="i8"创建Series对象，预期会引发异常

    msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
    arr[2] = np.nan  # 将数组的第三个元素设置为NaN
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(darr, dtype="i8")  # 使用dtype="i8"创建Series对象，预期会引发异常

    # 这与使用NumPy数组作为输入时的情况相同
    with pytest.raises(IntCastingNaNError, match=msg):
        Series(arr, dtype="i8")  # 使用dtype="i8"创建Series对象，预期会引发异常


def test_xarray(df):
    pytest.importorskip("xarray")  # 导入xarray模块，如果导入失败则跳过该测试

    assert df.to_xarray() is not None  # 断言确保DataFrame转换为xarray对象后不为None


def test_xarray_cftimeindex_nearest():
    # https://github.com/pydata/xarray/issues/3751
    cftime = pytest.importorskip("cftime")  # 导入cftime模块，如果导入失败则跳过该测试
    xarray = pytest.importorskip("xarray")  # 导入xarray模块，如果导入失败则跳过该测试

    times = xarray.cftime_range("0001", periods=2)  # 创建一个cftime时间范围对象
    key = cftime.DatetimeGregorian(2000, 1, 1)  # 创建一个cftime日期时间对象
    result = times.get_indexer([key], method="nearest")  # 使用nearest方法获取最近索引
    expected = 1  # 预期的结果为1
    # 使用断言检查变量 result 是否等于变量 expected
    assert result == expected
# 标记为单CPU环境的测试函数，用于检验是否可以优化
@pytest.mark.single_cpu
def test_oo_optimizable():
    # GH 21071
    # 执行一个子进程，使用-O标志运行Python解释器，加载pandas库
    subprocess.check_call([sys.executable, "-OO", "-c", "import pandas"])


# 标记为单CPU环境的测试函数，检验优化后的日期时间索引的反序列化
@pytest.mark.single_cpu
def test_oo_optimized_datetime_index_unpickle():
    # GH 42866
    # 执行一个子进程，使用-O标志运行Python解释器，序列化和反序列化日期时间索引
    subprocess.check_call(
        [
            sys.executable,
            "-OO",
            "-c",
            (
                "import pandas as pd, pickle; "
                "pickle.loads(pickle.dumps(pd.date_range('2021-01-01', periods=1)))"
            ),
        ]
    )


# 测试statsmodels库的线性回归功能
def test_statsmodels():
    smf = pytest.importorskip("statsmodels.formula.api")

    # 创建一个DataFrame对象，包含彩票、文化程度和人口数据
    df = DataFrame(
        {"Lottery": range(5), "Literacy": range(5), "Pop1831": range(100, 105)}
    )
    # 使用OLS方法拟合线性模型
    smf.ols("Lottery ~ Literacy + np.log(Pop1831)", data=df).fit()


# 测试scikit-learn库的支持向量机分类器
def test_scikit_learn():
    pytest.importorskip("sklearn")
    from sklearn import (
        datasets,
        svm,
    )

    # 加载手写数字数据集
    digits = datasets.load_digits()
    # 使用SVC模型拟合数据
    clf = svm.SVC(gamma=0.001, C=100.0)
    clf.fit(digits.data[:-1], digits.target[:-1])
    # 预测新的数据样本
    clf.predict(digits.data[-1:])


# 测试seaborn库的数据可视化功能
def test_seaborn(mpl_cleanup):
    seaborn = pytest.importorskip("seaborn")
    # 创建一个包含日期和账单总额的DataFrame对象
    tips = DataFrame(
        {"day": pd.date_range("2023", freq="D", periods=5), "total_bill": range(5)}
    )
    # 使用stripplot函数绘制天数与账单总额的散点图
    seaborn.stripplot(x="day", y="total_bill", data=tips)


# 测试pandas_datareader库的数据读取功能
def test_pandas_datareader():
    pytest.importorskip("pandas_datareader")


# 测试pyarrow库的数据转换功能
@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_pyarrow(df):
    pyarrow = pytest.importorskip("pyarrow")
    # 将DataFrame对象转换为pyarrow.Table对象
    table = pyarrow.Table.from_pandas(df)
    # 将pyarrow.Table对象转换回DataFrame对象
    result = table.to_pandas()
    # 断言转换后的DataFrame与原始DataFrame相等
    tm.assert_frame_equal(result, df)


# 测试yaml库的数据序列化和反序列化功能
def test_yaml_dump(df):
    # GH#42748
    yaml = pytest.importorskip("yaml")

    # 将DataFrame对象转换为YAML格式字符串
    dumped = yaml.dump(df)

    # 从YAML格式字符串反序列化出DataFrame对象
    loaded = yaml.load(dumped, Loader=yaml.Loader)
    tm.assert_frame_equal(df, loaded)

    # 使用不安全的加载器反序列化YAML格式字符串为DataFrame对象
    loaded2 = yaml.load(dumped, Loader=yaml.UnsafeLoader)
    tm.assert_frame_equal(df, loaded2)


# 检验是否缺少必要的依赖
@pytest.mark.single_cpu
def test_missing_required_dependency():
    # GH 23868
    # 为确保正确的隔离，传递以下标志
    # -S : 禁用site-packages
    # -s : 禁用用户site-packages
    # -E : 禁用PYTHON*环境变量，特别是PYTHONPATH
    # https://github.com/MacPython/pandas-wheels/pull/50

    pyexe = sys.executable.replace("\\", "/")

    # 如果pandas作为site package安装，则跳过该测试
    call = [pyexe, "-c", "import pandas;print(pandas.__file__)"]
    output = subprocess.check_output(call).decode()
    if "site-packages" in output:
        pytest.skip("pandas installed as site package")

    # 如果pandas作为site package安装，该测试将失败
    # 标志阻止导入pandas，测试将报告“Failed: DID NOT RAISE <class 'subprocess.CalledProcessError'>”
    call = [pyexe, "-sSE", "-c", "import pandas"]
    # 定义包含特定错误消息的字符串，使用原始字符串标记，以避免反斜杠转义字符被处理
    msg = (
        rf"Command '\['{pyexe}', '-sSE', '-c', 'import pandas'\]' "
        "returned non-zero exit status 1."
    )
    
    # 使用 pytest 模块中的 pytest.raises 上下文管理器捕获 subprocess.CalledProcessError 异常，并验证异常消息匹配预期的 msg
    with pytest.raises(subprocess.CalledProcessError, match=msg) as exc:
        # 执行系统命令 call，并将标准错误输出合并到标准输出中
        subprocess.check_output(call, stderr=subprocess.STDOUT)
    
    # 从捕获的异常对象中解码标准输出，并赋值给 output 变量
    output = exc.value.stdout.decode()
    
    # 遍历预定义的名称列表 ["numpy", "pytz", "dateutil"]
    for name in ["numpy", "pytz", "dateutil"]:
        # 断言每个名称存在于输出中
        assert name in output
# 定义一个测试函数，用于测试将 dask 数组中的数据设置到新列中
def test_frame_setitem_dask_array_into_new_col():
    # GH#47128

    # 获取当前 "compute.use_numexpr" 的选项值，并保存为旧值
    olduse = pd.get_option("compute.use_numexpr")

    try:
        # 导入 dask.array 模块，如果导入失败则跳过此测试
        da = pytest.importorskip("dask.array")

        # 创建 dask 数组对象
        dda = da.array([1, 2])
        # 创建一个 DataFrame 对象
        df = DataFrame({"a": ["a", "b"]})
        # 将 dask 数组设置到 DataFrame 的新列 'b' 和 'c' 中
        df["b"] = dda
        df["c"] = dda
        # 使用布尔索引选择部分行，并将 'b' 列中的值设置为 100
        df.loc[[False, True], "b"] = 100
        # 选择 DataFrame 中索引为 1 的行作为结果
        result = df.loc[[1], :]
        # 创建预期的 DataFrame 对象，以便与结果进行比较
        expected = DataFrame({"a": ["b"], "b": [100], "c": [2]}, index=[1])
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)
    finally:
        # 恢复旧的 "compute.use_numexpr" 选项值
        pd.set_option("compute.use_numexpr", olduse)


# 定义一个测试函数，用于测试 pandas 的优先级设置
def test_pandas_priority():
    # GH#48347

    # 定义一个类 MyClass，并设置 __pandas_priority__ 属性为 5000
    class MyClass:
        __pandas_priority__ = 5000

        # 定义 __radd__ 方法，用于反向加法操作
        def __radd__(self, other):
            return self

    # 创建 MyClass 的实例对象 left
    left = MyClass()
    # 创建 Series 对象 right，其值为 0、1、2
    right = Series(range(3))

    # 断言 right.__add__(left) 操作未实现
    assert right.__add__(left) is NotImplemented
    # 断言 right + left 操作的结果为 left 对象本身
    assert right + left is left


# 使用参数化装饰器定义一个测试函数，用于测试从不同类型的数组创建 pandas 对象
@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
@pytest.mark.parametrize(
    "box", [memoryview, partial(array.array, "i"), "dask", "xarray"]
)
def test_from_obscure_array(dtype, box):
    # GH#24539 recognize e.g xarray, dask, ...
    # Note: we dont do this for PeriodArray bc _from_sequence won't accept
    #  an array of integers
    # TODO: could check with arraylike of Period objects
    # GH#24539 recognize e.g xarray, dask, ...
    
    # 创建一个 numpy 数组 arr，其中包含整数 1、2、3
    arr = np.array([1, 2, 3], dtype=np.int64)
    
    # 根据 box 的值选择不同的操作
    if box == "dask":
        # 导入 dask.array 模块，如果导入失败则跳过此测试
        da = pytest.importorskip("dask.array")
        # 使用 dask.array 创建数据对象 data
        data = da.array(arr)
    elif box == "xarray":
        # 导入 xarray 模块，如果导入失败则跳过此测试
        xr = pytest.importorskip("xarray")
        # 使用 xarray.DataArray 创建数据对象 data
        data = xr.DataArray(arr)
    else:
        # 使用 box 函数创建数据对象 data
        data = box(arr)

    # 如果数据对象 data 不是 memoryview 类型
    if not isinstance(data, memoryview):
        # 根据 dtype 类型选择不同的函数操作
        func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]
        # 调用函数 func 处理数组 arr，将结果保存到 result 中
        result = func(arr).array
        # 调用函数 func 处理数据对象 data，将结果保存到 expected 中
        expected = func(data).array
        # 断言结果 result 与预期 expected 相等
        tm.assert_equal(result, expected)

    # 检查 Indexes 是否正确
    # 根据 dtype 类型选择不同的 Index 类
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]
    # 使用数组 arr 创建索引对象 result
    result = idx_cls(arr)
    # 使用数据对象 data 创建索引对象 expected
    expected = idx_cls(data)
    # 断言 Indexes result 和 expected 相等
    tm.assert_index_equal(result, expected)


# 定义一个测试函数，用于测试 xarray 数据类型的时间单位转换
def test_xarray_coerce_unit():
    # GH44053
    # 导入 xarray 模块，如果导入失败则跳过此测试
    xr = pytest.importorskip("xarray")

    # 创建 xarray.DataArray 对象 arr，其中包含整数 1、2、3
    arr = xr.DataArray([1, 2, 3])
    # 使用 pd.to_datetime 将 arr 转换为 DatetimeIndex，单位为 'ns'，保存为 result
    result = pd.to_datetime(arr, unit="ns")
    # 创建预期的 DatetimeIndex 对象 expected
    expected = DatetimeIndex(
        [
            "1970-01-01 00:00:00.000000001",
            "1970-01-01 00:00:00.000000002",
            "1970-01-01 00:00:00.000000003",
        ],
        dtype="datetime64[ns]",
        freq=None,
    )
    # 断言 result 和 expected 的索引相等
    tm.assert_index_equal(result, expected)
```