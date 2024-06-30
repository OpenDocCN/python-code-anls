# `D:\src\scipysrc\seaborn\tests\conftest.py`

```
# 导入 NumPy 库并使用 np 别名
import numpy as np
# 导入 Pandas 库并使用 pd 别名
import pandas as pd

# 导入 pytest 库
import pytest

# 自动使用的测试夹具，用于关闭所有 matplotlib 图形
@pytest.fixture(autouse=True)
def close_figs():
    # 生成器函数，暂时挂起直到测试完成后再继续执行
    yield
    # 导入 matplotlib.pyplot 库并关闭所有图形
    import matplotlib.pyplot as plt
    plt.close("all")

# 自动使用的测试夹具，设定随机种子
@pytest.fixture(autouse=True)
def random_seed():
    # 使用字符串 "seaborn random global" 计算种子值
    seed = sum(map(ord, "seaborn random global"))
    # 使用 NumPy 设置随机种子
    np.random.seed(seed)

# 测试夹具，返回一个随机数生成器对象
@pytest.fixture()
def rng():
    # 使用字符串 "seaborn random object" 计算种子值
    seed = sum(map(ord, "seaborn random object"))
    # 返回一个基于 NumPy 的随机数生成器对象
    return np.random.RandomState(seed)

# 测试夹具，返回一个包含随机数的宽数据框 DataFrame 对象
@pytest.fixture
def wide_df(rng):
    # 列名为 'abc'
    columns = list("abc")
    # 索引为 10 到 50（步长为 2），名称为 'wide_index'
    index = pd.RangeIndex(10, 50, 2, name="wide_index")
    # 使用随机数生成器生成正态分布随机数填充的数据
    values = rng.normal(size=(len(index), len(columns)))
    # 返回一个 Pandas DataFrame 对象
    return pd.DataFrame(values, index=index, columns=columns)

# 测试夹具，返回一个宽数组（基于宽数据框 DataFrame 对象）
@pytest.fixture
def wide_array(wide_df):
    # 将宽数据框转换为 NumPy 数组并返回
    return wide_df.to_numpy()

# 测试夹具，返回一个扁平化的序列 Series 对象
@pytest.fixture
def flat_series(rng):
    # 索引为 10 到 30，名称为 't'
    index = pd.RangeIndex(10, 30, name="t")
    # 返回一个带有正态分布随机数的 Pandas Series 对象
    return pd.Series(rng.normal(size=20), index, name="s")

# 测试夹具，返回一个扁平化的数组（基于扁平化序列 Series 对象）
@pytest.fixture
def flat_array(flat_series):
    # 将扁平化序列转换为 NumPy 数组并返回
    return flat_series.to_numpy()

# 测试夹具，返回一个扁平化的列表（基于扁平化序列 Series 对象）
@pytest.fixture
def flat_list(flat_series):
    # 将扁平化序列转换为 Python 列表并返回
    return flat_series.to_list()

# 测试夹具，根据参数返回不同形式的扁平化数据（序列、数组或列表）
@pytest.fixture(params=["series", "array", "list"])
def flat_data(rng, request):
    # 索引为 10 到 30，名称为 't'
    index = pd.RangeIndex(10, 30, name="t")
    # 返回不同形式的扁平化数据，根据参数决定返回序列、数组或列表
    series = pd.Series(rng.normal(size=20), index, name="s")
    if request.param == "series":
        data = series
    elif request.param == "array":
        data = series.to_numpy()
    elif request.param == "list":
        data = series.to_list()
    return data

# 测试夹具，返回包含两个系列对象的宽列表
@pytest.fixture
def wide_list_of_series(rng):
    # 返回包含两个正态分布随机数系列对象的列表
    return [pd.Series(rng.normal(size=20), np.arange(20), name="a"),
            pd.Series(rng.normal(size=10), np.arange(5, 15), name="b")]

# 测试夹具，返回包含两个数组的宽列表（基于宽列表的系列对象）
@pytest.fixture
def wide_list_of_arrays(wide_list_of_series):
    # 将宽列表的每个系列对象转换为 NumPy 数组并返回列表
    return [s.to_numpy() for s in wide_list_of_series]

# 测试夹具，返回包含两个列表的宽列表（基于宽列表的系列对象）
@pytest.fixture
def wide_list_of_lists(wide_list_of_series):
    # 将宽列表的每个系列对象转换为 Python 列表并返回列表
    return [s.to_list() for s in wide_list_of_series]

# 测试夹具，返回包含两个系列对象的宽字典（键为系列名称）
@pytest.fixture
def wide_dict_of_series(wide_list_of_series):
    # 将宽列表的每个系列对象转换为字典（键为系列名称）并返回
    return {s.name: s for s in wide_list_of_series}

# 测试夹具，返回包含两个数组的宽字典（键为系列名称）
@pytest.fixture
def wide_dict_of_arrays(wide_list_of_series):
    # 将宽列表的每个系列对象转换为 NumPy 数组，并以字典形式返回（键为系列名称）
    return {s.name: s.to_numpy() for s in wide_list_of_series}

# 测试夹具，返回包含两个列表的宽字典（键为系列名称）
@pytest.fixture
def wide_dict_of_lists(wide_list_of_series):
    # 将宽列表的每个系列对象转换为 Python 列表，并以字典形式返回（键为系列名称）
    return {s.name: s.to_list() for s in wide_list_of_series}

# 测试夹具，返回一个包含多种类型数据的长数据框 DataFrame 对象
@pytest.fixture
def long_df(rng):
    n = 100
    # 使用随机数生成器生成具有多种数据类型的数据框 DataFrame 对象
    df = pd.DataFrame(dict(
        x=rng.uniform(0, 20, n).round().astype("int"),
        y=rng.normal(size=n),
        z=rng.lognormal(size=n),
        a=rng.choice(list("abc"), n),
        b=rng.choice(list("mnop"), n),
        c=rng.choice([0, 1], n, [.3, .7]),
        d=rng.choice(np.arange("2004-07-30", "2007-07-30", dtype="datetime64[Y]"), n),
        t=rng.choice(np.arange("2004-07-30", "2004-07-31", dtype="datetime64[m]"), n),
        s=rng.choice([2, 4, 8], n),
        f=rng.choice([0.2, 0.3], n),
    ))

    # 将 'a' 列转换为分类数据类型，并重新排序分类
    a_cat = df["a"].astype("category")
    new_categories = np.roll(a_cat.cat.categories, 1)
    df["a_cat"] = a_cat.cat.reorder_categories(new_categories)
    # 将 DataFrame 中的列 "s" 转换为分类类型，并赋值给新列 "s_cat"
    df["s_cat"] = df["s"].astype("category")
    
    # 将 DataFrame 中的列 "s" 转换为字符串类型，并赋值给新列 "s_str"
    df["s_str"] = df["s"].astype(str)
    
    # 返回处理后的 DataFrame
    return df
# 使用 pytest.fixture 装饰器定义一个名为 long_dict 的测试固件，接受 long_df 作为参数
@pytest.fixture
def long_dict(long_df):
    # 将 long_df 转换为字典格式并返回
    return long_df.to_dict()


# 使用 pytest.fixture 装饰器定义一个名为 repeated_df 的测试固件，接受 rng 作为参数
@pytest.fixture
def repeated_df(rng):
    # 设置 DataFrame 的大小为 100 行
    n = 100
    # 创建 DataFrame，包含四列：x 是重复的 0 到 49，y 是随机正态分布数据，a 是从 "abc" 中选择的字符，u 是重复的 0 和 1
    return pd.DataFrame(dict(
        x=np.tile(np.arange(n // 2), 2),
        y=rng.normal(size=n),
        a=rng.choice(list("abc"), n),
        u=np.repeat(np.arange(2), n // 2),
    ))


# 使用 pytest.fixture 装饰器定义一个名为 null_df 的测试固件，接受 rng 和 long_df 作为参数
@pytest.fixture
def null_df(rng, long_df):
    # 复制 long_df 到 df
    df = long_df.copy()
    # 遍历 df 的每一列
    for col in df:
        # 如果列 col 是整数类型，则转换为浮点数
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        # 随机打乱 df 的索引并选择前 10 个索引
        idx = rng.permutation(df.index)[:10]
        # 在选定的索引位置上将列 col 的值设置为 NaN
        df.loc[idx, col] = np.nan
    # 返回修改后的 DataFrame df
    return df


# 使用 pytest.fixture 装饰器定义一个名为 object_df 的测试固件，接受 rng 和 long_df 作为参数
@pytest.fixture
def object_df(rng, long_df):
    # 复制 long_df 到 df
    df = long_df.copy()
    # 将列 "c", "s", "f" 的数据类型转换为对象类型
    for col in ["c", "s", "f"]:
        df[col] = df[col].astype(object)
    # 返回修改后的 DataFrame df
    return df


# 使用 pytest.fixture 装饰器定义一个名为 null_series 的测试固件，接受 flat_series 作为参数
@pytest.fixture
def null_series(flat_series):
    # 返回一个与 flat_series 索引相同、数据类型为 'float64' 的 Series
    return pd.Series(index=flat_series.index, dtype='float64')


# 定义一个 MockInterchangeableDataFrame 类，模拟一个不是 pandas.DataFrame 的对象，但可以通过 DataFrame 交换协议转换为 DataFrame
class MockInterchangeableDataFrame:
    # 初始化方法接受 data 参数
    def __init__(self, data):
        self._data = data

    # 定义 __dataframe__ 方法，用于将对象转换为 DataFrame
    def __dataframe__(self, *args, **kwargs):
        return self._data.__dataframe__(*args, **kwargs)


# 使用 pytest.fixture 装饰器定义一个名为 mock_long_df 的测试固件，接受 long_df 作为参数
@pytest.fixture
def mock_long_df(long_df):
    # 返回一个 MockInterchangeableDataFrame 实例，使用 long_df 初始化
    return MockInterchangeableDataFrame(long_df)
```