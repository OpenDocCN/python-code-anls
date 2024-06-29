# `D:\src\scipysrc\pandas\asv_bench\benchmarks\reshape.py`

```
# 导入 itertools 模块中的 product 函数，用于生成迭代器的笛卡尔积
# 导入 string 模块，包含字符串相关的常量和函数
from itertools import product
import string

# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 导入 pandas 库，并从中导入以下内容：
# - DataFrame: 用于操作和处理数据的二维标记数据结构
# - MultiIndex: 用于多层索引的数据结构
# - date_range: 用于生成日期范围的函数
# - melt: 用于数据的长格式转换
# - wide_to_long: 用于宽格式数据转换为长格式数据
from pandas import (
    DataFrame,
    MultiIndex,
    date_range,
    melt,
    wide_to_long,
)

# 从 pandas.api.types 模块中导入 CategoricalDtype 类型
from pandas.api.types import CategoricalDtype


# 定义一个名为 Melt 的类
class Melt:
    # 定义类变量 params，包含字符串 "float64" 和 "Float64"
    params = ["float64", "Float64"]
    # 定义类变量 param_names，包含字符串 "dtype"
    param_names = ["dtype"]

    # 定义实例方法 setup，接受参数 dtype
    def setup(self, dtype):
        # 创建一个 DataFrame 对象 self.df，包含以下内容：
        # - 100000 行，3 列的随机标准正态分布数据
        # - 列名分别为 "A", "B", "C"，数据类型为 dtype
        self.df = DataFrame(
            np.random.randn(100_000, 3), columns=["A", "B", "C"], dtype=dtype
        )
        # 为 DataFrame 对象 self.df 添加两列数据：
        # - 名为 "id1" 的 Series，包含10000个随机整数，范围为 [0, 10)
        # - 名为 "id2" 的 Series，包含10000个随机整数，范围为 [100, 1000)
        self.df["id1"] = pd.Series(np.random.randint(0, 10, 10000))
        self.df["id2"] = pd.Series(np.random.randint(100, 1000, 10000))

    # 定义实例方法 time_melt_dataframe，接受参数 dtype
    def time_melt_dataframe(self, dtype):
        # 调用 pandas 的 melt 函数，将 self.df 转换为长格式数据，
        # id_vars 参数为 ["id1", "id2"]
        melt(self.df, id_vars=["id1", "id2"])


# 定义一个名为 Pivot 的类
class Pivot:
    # 定义实例方法 setup，无参数
    def setup(self):
        # 创建一个包含 10000 个时间点的时间索引 index
        index = date_range("1/1/2000", periods=10000, freq="h")
        # 创建一个包含以下内容的字典 data：
        # - "value": 500000 个随机标准正态分布数据
        # - "variable": 重复的 0 到 49，每个重复10000次
        # - "date": 重复的时间索引 index 的值，每个重复50次
        data = {
            "value": np.random.randn(500000),
            "variable": np.arange(50).repeat(10000),
            "date": np.tile(index.values, 50),
        }
        # 创建 DataFrame 对象 self.df，包含 data 字典中的数据
        self.df = DataFrame(data)

    # 定义实例方法 time_reshape_pivot_time_series，无参数
    def time_reshape_pivot_time_series(self):
        # 使用 pandas 的 pivot 函数，将 self.df 转换为透视表形式，
        # index 参数为 "date"，columns 参数为 "variable"，values 参数为 "value"
        self.df.pivot(index="date", columns="variable", values="value")


# 定义一个名为 SimpleReshape 的类
class SimpleReshape:
    # 定义实例方法 setup，无参数
    def setup(self):
        # 创建一个包含多层索引的 DataFrame 对象 self.df，具体内容如下：
        # - 索引数组 arrays 包含两个数组：
        #   - 第一个数组是 [0, 1, ..., 99] 重复100次
        #   - 第二个数组是 [25, 26, ..., 99, 0, 1, ..., 24]，向左移动25个位置后重复100次
        # - 列数为4，包含10000行的随机标准正态分布数据
        arrays = [np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)]
        index = MultiIndex.from_arrays(arrays)
        self.df = DataFrame(np.random.randn(10000, 4), index=index)
        # 对 self.df 执行 unstack(1) 操作，将第二层索引展开为列
        self.udf = self.df.unstack(1)

    # 定义实例方法 time_stack，无参数
    def time_stack(self):
        # 对 self.udf 执行 stack 操作，将列索引展开为第二层索引
        self.udf.stack()

    # 定义实例方法 time_unstack，无参数
    def time_unstack(self):
        # 对 self.df 执行 unstack(1) 操作，将第二层索引展开为列
        self.df.unstack(1)


# 定义一个名为 ReshapeExtensionDtype 的类，继承自 ReshapeExtensionDtype
class ReshapeExtensionDtype(ReshapeExtensionDtype):
    # 定义类变量 params，包含字符串 "Int64" 和 "Float64"
    params = ["Int64", "Float64"]
    # 定义类变量 param_names，包含字符串 "dtype"
    param_names = ["dtype"]
    # 设置函数 `setup`，接受一个数据类型 `dtype` 作为参数
    def setup(self, dtype):
        # 创建一个包含大写字母 A 到 J 的索引对象
        lev = pd.Index(list("ABCDEFGHIJ"))
        # 创建一个包含从 0 到 999 的整数索引对象
        ri = pd.Index(range(1000))
        # 使用 lev 和 ri 创建一个多级索引对象 mi，级别分别为 "foo" 和 "bar"
        mi = MultiIndex.from_product([lev, ri], names=["foo", "bar"])

        # 生成一个包含 10000 个随机整数的 NumPy 数组，并将其转换为指定的数据类型
        values = np.random.randn(10_000).astype(int)

        # 使用 values、dtype 和 mi 创建一个 Pandas Series 对象 ser
        ser = pd.Series(values, dtype=dtype, index=mi)
        # 将 ser 按照 "bar" 索引列解堆叠，生成一个 DataFrame 对象 df
        df = ser.unstack("bar")
        # 检查 roundtrips 是否为真，即 df.stack().equals(ser)

        # 将 ser 和 df 分别存储到 self.ser 和 self.df 中
        self.ser = ser
        self.df = df
class Unstack:
    # 类属性，指定参数类型为"int"或"category"
    params = ["int", "category"]

    def setup(self, dtype):
        # 初始化变量m和n，设定初始值
        m = 100
        n = 1000

        # 创建多级索引对象，包含m个级别，每级两个标签
        levels = np.arange(m)
        index = MultiIndex.from_product([levels] * 2)
        columns = np.arange(n)

        if dtype == "int":
            # 如果dtype为"int"，创建m*m*n大小的数组，赋值给self.df
            values = np.arange(m * m * n).reshape(m * m, n)
            self.df = DataFrame(values, index, columns)
        else:
            # 如果dtype不为"int"，使用字符串字母生成n个分类数据的DataFrame
            # 这部分比整数分支慢大约20倍，因此减小数据规模以降低速度
            n = 50
            columns = columns[:n]
            indices = np.random.randint(0, 52, size=(m * m, n))
            values = np.take(list(string.ascii_letters), indices)
            values = [pd.Categorical(v) for v in values.T]

            self.df = DataFrame(dict(enumerate(values)), index, columns)

        # 将self.df的倒数第一行赋值给self.df2
        self.df2 = self.df.iloc[:-1]

    def time_full_product(self, dtype):
        # 测试self.df的unstack方法的运行时间
        self.df.unstack()

    def time_without_last_row(self, dtype):
        # 测试self.df2的unstack方法的运行时间
        self.df2.unstack()


class SparseIndex:
    def setup(self):
        NUM_ROWS = 1000
        # 创建具有五个列（A到E）的DataFrame，每列包含NUM_ROWS行的随机整数或浮点数
        self.df = DataFrame(
            {
                "A": np.random.randint(50, size=NUM_ROWS),
                "B": np.random.randint(50, size=NUM_ROWS),
                "C": np.random.randint(-10, 10, size=NUM_ROWS),
                "D": np.random.randint(-10, 10, size=NUM_ROWS),
                "E": np.random.randint(10, size=NUM_ROWS),
                "F": np.random.randn(NUM_ROWS),
            }
        )
        # 将"A", "B", "C", "D", "E"列设置为MultiIndex
        self.df = self.df.set_index(["A", "B", "C", "D", "E"])

    def time_unstack(self):
        # 测试self.df的unstack方法的运行时间
        self.df.unstack()


class WideToLong:
    def setup(self):
        nyrs = 20
        nidvars = 20
        N = 5000
        self.letters = list("ABCD")
        yrvars = [
            letter + str(num)
            for letter, num in product(self.letters, range(1, nyrs + 1))
        ]
        columns = [str(i) for i in range(nidvars)] + yrvars
        # 创建一个具有N行和nidvars+len(yrvars)列的随机数据DataFrame
        self.df = DataFrame(np.random.randn(N, nidvars + len(yrvars)), columns=columns)
        # 添加一个"id"列，内容为行索引
        self.df["id"] = self.df.index

    def time_wide_to_long_big(self):
        # 测试将self.df从宽格式转换为长格式的运行时间
        wide_to_long(self.df, self.letters, i="id", j="year")


class PivotTable:
    def setup(self):
        N = 100000
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        ind1 = np.random.randint(0, 3, size=N)
        ind2 = np.random.randint(0, 2, size=N)
        # 创建包含N行的DataFrame，包括key1、key2、key3三列和value1、value2、value3三列的随机数据
        self.df = DataFrame(
            {
                "key1": fac1.take(ind1),
                "key2": fac2.take(ind2),
                "key3": fac2.take(ind2),
                "value1": np.random.randn(N),
                "value2": np.random.randn(N),
                "value3": np.random.randn(N),
            }
        )
        # 创建一个具有5行和3列的DataFrame
        self.df2 = DataFrame(
            {"col1": list("abcde"), "col2": list("fghij"), "col3": [1, 2, 3, 4, 5]}
        )
        # 将self.df2的col1和col2列转换为category类型
        self.df2.col1 = self.df2.col1.astype("category")
        self.df2.col2 = self.df2.col2.astype("category")
    # 创建一个时间性能测试方法，用于生成基于 DataFrame 的透视表，仅指定索引列
    def time_pivot_table(self):
        self.df.pivot_table(index="key1", columns=["key2", "key3"])
    
    # 创建一个时间性能测试方法，用于生成基于 DataFrame 的透视表，指定索引列和聚合函数（sum 和 mean）
    def time_pivot_table_agg(self):
        self.df.pivot_table(
            index="key1", columns=["key2", "key3"], aggfunc=["sum", "mean"]
        )
    
    # 创建一个时间性能测试方法，用于生成基于 DataFrame 的透视表，指定索引列和包括边际汇总
    def time_pivot_table_margins(self):
        self.df.pivot_table(index="key1", columns=["key2", "key3"], margins=True)
    
    # 创建一个时间性能测试方法，用于生成基于 DataFrame 的透视表，指定索引、值和列，并指定聚合函数和填充缺失值
    def time_pivot_table_categorical(self):
        self.df2.pivot_table(
            index="col1", values="col3", columns="col2", aggfunc="sum", fill_value=0
        )
    
    # 创建一个时间性能测试方法，用于生成基于 DataFrame 的透视表，指定索引、值和列，并指定聚合函数、填充缺失值以及使用观测值
    def time_pivot_table_categorical_observed(self):
        self.df2.pivot_table(
            index="col1",
            values="col3",
            columns="col2",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
    
    # 创建一个时间性能测试方法，用于生成基于 DataFrame 的透视表，仅指定列，并包括边际汇总
    def time_pivot_table_margins_only_column(self):
        self.df.pivot_table(columns=["key1", "key2", "key3"], margins=True)
# 定义一个名为 Crosstab 的类，用于执行交叉表相关的基准测试
class Crosstab:
    # 设置方法，用于初始化数据
    def setup(self):
        # 设定数据量 N 为 100000
        N = 100000
        # 创建一个包含三个元素的数组 fac1，元素类型为对象
        fac1 = np.array(["A", "B", "C"], dtype="O")
        # 创建一个包含两个元素的数组 fac2，元素类型为对象
        fac2 = np.array(["one", "two"], dtype="O")
        # 生成 N 个随机整数作为 self.ind1 数组的内容，取值范围为 0 到 2
        self.ind1 = np.random.randint(0, 3, size=N)
        # 生成 N 个随机整数作为 self.ind2 数组的内容，取值范围为 0 到 1
        self.ind2 = np.random.randint(0, 2, size=N)
        # 根据 self.ind1 的索引从 fac1 中取值，构成 self.vec1 数组
        self.vec1 = fac1.take(self.ind1)
        # 根据 self.ind2 的索引从 fac2 中取值，构成 self.vec2 数组
        self.vec2 = fac2.take(self.ind2)

    # 执行 crosstab 操作的基准测试方法
    def time_crosstab(self):
        # 调用 pandas 的 crosstab 函数，根据 self.vec1 和 self.vec2 创建交叉表
        pd.crosstab(self.vec1, self.vec2)

    # 执行 crosstab 操作的基准测试方法，包括值和聚合函数参数
    def time_crosstab_values(self):
        # 调用 pandas 的 crosstab 函数，根据 self.vec1 和 self.vec2 创建交叉表，并使用 self.ind1 作为值，使用求和作为聚合函数
        pd.crosstab(self.vec1, self.vec2, values=self.ind1, aggfunc="sum")

    # 执行 crosstab 操作的基准测试方法，包括归一化参数
    def time_crosstab_normalize(self):
        # 调用 pandas 的 crosstab 函数，根据 self.vec1 和 self.vec2 创建归一化后的交叉表
        pd.crosstab(self.vec1, self.vec2, normalize=True)

    # 执行 crosstab 操作的基准测试方法，包括归一化和边际合计参数
    def time_crosstab_normalize_margins(self):
        # 调用 pandas 的 crosstab 函数，根据 self.vec1 和 self.vec2 创建归一化后的交叉表，并计算边际合计
        pd.crosstab(self.vec1, self.vec2, normalize=True, margins=True)


# 定义一个名为 GetDummies 的类，用于执行 get_dummies 相关的基准测试
class GetDummies:
    # 设置方法，用于初始化数据
    def setup(self):
        # 创建一个包含前12个字母的列表 categories
        categories = list(string.ascii_letters[:12])
        # 生成一个包含1000000个随机选择的分类元素的 pandas Series 对象 s
        s = pd.Series(
            np.random.choice(categories, size=1000000),
            dtype=CategoricalDtype(categories),
        )
        # 将生成的 Series 对象赋值给 self.s
        self.s = s

    # 执行 get_dummies 操作的基准测试方法，不使用稀疏矩阵
    def time_get_dummies_1d(self):
        # 调用 pandas 的 get_dummies 函数，对 self.s 进行独热编码，不使用稀疏矩阵表示
        pd.get_dummies(self.s, sparse=False)

    # 执行 get_dummies 操作的基准测试方法，使用稀疏矩阵
    def time_get_dummies_1d_sparse(self):
        # 调用 pandas 的 get_dummies 函数，对 self.s 进行独热编码，使用稀疏矩阵表示
        pd.get_dummies(self.s, sparse=True)


# 定义一个名为 Cut 的类，用于执行 cut 和 qcut 相关的基准测试
class Cut:
    # 定义 bins 参数的取值范围
    params = [[4, 10, 1000]]
    param_names = ["bins"]

    # 设置方法，用于初始化数据
    def setup(self, bins):
        # 设置数据量 N 为 10^5
        N = 10**5
        # 创建一个包含 N*5 个整数的 Series 对象 self.int_series
        self.int_series = pd.Series(np.arange(N).repeat(5))
        # 创建一个包含 N*5 个随机浮点数的 Series 对象 self.float_series
        self.float_series = pd.Series(np.random.randn(N).repeat(5))
        # 创建一个包含 N 个随机 timedelta 的 Series 对象 self.timedelta_series
        self.timedelta_series = pd.Series(
            np.random.randint(N, size=N), dtype="timedelta64[ns]"
        )
        # 创建一个包含 N 个随机 datetime 的 Series 对象 self.datetime_series
        self.datetime_series = pd.Series(
            np.random.randint(N, size=N), dtype="datetime64[ns]"
        )
        # 根据给定的 bins 创建一个时间间隔索引 self.interval_bins
        self.interval_bins = pd.IntervalIndex.from_breaks(np.linspace(0, N, bins))

    # 执行 cut 操作的基准测试方法，针对整数 Series
    def time_cut_int(self, bins):
        # 调用 pandas 的 cut 函数，对 self.int_series 进行分箱操作
        pd.cut(self.int_series, bins)

    # 执行 cut 操作的基准测试方法，针对浮点数 Series
    def time_cut_float(self, bins):
        # 调用 pandas 的 cut 函数，对 self.float_series 进行分箱操作
        pd.cut(self.float_series, bins)

    # 执行 cut 操作的基准测试方法，针对 timedelta Series
    def time_cut_timedelta(self, bins):
        # 调用 pandas 的 cut 函数，对 self.timedelta_series 进行分箱操作
        pd.cut(self.timedelta_series, bins)

    # 执行 cut 操作的基准测试方法，针对 datetime Series
    def time_cut_datetime(self, bins):
        # 调用 pandas 的 cut 函数，对 self.datetime_series 进行分箱操作
        pd.cut(self.datetime_series, bins)

    # 执行 qcut 操作的基准测试方法，针对整数 Series
    def time_qcut_int(self, bins):
        # 调用 pandas 的 qcut 函数，对 self.int_series 进行分位数分箱操作
        pd.qcut(self.int_series, bins)

    # 执行 qcut 操作的基准测试方法，针对浮点数 Series
    def time_qcut_float(self, bins):
        # 调用 pandas 的 qcut 函数，对 self.float_series 进行分位数分箱操作
        pd.qcut(self.float_series, bins)

    # 执行 qcut 操作的基准测试方法，针对 timedelta Series
    def time_qcut_timedelta(self, bins):
        # 调用 pandas 的 qcut 函数，对 self.timedelta_series 进行分位数分箱操作
        pd.qcut(self.timedelta_series, bins)

    # 执行 qcut 操作的基准测试方法，针对 datetime Series
    def time_qcut_datetime(self, bins):
        # 调用 pandas 的 qcut 函数，对 self.datetime_series 进行分位数分箱操作
        pd.qcut(self.datetime_series, bins)

    # 执行 cut 操作的基准测试方法，针对整数 Series 和 IntervalIndex
    def time_cut_interval(self, bins):
        # GH 27668
        # 调用 pandas 的 cut 函数，对 self.int_series 根据预先定义的 self.interval_bins 进行分箱操作
        pd.cut(self.int_series, self.interval_bins)

    # 测试内存峰值的基准测试方法，针对整数 Series 和 IntervalIndex
    def peakmem_cut_interval(self, bins):
        # GH 27668
        # 调用 pandas 的 cut 函数，对 self.int_series 根据预先定义的 self.interval_bins 进行分箱操作，并记录内存峰值
        pd.cut
# 从当前目录的 pandas_vb_common 模块中导入 setup 函数
# noqa: F401 告知 linter 忽略 F401 错误（即未使用的导入），isort:skip 告知 isort 忽略此行以保持排序
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```