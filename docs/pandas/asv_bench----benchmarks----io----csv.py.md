# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\csv.py`

```
from io import (
    BytesIO,
    StringIO,
)
import random  # 导入 random 模块，用于生成随机数
import string  # 导入 string 模块，用于字符串操作

import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 pandas 库中导入以下类和函数
    Categorical,
    DataFrame,
    Index,
    concat,
    date_range,
    period_range,
    read_csv,
    to_datetime,
)

from ..pandas_vb_common import BaseIO  # 从相对路径的 pandas_vb_common 模块导入 BaseIO 类


class ToCSV(BaseIO):
    fname = "__test__.csv"  # 定义类变量 fname 为 "__test__.csv"
    params = ["wide", "long", "mixed"]  # 定义类变量 params，包含字符串列表
    param_names = ["kind"]  # 定义类变量 param_names，包含字符串列表

    def setup(self, kind):
        wide_frame = DataFrame(np.random.randn(3000, 30))  # 创建一个 3000 行 30 列的随机数 DataFrame 对象
        long_frame = DataFrame(
            {
                "A": np.arange(50000),  # 列 "A" 包含从 0 到 49999 的整数
                "B": np.arange(50000) + 1.0,  # 列 "B" 包含从 1 到 50000 的浮点数
                "C": np.arange(50000) + 2.0,  # 列 "C" 包含从 2 到 50001 的浮点数
                "D": np.arange(50000) + 3.0,  # 列 "D" 包含从 3 到 50002 的浮点数
            }
        )
        mixed_frame = DataFrame(
            {
                "float": np.random.randn(5000),  # 列 "float" 包含 5000 个随机浮点数
                "int": np.random.randn(5000).astype(int),  # 列 "int" 包含 5000 个随机整数
                "bool": (np.arange(5000) % 2) == 0,  # 列 "bool" 包含 5000 个布尔值，每隔一行为 True
                "datetime": date_range("2001", freq="s", periods=5000),  # 列 "datetime" 包含 从 "2001-01-01 00:00:00" 开始，每秒增加一行的时间戳
                "object": ["foo"] * 5000,  # 列 "object" 包含 5000 个字符串 "foo"
            }
        )
        mixed_frame.loc[30:500, "float"] = np.nan  # 将 mixed_frame 中从第 30 行到第 500 行的 "float" 列设为 NaN
        data = {"wide": wide_frame, "long": long_frame, "mixed": mixed_frame}  # 创建数据字典 data，包含三个不同类型的 DataFrame
        self.df = data[kind]  # 根据传入的 kind 参数选择相应的 DataFrame 并赋值给实例变量 self.df

    def time_frame(self, kind):
        self.df.to_csv(self.fname)  # 将 self.df 对象写入 CSV 文件 self.fname


class ToCSVMultiIndexUnusedLevels(BaseIO):
    fname = "__test__.csv"  # 定义类变量 fname 为 "__test__.csv"

    def setup(self):
        df = DataFrame({"a": np.random.randn(100_000), "b": 1, "c": 1})  # 创建包含 100000 行的 DataFrame 对象 df
        self.df = df.set_index(["a", "b"])  # 将 df 设置为多级索引，索引包括 "a" 和 "b"
        self.df_unused_levels = self.df.iloc[:10_000]  # 从 self.df 中选择前 10000 行赋值给 self.df_unused_levels
        self.df_single_index = df.set_index(["a"]).iloc[:10_000]  # 将 df 设置为单级索引，然后选择前 10000 行赋值给 self.df_single_index

    def time_full_frame(self):
        self.df.to_csv(self.fname)  # 将 self.df 对象写入 CSV 文件 self.fname

    def time_sliced_frame(self):
        self.df_unused_levels.to_csv(self.fname)  # 将 self.df_unused_levels 对象写入 CSV 文件 self.fname

    def time_single_index_frame(self):
        self.df_single_index.to_csv(self.fname)  # 将 self.df_single_index 对象写入 CSV 文件 self.fname


class ToCSVDatetime(BaseIO):
    fname = "__test__.csv"  # 定义类变量 fname 为 "__test__.csv"

    def setup(self):
        rng = date_range("1/1/2000", periods=1000)  # 创建从 "2000-01-01" 开始的 1000 个时间戳
        self.data = DataFrame(rng, index=rng)  # 创建 DataFrame 对象 self.data，以 rng 为数据和索引

    def time_frame_date_formatting(self):
        self.data.to_csv(self.fname, date_format="%Y%m%d")  # 将 self.data 对象写入 CSV 文件 self.fname，并指定日期格式为 "%Y%m%d"


class ToCSVDatetimeIndex(BaseIO):
    fname = "__test__.csv"  # 定义类变量 fname 为 "__test__.csv"

    def setup(self):
        rng = date_range("2000", periods=100_000, freq="s")  # 创建从 "2000-01-01 00:00:00" 开始，每秒一个时间戳，共 100000 行
        self.data = DataFrame({"a": 1}, index=rng)  # 创建 DataFrame 对象 self.data，包含一列 "a" 和时间戳索引

    def time_frame_date_formatting_index(self):
        self.data.to_csv(self.fname, date_format="%Y-%m-%d %H:%M:%S")  # 将 self.data 对象写入 CSV 文件 self.fname，并指定日期格式为 "%Y-%m-%d %H:%M:%S"

    def time_frame_date_no_format_index(self):
        self.data.to_csv(self.fname)  # 将 self.data 对象写入 CSV 文件 self.fname
        

class ToCSVPeriod(BaseIO):
    fname = "__test__.csv"  # 定义类变量 fname 为 "__test__.csv"

    params = ([1000, 10000], ["D", "h"])  # 定义参数化测试的参数列表，包括两个参数 nobs 和 freq
    param_names = ["nobs", "freq"]  # 定义参数名称列表

    def setup(self, nobs, freq):
        rng = period_range(start="2000-01-01", periods=nobs, freq=freq)  # 根据传入的 nobs 和 freq 创建时间周期索引
        self.data = DataFrame(rng)  # 创建 DataFrame 对象 self.data，以 rng 为数据

        if freq == "D":
            self.default_fmt = "%Y-%m-%d"  # 如果 freq 是 "D"，设置默认日期格式为 "%Y-%m-%d"
        elif freq == "h":
            self.default_fmt = "%Y-%m-%d %H:00"  # 如果 freq 是 "h"，设置默认日期格式为 "%Y-%m-%d %H:00"
    # 将数据保存为 CSV 文件，使用默认的日期时间格式化选项
    def time_frame_period_formatting_default(self, nobs, freq):
        self.data.to_csv(self.fname)

    # 将数据保存为 CSV 文件，使用显式指定的日期时间格式化选项
    def time_frame_period_formatting_default_explicit(self, nobs, freq):
        self.data.to_csv(self.fname, date_format=self.default_fmt)

    # 将数据保存为 CSV 文件，使用特定的日期时间格式化选项
    def time_frame_period_formatting(self, nobs, freq):
        # 注意: 当前不会实际考虑 `date_format` 参数，因此性能与上面的 `time_frame_period_formatting_default` 相同。
        # 当 GH#51621 问题解决后，预计这段代码的性能将会下降。
        # (解决 GH#51621 问题后，删除此注释。)
        self.data.to_csv(self.fname, date_format="%Y-%m-%d___%H:%M:%S")
class ToCSVPeriodIndex(BaseIO):
    # 设置默认的文件名
    fname = "__test__.csv"

    # 设置参数的可能取值范围
    params = ([1000, 10000], ["D", "h"])

    # 设置参数的名称
    param_names = ["nobs", "freq"]

    # 初始化方法，根据给定的参数生成时间区间索引的数据框
    def setup(self, nobs, freq):
        # 根据参数生成时间范围索引
        rng = period_range(start="2000-01-01", periods=nobs, freq=freq)
        # 创建数据框，包含一列名为'a'的数据，索引为生成的时间范围
        self.data = DataFrame({"a": 1}, index=rng)
        # 根据频率设置默认的日期格式
        if freq == "D":
            self.default_fmt = "%Y-%m-%d"
        elif freq == "h":
            self.default_fmt = "%Y-%m-%d %H:00"

    # 将数据框保存为 CSV 文件，日期格式为"%Y-%m-%d___%H:%M:%S"
    def time_frame_period_formatting_index(self, nobs, freq):
        self.data.to_csv(self.fname, date_format="%Y-%m-%d___%H:%M:%S")

    # 将数据框保存为 CSV 文件，默认使用数据框索引的日期格式
    def time_frame_period_formatting_index_default(self, nobs, freq):
        self.data.to_csv(self.fname)

    # 将数据框保存为 CSV 文件，日期格式为实例属性self.default_fmt定义的格式
    def time_frame_period_formatting_index_default_explicit(self, nobs, freq):
        self.data.to_csv(self.fname, date_format=self.default_fmt)


class ToCSVDatetimeBig(BaseIO):
    # 设置默认的文件名
    fname = "__test__.csv"

    # 设置超时时间
    timeout = 1500

    # 设置参数的可能取值范围
    params = [1000, 10000, 100000]

    # 设置参数的名称
    param_names = ["nobs"]

    # 初始化方法，创建包含日期时间数据的数据框
    def setup(self, nobs):
        d = "2018-11-29"
        dt = "2018-11-26 11:18:27.0"
        # 创建数据框，包含'dt'和'd'列，每列都包含nobs行，'r'列包含随机数
        self.data = DataFrame(
            {
                "dt": [np.datetime64(dt)] * nobs,
                "d": [np.datetime64(d)] * nobs,
                "r": [np.random.uniform()] * nobs,
            }
        )

    # 将数据框保存为 CSV 文件
    def time_frame(self, nobs):
        self.data.to_csv(self.fname)


class ToCSVIndexes(BaseIO):
    # 设置默认的文件名
    fname = "__test__.csv"

    # 静态方法：创建包含指定行数和列数的数据框
    @staticmethod
    def _create_df(rows, cols):
        # 创建索引列和数据列的字典
        index_cols = {
            "index1": np.random.randint(0, rows, rows),
            "index2": np.full(rows, 1, dtype=int),
            "index3": np.full(rows, 1, dtype=int),
        }
        data_cols = {
            f"col{i}": np.random.uniform(0, 100000.0, rows) for i in range(cols)
        }
        # 合并索引列和数据列，创建数据框
        df = DataFrame({**index_cols, **data_cols})
        return df

    # 初始化方法，设置数据框的不同状态
    def setup(self):
        ROWS = 100000
        COLS = 5
        # 对于使用.head()的测试，创建初始数据框，包含多倍于ROWS的行数
        HEAD_ROW_MULTIPLIER = 10

        # 创建标准索引的数据框
        self.df_standard_index = self._create_df(ROWS, COLS)

        # 创建先自定义索引再使用.head()方法的数据框
        self.df_custom_index_then_head = (
            self._create_df(ROWS * HEAD_ROW_MULTIPLIER, COLS)
            .set_index(["index1", "index2", "index3"])
            .head(ROWS)
        )

        # 创建先使用.head()方法再自定义索引的数据框
        self.df_head_then_custom_index = (
            self._create_df(ROWS * HEAD_ROW_MULTIPLIER, COLS)
            .head(ROWS)
            .set_index(["index1", "index2", "index3"])
        )

    # 将标准索引的数据框保存为 CSV 文件
    def time_standard_index(self):
        self.df_standard_index.to_csv(self.fname)

    # 将多级索引的数据框保存为 CSV 文件
    def time_multiindex(self):
        self.df_head_then_custom_index.to_csv(self.fname)

    # 将多级索引数据框使用.head()方法后的部分保存为 CSV 文件
    def time_head_of_multiindex(self):
        self.df_custom_index_then_head.to_csv(self.fname)


class StringIORewind:
    # 重置StringIO对象的指针位置，并返回对象本身
    def data(self, stringio_object):
        stringio_object.seek(0)
        return stringio_object


class ReadCSVDInferDatetimeFormat(StringIORewind):
    # 设置参数的可能取值范围
    params = [None, "custom", "iso8601", "ymd"]
    param_names = ["format"]

# 定义一个列表param_names，包含字符串"format"

    def setup(self, format):

# 定义一个方法setup，接受参数self和format
        rng = date_range("1/1/2000", periods=1000)

# 使用pandas的date_range函数生成从"1/1/2000"开始的1000个日期时间对象，并赋值给rng变量

        formats = {
            None: None,
            "custom": "%m/%d/%Y %H:%M:%S.%f",
            "iso8601": "%Y-%m-%d %H:%M:%S",
            "ymd": "%Y%m%d",
        }

# 定义一个格式映射字典formats，将不同的格式字符串映射到对应的日期时间格式字符串

        dt_format = formats[format]

# 根据传入的format参数从formats字典中获取对应的日期时间格式字符串，并赋值给dt_format变量

        self.StringIO_input = StringIO("\n".join(rng.strftime(dt_format).tolist()))

# 将rng中的日期时间对象按照dt_format格式转换为字符串，并用换行符连接成一个字符串，然后使用StringIO封装成一个可读写的文本缓冲区，并赋值给self.StringIO_input变量

    def time_read_csv(self, format):

# 定义一个方法time_read_csv，接受参数self和format

        read_csv(
            self.data(self.StringIO_input),

# 调用read_csv函数，传入以下参数：
# - self.data(self.StringIO_input)：调用self.data方法并传入self.StringIO_input作为参数，返回读取的数据
# - header=None：指定CSV文件没有头部行
# - names=["foo"]：指定CSV文件的列名为["foo"]
# - parse_dates=["foo"]：指定解析["foo"]列为日期时间类型
        )
# 定义一个继承自 StringIORewind 的类 ReadCSVConcatDatetime，用于读取 CSV 并处理日期时间的连接操作
class ReadCSVConcatDatetime(StringIORewind):
    # ISO 8601 格式的日期时间字符串
    iso8601 = "%Y-%m-%d %H:%M:%S"

    # 设置方法，在其中生成一个包含 50000 个日期时间字符串的 StringIO 对象
    def setup(self):
        # 生成从 "1/1/2000" 开始，每秒一个的日期时间范围
        rng = date_range("1/1/2000", periods=50000, freq="s")
        # 将日期时间格式化为 ISO 8601 格式，并用换行符连接成字符串，存储在 StringIO 中
        self.StringIO_input = StringIO("\n".join(rng.strftime(self.iso8601).tolist()))

    # 读取 CSV 文件的方法，解析日期时间列，并未指定列名
    def time_read_csv(self):
        read_csv(
            self.data(self.StringIO_input),  # 传入 StringIO 对象作为数据源
            header=None,                     # 不指定列名行
            names=["foo"],                   # 列名为 "foo"
            parse_dates=["foo"],             # 解析 "foo" 列作为日期时间
        )


# 定义一个继承自 StringIORewind 的类 ReadCSVConcatDatetimeBadDateValue，用于处理包含错误日期值的 CSV 文件
class ReadCSVConcatDatetimeBadDateValue(StringIORewind):
    # 参数列表，包含不同的错误日期值
    params = (["nan", "0", ""],)
    # 参数名称
    param_names = ["bad_date_value"]

    # 设置方法，根据不同的错误日期值生成包含 50000 行数据的 StringIO 对象
    def setup(self, bad_date_value):
        self.StringIO_input = StringIO((f"{bad_date_value},\n") * 50000)

    # 读取 CSV 文件的方法，解析日期时间列，并未指定列名
    def time_read_csv(self, bad_date_value):
        read_csv(
            self.data(self.StringIO_input),  # 传入 StringIO 对象作为数据源
            header=None,                     # 不指定列名行
            names=["foo", "bar"],            # 列名为 "foo" 和 "bar"
            parse_dates=["foo"],             # 解析 "foo" 列作为日期时间
        )


# 定义一个继承自 BaseIO 的类 ReadCSVSkipRows，用于测试跳过行数的 CSV 读取操作
class ReadCSVSkipRows(BaseIO):
    # CSV 文件名
    fname = "__test__.csv"
    # 参数列表，包含跳过的行数和引擎类型
    params = ([None, 10000], ["c", "python", "pyarrow"])
    # 参数名称
    param_names = ["skiprows", "engine"]

    # 设置方法，生成包含随机数据的 DataFrame，并将其保存为 CSV 文件
    def setup(self, skiprows, engine):
        N = 20000
        # 生成对象类型的索引
        index = Index([f"i-{i}" for i in range(N)], dtype=object)
        # 生成包含随机数据的 DataFrame
        df = DataFrame(
            {
                "float1": np.random.randn(N),
                "float2": np.random.randn(N),
                "string1": ["foo"] * N,
                "bool1": [True] * N,
                "int1": np.random.randint(0, N, size=N),
            },
            index=index,
        )
        # 将 DataFrame 写入 CSV 文件
        df.to_csv(self.fname)

    # 测试跳过行数的 CSV 读取方法
    def time_skipprows(self, skiprows, engine):
        read_csv(self.fname, skiprows=skiprows, engine=engine)


# 定义一个继承自 StringIORewind 的类 ReadUint64Integers，用于读取包含 uint64 整数的 CSV 文件
class ReadUint64Integers(StringIORewind):
    # 设置方法，生成包含 uint64 整数的 StringIO 对象，并包含一个特定的 NA 值
    def setup(self):
        self.na_values = [2**63 + 500]
        arr = np.arange(10000).astype("uint64") + 2**63
        self.data1 = StringIO("\n".join(arr.astype(str).tolist()))
        arr = arr.astype(object)
        arr[500] = -1
        self.data2 = StringIO("\n".join(arr.astype(str).tolist()))

    # 测试读取包含 uint64 整数的 CSV 文件的方法
    def time_read_uint64(self):
        read_csv(self.data(self.data1), header=None, names=["foo"])

    # 测试读取包含负数的 uint64 整数的 CSV 文件的方法
    def time_read_uint64_neg_values(self):
        read_csv(self.data(self.data2), header=None, names=["foo"])

    # 测试读取包含 NA 值的 uint64 整数的 CSV 文件的方法
    def time_read_uint64_na_values(self):
        read_csv(
            self.data(self.data1),         # 传入包含 NA 值的 StringIO 对象作为数据源
            header=None,                   # 不指定列名行
            names=["foo"],                 # 列名为 "foo"
            na_values=self.na_values       # 指定 NA 值的列表
        )


# 定义一个继承自 BaseIO 的类 ReadCSVThousands，用于测试读取包含千分位符号的 CSV 文件
class ReadCSVThousands(BaseIO):
    # CSV 文件名
    fname = "__test__.csv"
    # 参数列表，包含分隔符、千分位符号和引擎类型
    params = ([",", "|"], [None, ","], ["c", "python"])
    # 参数名称
    param_names = ["sep", "thousands", "engine"]

    # 设置方法，生成包含随机数据的 DataFrame，并将其保存为 CSV 文件
    def setup(self, sep, thousands, engine):
        N = 10000
        K = 8
        # 生成随机数据的 DataFrame
        data = np.random.randn(N, K) * np.random.randint(100, 10000, (N, K))
        df = DataFrame(data)
        # 如果指定了千分位符号，则格式化每个数据
        if thousands is not None:
            fmt = f":{thousands}"
            fmt = "{" + fmt + "}"
            df = df.map(lambda x: fmt.format(x))
        # 将 DataFrame 写入 CSV 文件
        df.to_csv(self.fname, sep=sep)
    # 定义一个方法 `time_thousands`，接受三个参数 `sep`（分隔符）、`thousands`（千位分隔符）、`engine`（引擎）
    def time_thousands(self, sep, thousands, engine):
        # 调用外部函数 `read_csv` 读取文件内容，并传入参数 `sep`、`thousands`、`engine`
        read_csv(self.fname, sep=sep, thousands=thousands, engine=engine)
# 继承自 StringIORewind 类的 ReadCSVComment 类
class ReadCSVComment(StringIORewind):
    # 参数列表，包括 "c" 和 "python"
    params = ["c", "python"]
    # 参数名列表，只有一个元素 "engine"
    param_names = ["engine"]

    # 设置方法，接受 engine 参数
    def setup(self, engine):
        # 创建包含 100001 行的 CSV 数据，其中第 2 行包含注释字符串
        data = ["A,B,C"] + (["1,2,3 # comment"] * 100000)
        # 创建 StringIO_input 属性，用于存储数据的 StringIO 对象
        self.StringIO_input = StringIO("\n".join(data))

    # 测试方法，接受 engine 参数
    def time_comment(self, engine):
        # 调用 read_csv 函数，读取 self.StringIO_input 中的数据
        read_csv(
            self.data(self.StringIO_input),  # 读取的数据来源
            comment="#",                    # 注释符号为 #
            header=None,                    # 没有标题行
            names=list("abc")               # 列名为 'a', 'b', 'c'
        )


# 继承自 StringIORewind 类的 ReadCSVFloatPrecision 类
class ReadCSVFloatPrecision(StringIORewind):
    # 多个参数的组合：sep, decimal, float_precision
    params = ([",", ";"], [".", "_"], [None, "high", "round_trip"])
    # 参数名列表：sep, decimal, float_precision
    param_names = ["sep", "decimal", "float_precision"]

    # 设置方法，接受 sep, decimal, float_precision 参数
    def setup(self, sep, decimal, float_precision):
        # 创建包含随机生成的 15 个长数字字符串的列表
        floats = [
            "".join([random.choice(string.digits) for _ in range(28)])
            for _ in range(15)
        ]
        # 创建包含 1000 行数据的 CSV 格式字符串，每行有 3 列
        rows = sep.join([f"0{decimal}{{}}"] * 3) + "\n"
        data = rows * 5
        # 使用格式化将随机数插入到数据中，总共有 1000 x 3 个数据行
        data = data.format(*floats) * 200
        # 创建 StringIO_input 属性，用于存储数据的 StringIO 对象
        self.StringIO_input = StringIO(data)

    # 测试方法，接受 sep, decimal, float_precision 参数
    def time_read_csv(self, sep, decimal, float_precision):
        # 调用 read_csv 函数，读取 self.StringIO_input 中的数据
        read_csv(
            self.data(self.StringIO_input),  # 读取的数据来源
            sep=sep,                        # CSV 文件的分隔符
            header=None,                    # 没有标题行
            names=list("abc"),              # 列名为 'a', 'b', 'c'
            float_precision=float_precision  # 浮点数精度控制
        )

    # 测试方法，接受 sep, decimal, float_precision 参数
    def time_read_csv_python_engine(self, sep, decimal, float_precision):
        # 调用 read_csv 函数，读取 self.StringIO_input 中的数据
        read_csv(
            self.data(self.StringIO_input),  # 读取的数据来源
            sep=sep,                        # CSV 文件的分隔符
            header=None,                    # 没有标题行
            engine="python",                # 强制使用 Python 引擎
            float_precision=None,           # 不控制浮点数精度
            names=list("abc")               # 列名为 'a', 'b', 'c'
        )


# 继承自 StringIORewind 类的 ReadCSVEngine 类
class ReadCSVEngine(StringIORewind):
    # 参数列表，包括 "c", "python", "pyarrow"
    params = ["c", "python", "pyarrow"]
    # 参数名列表，只有一个元素 "engine"
    param_names = ["engine"]

    # 设置方法，接受 engine 参数
    def setup(self, engine):
        # 创建包含 100001 行的 CSV 数据，有五列
        data = ["A,B,C,D,E"] + (["1,2,3,4,5"] * 100000)
        # 创建 StringIO_input 属性，用于存储数据的 StringIO 对象
        self.StringIO_input = StringIO("\n".join(data))
        # 模拟从文件中读取数据，创建 BytesIO_input 属性
        self.BytesIO_input = BytesIO(self.StringIO_input.read().encode("utf-8"))

    # 测试方法，接受 engine 参数
    def time_read_stringcsv(self, engine):
        # 调用 read_csv 函数，从 self.StringIO_input 中读取数据
        read_csv(self.data(self.StringIO_input), engine=engine)

    # 测试方法，接受 engine 参数
    def time_read_bytescsv(self, engine):
        # 调用 read_csv 函数，从 self.BytesIO_input 中读取数据
        read_csv(self.data(self.BytesIO_input), engine=engine)

    # 峰值内存使用测试方法，接受 engine 参数
    def peakmem_read_csv(self, engine):
        # 调用 read_csv 函数，从 self.BytesIO_input 中读取数据
        read_csv(self.data(self.BytesIO_input), engine=engine)


# 继承自 BaseIO 类的 ReadCSVCategorical 类
class ReadCSVCategorical(BaseIO):
    # CSV 文件名
    fname = "__test__.csv"
    # 参数列表，包括 "c" 和 "python"
    params = ["c", "python"]
    # 参数名列表，只有一个元素 "engine"
    param_names = ["engine"]

    # 设置方法，接受 engine 参数
    def setup(self, engine):
        # 创建包含 100000 行的 DataFrame，每列数据为随机选择的字符串
        N = 100000
        group1 = ["aaaaaaaa", "bbbbbbb", "cccccccc", "dddddddd", "eeeeeeee"]
        df = DataFrame(np.random.choice(group1, (N, 3)), columns=list("abc"))
        # 将 DataFrame 写入 CSV 文件中
        df.to_csv(self.fname, index=False)

    # 测试方法，接受 engine 参数
    def time_convert_post(self, engine):
        # 读取 CSV 文件并将其转换为分类数据
        read_csv(self.fname, engine=engine).apply(Categorical)

    # 测试方法，接受 engine 参数
    def time_convert_direct(self, engine):
        # 读取 CSV 文件并将其直接转换为分类数据
        read_csv(self.fname, engine=engine, dtype="category")


# 继承自 StringIORewind 类的 ReadCSVParseDates 类
class ReadCSVParseDates(StringIORewind):
    # 参数列表，包括 "c" 和 "python"
    params = ["c", "python"]
    # 参数名列表，只有一个元素 "engine"
    param_names = ["engine"]
    # 在设置方法中初始化数据字符串，使用大括号占位符填充五行数据，每行数据包含时间和数值
    data = """{},19:00:00,18:56:00,0.8100,2.8100,7.2000,0.0000,280.0000\n
              {},20:00:00,19:56:00,0.0100,2.2100,7.2000,0.0000,260.0000\n
              {},21:00:00,20:56:00,-0.5900,2.2100,5.7000,0.0000,280.0000\n
              {},21:00:00,21:18:00,-0.9900,2.0100,3.6000,0.0000,270.0000\n
              {},22:00:00,21:56:00,-0.5900,1.7100,5.1000,0.0000,290.0000\n
           """
    # 创建包含五个重复条目的列表，每个条目由"KORD,19990127"组成
    two_cols = ["KORD,19990127"] * 5
    # 使用数据字符串填充占位符，生成完整的数据字符串
    data = data.format(*two_cols)
    # 将数据字符串转换为类似文件对象的 StringIO 对象并保存在实例变量中
    self.StringIO_input = StringIO(data)

def time_baseline(self, engine):
    # 调用 read_csv 函数，传入 StringIO 对象作为输入数据流
    read_csv(
        self.data(self.StringIO_input),  # 使用预先设定的 StringIO 数据
        engine=engine,  # 引擎参数传递给 read_csv 函数
        sep=",",  # 列分隔符为逗号
        header=None,  # 没有头部行
        parse_dates=[1],  # 解析第二列为日期时间格式
        names=list(string.digits[:9]),  # 列名为前 9 个数字的列表
    )
class ReadCSVCachedParseDates(StringIORewind):
    params = ([True, False], ["c", "python"])  # 参数组合：缓存选项和引擎选项
    param_names = ["do_cache", "engine"]  # 参数名称：缓存选项和引擎选项

    def setup(self, do_cache, engine):
        # 准备数据：生成一个长字符串，包含多行 "10/年份" 的数据
        data = ("\n".join([f"10/{year}" for year in range(2000, 2100)]) + "\n") * 10
        self.StringIO_input = StringIO(data)  # 将数据封装为 StringIO 对象

    def time_read_csv_cached(self, do_cache, engine):
        try:
            read_csv(
                self.data(self.StringIO_input),  # 读取 StringIO 中的数据
                engine=engine,  # 指定 CSV 解析引擎
                header=None,  # 没有表头行
                parse_dates=[0],  # 将第一列解析为日期
                cache_dates=do_cache,  # 是否缓存日期解析结果
            )
        except TypeError:
            # 在 pandas 0.25 版本中新增了 cache_dates 参数
            pass


class ReadCSVMemoryGrowth(BaseIO):
    chunksize = 20  # 每块的行数
    num_rows = 1000  # 总行数
    fname = "__test__.csv"  # 测试用的 CSV 文件名
    params = ["c", "python"]  # CSV 解析引擎选项
    param_names = ["engine"]  # 参数名称：CSV 解析引擎选项

    def setup(self, engine):
        # 创建测试用 CSV 文件
        with open(self.fname, "w", encoding="utf-8") as f:
            for i in range(self.num_rows):
                f.write(f"{i}\n")

    def mem_parser_chunks(self, engine):
        # 使用 CSV 文件的内存映射方式读取数据块
        result = read_csv(self.fname, chunksize=self.chunksize, engine=engine)

        for _ in result:
            pass


class ReadCSVParseSpecialDate(StringIORewind):
    params = (["mY", "mdY", "hm"], ["c", "python"])  # 参数组合：日期格式和引擎选项
    param_names = ["value", "engine"]  # 参数名称：日期格式和引擎选项
    objects = {
        "mY": "01-2019\n10-2019\n02/2000\n",
        "mdY": "12/02/2010\n",
        "hm": "21:34\n",
    }  # 各种日期格式的示例数据

    def setup(self, value, engine):
        count_elem = 10000  # 数据重复次数
        data = self.objects[value] * count_elem  # 根据选项生成数据
        self.StringIO_input = StringIO(data)  # 将数据封装为 StringIO 对象

    def time_read_special_date(self, value, engine):
        read_csv(
            self.data(self.StringIO_input),  # 读取 StringIO 中的数据
            engine=engine,  # 指定 CSV 解析引擎
            sep=",",  # 指定分隔符
            header=None,  # 没有表头行
            names=["Date"],  # 列名为 "Date"
            parse_dates=["Date"],  # 将 "Date" 列解析为日期
        )


class ReadCSVMemMapUTF8:
    fname = "__test__.csv"  # 测试用的 CSV 文件名
    number = 5  # 每行的字符数

    def setup(self):
        lines = []
        line_length = 128  # 每行的字符长度
        start_char = " "  # 起始字符
        end_char = "\U00010080"  # 结束字符
        # 创建包含连续 Unicode 字符的 128 个字符字符串列表
        for lnum in range(ord(start_char), ord(end_char), line_length):
            line = "".join([chr(c) for c in range(lnum, lnum + 0x80)]) + "\n"
            try:
                line.encode("utf-8")  # 尝试将字符串编码为 UTF-8
            except UnicodeEncodeError:
                # 部分 16 位词不是有效的 Unicode 字符，需要跳过
                continue
            lines.append(line)
        df = DataFrame(lines)  # 创建 DataFrame 对象
        df = concat([df for n in range(100)], ignore_index=True)  # 拼接多次创建的 DataFrame
        df.to_csv(self.fname, index=False, header=False, encoding="utf-8")  # 将 DataFrame 写入 CSV 文件

    def time_read_memmapped_utf8(self):
        read_csv(self.fname, header=None, memory_map=True, encoding="utf-8", engine="c")
        # 使用内存映射方式读取 UTF-8 编码的 CSV 文件


class ParseDateComparison(StringIORewind):
    params = ([False, True],)  # 参数：日期缓存选项
    param_names = ["cache_dates"]  # 参数名称：日期缓存选项
    # 设置测试环境的初始数据
    def setup(self, cache_dates):
        # 创建一个包含大量日期数据的字符串
        count_elem = 10000
        data = "12-02-2010\n" * count_elem
        # 将数据放入内存中的 StringIO 对象中，以便后续测试使用
        self.StringIO_input = StringIO(data)

    # 测试读取 CSV 文件时，指定日期格式为 dayfirst=True
    def time_read_csv_dayfirst(self, cache_dates):
        try:
            # 读取 CSV 数据，解析其中的日期列，使用 dayfirst 格式
            read_csv(
                self.data(self.StringIO_input),
                sep=",",
                header=None,
                names=["Date"],
                parse_dates=["Date"],
                cache_dates=cache_dates,
                dayfirst=True,
            )
        except TypeError:
            # 处理在旧版本中 cache_dates 是新的关键字参数的情况
            pass

    # 测试将 CSV 中的日期字符串转换为 datetime 对象，使用 dayfirst=True
    def time_to_datetime_dayfirst(self, cache_dates):
        # 读取 CSV 数据并创建 DataFrame
        df = read_csv(
            self.data(self.StringIO_input), dtype={"date": str}, names=["date"]
        )
        # 将日期字符串转换为 datetime 对象，使用 dayfirst 格式
        to_datetime(df["date"], cache=cache_dates, dayfirst=True)

    # 测试将 CSV 中的日期字符串按指定格式转换为 datetime 对象
    def time_to_datetime_format_DD_MM_YYYY(self, cache_dates):
        # 读取 CSV 数据并创建 DataFrame
        df = read_csv(
            self.data(self.StringIO_input), dtype={"date": str}, names=["date"]
        )
        # 将日期字符串按照指定格式 "%d-%m-%Y" 转换为 datetime 对象
        to_datetime(df["date"], cache=cache_dates, format="%d-%m-%Y")
class ReadCSVIndexCol(StringIORewind):
    # 继承自 StringIORewind 类，用于读取 CSV 格式数据并定位索引列
    def setup(self):
        # 设定数据行数
        count_elem = 100_000
        # 生成包含指定内容的测试数据
        data = "a,b\n" + "1,2\n" * count_elem
        # 将数据封装到 StringIO 对象中
        self.StringIO_input = StringIO(data)

    # 测试读取 CSV 数据时的性能，指定索引列为 "a"
    def time_read_csv_index_col(self):
        read_csv(self.StringIO_input, index_col="a")


class ReadCSVDatePyarrowEngine(StringIORewind):
    # 继承自 StringIORewind 类，用于读取 CSV 格式数据并使用 pyarrow 引擎解析日期
    def setup(self):
        # 设定数据行数
        count_elem = 100_000
        # 生成包含指定内容的测试数据
        data = "a\n" + "2019-12-31\n" * count_elem
        # 将数据封装到 StringIO 对象中
        self.StringIO_input = StringIO(data)

    # 测试使用 pyarrow 引擎读取 CSV 数据，同时解析列 "a" 为日期
    def time_read_csv_index_col(self):
        read_csv(
            self.StringIO_input,
            parse_dates=["a"],
            engine="pyarrow",
            dtype_backend="pyarrow",
        )


class ReadCSVCParserLowMemory:
    # GH 16798
    # 用于测试读取大文件时内存占用情况的特定用例
    def setup(self):
        # 创建包含大量数据的 StringIO 对象，模拟大文件
        self.csv = StringIO(
            "strings\n" + "\n".join(["x" * (1 << 20) for _ in range(2100)])
        )

    # 测试使用 C 解析引擎读取 CSV 数据时的内存占用，关闭低内存模式
    def peakmem_over_2gb_input(self):
        read_csv(self.csv, engine="c", low_memory=False)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
```