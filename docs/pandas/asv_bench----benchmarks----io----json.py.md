# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\json.py`

```
# 导入系统模块
import sys

# 导入第三方库 numpy，并重命名为 np
import numpy as np

# 从 pandas 库中导入多个子模块
from pandas import (
    DataFrame,    # 导入 DataFrame 类
    Index,        # 导入 Index 类
    concat,       # 导入 concat 函数
    date_range,   # 导入 date_range 函数
    json_normalize,  # 导入 json_normalize 函数
    read_json,    # 导入 read_json 函数
    timedelta_range,  # 导入 timedelta_range 函数
)

# 从当前包的 pandas_vb_common 模块中导入 BaseIO 类
from ..pandas_vb_common import BaseIO


# 定义一个 ReadJSON 类，继承自 BaseIO 类
class ReadJSON(BaseIO):
    # 定义类属性 fname，表示文件名为 "__test__.json"
    fname = "__test__.json"
    # 定义类属性 params，包含两个列表作为元组元素，表示参数选项
    params = (["split", "index", "records"], ["int", "datetime"])
    # 定义类属性 param_names，表示参数名称列表
    param_names = ["orient", "index"]

    # 定义 setup 方法，设置数据和生成 JSON 文件
    def setup(self, orient, index):
        # 设置数据量 N 为 100000
        N = 100000
        # 定义 indexes 字典，包含 "int" 和 "datetime" 两个键，对应不同的索引类型
        indexes = {
            "int": np.arange(N),  # "int" 索引为 0 到 N-1 的整数
            "datetime": date_range("20000101", periods=N, freq="h"),  # "datetime" 索引为指定日期范围的时间戳
        }
        # 创建 DataFrame 对象 df，包含 N 行随机数和 5 列浮点数，以指定索引类型为索引
        df = DataFrame(
            np.random.randn(N, 5),  # N 行 5 列的随机数
            columns=[f"float_{i}" for i in range(5)],  # 列名为 "float_0" 到 "float_4"
            index=indexes[index],  # 使用指定的索引类型作为索引
        )
        # 将 DataFrame 对象 df 以指定的 orient 形式写入 JSON 文件 self.fname
        df.to_json(self.fname, orient=orient)

    # 定义 time_read_json 方法，用于读取 JSON 文件
    def time_read_json(self, orient, index):
        # 调用 pandas 的 read_json 函数读取 JSON 文件 self.fname，以指定的 orient 形式
        read_json(self.fname, orient=orient)


# 定义一个 ReadJSONLines 类，继承自 BaseIO 类
class ReadJSONLines(BaseIO):
    # 定义类属性 fname，表示文件名为 "__test_lines__.json"
    fname = "__test_lines__.json"
    # 定义类属性 params，包含 "int" 和 "datetime" 两个字符串，作为参数选项
    params = ["int", "datetime"]
    # 定义类属性 param_names，表示参数名称列表，只包含 "index"
    param_names = ["index"]

    # 定义 setup 方法，设置数据和生成 JSON Lines 文件
    def setup(self, index):
        # 设置数据量 N 为 100000
        N = 100000
        # 定义 indexes 字典，包含 "int" 和 "datetime" 两个键，对应不同的索引类型
        indexes = {
            "int": np.arange(N),  # "int" 索引为 0 到 N-1 的整数
            "datetime": date_range("20000101", periods=N, freq="h"),  # "datetime" 索引为指定日期范围的时间戳
        }
        # 创建 DataFrame 对象 df，包含 N 行随机数和 5 列浮点数，以指定索引类型为索引
        df = DataFrame(
            np.random.randn(N, 5),  # N 行 5 列的随机数
            columns=[f"float_{i}" for i in range(5)],  # 列名为 "float_0" 到 "float_4"
            index=indexes[index],  # 使用指定的索引类型作为索引
        )
        # 将 DataFrame 对象 df 以 "records" 形式写入 JSON Lines 文件 self.fname
        df.to_json(self.fname, orient="records", lines=True)

    # 定义 time_read_json_lines 方法，用于读取 JSON Lines 文件
    def time_read_json_lines(self, index):
        # 调用 pandas 的 read_json 函数读取 JSON Lines 文件 self.fname，以 "records" 形式
        read_json(self.fname, orient="records", lines=True)

    # 定义 time_read_json_lines_concat 方法，用于读取并合并 JSON Lines 文件内容
    def time_read_json_lines_concat(self, index):
        # 调用 pandas 的 read_json 函数读取 JSON Lines 文件 self.fname 的内容，并使用 concat 函数合并
        concat(read_json(self.fname, orient="records", lines=True, chunksize=25000))

    # 定义 time_read_json_lines_nrows 方法，用于读取指定行数的 JSON Lines 文件内容
    def time_read_json_lines_nrows(self, index):
        # 调用 pandas 的 read_json 函数读取 JSON Lines 文件 self.fname 的前 25000 行内容
        read_json(self.fname, orient="records", lines=True, nrows=25000)

    # 定义 peakmem_read_json_lines 方法，用于记录 JSON Lines 文件的内存峰值
    def peakmem_read_json_lines(self, index):
        # 调用 pandas 的 read_json 函数读取 JSON Lines 文件 self.fname 的内容，并记录内存峰值
        read_json(self.fname, orient="records", lines=True)

    # 定义 peakmem_read_json_lines_concat 方法，用于记录并合并 JSON Lines 文件内容的内存峰值
    def peakmem_read_json_lines_concat(self, index):
        # 调用 pandas 的 read_json 函数读取 JSON Lines 文件 self.fname 的内容，并使用 concat 函数合并，并记录内存峰值
        concat(read_json(self.fname, orient="records", lines=True, chunksize=25000))

    # 定义 peakmem_read_json_lines_nrows 方法，用于记录指定行数的 JSON Lines 文件内容的内存峰值
    def peakmem_read_json_lines_nrows(self, index):
        # 调用 pandas 的 read_json 函数读取 JSON Lines 文件 self.fname 的前 15000 行内容，并记录内存峰值
        read_json(self.fname, orient="records", lines=True, nrows=15000)


# 定义一个 NormalizeJSON 类，继承自 BaseIO 类
class NormalizeJSON(BaseIO):
    # 定义类属性 fname，表示文件名为 "__test__.json"
    fname = "__test__.json"
    # 定义类属性 params，包含多个列表作为元组元素，表示参数选项
    params = [
        ["split", "columns", "index", "values", "records"],  # 不同的 orient 选项
        ["df", "df_date_idx", "df_td_int_ts", "df_int_floats", "df_int_float_str"],  # 不同的 frame 选项
    ]
    # 定义类属性 param_names，表示参数名称列表，包含 "orient" 和 "frame" 两个名称
    param_names = ["orient", "frame"]

    # 定义 setup 方法，设置数据和生成 JSON 数据
    def setup(self, orient, frame):
        # 定义示例数据，包含一个字典，作为 JSON 数据的样例
        data = {
            "hello": ["thisisatest", 999898, "mixed types"],  # 字典包含一个列表，包含不同类型的数据
            "nest1": {"nest2": {"nest3": "nest3_value", "nest3_int": 3445}},  # 嵌套字典
            "nest1_list": {"nest2": ["blah", 32423, 546456.876, 92030234]},  # 嵌套列表
            "hello2": "string",  # 包含一个字符串
        }
        # 将示例数据作为列表的元素，重复 10000 次，赋值给实例属性 self.data
        self.data = [data for i in range(10000)]

    # 定义 time_normalize_json 方法，用于规
    # 定义一个包含两个子列表的参数列表，每个子列表包含多个字符串
    params = [
        ["split", "columns", "index", "values", "records"],  # 第一个子列表包含不同的参数选项
        ["df", "df_date_idx", "df_td_int_ts", "df_int_floats", "df_int_float_str"],  # 第二个子列表包含不同的数据框名称
    ]
    
    # 定义一个包含两个字符串的参数名称列表
    param_names = ["orient", "frame"]
    
    
    # 定义一个设置函数，初始化各种数据结构和数据框
    def setup(self, orient, frame):
        # 定义常量 N，并生成一个时间序列作为索引
        N = 10**5
        ncols = 5
        index = date_range("20000101", periods=N, freq="h")
        timedeltas = timedelta_range(start=1, periods=N, freq="s")
        datetimes = date_range(start=1, periods=N, freq="s")
        ints = np.random.randint(100000000, size=N)
        longints = sys.maxsize * np.random.randint(100000000, size=N)
        floats = np.random.randn(N)
        strings = Index([f"i-{i}" for i in range(N)], dtype=object)
        
        # 初始化各个数据框对象，存储到 setup 函数的实例属性中
        self.df = DataFrame(np.random.randn(N, ncols), index=np.arange(N))
        self.df_date_idx = DataFrame(np.random.randn(N, ncols), index=index)
        self.df_td_int_ts = DataFrame(
            {
                "td_1": timedeltas,
                "td_2": timedeltas,
                "int_1": ints,
                "int_2": ints,
                "ts_1": datetimes,
                "ts_2": datetimes,
            },
            index=index,
        )
        self.df_int_floats = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "int_3": ints,
                "float_1": floats,
                "float_2": floats,
                "float_3": floats,
            },
            index=index,
        )
        self.df_int_float_str = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )
    
        self.df_longint_float_str = DataFrame(
            {
                "longint_1": longints,
                "longint_2": longints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )
    
    # 定义一个将数据框转换为 JSON 格式并保存的方法
    def time_to_json(self, orient, frame):
        # 使用 getattr 方法获取指定属性名称的数据框，并将其转换为 JSON 格式保存到指定文件中
        getattr(self, frame).to_json(self.fname, orient=orient)
    
    # 定义一个将数据框转换为 JSON 格式并保存内存峰值的方法
    def peakmem_to_json(self, orient, frame):
        # 使用 getattr 方法获取指定属性名称的数据框，并将其转换为 JSON 格式保存内存峰值
        getattr(self, frame).to_json(self.fname, orient=orient)
class ToJSONWide(ToJSON):
    # ToJSONWide 类继承自 ToJSON 类，用于生成宽格式的 JSON 输出
    def setup(self, orient, frame):
        # 设置函数，初始化对象状态
        super().setup(orient, frame)
        # 获取 self 中名为 frame 的属性，复制到 base_df
        base_df = getattr(self, frame).copy()
        # 将 base_df 的前100行复制1000次，构成宽格式的 DataFrame，忽略索引
        df_wide = concat([base_df.iloc[:100]] * 1000, ignore_index=True, axis=1)
        # 将构造好的宽格式 DataFrame 赋值给对象的 df_wide 属性
        self.df_wide = df_wide

    def time_to_json_wide(self, orient, frame):
        # 将 df_wide 对象以 JSON 格式写入到 self.fname 指定的文件中，按照指定的 orient 格式
        self.df_wide.to_json(self.fname, orient=orient)

    def peakmem_to_json_wide(self, orient, frame):
        # 将 df_wide 对象以 JSON 格式写入到 self.fname 指定的文件中，按照指定的 orient 格式
        self.df_wide.to_json(self.fname, orient=orient)


class ToJSONISO(BaseIO):
    # ToJSONISO 类继承自 BaseIO 类，用于生成 ISO 格式的 JSON 输出
    fname = "__test__.json"
    # 支持的参数列表
    params = [["split", "columns", "index", "values", "records"]]
    # 参数名列表
    param_names = ["orient"]

    def setup(self, orient):
        # 初始化函数，生成包含时间序列的 DataFrame 对象
        N = 10**5
        index = date_range("20000101", periods=N, freq="h")
        timedeltas = timedelta_range(start=1, periods=N, freq="s")
        datetimes = date_range(start=1, periods=N, freq="s")
        # 创建包含时间序列的 DataFrame
        self.df = DataFrame(
            {
                "td_1": timedeltas,
                "td_2": timedeltas,
                "ts_1": datetimes,
                "ts_2": datetimes,
            },
            index=index,
        )

    def time_iso_format(self, orient):
        # 将 df 对象以 JSON 格式写入到文件，采用 ISO 格式的日期时间格式化
        self.df.to_json(orient=orient, date_format="iso")


class ToJSONLines(BaseIO):
    # ToJSONLines 类继承自 BaseIO 类，用于生成按行格式的 JSON 输出
    fname = "__test__.json"
    # 在测试设置中，生成要使用的行数
    def setup(self):
        N = 10**5
        # 每行的列数
        ncols = 5
        # 创建一个日期范围，间隔为1小时，共N个时间点
        index = date_range("20000101", periods=N, freq="h")
        # 创建一个时间增量范围，每秒1个增量，共N个时间增量
        timedeltas = timedelta_range(start=1, periods=N, freq="s")
        # 创建一个日期时间范围，每秒1个日期时间，共N个日期时间
        datetimes = date_range(start=1, periods=N, freq="s")
        # 生成N个随机整数
        ints = np.random.randint(100000000, size=N)
        # 生成N个随机长整数，每个长整数为sys.maxsize的随机倍数
        longints = sys.maxsize * np.random.randint(100000000, size=N)
        # 生成N个随机浮点数
        floats = np.random.randn(N)
        # 生成N个字符串索引，格式为"i-{i}"
        strings = Index([f"i-{i}" for i in range(N)], dtype=object)
        
        # 创建一个DataFrame对象，包含N行、ncols列的随机标准正态分布数据
        self.df = DataFrame(np.random.randn(N, ncols), index=np.arange(N))
        # 创建一个DataFrame对象，包含N行、ncols列的随机标准正态分布数据，索引为日期时间index
        self.df_date_idx = DataFrame(np.random.randn(N, ncols), index=index)
        # 创建一个DataFrame对象，包含多列数据，包括时间增量、整数和日期时间
        self.df_td_int_ts = DataFrame(
            {
                "td_1": timedeltas,
                "td_2": timedeltas,
                "int_1": ints,
                "int_2": ints,
                "ts_1": datetimes,
                "ts_2": datetimes,
            },
            index=index,
        )
        # 创建一个DataFrame对象，包含多列数据，包括整数和浮点数
        self.df_int_floats = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "int_3": ints,
                "float_1": floats,
                "float_2": floats,
                "float_3": floats,
            },
            index=index,
        )
        # 创建一个DataFrame对象，包含多列数据，包括整数、浮点数和字符串
        self.df_int_float_str = DataFrame(
            {
                "int_1": ints,
                "int_2": ints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )
        # 创建一个DataFrame对象，包含多列数据，包括长整数、浮点数和字符串
        self.df_longint_float_str = DataFrame(
            {
                "longint_1": longints,
                "longint_2": longints,
                "float_1": floats,
                "float_2": floats,
                "str_1": strings,
                "str_2": strings,
            },
            index=index,
        )

    # 将DataFrame对象以JSON格式写入文件，每行一个记录，包含浮点数和整数
    def time_floats_with_int_idex_lines(self):
        self.df.to_json(self.fname, orient="records", lines=True)

    # 将DataFrame对象以JSON格式写入文件，每行一个记录，包含浮点数和日期时间索引
    def time_floats_with_dt_index_lines(self):
        self.df_date_idx.to_json(self.fname, orient="records", lines=True)

    # 将DataFrame对象以JSON格式写入文件，每行一个记录，包含时间增量、整数和日期时间索引
    def time_delta_int_tstamp_lines(self):
        self.df_td_int_ts.to_json(self.fname, orient="records", lines=True)

    # 将DataFrame对象以JSON格式写入文件，每行一个记录，包含整数和浮点数
    def time_float_int_lines(self):
        self.df_int_floats.to_json(self.fname, orient="records", lines=True)

    # 将DataFrame对象以JSON格式写入文件，每行一个记录，包含整数、浮点数和字符串
    def time_float_int_str_lines(self):
        self.df_int_float_str.to_json(self.fname, orient="records", lines=True)

    # 将DataFrame对象以JSON格式写入文件，每行一个记录，包含长整数、浮点数和字符串
    def time_float_longint_str_lines(self):
        self.df_longint_float_str.to_json(self.fname, orient="records", lines=True)
# 定义一个名为 ToJSONMem 的类，用于处理数据转换为 JSON 格式的内存操作
class ToJSONMem:
    
    # 设置缓存，返回包含不同数据类型 DataFrame 的字典 frames
    def setup_cache(self):
        # 创建包含单个元素的 DataFrame df
        df = DataFrame([[1]])
        # 创建时间序列 DataFrame df2，包含从 "2000年1月1日" 开始的8个时间点，频率为每分钟
        df2 = DataFrame(range(8), date_range("1/1/2000", periods=8, freq="min"))
        # 创建包含不同数据类型 DataFrame 的字典 frames
        frames = {"int": df, "float": df.astype(float), "datetime": df2}

        return frames

    # 处理 frames 中 "int" 类型 DataFrame 的内存峰值操作
    def peakmem_int(self, frames):
        # 获取 frames 中 "int" 类型的 DataFrame df
        df = frames["int"]
        # 执行10万次将 DataFrame 转换为 JSON 格式的操作
        for _ in range(100_000):
            df.to_json()

    # 处理 frames 中 "float" 类型 DataFrame 的内存峰值操作
    def peakmem_float(self, frames):
        # 获取 frames 中 "float" 类型的 DataFrame df
        df = frames["float"]
        # 执行10万次将 DataFrame 转换为 JSON 格式的操作
        for _ in range(100_000):
            df.to_json()

    # 处理 frames 中 "datetime" 类型 DataFrame 的内存峰值操作
    def peakmem_time(self, frames):
        # 获取 frames 中 "datetime" 类型的 DataFrame df
        df = frames["datetime"]
        # 执行1万次将 DataFrame 转换为 JSON 格式的操作，指定 orient 为 "table"
        for _ in range(10_000):
            df.to_json(orient="table")


# 导入 setup 函数，来自 pandas_vb_common 模块的相对路径，忽略未使用的导入警告
from ..pandas_vb_common import setup  # noqa: F401 isort:skip
```