# `D:\src\scipysrc\pandas\asv_bench\benchmarks\strings.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库

from pandas import (  # 从Pandas库中导入以下模块：
    NA,  # 缺失值标识符
    Categorical,  # 分类数据类型
    DataFrame,  # 数据帧
    Index,  # 索引对象
    Series,  # 系列对象
)
from pandas.arrays import StringArray  # 从Pandas的arrays模块中导入StringArray

class Dtypes:
    params = ["str", "string[python]", "string[pyarrow]"]  # 参数列表
    param_names = ["dtype"]  # 参数名称列表

    def setup(self, dtype):
        try:
            # 创建一个包含10000个元素的Series对象
            self.s = Series(
                Index([f"i-{i}" for i in range(10000)], dtype=object)._values,
                dtype=dtype,
            )
        except ImportError as err:
            raise NotImplementedError from err  # 抛出未实现错误

class Construction:
    params = (
        ["series", "frame", "categorical_series"],  # 参数列表
        ["str", "string[python]", "string[pyarrow]"],  # 参数名称列表
    )
    param_names = ["pd_type", "dtype"]  # 参数名称

    pd_mapping = {"series": Series, "frame": DataFrame, "categorical_series": Series}  # 映射Pandas类型到类
    dtype_mapping = {"str": "str", "string[python]": object, "string[pyarrow]": object}  # 映射dtype到Python对象

    def setup(self, pd_type, dtype):
        series_arr = np.array(
            [str(i) * 10 for i in range(100_000)], dtype=self.dtype_mapping[dtype]
        )  # 创建包含100000个元素的数组
        if pd_type == "series":
            self.arr = series_arr  # 如果是series类型，使用series_arr数组
        elif pd_type == "frame":
            self.arr = series_arr.reshape((50_000, 2)).copy()  # 如果是frame类型，使用reshape后的数组
        elif pd_type == "categorical_series":
            # 创建分类系列对象
            self.arr = Categorical(series_arr)

    def time_construction(self, pd_type, dtype):
        self.pd_mapping[pd_type](self.arr, dtype=dtype)  # 测试构造函数时间

    def peakmem_construction(self, pd_type, dtype):
        self.pd_mapping[pd_type](self.arr, dtype=dtype)  # 测试内存峰值

class Methods(Dtypes):
    def time_center(self, dtype):
        self.s.str.center(100)  # 测试字符串居中方法的时间

    def time_count(self, dtype):
        self.s.str.count("A")  # 测试字符串计数方法的时间

    def time_endswith(self, dtype):
        self.s.str.endswith("A")  # 测试字符串是否以指定后缀结尾的时间

    def time_extract(self, dtype):
        with warnings.catch_warnings(record=True):
            self.s.str.extract("(\\w*)A(\\w*)")  # 测试字符串提取方法的时间

    def time_findall(self, dtype):
        self.s.str.findall("[A-Z]+")  # 测试字符串查找所有匹配项的时间

    def time_find(self, dtype):
        self.s.str.find("[A-Z]+")  # 测试字符串查找首个匹配项的时间

    def time_rfind(self, dtype):
        self.s.str.rfind("[A-Z]+")  # 测试字符串反向查找首个匹配项的时间

    def time_fullmatch(self, dtype):
        self.s.str.fullmatch("A")  # 测试字符串是否完全匹配指定模式的时间

    def time_get(self, dtype):
        self.s.str.get(0)  # 测试获取字符串指定位置字符的时间

    def time_len(self, dtype):
        self.s.str.len()  # 测试字符串长度方法的时间

    def time_join(self, dtype):
        self.s.str.join(" ")  # 测试连接字符串序列的时间

    def time_match(self, dtype):
        self.s.str.match("A")  # 测试字符串是否匹配指定模式的时间

    def time_normalize(self, dtype):
        self.s.str.normalize("NFC")  # 测试字符串规范化的时间

    def time_pad(self, dtype):
        self.s.str.pad(100, side="both")  # 测试字符串填充的时间

    def time_partition(self, dtype):
        self.s.str.partition("A")  # 测试字符串分割的时间

    def time_rpartition(self, dtype):
        self.s.str.rpartition("A")  # 测试字符串反向分割的时间

    def time_replace(self, dtype):
        self.s.str.replace("A", "\x01\x01")  # 测试字符串替换的时间

    def time_translate(self, dtype):
        self.s.str.translate({"A": "\x01\x01"})  # 测试字符串翻译的时间
    # 切片操作：从字符串中提取指定范围内的字符，起始索引为5（包含），结束索引为15（不包含），步长为2
    def time_slice(self, dtype):
        self.s.str.slice(5, 15, 2)
    
    # 判断字符串是否以指定前缀开头，返回布尔值
    def time_startswith(self, dtype):
        self.s.str.startswith("A")
    
    # 去除字符串两侧指定字符（在这里是'A'）并返回新字符串
    def time_strip(self, dtype):
        self.s.str.strip("A")
    
    # 去除字符串右侧指定字符（在这里是'A'）并返回新字符串
    def time_rstrip(self, dtype):
        self.s.str.rstrip("A")
    
    # 去除字符串左侧指定字符（在这里是'A'）并返回新字符串
    def time_lstrip(self, dtype):
        self.s.str.lstrip("A")
    
    # 将字符串中每个单词的首字母大写，并返回新字符串
    def time_title(self, dtype):
        self.s.str.title()
    
    # 将字符串中所有字符转换为大写，并返回新字符串
    def time_upper(self, dtype):
        self.s.str.upper()
    
    # 将字符串中所有字符转换为小写，并返回新字符串
    def time_lower(self, dtype):
        self.s.str.lower()
    
    # 在字符串中插入换行符以使其每行不超过指定长度，并返回新字符串
    def time_wrap(self, dtype):
        self.s.str.wrap(10)
    
    # 将字符串左侧填充0（零）直到字符串长度达到指定长度，并返回新字符串
    def time_zfill(self, dtype):
        self.s.str.zfill(10)
    
    # 判断字符串是否只包含字母和数字，并返回布尔值
    def time_isalnum(self, dtype):
        self.s.str.isalnum()
    
    # 判断字符串是否只包含字母，并返回布尔值
    def time_isalpha(self, dtype):
        self.s.str.isalpha()
    
    # 判断字符串是否只包含十进制数字，并返回布尔值
    def time_isdecimal(self, dtype):
        self.s.str.isdecimal()
    
    # 判断字符串是否只包含数字，并返回布尔值
    def time_isdigit(self, dtype):
        self.s.str.isdigit()
    
    # 判断字符串是否只包含小写字母，并返回布尔值
    def time_islower(self, dtype):
        self.s.str.islower()
    
    # 判断字符串是否只包含数字字符，并返回布尔值
    def time_isnumeric(self, dtype):
        self.s.str.isnumeric()
    
    # 判断字符串是否只包含空白字符（如空格、制表符等），并返回布尔值
    def time_isspace(self, dtype):
        self.s.str.isspace()
    
    # 判断字符串是否符合标题格式（每个单词首字母大写），并返回布尔值
    def time_istitle(self, dtype):
        self.s.str.istitle()
    
    # 判断字符串是否只包含大写字母，并返回布尔值
    def time_isupper(self, dtype):
        self.s.str.isupper()
class Repeat:
    # 参数定义：int 和 array 类型的参数
    params = ["int", "array"]
    # 参数名称：repeats
    param_names = ["repeats"]

    # 初始化设置方法，接受 repeats 参数
    def setup(self, repeats):
        # 创建一个包含 10^5 个索引的 Series 对象
        N = 10**5
        self.s = Series(Index([f"i-{i}" for i in range(N)], dtype=object))
        # 根据 repeats 参数设置不同的重复值
        repeat = {"int": 1, "array": np.random.randint(1, 3, N)}
        self.values = repeat[repeats]

    # 测试重复操作的方法，接受 repeats 参数
    def time_repeat(self, repeats):
        # 对 Series 对象的字符串内容进行重复操作，重复次数由 self.values 决定
        self.s.str.repeat(self.values)


class Cat:
    # 参数定义：other_cols, sep, na_rep, na_frac 四个参数的组合
    params = ([0, 3], [None, ","], [None, "-"], [0.0, 0.001, 0.15])
    # 参数名称：other_cols, sep, na_rep, na_frac
    param_names = ["other_cols", "sep", "na_rep", "na_frac"]

    # 初始化设置方法，接受 other_cols, sep, na_rep, na_frac 四个参数
    def setup(self, other_cols, sep, na_rep, na_frac):
        # 创建一个包含 10^5 个索引的 Series 对象，并根据概率 na_frac 生成 NaN 值的掩码
        N = 10**5
        mask_gen = lambda: np.random.choice([True, False], N, p=[1 - na_frac, na_frac])
        self.s = Series(Index([f"i-{i}" for i in range(N)], dtype=object)).where(
            mask_gen()
        )
        if other_cols == 0:
            # 如果 other_cols 参数为 0，则 self.others 设置为 None
            self.others = None
        else:
            # 否则创建包含 other_cols 列的 DataFrame 对象，每列包含 10^5 个索引，并根据概率 na_frac 生成 NaN 值的掩码
            self.others = DataFrame(
                {
                    i: Index([f"i-{i}" for i in range(N)], dtype=object).where(
                        mask_gen()
                    )
                    for i in range(other_cols)
                }
            )

    # 测试字符串连接操作的方法，接受 other_cols, sep, na_rep, na_frac 四个参数
    def time_cat(self, other_cols, sep, na_rep, na_frac):
        # 对 Series 对象的字符串内容进行连接操作，包括 self.others 列的连接，使用 sep 分隔符，na_rep 用于替换 NaN 值
        self.s.str.cat(others=self.others, sep=sep, na_rep=na_rep)


class Contains(Dtypes):
    # 参数定义：继承自 Dtypes 类的参数组合，以及 regex 参数
    params = (Dtypes.params, [True, False])
    # 参数名称：dtype, regex
    param_names = ["dtype", "regex"]

    # 初始化设置方法，接受 dtype, regex 两个参数
    def setup(self, dtype, regex):
        super().setup(dtype)

    # 测试字符串包含操作的方法，接受 dtype, regex 两个参数
    def time_contains(self, dtype, regex):
        # 在 Series 对象的字符串内容中查找 "A"，根据 regex 参数决定是否使用正则表达式
        self.s.str.contains("A", regex=regex)


class Split(Dtypes):
    # 参数定义：继承自 Dtypes 类的参数组合，以及 expand 参数
    params = (Dtypes.params, [True, False])
    # 参数名称：dtype, expand
    param_names = ["dtype", "expand"]

    # 初始化设置方法，接受 dtype, expand 两个参数
    def setup(self, dtype, expand):
        super().setup(dtype)
        # 对 Series 对象的字符串内容进行连接操作，使用 "--" 分隔符
        self.s = self.s.str.join("--")

    # 测试字符串分割操作的方法，接受 dtype, expand 两个参数
    def time_split(self, dtype, expand):
        # 对 Series 对象的字符串内容进行分割操作，使用 "--" 分隔符，根据 expand 参数决定是否扩展为 DataFrame
        self.s.str.split("--", expand=expand)

    # 测试字符串反向分割操作的方法，接受 dtype, expand 两个参数
    def time_rsplit(self, dtype, expand):
        # 对 Series 对象的字符串内容进行反向分割操作，使用 "--" 分隔符，根据 expand 参数决定是否扩展为 DataFrame
        self.s.str.rsplit("--", expand=expand)


class Extract(Dtypes):
    # 参数定义：继承自 Dtypes 类的参数组合，以及 expand 参数
    params = (Dtypes.params, [True, False])
    # 参数名称：dtype, expand
    param_names = ["dtype", "expand"]

    # 初始化设置方法，接受 dtype, expand 两个参数
    def setup(self, dtype, expand):
        super().setup(dtype)

    # 测试字符串提取操作的方法，接受 dtype, expand 两个参数
    def time_extract_single_group(self, dtype, expand):
        # 在 Series 对象的字符串内容中进行正则表达式提取，提取模式为 "(\\w*)A"，根据 expand 参数决定是否扩展为 DataFrame
        with warnings.catch_warnings(record=True):
            self.s.str.extract("(\\w*)A", expand=expand)


class Dummies(Dtypes):
    # 初始化设置方法，接受 dtype 参数
    def setup(self, dtype):
        super().setup(dtype)
        # 对 Series 对象进行切片操作，取前 1/5 部分，并使用 "|" 连接
        N = len(self.s) // 5
        self.s = self.s[:N].str.join("|")

    # 测试获取虚拟变量（哑变量）的方法，接受 dtype 参数
    def time_get_dummies(self, dtype):
        # 对 Series 对象的字符串内容进行分割，根据 "|" 分隔，生成哑变量 DataFrame
        self.s.str.get_dummies("|")


class Encode:
    # 初始化设置方法
    def setup(self):
        # 创建一个包含 10,000 个索引的 Series 对象
        self.ser = Series(Index([f"i-{i}" for i in range(10_000)], dtype=object))
    # 定义一个方法 `time_encode_decode`，该方法属于类的实例方法，需要通过实例来调用
    def time_encode_decode(self):
        # 对实例变量 `ser` 中的字符串进行 utf-8 编码，然后再解码为 utf-8 编码的字符串
        self.ser.str.encode("utf-8").str.decode("utf-8")
# 定义 Slice 类，用于演示切片操作
class Slice:
    # 设置方法，初始化一个 Series 对象，包含大量字符串和 NaN 值
    def setup(self):
        self.s = Series(["abcdefg", np.nan] * 500000)

    # 时间性能测试方法，演示字符串切片操作
    def time_vector_slice(self):
        # GH 2602：执行字符串切片操作，获取字符串的前五个字符
        self.s.str[:5]


# 定义 Iter 类，继承自 Dtypes 类
class Iter(Dtypes):
    # 时间性能测试方法，演示迭代操作
    def time_iter(self, dtype):
        # 遍历 self.s 中的每个元素，不执行具体操作
        for i in self.s:
            pass


# 定义 StringArrayConstruction 类
class StringArrayConstruction:
    # 设置方法，初始化包含大量重复字符串的 NumPy 数组和包含 NaN 值的 Series 数组
    def setup(self):
        self.series_arr = np.array([str(i) * 10 for i in range(10**5)], dtype=object)
        self.series_arr_nan = np.concatenate([self.series_arr, np.array([NA] * 1000)])

    # 时间性能测试方法，演示 StringArray 对象的构建
    def time_string_array_construction(self):
        # 使用 self.series_arr 构建 StringArray 对象
        StringArray(self.series_arr)

    # 时间性能测试方法，演示包含 NaN 值的 StringArray 对象的构建
    def time_string_array_with_nan_construction(self):
        # 使用 self.series_arr_nan 构建 StringArray 对象
        StringArray(self.series_arr_nan)

    # 峰值内存使用量测试方法，演示 StringArray 对象的构建
    def peakmem_stringarray_construction(self):
        # 使用 self.series_arr 构建 StringArray 对象
        StringArray(self.series_arr)
```