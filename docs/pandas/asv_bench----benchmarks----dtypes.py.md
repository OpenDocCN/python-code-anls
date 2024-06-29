# `D:\src\scipysrc\pandas\asv_bench\benchmarks\dtypes.py`

```
import string  # 导入字符串模块

import numpy as np  # 导入 NumPy 库

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入 DataFrame 和 Index 类
    DataFrame,
    Index,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.api.types import (  # 从 Pandas API 中导入类型相关函数
    is_extension_array_dtype,
    pandas_dtype,
)

from .pandas_vb_common import (  # 从本地模块导入特定数据类型
    datetime_dtypes,
    extension_dtypes,
    numeric_dtypes,
    string_dtypes,
)

_numpy_dtypes = [  # 创建包含 NumPy 数据类型的列表
    np.dtype(dtype) for dtype in (numeric_dtypes + datetime_dtypes + string_dtypes)
]
_dtypes = _numpy_dtypes + extension_dtypes  # 将扩展数据类型加入总数据类型列表


class Dtypes:  # 定义类型类
    params = _dtypes + [dt.name for dt in _dtypes]  # 设置类型参数为总数据类型列表
    param_names = ["dtype"]  # 参数名称为 "dtype"

    def time_pandas_dtype(self, dtype):  # 定义函数以计时 Pandas 数据类型函数
        pandas_dtype(dtype)  # 调用 Pandas 数据类型函数


class DtypesInvalid:  # 定义无效类型类
    param_names = ["dtype"]  # 参数名称为 "dtype"
    params = ["scalar-string", "scalar-int", "list-string", "array-string"]  # 参数为不同的数据类型
    data_dict = {  # 数据字典包含不同类型的数据
        "scalar-string": "foo",
        "scalar-int": 1,
        "list-string": ["foo"] * 1000,
        "array-string": np.array(["foo"] * 1000),
    }

    def time_pandas_dtype_invalid(self, dtype):  # 定义函数以计时处理无效的 Pandas 数据类型
        try:
            pandas_dtype(self.data_dict[dtype])  # 尝试调用 Pandas 数据类型函数
        except TypeError:
            pass  # 捕获类型错误异常并忽略


class SelectDtypes:  # 定义选择数据类型类
    try:
        params = [  # 设置参数为 Pandas 测试模块定义的数据类型
            tm.ALL_INT_NUMPY_DTYPES
            + tm.ALL_INT_EA_DTYPES
            + tm.FLOAT_NUMPY_DTYPES
            + tm.COMPLEX_DTYPES
            + tm.DATETIME64_DTYPES
            + tm.TIMEDELTA64_DTYPES
            + tm.BOOL_DTYPES
        ]
    except AttributeError:
        params = [  # 处理属性错误异常时的备选参数
            tm.ALL_INT_DTYPES
            + tm.ALL_EA_INT_DTYPES
            + tm.FLOAT_DTYPES
            + tm.COMPLEX_DTYPES
            + tm.DATETIME64_DTYPES
            + tm.TIMEDELTA64_DTYPES
            + tm.BOOL_DTYPES
        ]
    param_names = ["dtype"]  # 参数名称为 "dtype"

    def setup(self, dtype):  # 设置测试数据的初始化
        N, K = 5000, 50  # 定义测试数据的维度
        self.index = Index([f"i-{i}" for i in range(N)], dtype=object)  # 创建索引对象
        self.columns = Index([f"i-{i}" for i in range(K)], dtype=object)  # 创建列对象

        def create_df(data):  # 定义创建 DataFrame 的辅助函数
            return DataFrame(data, index=self.index, columns=self.columns)

        # 初始化不同类型的测试 DataFrame
        self.df_int = create_df(np.random.randint(low=100, size=(N, K)))
        self.df_float = create_df(np.random.randn(N, K))
        self.df_bool = create_df(np.random.choice([True, False], size=(N, K)))
        self.df_string = create_df(
            np.random.choice(list(string.ascii_letters), size=(N, K))
        )

    def time_select_dtype_int_include(self, dtype):  # 计时选择整数数据类型包含操作
        self.df_int.select_dtypes(include=dtype)

    def time_select_dtype_int_exclude(self, dtype):  # 计时选择整数数据类型排除操作
        self.df_int.select_dtypes(exclude=dtype)

    def time_select_dtype_float_include(self, dtype):  # 计时选择浮点数数据类型包含操作
        self.df_float.select_dtypes(include=dtype)

    def time_select_dtype_float_exclude(self, dtype):  # 计时选择浮点数数据类型排除操作
        self.df_float.select_dtypes(exclude=dtype)

    def time_select_dtype_bool_include(self, dtype):  # 计时选择布尔数据类型包含操作
        self.df_bool.select_dtypes(include=dtype)

    def time_select_dtype_bool_exclude(self, dtype):  # 计时选择布尔数据类型排除操作
        self.df_bool.select_dtypes(exclude=dtype)
    # 定义一个方法用于选择数据框中特定数据类型的列，并包含指定的数据类型
    def time_select_dtype_string_include(self, dtype):
        # 使用 Pandas 中的 select_dtypes 方法，选择数据框中包含特定数据类型的列
        self.df_string.select_dtypes(include=dtype)
    
    # 定义一个方法用于选择数据框中特定数据类型的列，并排除指定的数据类型
    def time_select_dtype_string_exclude(self, dtype):
        # 使用 Pandas 中的 select_dtypes 方法，选择数据框中排除特定数据类型的列
        self.df_string.select_dtypes(exclude=dtype)
class CheckDtypes:
    # 设置方法，初始化 self.ext_dtype 为 Pandas 的 Int64Dtype 类型
    def setup(self):
        self.ext_dtype = pd.Int64Dtype()
        # 初始化 self.np_dtype 为 NumPy 的 int64 数据类型
        self.np_dtype = np.dtype("int64")

    # 用于测试是否为扩展数组数据类型的方法，传入 self.ext_dtype
    def time_is_extension_array_dtype_true(self):
        is_extension_array_dtype(self.ext_dtype)

    # 用于测试是否为扩展数组数据类型的方法，传入 self.np_dtype
    def time_is_extension_array_dtype_false(self):
        is_extension_array_dtype(self.np_dtype)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
```