# `D:\src\scipysrc\pandas\pandas\tests\api\test_types.py`

```
# 从未来模块导入 annotations，以便支持注解类型的声明
from __future__ import annotations

# 导入 pandas 测试工具模块中的 _testing 子模块，简称为 tm
import pandas._testing as tm

# 从 pandas.api 中导入 types 模块
from pandas.api import types

# 从 pandas.tests.api.test_api 中导入 Base 类
from pandas.tests.api.test_api import Base

# 定义一个 TestTypes 类，继承自 Base 类
class TestTypes(Base):
    # 允许的类型名称列表
    allowed = [
        "is_any_real_numeric_dtype",
        "is_bool",
        "is_bool_dtype",
        "is_categorical_dtype",
        "is_complex",
        "is_complex_dtype",
        "is_datetime64_any_dtype",
        "is_datetime64_dtype",
        "is_datetime64_ns_dtype",
        "is_datetime64tz_dtype",
        "is_dtype_equal",
        "is_float",
        "is_float_dtype",
        "is_int64_dtype",
        "is_integer",
        "is_integer_dtype",
        "is_number",
        "is_numeric_dtype",
        "is_object_dtype",
        "is_scalar",
        "is_sparse",
        "is_string_dtype",
        "is_signed_integer_dtype",
        "is_timedelta64_dtype",
        "is_timedelta64_ns_dtype",
        "is_unsigned_integer_dtype",
        "is_period_dtype",
        "is_interval_dtype",
        "is_re",
        "is_re_compilable",
        "is_dict_like",
        "is_iterator",
        "is_file_like",
        "is_list_like",
        "is_hashable",
        "is_array_like",
        "is_named_tuple",
        "pandas_dtype",
        "union_categoricals",
        "infer_dtype",
        "is_extension_array_dtype",
    ]
    
    # 废弃的类型名称列表，初始为空列表
    deprecated: list[str] = []

    # 数据类型名称列表，包括 'CategoricalDtype', 'DatetimeTZDtype', 'PeriodDtype', 'IntervalDtype'
    dtypes = ["CategoricalDtype", "DatetimeTZDtype", "PeriodDtype", "IntervalDtype"]

    # 定义一个测试方法 test_types，用于检查 types 模块中允许的类型和数据类型
    def test_types(self):
        # 调用父类方法 check，传入 types 模块、允许的类型列表和数据类型列表
        self.check(types, self.allowed + self.dtypes + self.deprecated)

    # 定义一个测试方法 test_deprecated_from_api_types，用于检查废弃的类型从 api.types 模块中
    def test_deprecated_from_api_types(self):
        # 遍历废弃类型列表 self.deprecated
        for t in self.deprecated:
            # 使用 assert_produces_warning 上下文管理器，检查是否触发 FutureWarning 警告
            with tm.assert_produces_warning(FutureWarning):
                # 动态获取 types 模块中的废弃类型 t，并调用它，传入参数 1
                getattr(types, t)(1)
```