# `.\pytorch\torch\utils\data\datapipes\dataframe\dataframe_wrapper.py`

```py
# mypy: allow-untyped-defs
# 引入类型定义中的 Any 和 Optional
from typing import Any, Optional

# 全局变量 _pandas 初始化为 None
_pandas: Any = None
# 全局变量 _WITH_PANDAS 初始化为 None 或者 Optional[bool] 类型
_WITH_PANDAS: Optional[bool] = None


# 尝试导入 pandas 模块的函数，返回是否成功导入的布尔值
def _try_import_pandas() -> bool:
    try:
        # 尝试导入 pandas 模块，忽略类型检查
        import pandas  # type: ignore[import]
        # 将成功导入的 pandas 模块赋值给全局变量 _pandas
        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        # 捕获 ImportError 异常，表示未成功导入 pandas 模块
        return False


# 判断是否已经导入 pandas 模块的函数
def _with_pandas() -> bool:
    global _WITH_PANDAS
    # 如果 _WITH_PANDAS 为 None，则调用 _try_import_pandas 函数进行导入
    if _WITH_PANDAS is None:
        _WITH_PANDAS = _try_import_pandas()
    return _WITH_PANDAS


# PandasWrapper 类，用于封装与 pandas 相关的操作
class PandasWrapper:
    # 创建 DataFrame 的类方法
    @classmethod
    def create_dataframe(cls, data, columns):
        # 如果未导入 pandas，则抛出 RuntimeError 异常
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        # 使用 _pandas 创建 DataFrame 对象，并指定列名
        return _pandas.DataFrame(data, columns=columns)  # type: ignore[union-attr]

    # 判断是否为 DataFrame 的类方法
    @classmethod
    def is_dataframe(cls, data):
        # 如果未导入 pandas，则返回 False
        if not _with_pandas():
            return False
        # 判断数据是否为 _pandas.core.frame.DataFrame 类型
        return isinstance(data, _pandas.core.frame.DataFrame)  # type: ignore[union-attr]

    # 判断是否为 Series 的类方法
    @classmethod
    def is_column(cls, data):
        # 如果未导入 pandas，则返回 False
        if not _with_pandas():
            return False
        # 判断数据是否为 _pandas.core.series.Series 类型
        return isinstance(data, _pandas.core.series.Series)  # type: ignore[union-attr]

    # 迭代数据的类方法
    @classmethod
    def iterate(cls, data):
        # 如果未导入 pandas，则抛出 RuntimeError 异常
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        # 使用 data 的 itertuples 方法迭代数据
        yield from data.itertuples(index=False)

    # 合并数据的类方法
    @classmethod
    def concat(cls, buffer):
        # 如果未导入 pandas，则抛出 RuntimeError 异常
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        # 使用 _pandas 的 concat 方法合并 buffer 中的数据
        return _pandas.concat(buffer)  # type: ignore[union-attr]

    # 获取指定索引位置数据的类方法
    @classmethod
    def get_item(cls, data, idx):
        # 如果未导入 pandas，则抛出 RuntimeError 异常
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        # 返回指定索引位置的 data 数据
        return data[idx : idx + 1]

    # 获取 DataFrame 的行数的类方法
    @classmethod
    def get_len(cls, df):
        # 如果未导入 pandas，则抛出 RuntimeError 异常
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        # 返回 DataFrame 的行数
        return len(df.index)

    # 获取 DataFrame 的列名的类方法
    @classmethod
    def get_columns(cls, df):
        # 如果未导入 pandas，则抛出 RuntimeError 异常
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        # 返回 DataFrame 的列名列表
        return list(df.columns.values.tolist())


# 默认的 wrapper 类为 PandasWrapper
default_wrapper = PandasWrapper


# 获取当前的 wrapper 实例的函数
def get_df_wrapper():
    return default_wrapper


# 设置新的 wrapper 类的函数
def set_df_wrapper(wrapper):
    global default_wrapper
    default_wrapper = wrapper


# 创建 DataFrame 的函数
def create_dataframe(data, columns=None):
    # 获取当前的 wrapper 实例
    wrapper = get_df_wrapper()
    # 调用 wrapper 实例的 create_dataframe 方法创建 DataFrame
    return wrapper.create_dataframe(data, columns)


# 判断是否为 DataFrame 的函数
def is_dataframe(data):
    # 获取当前的 wrapper 实例
    wrapper = get_df_wrapper()
    # 调用 wrapper 实例的 is_dataframe 方法判断数据是否为 DataFrame
    return wrapper.is_dataframe(data)


# 获取 DataFrame 的列名的函数
def get_columns(data):
    # 获取当前的 wrapper 实例
    wrapper = get_df_wrapper()
    # 调用 wrapper 实例的 get_columns 方法获取 DataFrame 的列名
    return wrapper.get_columns(data)


# 判断是否为 Series 的函数
def is_column(data):
    # 获取当前的 wrapper 实例
    wrapper = get_df_wrapper()
    # 调用 wrapper 实例的 is_column 方法判断数据是否为 Series
    return wrapper.is_column(data)


# 合并数据的函数
def concat(buffer):
    # 获取当前的 wrapper 实例
    wrapper = get_df_wrapper()
    # 返回 wrapper 对象的 concat 方法应用于 buffer 参数后的结果
    return wrapper.concat(buffer)
# 从全局函数获取数据框架包装器，用于迭代数据
def iterate(data):
    # 获取数据框架包装器对象
    wrapper = get_df_wrapper()
    # 使用包装器对象迭代数据
    return wrapper.iterate(data)


# 从全局函数获取数据框架包装器，用于获取指定索引的数据项
def get_item(data, idx):
    # 获取数据框架包装器对象
    wrapper = get_df_wrapper()
    # 使用包装器对象获取指定索引的数据项
    return wrapper.get_item(data, idx)


# 从全局函数获取数据框架包装器，用于获取数据框架的长度
def get_len(df):
    # 获取数据框架包装器对象
    wrapper = get_df_wrapper()
    # 使用包装器对象获取数据框架的长度
    return wrapper.get_len(df)
```