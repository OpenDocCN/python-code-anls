# `D:\src\scipysrc\pandas\pandas\util\__init__.py`

```
# 定义一个特殊方法 __getattr__()，用于在运行时动态获取属性
def __getattr__(key: str):
    # 需要延迟导入这些模块，以避免循环导入错误
    # 如果请求的属性是 "hash_array"，则导入并返回 pandas 库中的 hash_array 函数
    if key == "hash_array":
        from pandas.core.util.hashing import hash_array
        return hash_array
    
    # 如果请求的属性是 "hash_pandas_object"，则导入并返回 pandas 库中的 hash_pandas_object 函数
    if key == "hash_pandas_object":
        from pandas.core.util.hashing import hash_pandas_object
        return hash_pandas_object
    
    # 如果请求的属性是 "Appender"，则导入并返回 pandas 库中的 Appender 类
    if key == "Appender":
        from pandas.util._decorators import Appender
        return Appender
    
    # 如果请求的属性是 "Substitution"，则导入并返回 pandas 库中的 Substitution 类
    if key == "Substitution":
        from pandas.util._decorators import Substitution
        return Substitution
    
    # 如果请求的属性是 "cache_readonly"，则导入并返回 pandas 库中的 cache_readonly 装饰器函数
    if key == "cache_readonly":
        from pandas.util._decorators import cache_readonly
        return cache_readonly
    
    # 如果请求的属性不在上述定义的几种情况中，抛出 AttributeError 异常
    raise AttributeError(f"module 'pandas.util' has no attribute '{key}'")

# 定义一个特殊方法 __dir__()，返回当前模块的全局变量名列表以及一些特定字符串列表
def __dir__() -> list[str]:
    # 返回当前模块中的全局变量名列表以及包含 "hash_array" 和 "hash_pandas_object" 字符串的列表
    return list(globals().keys()) + ["hash_array", "hash_pandas_object"]
```