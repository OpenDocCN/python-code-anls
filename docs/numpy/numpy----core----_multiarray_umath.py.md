# `.\numpy\numpy\core\_multiarray_umath.py`

```py
# 从 numpy._core 模块中导入 _multiarray_umath 对象
from numpy._core import _multiarray_umath
# 从 numpy 模块中导入 ufunc 类型
from numpy import ufunc

# 遍历 _multiarray_umath 对象的所有属性名称
for item in _multiarray_umath.__dir__():
    # 对于每个属性名称，获取其对应的属性对象
    attr = getattr(_multiarray_umath, item)
    # 如果该属性对象属于 ufunc 类型
    if isinstance(attr, ufunc):
        # 将该属性对象添加到全局命名空间中，属性名称和对象相同
        globals()[item] = attr


# 定义一个特殊的 __getattr__ 函数
def __getattr__(attr_name):
    # 从 numpy._core 模块中导入 _multiarray_umath 对象
    from numpy._core import _multiarray_umath
    # 从当前模块的 ._utils 子模块中导入 _raise_warning 函数

    # 如果 attr_name 是 "_ARRAY_API" 或 "_UFUNC_API"
    if attr_name in {"_ARRAY_API", "_UFUNC_API"}:
        # 从 numpy.version 模块中导入 short_version 变量
        from numpy.version import short_version
        # 导入 textwrap 模块，用于格式化消息文本
        import textwrap
        # 导入 traceback 模块，用于生成调用堆栈信息
        import traceback
        # 导入 sys 模块，用于访问标准错误流
        import sys

        # 创建一条包含详细信息的消息字符串
        msg = textwrap.dedent(f"""
            A module that was compiled using NumPy 1.x cannot be run in
            NumPy {short_version} as it may crash. To support both 1.x and 2.x
            versions of NumPy, modules must be compiled with NumPy 2.0.
            Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

            If you are a user of the module, the easiest solution will be to
            downgrade to 'numpy<2' or try to upgrade the affected module.
            We expect that some modules will need time to support NumPy 2.

            """)
        # 创建一个包含调用堆栈的消息字符串
        tb_msg = "Traceback (most recent call last):"
        for line in traceback.format_stack()[:-1]:
            if "frozen importlib" in line:
                continue
            tb_msg += line

        # 将消息和调用堆栈信息写入标准错误流
        sys.stderr.write(msg + tb_msg)
        # 抛出 ImportError 异常，包含详细消息
        raise ImportError(msg)

    # 尝试从 _multiarray_umath 对象中获取指定名称的属性对象
    ret = getattr(_multiarray_umath, attr_name, None)
    # 如果获取的属性对象为 None，则抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            "module 'numpy.core._multiarray_umath' has no attribute "
            f"{attr_name}")
    # 调用 _raise_warning 函数，向用户发出警告
    _raise_warning(attr_name, "_multiarray_umath")
    # 返回获取到的属性对象
    return ret

# 从全局命名空间中删除 _multiarray_umath 和 ufunc 变量
del _multiarray_umath, ufunc
```