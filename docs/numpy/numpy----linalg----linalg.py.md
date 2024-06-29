# `.\numpy\numpy\linalg\linalg.py`

```
# 定义一个特殊的方法 __getattr__，用于在获取不存在的属性时执行
def __getattr__(attr_name):
    # 导入警告模块，用于输出警告信息
    import warnings
    # 从 numpy.linalg 模块中导入 _linalg 对象
    from numpy.linalg import _linalg
    # 尝试获取 _linalg 对象的属性 attr_name
    ret = getattr(_linalg, attr_name, None)
    # 如果未找到该属性，则引发 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.linalg.linalg' has no attribute {attr_name}")
    # 发出警告，说明 numpy.linalg.linalg 已被标记为私有，推荐使用 numpy.linalg._linalg 替代
    warnings.warn(
        "The numpy.linalg.linalg has been made private and renamed to "
        "numpy.linalg._linalg. All public functions exported by it are "
        f"available from numpy.linalg. Please use numpy.linalg.{attr_name} "
        "instead.",
        DeprecationWarning,
        stacklevel=3
    )
    # 返回获取到的属性对象
    return ret
```