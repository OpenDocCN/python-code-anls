# `.\numpy\numpy\fft\helper.py`

```py
# 定义一个特殊的属性获取方法，用于获取 numpy.fft._helper 模块中的特定属性
def __getattr__(attr_name):
    # 导入警告模块，用于发出警告信息
    import warnings
    # 从 numpy.fft._helper 模块中获取指定名称的属性
    ret = getattr(_helper, attr_name, None)
    # 如果未找到指定属性，抛出 AttributeError 异常
    if ret is None:
        raise AttributeError(
            f"module 'numpy.fft.helper' has no attribute {attr_name}")
    # 发出警告，说明 numpy.fft.helper 模块已更名并且变为私有，建议使用 numpy.fft._helper
    warnings.warn(
        "The numpy.fft.helper has been made private and renamed to "
        "numpy.fft._helper. All four functions exported by it (i.e. fftshift, "
        "ifftshift, fftfreq, rfftfreq) are available from numpy.fft. "
        f"Please use numpy.fft.{attr_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )
    # 返回获取到的属性
    return ret
```