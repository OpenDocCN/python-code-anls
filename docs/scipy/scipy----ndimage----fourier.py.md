# `D:\src\scipysrc\scipy\scipy\ndimage\fourier.py`

```
# 导入 `_sub_module_deprecation` 函数从 `scipy._lib.deprecation` 模块
# 定义 __all__ 列表，这些函数名将会在该模块被导入时公开，同时禁用 Flake8 的 F822 错误检查
__all__ = [  # noqa: F822
    'fourier_gaussian', 'fourier_uniform',
    'fourier_ellipsoid', 'fourier_shift'
]


# 定义 __dir__() 函数，用于返回当前模块的公开函数列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，当试图访问当前模块中未定义的属性时调用
def __getattr__(name):
    # 调用 `_sub_module_deprecation` 函数，发出关于子模块 'ndimage' 和模块 'fourier'
    # 的废弃警告，并提到私有模块 '_fourier' 和所有列在 __all__ 中的函数名
    return _sub_module_deprecation(sub_package='ndimage', module='fourier',
                                   private_modules=['_fourier'], all=__all__,
                                   attribute=name)
```