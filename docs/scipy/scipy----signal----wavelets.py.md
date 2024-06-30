# `D:\src\scipysrc\scipy\scipy\signal\wavelets.py`

```
# 引入从 SciPy 库中导入私有子模块 _sub_module_deprecation
from scipy._lib.deprecation import _sub_module_deprecation

# 定义空列表 __all__，用于存储模块中所有公开的符号（symbol）
__all__: list[str] = []

# 定义特殊方法 __dir__()，返回模块的公开符号列表
def __dir__():
    return __all__

# 定义特殊方法 __getattr__(name)，处理模块中未定义的属性访问
# 返回 _sub_module_deprecation 函数的调用结果，指定信号处理子模块的过时信息
# sub_package="signal" 表示子模块是信号处理模块
# module="wavelets" 表示模块是小波变换模块
# private_modules=["_wavelets"] 表示私有模块 _wavelets 是被处理的模块
# all=__all__ 表示将当前模块的所有公开符号传递给 _sub_module_deprecation 函数
# attribute=name 表示未定义的属性名将作为参数传递给 _sub_module_deprecation 函数
def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="wavelets",
                                   private_modules=["_wavelets"], all=__all__,
                                   attribute=name)
```