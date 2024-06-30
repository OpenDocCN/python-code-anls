# `D:\src\scipysrc\scipy\scipy\integrate\vode.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个空列表，用于存放当前模块中需要导出的符号名称
__all__: list[str] = []

# 定义一个特殊函数 __dir__()，用于返回当前模块中可以公开的符号列表
def __dir__():
    return __all__

# 定义一个特殊函数 __getattr__(name)，用于在获取不存在的属性时调用
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，向用户显示子模块的废弃警告信息
    return _sub_module_deprecation(
        sub_package="integrate",  # 废弃警告中涉及的子包名称
        module="vode",  # 废弃警告中涉及的模块名称
        private_modules=["_vode"],  # 废弃警告中提到的私有模块列表
        all=__all__,  # 当前模块中可以公开的符号列表
        attribute=name  # 用户试图获取的属性名称
    )
```