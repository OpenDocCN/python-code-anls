# `D:\src\scipysrc\scipy\scipy\odr\models.py`

```
# 导入模块_scipy._lib.deprecation中的子模块_deprecation
from scipy._lib.deprecation import _sub_module_deprecation

# 定义全局变量__all__，用于指定当前模块中导出的公共对象列表，F822用于禁止检查格式
__all__ = [
    'Model', 'exponential', 'multilinear', 'unilinear',
    'quadratic', 'polynomial'
]


# 定义特殊方法__dir__()，用于返回当前模块中定义的公共对象列表
def __dir__():
    return __all__


# 定义特殊方法__getattr__(name)，用于动态获取模块中的属性，当属性未找到时调用_sub_module_deprecation
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="odr",   # 子模块名称为'odr'
        module="models",     # 模块名称为'models'
        private_modules=["_models"],  # 私有模块列表为['_models']
        all=__all__,         # 公共对象列表为定义的__all__
        attribute=name       # 请求的属性名称
    )
```