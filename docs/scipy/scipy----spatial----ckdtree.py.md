# `D:\src\scipysrc\scipy\scipy\spatial\ckdtree.py`

```
# 引入 _sub_module_deprecation 函数，用于处理子模块的弃用警告和替换
from scipy._lib.deprecation import _sub_module_deprecation

# 定义导出模块的列表，这些模块名将被导出
__all__ = ["cKDTree"]  # noqa: F822

# 定义一个特殊的函数 __dir__()，返回当前模块的导出成员列表
def __dir__():
    return __all__

# 定义一个特殊的函数 __getattr__(name)，用于动态获取属性
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，传递参数指定替换的子包、模块、私有模块列表和导出列表等信息
    return _sub_module_deprecation(sub_package="spatial", module="ckdtree",
                                   private_modules=["_ckdtree"], all=__all__,
                                   attribute=name)
```