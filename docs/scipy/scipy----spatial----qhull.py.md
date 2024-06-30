# `D:\src\scipysrc\scipy\scipy\spatial\qhull.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块的弃用警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义了该模块中公开的类和函数名称列表
__all__ = [  # noqa: F822
    'ConvexHull',                     # 凸包计算类
    'Delaunay',                       # Delaunay 三角剖分类
    'HalfspaceIntersection',          # 半空间交集类
    'QhullError',                     # Qhull 出错异常类
    'Voronoi',                        # Voronoi 图计算类
    'tsearch',                        # tsearch 函数
]


# 定义 __dir__() 函数，返回模块公开的所有名称列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，用于动态获取指定名称的属性
def __getattr__(name):
    return _sub_module_deprecation(sub_package="spatial", module="qhull",
                                   private_modules=["_qhull"], all=__all__,
                                   attribute=name)
```