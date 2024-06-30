# `D:\src\scipysrc\scipy\scipy\spatial\kdtree.py`

```
# 这个文件不适合公共使用，并且将在 SciPy v2.0.0 中移除。
# 使用 `scipy.spatial` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，指定导出的公共接口，禁止 Flake8 linter 报错 F822
__all__ = [
    'KDTree',              # KDTree 类
    'Rectangle',           # Rectangle 类
    'cKDTree',             # cKDTree 类
    'distance_matrix',     # distance_matrix 函数
    'minkowski_distance',  # minkowski_distance 函数
    'minkowski_distance_p' # minkowski_distance_p 函数
]

# 定义 __dir__() 函数，返回 __all__ 列表，用于指定对象的属性列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理动态属性访问
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="spatial",  # 子包名为 "spatial"
        module="kdtree",        # 模块名为 "kdtree"
        private_modules=["_kdtree"],  # 私有模块列表，包括 "_kdtree"
        all=__all__,            # 所有导出的公共接口列表
        attribute=name          # 请求的属性名
    )
```