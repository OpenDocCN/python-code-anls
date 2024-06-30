# `D:\src\scipysrc\scipy\scipy\ndimage\morphology.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块被弃用的情况
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，声明模块中公开的函数和变量名，用于导入时的限定
__all__ = [
    'iterate_structure', 'generate_binary_structure',
    'binary_erosion', 'binary_dilation', 'binary_opening',
    'binary_closing', 'binary_hit_or_miss', 'binary_propagation',
    'binary_fill_holes', 'grey_erosion', 'grey_dilation',
    'grey_opening', 'grey_closing', 'morphological_gradient',
    'morphological_laplace', 'white_tophat', 'black_tophat',
    'distance_transform_bf', 'distance_transform_cdt',
    'distance_transform_edt'
]

# 定义 __dir__() 函数，返回当前模块的所有公开成员
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理对未定义属性的访问
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，标记子模块 'ndimage' 中的 'morphology' 模块作为已弃用
    # private_modules 参数指定要弃用的私有模块列表，all 参数指定公开的成员列表，attribute 参数是被访问的属性名
    return _sub_module_deprecation(sub_package='ndimage', module='morphology',
                                   private_modules=['_morphology'], all=__all__,
                                   attribute=name)
```