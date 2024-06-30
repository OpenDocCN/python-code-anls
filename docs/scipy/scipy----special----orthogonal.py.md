# `D:\src\scipysrc\scipy\scipy\special\orthogonal.py`

```
# 导入 _sub_module_deprecation 函数，用于处理模块中废弃的子模块和函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义多项式函数名称列表
_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys',
             'jacobi', 'laguerre', 'genlaguerre', 'hermite',
             'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt',
             'sh_chebyu', 'sh_jacobi']

# 新旧函数名称映射字典，用于根据新名称查找旧名称
_rootfuns_map = {'roots_legendre': 'p_roots',
               'roots_chebyt': 't_roots',
               'roots_chebyu': 'u_roots',
               'roots_chebyc': 'c_roots',
               'roots_chebys': 's_roots',
               'roots_jacobi': 'j_roots',
               'roots_laguerre': 'l_roots',
               'roots_genlaguerre': 'la_roots',
               'roots_hermite': 'h_roots',
               'roots_hermitenorm': 'he_roots',
               'roots_gegenbauer': 'cg_roots',
               'roots_sh_legendre': 'ps_roots',
               'roots_sh_chebyt': 'ts_roots',
               'roots_sh_chebyu': 'us_roots',
               'roots_sh_jacobi': 'js_roots'}

# 将所有公开的函数名汇总到 __all__ 列表中，用于 from ... import * 语句的支持
__all__ = _polyfuns + list(_rootfuns_map.keys()) + [
    'airy', 'p_roots', 't_roots', 'u_roots', 'c_roots', 's_roots',
    'j_roots', 'l_roots', 'la_roots', 'h_roots', 'he_roots', 'cg_roots',
    'ps_roots', 'ts_roots', 'us_roots', 'js_roots'
]


# 定义 __dir__ 函数，返回包含所有公开函数名称的列表 __all__
def __dir__():
    return __all__


# 定义 __getattr__ 函数，用于动态获取特定属性名称对应的对象，支持废弃模块和函数的警告
def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="orthogonal",
                                   private_modules=["_orthogonal"], all=__all__,
                                   attribute=name)
```