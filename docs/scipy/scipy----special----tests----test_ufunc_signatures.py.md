# `D:\src\scipysrc\scipy\scipy\special\tests\test_ufunc_signatures.py`

```
"""
Test that all ufuncs have float32-preserving signatures.

This was once guaranteed through the code generation script for
generating ufuncs, `scipy/special/_generate_pyx.py`. Starting with
gh-20260, SciPy developers have begun moving to generate ufuncs
through direct use of the NumPy C API (through C++). Existence of
float32 preserving signatures must now be tested since it is no
longer guaranteed.
"""

# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架
import scipy.special._ufuncs  # 导入 scipy.special._ufuncs 模块
import scipy.special._gufuncs  # 导入 scipy.special._gufuncs 模块

# 初始化一个空列表来存储所有的 ufunc
_ufuncs = []

# 遍历 scipy.special._ufuncs 模块中的所有成员
for funcname in dir(scipy.special._ufuncs):
    _ufuncs.append(getattr(scipy.special._ufuncs, funcname))

# 遍历 scipy.special._gufuncs 模块中的所有成员
for funcname in dir(scipy.special._gufuncs):
    _ufuncs.append(getattr(scipy.special._gufuncs, funcname))

# 筛选出 _ufuncs 列表中的 ufunc 对象
_ufuncs = [func for func in _ufuncs if isinstance(func, np.ufunc)]

# 使用 @pytest.mark.parametrize 装饰器对每个 ufunc 进行参数化测试
@pytest.mark.parametrize("ufunc", _ufuncs)
def test_ufunc_signatures(ufunc):

    # From _generate_pyx.py
    # "Don't add float32 versions of ufuncs with integer arguments, as this
    # can lead to incorrect dtype selection if the integer arguments are
    # arrays, but float arguments are scalars.
    # For instance sph_harm(0,[0],0,0).dtype == complex64
    # This may be a NumPy bug, but we need to work around it.
    # cf. gh-4895, https://github.com/numpy/numpy/issues/5895"
    
    # 从 _generate_pyx.py 中引用的注释，解释不应该为带有整数参数的 ufuncs 添加 float32 版本，
    # 因为这可能导致 dtype 选择错误，特别是如果整数参数是数组而浮点参数是标量时。
    types = set(sig for sig in ufunc.types if not ("l" in sig or "i" in sig))

    # 生成应该存在的完整扩展签名集合。对于任何现有的签名，应该存在匹配的 float 和 double 版本。
    expanded_types = set()
    for sig in types:
        expanded_types.update(
            [sig.replace("d", "f").replace("D", "F"),
             sig.replace("f", "d").replace("F", "D")]
        )

    # 断言：types 和 expanded_types 应该相等，即每个现有的签名应该有匹配的 float 和 double 版本。
    assert types == expanded_types
```