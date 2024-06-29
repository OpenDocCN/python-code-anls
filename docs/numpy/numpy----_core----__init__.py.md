# `.\numpy\numpy\_core\__init__.py`

```py
"""
Contains the core of NumPy: ndarray, ufuncs, dtypes, etc.

Please note that this module is private.  All functions and objects
are available in the main ``numpy`` namespace - use that instead.

"""

# 导入标准库 os
import os

# 从 numpy.version 模块导入版本号作为 __version__
from numpy.version import version as __version__

# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
# 禁用 OpenBLAS 对主线程的亲和性设置，以免限制 Python 线程或进程只能使用一个核心
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)

try:
    # 尝试从当前包中导入 multiarray 模块
    from . import multiarray
except ImportError as exc:
    # 如果导入失败，处理 ImportError 异常
    import sys
    msg = """

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python%d.%d from "%s"
  * The NumPy version is: "%s"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: %s
""" % (sys.version_info[0], sys.version_info[1], sys.executable,
        __version__, exc)
    raise ImportError(msg)
finally:
    # 无论是否发生异常，都要删除添加的环境变量
    for envkey in env_added:
        del os.environ[envkey]
# 删除临时变量和模块
del envkey
del env_added
del os

# 从当前包中导入 umath 模块
from . import umath

# Check that multiarray, umath are pure python modules wrapping
# _multiarray_umath and not either of the old c-extension modules
# 检查 multiarray 和 umath 是否是纯 Python 模块，包装了 _multiarray_umath，而不是旧的 C 扩展模块
if not (hasattr(multiarray, '_multiarray_umath') and
        hasattr(umath, '_multiarray_umath')):
    import sys
    path = sys.modules['numpy'].__path__
    msg = ("Something is wrong with the numpy installation. "
        "While importing we detected an older version of "
        "numpy in {}. One method of fixing this is to repeatedly uninstall "
        "numpy until none is found, then reinstall this version.")
    raise ImportError(msg.format(path))

# 从当前包中导入 numerictypes 模块，并引入其 sctypes 和 sctypeDict
from . import numerictypes as nt
from .numerictypes import sctypes, sctypeDict
# 设置 multiarray 模块的 typeDict 属性为 nt 模块的 sctypeDict
multiarray.set_typeDict(nt.sctypeDict)
# 从当前包中导入 numeric 模块的所有内容
from . import numeric
from .numeric import *
# 从当前包中导入 fromnumeric 模块的所有内容
from . import fromnumeric
from .fromnumeric import *
# 从当前包中导入 records 模块的 record 和 recarray 类
from .records import record, recarray
# Note: module name memmap is overwritten by a class with same name
# 从 memmap 模块导入所有内容（注意：模块名 memmap 被同名类覆盖）
from .memmap import *
# 从当前包中导入 function_base 模块的所有内容
from . import function_base
from .function_base import *
# 从当前包中导入 _machar 模块
from . import _machar
# 从当前包中导入 getlimits 模块的所有内容
from . import getlimits
from .getlimits import *
# 从当前包中导入 shape_base 模块的所有内容
from . import shape_base
from .shape_base import *
# 从当前包中导入 einsumfunc 模块的所有内容
from . import einsumfunc
from .einsumfunc import *
# 删除 nt 变量，清理命名空间
del nt

# 从 numeric 模块中导入 absolute 函数并命名为 abs
from .numeric import absolute as abs

# do this after everything else, to minimize the chance of this misleadingly
# appearing in an import-time traceback
# 从当前包中导入 _add_newdocs 模块
from . import _add_newdocs
# 从当前包中导入 _add_newdocs_scalars 模块
from . import _add_newdocs_scalars
# add these for module-freeze analysis (like PyInstaller)
# 从当前包中导入 _dtype_ctypes 模块
from . import _dtype_ctypes
# 从当前包中导入 _internal 模块
from . import _internal
from . import _dtype
from . import _methods


# 导入模块 _dtype 和 _methods，它们位于当前包中的子模块



acos = numeric.arccos
acosh = numeric.arccosh
asin = numeric.arcsin
asinh = numeric.arcsinh
atan = numeric.arctan
atanh = numeric.arctanh
atan2 = numeric.arctan2
concat = numeric.concatenate
bitwise_left_shift = numeric.left_shift
bitwise_invert = numeric.invert
bitwise_right_shift = numeric.right_shift
permute_dims = numeric.transpose
pow = numeric.power


# 给一些函数和方法赋值，以便后续使用，这些函数和方法来自 numeric 模块
# numeric 是一个导入的模块，这些函数和方法在后续代码中可能会被使用



__all__ = [
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "atan2",
    "bitwise_invert", "bitwise_left_shift", "bitwise_right_shift", "concat",
    "pow", "permute_dims", "memmap", "sctypeDict", "record", "recarray"
]
__all__ += numeric.__all__
__all__ += function_base.__all__
__all__ += getlimits.__all__
__all__ += shape_base.__all__
__all__ += einsumfunc.__all__


# 定义 __all__ 列表，包含导出的模块、函数和方法名称
# 这些名称被视为模块的公共 API，可以通过 `from module import *` 导入
# 还包括其他模块的公共 API，如 numeric、function_base、getlimits、shape_base 和 einsumfunc



def _ufunc_reduce(func):
    # 返回函数的 __name__ 属性，用于在 pickle 模块中找到相应的模块
    # pickle 模块支持使用 `__qualname__` 来查找模块，这对于明确指定 ufuncs（通用函数）很有用
    # 参考：https://github.com/dask/distributed/issues/3450
    return func.__name__


def _DType_reconstruct(scalar_type):
    # 这是 pickle np.dtype(np.float64) 等类型的一个解决方法
    # 应该用更好的解决方法替代，比如当 DTypes 变为 HeapTypes 时
    return type(dtype(scalar_type))


def _DType_reduce(DType):
    # 大多数 DTypes 可以简单地通过它们的名称来 pickle
    if not DType._legacy or DType.__module__ == "numpy.dtypes":
        return DType.__name__

    # 对于用户定义的 legacy DTypes（如 rational），它们不在 numpy.dtypes 中并且没有公共类
    # 对于这些情况，我们通过从标量类型重建它们来 pickle 它们
    scalar_type = DType.type
    return _DType_reconstruct, (scalar_type,)


# 定义了两个辅助函数，用于 pickle 特定类型的对象
# _ufunc_reduce 用于通用函数的 pickle
# _DType_reduce 和 _DType_reconstruct 用于数据类型对象的 pickle



def __getattr__(name):
    # Deprecated 2022-11-22, NumPy 1.25.
    if name == "MachAr":
        import warnings
        warnings.warn(
            "The `np._core.MachAr` is considered private API (NumPy 1.24)",
            DeprecationWarning, stacklevel=2,
        )
        return _machar.MachAr
    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}")


# __getattr__ 方法用于动态获取模块的属性
# 如果请求的属性名是 "MachAr"，则发出警告，并返回 _machar.MachAr
# 否则，引发 AttributeError 异常，指示模块没有该属性
# 注意：该方法已于 NumPy 1.25 弃用于 2022-11-22



import copyreg

copyreg.pickle(ufunc, _ufunc_reduce)
copyreg.pickle(type(dtype), _DType_reduce, _DType_reconstruct)

# Unclutter namespace (must keep _*_reconstruct for unpickling)
del copyreg, _ufunc_reduce, _DType_reduce


# 使用 copyreg.pickle 方法注册用于序列化的函数
# ufunc 对象使用 _ufunc_reduce 函数进行 pickle
# dtype 类型使用 _DType_reduce 和 _DType_reconstruct 进行 pickle
# 删除 copyreg、_ufunc_reduce 和 _DType_reduce 变量，以清理命名空间



from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester


# 导入 PytestTester 类，并将当前模块的名称传递给它来进行测试
# 将 PytestTester 实例赋值给 test 变量
# 删除 PytestTester 类，以清理命名空间
```