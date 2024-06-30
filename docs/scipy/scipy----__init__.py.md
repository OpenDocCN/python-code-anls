# `D:\src\scipysrc\scipy\scipy\__init__.py`

```
"""
SciPy: A scientific computing package for Python
================================================

Documentation is available in the docstrings and
online at https://docs.scipy.org.

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import scipy.cluster``.

::

 cluster                      --- Vector Quantization / Kmeans
 constants                    --- Physical and mathematical constants and units
 datasets                     --- Dataset methods
 fft                          --- Discrete Fourier transforms
 fftpack                      --- Legacy discrete Fourier transforms
 integrate                    --- Integration routines
 interpolate                  --- Interpolation Tools
 io                           --- Data input and output
 linalg                       --- Linear algebra routines
 misc                         --- Utilities that don't have another home.
 ndimage                      --- N-D image package
 odr                          --- Orthogonal Distance Regression
 optimize                     --- Optimization Tools
 signal                       --- Signal Processing Tools
 sparse                       --- Sparse Matrices
 spatial                      --- Spatial data structures and algorithms
 special                      --- Special functions
 stats                        --- Statistical Functions

Public API in the main SciPy namespace
--------------------------------------
::

 __version__       --- SciPy version string
 LowLevelCallable  --- Low-level callback function
 show_config       --- Show scipy build configuration
 test              --- Run scipy unittests

"""

# 导入标准库中的 importlib 模块，并将其重命名为 _importlib
import importlib as _importlib

# 从 numpy 模块中导入 __version__ 别名为 __numpy_version__
from numpy import __version__ as __numpy_version__

# 尝试导入 scipy.__config__ 模块中的 show 函数，并将其命名为 show_config
try:
    from scipy.__config__ import show as show_config
# 如果导入失败，将错误信息存储在变量 msg 中，并引发 ImportError 异常
except ImportError as e:
    msg = """Error importing SciPy: you cannot import SciPy while
    being in scipy source directory; please exit the SciPy source
    tree first and relaunch your Python interpreter."""
    raise ImportError(msg) from e

# 从 scipy.version 模块中导入 version 别名为 __version__
from scipy.version import version as __version__

# 允许分发商运行自定义的初始化代码，从当前模块中导入 _distributor_init 后立即删除
from . import _distributor_init
del _distributor_init

# 从 scipy._lib 模块中导入 _pep440
from scipy._lib import _pep440

# 在维护分支中，如果 numpy 的版本为 N，则将 np_maxversion 更改为 N+3
np_minversion = '1.23.5'
np_maxversion = '9.9.99'

# 如果当前 numpy 版本低于 np_minversion 或高于等于 np_maxversion，则发出警告
if (_pep440.parse(__numpy_version__) < _pep440.Version(np_minversion) or
        _pep440.parse(__numpy_version__) >= _pep440.Version(np_maxversion)):
    import warnings
    # 发出用户警告，提醒需要特定版本的 numpy 才能正常使用当前版本的 SciPy
    warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
                  f" is required for this version of SciPy (detected "
                  f"version {__numpy_version__})",
                  UserWarning, stacklevel=2)
del _pep440

# 这是 SciPy 中第一个扩展模块的导入。如果安装存在一般性问题，例如缺少扩展模块，则会报错。
# 尝试导入 LowLevelCallable 类，如果导入失败会引发 ImportError
try:
    from scipy._lib._ccallback import LowLevelCallable
# 如果 ImportError 发生，生成详细的错误消息并抛出新的 ImportError
except ImportError as e:
    # 错误消息说明 scipy 安装似乎存在问题，无法导入扩展模块
    msg = "The `scipy` install you are using seems to be broken, " + \
          "(extension modules cannot be imported), " + \
          "please try reinstalling."
    raise ImportError(msg) from e


# 导入 PytestTester 类并用当前模块名初始化测试对象
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，仅保留测试对象的引用
del PytestTester


# 定义 scipy 的子模块列表
submodules = [
    'cluster',
    'constants',
    'datasets',
    'fft',
    'fftpack',
    'integrate',
    'interpolate',
    'io',
    'linalg',
    'misc',
    'ndimage',
    'odr',
    'optimize',
    'signal',
    'sparse',
    'spatial',
    'special',
    'stats'
]

# 将所有子模块加入到 __all__ 列表中，并加入其他需要导出的对象
__all__ = submodules + [
    'LowLevelCallable',  # 导出 LowLevelCallable 类
    'test',               # 导出测试对象
    'show_config',        # 导出 show_config 函数
    '__version__',        # 导出 __version__ 变量
]


# 定义 __dir__ 特殊方法，返回所有需要导出的对象列表
def __dir__():
    return __all__


# 定义 __getattr__ 特殊方法，根据属性名动态返回对应的模块或全局对象
def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'scipy.{name}')  # 动态导入对应的子模块
    else:
        try:
            return globals()[name]  # 尝试返回全局变量中的对象
        except KeyError:
            raise AttributeError(
                f"Module 'scipy' has no attribute '{name}'"  # 如果对象不存在，抛出 AttributeError
            )
```