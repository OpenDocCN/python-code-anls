# `D:\src\scipysrc\sympy\sympy\external\gmpy.py`

```
#
# 导入所需的标准库和第三方模块
#
import os  # 导入操作系统相关功能模块
from ctypes import c_long, sizeof  # 导入 ctypes 模块中的 c_long 和 sizeof 函数
from functools import reduce  # 导入 functools 模块中的 reduce 函数
from typing import Tuple as tTuple, Type  # 导入 typing 模块中的 Tuple 别名 tTuple 和 Type 类型
from warnings import warn  # 导入 warnings 模块中的 warn 函数

#
# 导入 Sympy 所需的外部模块
#
from sympy.external import import_module  # 从 sympy.external 导入 import_module 函数

#
# 导入自定义模块
#
from .pythonmpq import PythonMPQ  # 从当前包中导入 pythonmpq 模块的 PythonMPQ 类
from .ntheory import (  # 从当前包中导入 ntheory 模块的以下函数和别名
    bit_scan1 as python_bit_scan1,
    bit_scan0 as python_bit_scan0,
    remove as python_remove,
    factorial as python_factorial,
    sqrt as python_sqrt,
    sqrtrem as python_sqrtrem,
    gcd as python_gcd,
    lcm as python_lcm,
    gcdext as python_gcdext,
    is_square as python_is_square,
    invert as python_invert,
    legendre as python_legendre,
    jacobi as python_jacobi,
    kronecker as python_kronecker,
    iroot as python_iroot,
    is_fermat_prp as python_is_fermat_prp,
    is_euler_prp as python_is_euler_prp,
    is_strong_prp as python_is_strong_prp,
    is_fibonacci_prp as python_is_fibonacci_prp,
    is_lucas_prp as python_is_lucas_prp,
    is_selfridge_prp as python_is_selfridge_prp,
    is_strong_lucas_prp as python_is_strong_lucas_prp,
    is_strong_selfridge_prp as python_is_strong_selfridge_prp,
    is_bpsw_prp as python_is_bpsw_prp,
    is_strong_bpsw_prp as python_is_strong_bpsw_prp,
)

#
# 定义 __all__ 列表，包含导出的符号名
#
__all__ = [
    # 根据 SYMPY_GROUND_TYPES 自动选择 GROUND_TYPES 的值，可能为 'gmpy' 或 'python'
    'GROUND_TYPES',

    # 检查是否安装了 gmpy，如果是，则 HAS_GMPY 为 2；否则为 0
    'HAS_GMPY',

    # 根据 GROUND_TYPES 选择整数类型的基本类型，可能为 (int,) 或 (int, type(mpz(0)))
    'SYMPY_INTS',

    # MPQ 类型，根据 GROUND_TYPES 可能是 gmpy.mpq 或 sympy.external.pythonmpq 中的 PythonMPQ
    'MPQ',

    # MPZ 类型，根据 GROUND_TYPES 可能是 gmpy.mpz 或 int
    'MPZ',

    # 下列函数和别名用于数论运算，从 ntheory 模块导入
    'bit_scan1',
    'bit_scan0',
    'remove',
    'factorial',
    'sqrt',
    'is_square',
    'sqrtrem',
    'gcd',
    'lcm',
    'gcdext',
    'invert',
    'legendre',
    'jacobi',
    'kronecker',
    'iroot',
    'is_fermat_prp',
    'is_euler_prp',
    'is_strong_prp',
    'is_fibonacci_prp',
    'is_lucas_prp',
    'is_selfridge_prp',
    'is_strong_lucas_prp',
    'is_strong_selfridge_prp',
    'is_bpsw_prp',
    'is_strong_bpsw_prp',
]

#
# python-flint 版本检查，当前版本为 "0.6.*"，未来版本可能支持但需通过 SYMPY_GROUND_TYPES=flint 明确指定
#
_PYTHON_FLINT_VERSION_NEEDED = "0.6.*"


def _flint_version_okay(flint_version):
    # 检查 python-flint 版本是否符合要求，只比较前两个版本号
    flint_ver = flint_version.split('.')[:2]
    needed_ver = _PYTHON_FLINT_VERSION_NEEDED.split('.')[:2]
    return flint_ver == needed_ver

#
# 仅当 gmpy2 版本 >= 2.0.0 时才使用
#
_GMPY2_MIN_VERSION = '2.0.0'


def _get_flint(sympy_ground_types):
    # 检查 sympy_ground_types 是否不是 'auto' 或 'flint'，如果是则返回 None
    if sympy_ground_types not in ('auto', 'flint'):
        return None

    try:
        # 尝试导入 flint 模块
        import flint
        # 导入 flint 模块的版本信息，如果早期版本可能没有 __version__ 属性
        from flint import __version__ as _flint_version
    except ImportError:
        # 如果导入失败且 sympy_ground_types 是 'flint'，发出警告并返回 None
        if sympy_ground_types == 'flint':
            warn("SYMPY_GROUND_TYPES was set to flint but python-flint is not "
                 "installed. Falling back to other ground types.")
        return None

    # 检查 _flint_version 是否符合要求
    if _flint_version_okay(_flint_version):
        # 如果版本符合要求，返回 flint 模块对象
        return flint
    elif sympy_ground_types == 'auto':
        # 如果 sympy_ground_types 是 'auto'，发出警告说明 python-flint 的版本不符合默认要求
        warn(f"python-flint {_flint_version} is installed but only version "
             f"{_PYTHON_FLINT_VERSION_NEEDED} will be used by default. "
             f"Falling back to other ground types. Use "
             f"SYMPY_GROUND_TYPES=flint to force the use of python-flint.")
        return None
    else:
        # 如果 sympy_ground_types 是 'flint'，但版本不符合要求，发出警告
        warn(f"Using python-flint {_flint_version} because SYMPY_GROUND_TYPES "
             f"is set to flint but this version of SymPy has only been tested "
             f"with python-flint {_PYTHON_FLINT_VERSION_NEEDED}.")
        return flint
# 检查 SYMPY_GROUND_TYPES 是否为 'auto', 'gmpy', 或 'gmpy2'，如果不是则返回 None
def _get_gmpy2(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'gmpy', 'gmpy2'):
        return None

    # 动态导入 gmpy2 模块，并确保其版本符合最低要求
    gmpy = import_module('gmpy2', min_module_version=_GMPY2_MIN_VERSION,
            module_version_attr='version', module_version_attr_call_args=())

    # 如果指定了具体的 gmpy 类型但 gmpy2 模块未安装，则警告并切换至 'python' 类型
    if sympy_ground_types != 'auto' and gmpy is None:
        warn("gmpy2 library is not installed, switching to 'python' ground types")

    # 返回导入的 gmpy2 模块对象或 None
    return gmpy


#
# 从环境变量 SYMPY_GROUND_TYPES 获取 Sympy 的地面类型，默认为 'auto'
#
_SYMPY_GROUND_TYPES = os.environ.get('SYMPY_GROUND_TYPES', 'auto').lower()
_flint = None
_gmpy = None

#
# 首先处理 flint/gmpy2 的自动检测。优先选择 flint（如果可用），然后是 gmpy2，最后是 python 类型。
#
if _SYMPY_GROUND_TYPES in ('auto', 'flint'):
    # 尝试获取 flint 类型
    _flint = _get_flint(_SYMPY_GROUND_TYPES)
    if _flint is not None:
        _SYMPY_GROUND_TYPES = 'flint'
    else:
        _SYMPY_GROUND_TYPES = 'auto'

if _SYMPY_GROUND_TYPES in ('auto', 'gmpy', 'gmpy2'):
    # 尝试获取 gmpy2 类型
    _gmpy = _get_gmpy2(_SYMPY_GROUND_TYPES)
    if _gmpy is not None:
        _SYMPY_GROUND_TYPES = 'gmpy'
    else:
        _SYMPY_GROUND_TYPES = 'python'

if _SYMPY_GROUND_TYPES not in ('flint', 'gmpy', 'python'):
    # 如果 SYMPY_GROUND_TYPES 环境变量值不被识别，则发出警告并将其设为 'python'
    warn("SYMPY_GROUND_TYPES environment variable unrecognised. "
         "Should be 'auto', 'flint', 'gmpy', 'gmpy2' or 'python'.")
    _SYMPY_GROUND_TYPES = 'python'

#
# 到此为止，_SYMPY_GROUND_TYPES 必定为 'flint', 'gmpy' 或 'python'。以下块定义了每种情况下导出的值。
#

#
# 在 gmpy2 和 flint 中，有些函数接受 long（或 unsigned long）类型的参数，即不能超过此值。
#
LONG_MAX = (1 << (8*sizeof(c_long) - 1)) - 1

#
# 类型检查器对 SYMPY_INTS 的含义可能有些困惑。可能有更好的类型提示，如 Type[Integral] 或类似的。
#
SYMPY_INTS: tTuple[Type, ...]

if _SYMPY_GROUND_TYPES == 'gmpy':

    assert _gmpy is not None

    flint = None
    gmpy = _gmpy

    # 设置 gmpy2 模块的相关常量和函数
    HAS_GMPY = 2
    GROUND_TYPES = 'gmpy'
    SYMPY_INTS = (int, type(gmpy.mpz(0)))
    MPZ = gmpy.mpz
    MPQ = gmpy.mpq

    bit_scan1 = gmpy.bit_scan1
    bit_scan0 = gmpy.bit_scan0
    remove = gmpy.remove
    factorial = gmpy.fac
    sqrt = gmpy.isqrt
    is_square = gmpy.is_square
    sqrtrem = gmpy.isqrt_rem
    gcd = gmpy.gcd
    lcm = gmpy.lcm
    gcdext = gmpy.gcdext
    invert = gmpy.invert
    legendre = gmpy.legendre
    jacobi = gmpy.jacobi
    kronecker = gmpy.kronecker

    def iroot(x, n):
        # 在最新的 gmpy2 中，对于 n 的阈值是 ULONG_MAX，
        # 但这里适应旧版本的设置。
        if n <= LONG_MAX:
            return gmpy.iroot(x, n)
        return python_iroot(x, n)

    is_fermat_prp = gmpy.is_fermat_prp
    is_euler_prp = gmpy.is_euler_prp
    is_strong_prp = gmpy.is_strong_prp
    is_fibonacci_prp = gmpy.is_fibonacci_prp
    is_lucas_prp = gmpy.is_lucas_prp
    is_selfridge_prp = gmpy.is_selfridge_prp
    # 将 gmpy 模块中的函数赋值给对应的变量，用于进行不同类型的素数性质测试
    is_strong_lucas_prp = gmpy.is_strong_lucas_prp
    is_strong_selfridge_prp = gmpy.is_strong_selfridge_prp
    is_bpsw_prp = gmpy.is_bpsw_prp
    is_strong_bpsw_prp = gmpy.is_strong_bpsw_prp
elif _SYMPY_GROUND_TYPES == 'flint':
    # 如果使用 flint 数字类型，进行以下设置

    assert _flint is not None
    # 确保 _flint 已经被正确地导入和定义

    flint = _flint
    # 将 _flint 赋值给 flint 变量，用于后续的 flint 数字类型操作

    gmpy = None
    # 设置 gmpy 为 None，因为没有使用 gmpy 库

    HAS_GMPY = 0
    # HAS_GMPY 标志设置为 0，表示没有使用 gmpy 库

    GROUND_TYPES = 'flint'
    # 将 GROUND_TYPES 设置为 'flint'，表示使用 flint 数字类型

    SYMPY_INTS = (int, flint.fmpz) # type: ignore
    # 定义 SYMPY_INTS 为一个元组，包含 int 和 flint.fmpz 类型

    MPZ = flint.fmpz # type: ignore
    # 将 MPZ 定义为 flint.fmpz 类型，用于整数运算

    bit_scan1 = python_bit_scan1
    # 将 bit_scan1 定义为 python_bit_scan1 函数，用于位操作

    bit_scan0 = python_bit_scan0
    # 将 bit_scan0 定义为 python_bit_scan0 函数，用于位操作

    remove = python_remove
    # 将 remove 定义为 python_remove 函数，用于列表元素移除

    factorial = python_factorial
    # 将 factorial 定义为 python_factorial 函数，用于计算阶乘

    def sqrt(x):
        return flint.fmpz(x).isqrt()
    # 定义 sqrt 函数，使用 flint.fmpz 类型计算平方根

    def is_square(x):
        if x < 0:
            return False
        return flint.fmpz(x).sqrtrem()[1] == 0
    # 定义 is_square 函数，使用 flint.fmpz 类型判断是否为完全平方数

    def sqrtrem(x):
        return flint.fmpz(x).sqrtrem()
    # 定义 sqrtrem 函数，使用 flint.fmpz 类型计算平方根余数

    def gcd(*args):
        return reduce(flint.fmpz.gcd, args, flint.fmpz(0))
    # 定义 gcd 函数，使用 flint.fmpz 类型计算多个数的最大公约数

    def lcm(*args):
        return reduce(flint.fmpz.lcm, args, flint.fmpz(1))
    # 定义 lcm 函数，使用 flint.fmpz 类型计算多个数的最小公倍数

    gcdext = python_gcdext
    # 将 gcdext 定义为 python_gcdext 函数，用于扩展的最大公约数计算

    invert = python_invert
    # 将 invert 定义为 python_invert 函数，用于模反元素的计算

    legendre = python_legendre
    # 将 legendre 定义为 python_legendre 函数，用于 Legendre 符号的计算

    def jacobi(x, y):
        if y <= 0 or not y % 2:
            raise ValueError("y should be an odd positive integer")
        return flint.fmpz(x).jacobi(y)
    # 定义 jacobi 函数，使用 flint.fmpz 类型计算雅可比符号

    kronecker = python_kronecker
    # 将 kronecker 定义为 python_kronecker 函数，用于 Kronecker 符号的计算

    def iroot(x, n):
        if n <= LONG_MAX:
            y = flint.fmpz(x).root(n)
            return y, y**n == x
        return python_iroot(x, n)
    # 定义 iroot 函数，使用 flint.fmpz 类型计算整数 n 次方根

    is_fermat_prp = python_is_fermat_prp
    # 将 is_fermat_prp 定义为 python_is_fermat_prp 函数，用于费马伪素数判断

    is_euler_prp = python_is_euler_prp
    # 将 is_euler_prp 定义为 python_is_euler_prp 函数，用于欧拉伪素数判断

    is_strong_prp = python_is_strong_prp
    # 将 is_strong_prp 定义为 python_is_strong_prp 函数，用于强伪素数判断

    is_fibonacci_prp = python_is_fibonacci_prp
    # 将 is_fibonacci_prp 定义为 python_is_fibonacci_prp 函数，用于斐波那契伪素数判断

    is_lucas_prp = python_is_lucas_prp
    # 将 is_lucas_prp 定义为 python_is_lucas_prp 函数，用于卢卡斯伪素数判断

    is_selfridge_prp = python_is_selfridge_prp
    # 将 is_selfridge_prp 定义为 python_is_selfridge_prp 函数，用于塞尔矩伪素数判断

    is_strong_lucas_prp = python_is_strong_lucas_prp
    # 将 is_strong_lucas_prp 定义为 python_is_strong_lucas_prp 函数，用于强卢卡斯伪素数判断

    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    # 将 is_strong_selfridge_prp 定义为 python_is_strong_selfridge_prp 函数，用于强塞尔矩伪素数判断

    is_bpsw_prp = python_is_bpsw_prp
    # 将 is_bpsw_prp 定义为 python_is_bpsw_prp 函数，用于 BPSW 伪素数判断

    is_strong_bpsw_prp = python_is_strong_bpsw_prp
    # 将 is_strong_bpsw_prp 定义为 python_is_strong_bpsw_prp 函数，用于强 BPSW 伪素数判断

elif _SYMPY_GROUND_TYPES == 'python':
    # 如果使用 python 数字类型，进行以下设置

    flint = None
    # 将 flint 设置为 None，因为没有使用 flint 库

    gmpy = None
    # 将 gmpy 设置为 None，因为没有使用 gmpy 库

    HAS_GMPY = 0
    # HAS_GMPY 标志设置为 0，表示没有使用 gmpy 库

    GROUND_TYPES = 'python'
    # 将 GROUND_TYPES 设置为 'python'，表示使用 python 原生整数类型

    SYMPY_INTS = (int,)
    # 定义 SYMPY_INTS 为一个元组，只包含 int 类型

    MPZ = int
    # 将 MPZ 定义为 int 类型，用于整数运算

    MPQ = PythonMPQ
    # 将 MPQ 定义为 PythonMPQ 类型，用于有理数运算

    bit_scan1 = python_bit_scan1
    # 将 bit_scan1 定义为 python_bit_scan1 函数，用于位操作

    bit_scan0 = python_bit_scan0
    # 将 bit_scan0 定义为 python_bit_scan0 函数，用于位操作

    remove = python_remove
    # 将 remove 定义为 python_remove 函数，用于列表元素移除

    factorial = python_factorial
    # 将 factorial 定义为 python_factorial 函数，用于计算阶乘

    sqrt = python_sqrt
    # 将 sqrt 定义为 python_sqrt 函数，用于计算平方根

    is_square = python_is_square
    # 将 is_square 定义为 python_is_square 函数，用于判断是否为完全平方数

    sqrtrem = python_sqrtrem
    # 将 sqrtrem 定义为 python_sqrtrem 函数，用于计算平方根余数

    gcd = python_gcd
    # 将 gcd 定义为 python_gcd 函数，用于计算最大公约数

    lcm = python_lcm
    # 将 lcm 定义为 python_lcm 函数，用于计算最小公倍数

    gcdext = python_gcdext
    # 将 gcdext 定义为 python_gcdext 函数，用于扩展的最大公约数计算

    invert
```