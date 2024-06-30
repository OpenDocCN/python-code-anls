# `D:\src\scipysrc\sympy\sympy\discrete\convolutions.py`

```
"""
Convolution (using **FFT**, **NTT**, **FWHT**), Subset Convolution,
Covering Product, Intersecting Product
"""

from sympy.core import S, sympify, Rational  # 导入符号计算相关的核心模块
from sympy.core.function import expand_mul  # 导入用于函数展开的模块
from sympy.discrete.transforms import (  # 导入离散变换相关的模块
    fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from sympy.external.gmpy import MPZ, lcm  # 导入外部库中的整数运算相关模块
from sympy.utilities.iterables import iterable  # 导入用于处理可迭代对象的工具模块
from sympy.utilities.misc import as_int  # 导入将对象转换为整数的工具函数


def convolution(a, b, cycle=0, dps=None, prime=None, dyadic=None, subset=None):
    """
    Performs convolution by determining the type of desired
    convolution using hints.

    Exactly one of ``dps``, ``prime``, ``dyadic``, ``subset`` arguments
    should be specified explicitly for identifying the type of convolution,
    and the argument ``cycle`` can be specified optionally.

    For the default arguments, linear convolution is performed using **FFT**.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    cycle : Integer
        Specifies the length for doing cyclic convolution.
    dps : Integer
        Specifies the number of decimal digits for precision for
        performing **FFT** on the sequence.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for
        performing **NTT** on the sequence.
    dyadic : bool
        Identifies the convolution type as dyadic (*bitwise-XOR*)
        convolution, which is performed using **FWHT**.
    subset : bool
        Identifies the convolution type as subset convolution.

    Examples
    ========

    >>> from sympy import convolution, symbols, S, I
    >>> u, v, w, x, y, z = symbols('u v w x y z')

    >>> convolution([1 + 2*I, 4 + 3*I], [S(5)/4, 6], dps=3)
    [1.25 + 2.5*I, 11.0 + 15.8*I, 24.0 + 18.0*I]
    >>> convolution([1, 2, 3], [4, 5, 6], cycle=3)
    [31, 31, 28]

    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1)
    [1283, 19351, 14219]
    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1, cycle=2)
    [15502, 19351]

    >>> convolution([u, v], [x, y, z], dyadic=True)
    [u*x + v*y, u*y + v*x, u*z, v*z]
    >>> convolution([u, v], [x, y, z], dyadic=True, cycle=2)
    [u*x + u*z + v*y, u*y + v*x + v*z]

    >>> convolution([u, v, w], [x, y, z], subset=True)
    [u*x, u*y + v*x, u*z + w*x, v*z + w*y]
    >>> convolution([u, v, w], [x, y, z], subset=True, cycle=3)
    [u*x + v*z + w*y, u*y + v*x, u*z + w*x]

    """

    c = as_int(cycle)  # 将周期参数转换为整数类型
    if c < 0:
        raise ValueError("The length for cyclic convolution "
                        "must be non-negative")  # 如果周期参数为负数，则引发值错误异常

    dyadic = True if dyadic else None  # 如果dyadic为True，则设置为True，否则为None
    subset = True if subset else None  # 如果subset为True，则设置为True，否则为None
    if sum(x is not None for x in (prime, dps, dyadic, subset)) > 1:
        raise TypeError("Ambiguity in determining the type of convolution")  # 如果指定多于一个类型参数，则引发类型错误异常
    # 如果给定了素数参数，则使用 NTT 进行卷积计算
    if prime is not None:
        ls = convolution_ntt(a, b, prime=prime)
        # 如果需要对结果进行分组求和，则进行分组操作并对素数取模
        return ls if not c else [sum(ls[i::c]) % prime for i in range(c)]

    # 如果指定了 dyadic 参数，则使用快速 Walsh-Hadamard 变换进行卷积计算
    if dyadic:
        ls = convolution_fwht(a, b)
    # 如果指定了 subset 参数，则使用子集卷积算法进行计算
    elif subset:
        ls = convolution_subset(a, b)
    else:
        # 定义一个内部函数 loop，用于处理有理数列表的特殊情况
        def loop(a):
            dens = []
            for i in a:
                if isinstance(i, Rational) and i.q - 1:
                    dens.append(i.q)
                elif not isinstance(i, int):
                    return
            # 如果存在有理数且需要求最小公倍数，则计算最小公倍数并调整列表元素
            if dens:
                l = lcm(*dens)
                return [i*l if type(i) is int else i.p*(l//i.q) for i in a], l
            # 没有需要处理的最小公倍数情况
            return a, 1
        
        ls = None
        # 对 a 应用 loop 函数，获取处理后的结果 da
        da = loop(a)
        if da is not None:
            # 对 b 应用 loop 函数，获取处理后的结果 db
            db = loop(b)
            if db is not None:
                # 分别解包 da 和 db 的结果
                (ia, ma), (ib, mb) = da, db
                den = ma * mb
                # 使用整数卷积算法处理处理后的 ia 和 ib
                ls = convolution_int(ia, ib)
                if den != 1:
                    # 如果需要进行有理数转换，则转换为有理数列表
                    ls = [Rational(i, den) for i in ls]
        # 如果 ls 仍然为 None，则使用 FFT 进行卷积计算，使用给定的 dps 参数
        if ls is None:
            ls = convolution_fft(a, b, dps)

    # 如果需要对结果进行分组求和，则进行分组操作
    return ls if not c else [sum(ls[i::c]) for i in range(c)]
#----------------------------------------------------------------------------#
#                                                                            #
#                       Convolution for Complex domain                       #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_fft(a, b, dps=None):
    """
    Performs linear convolution using Fast Fourier Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import S, I
    >>> from sympy.discrete.convolutions import convolution_fft

    >>> convolution_fft([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_fft([2, 5], [6, 7, 3])
    [12, 44, 41, 15]
    >>> convolution_fft([1 + 2*I, 4 + 3*I], [S(5)/4, 6])
    [5/4 + 5*I/2, 11 + 63*I/4, 24 + 18*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    a, b = a[:], b[:]  # Copy input sequences to avoid modification of original data
    n = m = len(a) + len(b) - 1  # Determine the size of the resulting convolution

    if n > 0 and n&(n - 1):  # Check if n is not a power of 2
        n = 2**n.bit_length()  # Find the next power of 2 greater than n

    # Padding sequences with zeros to match the required size for FFT
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    # Compute FFT of sequences a and b
    a, b = fft(a, dps), fft(b, dps)

    # Perform element-wise multiplication in frequency domain
    a = [expand_mul(x*y) for x, y in zip(a, b)]

    # Compute inverse FFT and retain only the first m elements for the convolution result
    a = ifft(a, dps)[:m]

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                           Convolution for GF(p)                            #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_ntt(a, b, prime):
    """
    Performs linear convolution using Number Theoretic Transform (NTT).

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` used for performing NTT on the sequence.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_ntt
    >>> convolution_ntt([2, 3], [4, 5], prime=19*2**10 + 1)
    [8, 22, 15]
    >>> convolution_ntt([2, 5], [6, 7, 3], prime=19*2**10 + 1)
    [12, 44, 41, 15]
    >>> convolution_ntt([333, 555], [222, 666], prime=19*2**10 + 1)
    [15555, 14219, 19404]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """

    a, b, p = a[:], b[:], as_int(prime)  # Copy input sequences and convert prime to integer
    n = m = len(a) + len(b) - 1  # Determine the size of the resulting convolution
    # 如果 n 大于 0 并且 n 不是 2 的幂次方，将 n 调整为大于等于 n 的最小的 2 的幂次方
    if n > 0 and n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # 使用零填充数组 a 和 b，使它们的长度变为 n
    a += [0]*(n - len(a))
    b += [0]*(n - len(b))

    # 对数组 a 和 b 分别进行数论变换（NTT），使用模数 p
    a, b = ntt(a, p), ntt(b, p)

    # 将数组 a 中的每个元素与数组 b 中对应位置的元素相乘，结果对 p 取模
    a = [x*y % p for x, y in zip(a, b)]

    # 对结果数组 a 进行反数论变换（INTT），使用模数 p，并截取前 m 个元素
    a = intt(a, p)[:m]

    # 返回处理后的数组 a 作为函数结果
    return a
#----------------------------------------------------------------------------#
#                                                                            #
#                         Convolution for 2**n-group                         #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_fwht(a, b):
    """
    Performs dyadic (*bitwise-XOR*) convolution using Fast Walsh Hadamard
    Transform.

    The convolution is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S, I
    >>> from sympy.discrete.convolutions import convolution_fwht

    >>> u, v, x, y = symbols('u v x y')
    >>> convolution_fwht([u, v], [x, y])
    [u*x + v*y, u*y + v*x]

    >>> convolution_fwht([2, 3], [4, 5])
    [23, 22]
    >>> convolution_fwht([2, 5 + 4*I, 7], [6*I, 7, 3 + 4*I])
    [56 + 68*I, -10 + 30*I, 6 + 50*I, 48 + 32*I]

    >>> convolution_fwht([S(33)/7, S(55)/6, S(7)/4], [S(2)/3, 5])
    [2057/42, 1870/63, 7/6, 35/4]

    References
    ==========

    .. [1] https://www.radioeng.cz/fulltexts/2002/02_03_40_42.pdf
    .. [2] https://en.wikipedia.org/wiki/Hadamard_transform

    """

    # Check if either sequence is empty; return empty list if true
    if not a or not b:
        return []

    # Create local copies of sequences a and b
    a, b = a[:], b[:]

    # Determine the maximum length of sequences a and b
    n = max(len(a), len(b))

    # Check if n is not a power of 2, then adjust n to the next power of 2
    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # Pad sequences a and b with zeros to length n
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    # Perform Fast Walsh Hadamard Transform (FWHT) on sequences a and b
    a, b = fwht(a), fwht(b)

    # Compute element-wise multiplication and expand the results
    a = [expand_mul(x*y) for x, y in zip(a, b)]

    # Perform inverse FWHT on the results
    a = ifwht(a)

    return a


#----------------------------------------------------------------------------#
#                                                                            #
#                            Subset Convolution                              #
#                                                                            #
#----------------------------------------------------------------------------#

def convolution_subset(a, b):
    """
    Performs Subset Convolution of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.discrete.convolutions import convolution_subset
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> convolution_subset([u, v], [x, y])
    [u*x, u*y + v*x]
    >>> convolution_subset([u, v, x], [y, z])

    """

    # Check if either sequence is empty; return empty list if true
    if not a or not b:
        return []

    # Create local copies of sequences a and b
    a, b = a[:], b[:]

    # Determine the maximum length of sequences a and b
    n = max(len(a), len(b))

    # Check if n is not a power of 2, then adjust n to the next power of 2
    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # Pad sequences a and b with zeros to length n
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    # Perform Subset Convolution
    result = [S.Zero] * (2**n)
    for i in range(2**n):
        for j in range(2**n):
            if (i & j) == i:
                result[i] += a[j] * b[i ^ j]

    return result
    # 定义一个列表，包含四个元素，每个元素都是乘法表达式
    [u*y, u*z + v*y, x*y, x*z]
    
    >>> convolution_subset([1, S(2)/3], [3, 4])
    # 调用 convolution_subset 函数，传入参数 [1, 2/3] 和 [3, 4]，返回结果为 [3, 6]
    [3, 6]
    >>> convolution_subset([1, 3, S(5)/7], [7])
    # 调用 convolution_subset 函数，传入参数 [1, 3, 5/7] 和 [7]，返回结果为 [7, 21, 5, 0]
    [7, 21, 5, 0]
    
    References
    ==========
    
    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf
    
    """
    
    # 检查输入的 a 和 b 是否为空列表，若有任一为空则返回空列表
    if not a or not b:
        return []
    
    # 检查 a 和 b 是否为可迭代对象，若不是则抛出类型错误异常
    if not iterable(a) or not iterable(b):
        raise TypeError("Expected a sequence of coefficients for convolution")
    
    # 将 a 和 b 中的每个元素转换为 SymPy 的表达式对象
    a = [sympify(arg) for arg in a]
    b = [sympify(arg) for arg in b]
    
    # 计算 a 和 b 的长度，并取其中较大值作为 n
    n = max(len(a), len(b))
    
    # 如果 n 不是 2 的幂，则将 n 调整为大于等于 n 的最小的 2 的幂
    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()
    
    # 如果 a 的长度小于 n，则用 S.Zero 填充到长度为 n
    a += [S.Zero]*(n - len(a))
    # 如果 b 的长度小于 n，则用 S.Zero 填充到长度为 n
    b += [S.Zero]*(n - len(b))
    
    # 初始化结果列表 c，长度为 n，每个元素初始为 S.Zero
    c = [S.Zero]*n
    
    # 开始卷积运算
    for mask in range(n):
        smask = mask
        while smask > 0:
            # 将 a[smask] 和 b[mask^smask] 的乘积加到 c[mask] 中
            c[mask] += expand_mul(a[smask] * b[mask^smask])
            # 更新 smask 的值，通过 (smask - 1) & mask 实现
            smask = (smask - 1)&mask
    
        # 将 a[smask] 和 b[mask^smask] 的乘积加到 c[mask] 中
        c[mask] += expand_mul(a[smask] * b[mask^smask])
    
    # 返回卷积结果列表 c
    return c
def intersecting_product(a, b):
    """
    Returns the intersecting product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The intersecting product of given sequences is the sequence which
    contains the sum of products of the elements of the given sequences
    grouped by the *bitwise-AND* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which intersecting product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, intersecting_product
    >>> u, v, x, y = symbols('u v x y')

    >>> intersecting_product([u, v], [x, y])
    [u*x, u*y + v*x]
    >>> intersecting_product([u, v, x], [y])
    [u*y, v*y, x*y]

    >>> intersecting_product([1, S(2)/3], [3, 4])
    [3, 8/3]
    >>> intersecting_product([1, 3, S(5)/7], [7])
    [7, 21/7, 5]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """

    if not a or not b:
        return []

    a, b = a[:], b[:]
    n = max(len(a), len(b))

    if n&(n - 1): # not a power of 2
        n = 2**n.bit_length()

    # padding with zeros
    a += [S.Zero]*(n - len(a))
    b += [S.Zero]*(n - len(b))

    a, b = mobius_transform(a), mobius_transform(b)
    a = [expand_mul(x*y) for x, y in zip(a, b)]
    a = inverse_mobius_transform(a)

    return a
    # 如果输入的序列 a 或 b 为空，则直接返回空列表
    if not a or not b:
        return []
    
    # 复制序列 a 和 b，避免修改原始输入
    a, b = a[:], b[:]
    
    # 计算序列 a 和 b 的长度的最大值
    n = max(len(a), len(b))
    
    # 如果 n 不是 2 的幂，则将 n 调整为大于等于 n 的最小的 2 的幂次方
    if n & (n - 1):  # 检查 n 是否为 2 的幂
        n = 2 ** n.bit_length()
    
    # 使用 0 填充序列 a 和 b，使它们的长度都变为 n
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    
    # 对序列 a 和 b 进行莫比乌斯变换（Mobius Transform），subset=False 表示不是子集变换
    a, b = mobius_transform(a, subset=False), mobius_transform(b, subset=False)
    
    # 计算每对元素的乘积并展开结果
    a = [expand_mul(x * y) for x, y in zip(a, b)]
    
    # 对结果进行逆莫比乌斯变换
    a = inverse_mobius_transform(a, subset=False)
    
    # 返回计算得到的交错乘积列表 a
    return a
# 定义一个函数，计算两个整数序列的卷积，并返回结果作为列表

def convolution_int(a, b):
    """Return the convolution of two sequences as a list.

    The iterables must consist solely of integers.

    Parameters
    ==========

    a, b : Sequence
        The sequences for which convolution is performed.

    Explanation
    ===========

    This function performs the convolution of ``a`` and ``b`` by packing
    each into a single integer, multiplying them together, and then
    unpacking the result from the product. The idea leverages efficient
    large integer multiplication available in libraries such as GMP.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_int

    >>> convolution_int([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_int([1, 1, -1], [1, 1])
    [1, 2, 0, -1]

    References
    ==========

    .. [1] Fateman, Richard J.
           Can you save time in multiplying polynomials by encoding them as integers?
           University of California, Berkeley, California (2004).
           https://people.eecs.berkeley.edu/~fateman/papers/polysbyGMP.pdf
    """

    # 计算多项式乘积中的上界，根据文献 [1] 的推导
    B = max(abs(c) for c in a) * max(abs(c) for c in b) * (1 + min(len(a) - 1, len(b) - 1))
    
    # 初始化变量 x 和 power，用于多项式乘法的位操作
    x, power = MPZ(1), 0
    while x <= (2 * B):  # 为了处理负系数，乘以二，参见文献 [1]
        x <<= 1
        power += 1

    # 将多项式转换为整数表示
    def to_integer(poly):
        n, mul = MPZ(0), 0
        for c in reversed(poly):
            if c and not mul:
                mul = -1 if c < 0 else 1
            n <<= power
            n += mul * int(c)
        return mul, n

    # 执行多项式乘法：打包和相乘
    (a_mul, a_packed), (b_mul, b_packed) = to_integer(a), to_integer(b)
    result = a_packed * b_packed

    # 解包结果
    mul = a_mul * b_mul
    mask, half, borrow, poly = x - 1, x >> 1, 0, []
    while result or borrow:
        coeff = (result & mask) + borrow
        result >>= power
        borrow = coeff >= half
        poly.append(mul * int(coeff if coeff < half else coeff - x))
    # 如果 poly 为真值（非空、非零），则返回 poly；否则返回一个包含单个元素 0 的列表作为默认值
    return poly or [0]
```