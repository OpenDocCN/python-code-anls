# `D:\src\scipysrc\sympy\sympy\discrete\transforms.py`

```
"""
Discrete Fourier Transform, Number Theoretic Transform,
Walsh Hadamard Transform, Mobius Transform
"""

from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int


#----------------------------------------------------------------------------#
#                                                                            #
#                         Discrete Fourier Transform                         #
#                                                                            #
#----------------------------------------------------------------------------#

def _fourier_transform(seq, dps, inverse=False):
    """Utility function for the Discrete Fourier Transform"""
    
    # 检查输入序列是否可迭代，否则抛出类型错误
    if not iterable(seq):
        raise TypeError("Expected a sequence of numeric coefficients "
                        "for Fourier Transform")

    # 将序列中的每个参数转换为 SymPy 表达式
    a = [sympify(arg) for arg in seq]
    # 如果参数包含符号变量，则抛出值错误
    if any(x.has(Symbol) for x in a):
        raise ValueError("Expected non-symbolic coefficients")

    # 获取序列长度
    n = len(a)
    # 如果序列长度小于 2，直接返回序列
    if n < 2:
        return a

    # 计算序列长度的比特位长度
    b = n.bit_length() - 1
    # 如果序列长度不是 2 的幂，则补齐为最接近的 2 的幂
    if n&(n - 1): # not a power of 2
        b += 1
        n = 2**b

    # 将序列补齐至 2 的幂的长度，不足部分补零
    a += [S.Zero]*(n - len(a))
    # 执行索引位反转操作
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    # 计算角频率
    ang = -2*pi/n if inverse else 2*pi/n

    # 如果指定了精度，则计算角频率的数值近似
    if dps is not None:
        ang = ang.evalf(dps + 2)

    # 计算旋转因子
    w = [cos(ang*i) + I*sin(ang*i) for i in range(n // 2)]

    # 进行蝶形运算
    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], expand_mul(a[i + j + hf]*w[ut * j])
                a[i + j], a[i + j + hf] = u + v, u - v
        h *= 2

    # 如果是反变换，则进行归一化处理
    if inverse:
        a = [(x/n).evalf(dps) for x in a] if dps is not None \
                            else [x/n for x in a]

    # 返回变换后的序列
    return a


def fft(seq, dps=None):
    r"""
    Performs the Discrete Fourier Transform (**DFT**) in the complex domain.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FFT* requires the number of sample points to be a power of 2.

    This method should be used with default arguments only for short sequences
    as the complexity of expressions increases with the size of the sequence.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import fft, ifft

    >>> fft([1, 2, 3, 4])
    [10, -2 - 2*I, -2, -2 + 2*I]
    >>> ifft(_)
    [1, 2, 3, 4]

    >>> ifft([1, 2, 3, 4])
    [5/2, -1/2 + I/2, -1/2, -1/2 - I/2]
    >>> fft(_)
    [1, 2, 3, 4]

    """

    # 调用内部 Fourier 变换函数，执行 DFT
    return _fourier_transform(seq, dps)
    # 对给定序列进行反离散傅立叶变换（IFFT），并指定小数点后位数为15
    >>> ifft([1, 7, 3, 4], dps=15)
    # 返回结果为反傅立叶变换后的复数列表
    [3.75, -0.5 - 0.75*I, -1.75, -0.5 + 0.75*I]
    # 使用上一步计算结果作为输入进行傅立叶变换（FFT）
    >>> fft(_)
    # 返回结果为傅立叶变换后的复数列表
    [1.0, 7.0, 3.0, 4.0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
    .. [2] https://mathworld.wolfram.com/FastFourierTransform.html

    """

    # 返回给定序列经过傅立叶变换后的结果
    return _fourier_transform(seq, dps=dps)
# 定义函数 ifft(seq, dps=None)，用于执行逆傅里叶变换，调用了 _fourier_transform 函数，并指定逆变换操作
def ifft(seq, dps=None):
    return _fourier_transform(seq, dps=dps, inverse=True)

# 将 ifft 函数的文档字符串设为与 fft 函数相同的文档字符串
ifft.__doc__ = fft.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                         Number Theoretic Transform                         #
#                                                                            #
#----------------------------------------------------------------------------#

# 定义函数 _number_theoretic_transform(seq, prime, inverse=False)，用于执行数论变换的实用函数
def _number_theoretic_transform(seq, prime, inverse=False):
    """Utility function for the Number Theoretic Transform"""

    # 检查 seq 是否为可迭代对象，否则抛出 TypeError 异常
    if not iterable(seq):
        raise TypeError("Expected a sequence of integer coefficients "
                        "for Number Theoretic Transform")

    # 将 prime 转换为整数类型
    p = as_int(prime)
    # 检查 p 是否为素数，若不是则抛出 ValueError 异常
    if not isprime(p):
        raise ValueError("Expected prime modulus for "
                        "Number Theoretic Transform")

    # 将 seq 中的每个元素转换为 p 模数下的整数，并存储在列表 a 中
    a = [as_int(x) % p for x in seq]

    # 获取序列 a 的长度 n
    n = len(a)
    # 若 n 小于 1，则直接返回列表 a
    if n < 1:
        return a

    # 计算 n 的二进制位数减一，存储在变量 b 中
    b = n.bit_length() - 1
    # 若 n 不是 2 的幂，则将 b 加一，并将 n 设为 2 的 b 次方
    if n&(n - 1):
        b += 1
        n = 2**b

    # 若 (p - 1) 不能整除 n，则抛出 ValueError 异常
    if (p - 1) % n:
        raise ValueError("Expected prime modulus of the form (m*2**k + 1)")

    # 将列表 a 扩展到长度为 n，用 0 填充剩余部分
    a += [0]*(n - len(a))
    # 执行置换操作，根据 i 和 j 的关系交换列表 a 中的元素
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    # 计算模 p 下的原根 pr
    pr = primitive_root(p)

    # 计算 pr 的 (p-1)/n 次幂，得到 rt，若 inverse 为 True，则计算其模 p 的逆元
    rt = pow(pr, (p - 1) // n, p)
    if inverse:
        rt = pow(rt, p - 2, p)

    # 初始化长度为 n//2 的单位根列表 w，并计算 w 中每个元素的值
    w = [1]*(n // 2)
    for i in range(1, n // 2):
        w[i] = w[i - 1]*rt % p

    # 初始化 h 为 2，执行 NTT 主循环，直到 h 大于 n 为止
    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]*w[ut * j]
                a[i + j], a[i + j + hf] = (u + v) % p, (u - v) % p
        h *= 2

    # 若 inverse 为 True，则计算乘以 n 的模 p 的逆元，并更新列表 a 中的元素
    if inverse:
        rv = pow(n, p - 2, p)
        a = [x*rv % p for x in a]

    # 返回计算结果列表 a
    return a


# 定义函数 ntt(seq, prime)，执行数论变换（NTT），特化于 Z/pZ 上的离散傅里叶变换（DFT）
def ntt(seq, prime):
    r"""
    Performs the Number Theoretic Transform (**NTT**), which specializes the
    Discrete Fourier Transform (**DFT**) over quotient ring `Z/pZ` for prime
    `p` instead of complex numbers `C`.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 NTT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which **DFT** is to be applied.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy import ntt, intt
    >>> ntt([1, 2, 3, 4], prime=3*2**8 + 1)
    [10, 643, 767, 122]
    >>> intt(_, 3*2**8 + 1)
    [1, 2, 3, 4]
    >>> intt([1, 2, 3, 4], prime=3*2**8 + 1)
    [387, 415, 384, 353]
    >>> ntt(_, prime=3*2**8 + 1)
    [1, 2, 3, 4]

    References
    ==========

    .. [1] http://www.apfloat.org/ntt.html
    .. [2] https://mathworld.wolfram.com/NumberTheoreticTransform.html

    """
    """
    调用一个特定的函数 `_number_theoretic_transform`，进行数论变换处理。
    函数参数 `seq` 是输入的序列，`prime` 是一个可选参数，用于指定质数。
    返回 `_number_theoretic_transform` 函数的结果。
    """

    return _number_theoretic_transform(seq, prime=prime)
# 定义一个函数 intt，用于执行数论变换的逆操作
def intt(seq, prime):
    return _number_theoretic_transform(seq, prime=prime, inverse=True)

# 将 intt 函数的文档字符串设为与 ntt 函数相同的文档字符串
intt.__doc__ = ntt.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                          Walsh Hadamard Transform                          #
#                                                                            #
#----------------------------------------------------------------------------#

# 定义一个用于执行 Walsh Hadamard 变换的内部函数 _walsh_hadamard_transform
def _walsh_hadamard_transform(seq, inverse=False):
    """Utility function for the Walsh Hadamard Transform"""

    # 检查 seq 是否可迭代，若不是则抛出类型错误异常
    if not iterable(seq):
        raise TypeError("Expected a sequence of coefficients "
                        "for Walsh Hadamard Transform")

    # 将 seq 中的每个元素转换为符号对象，并放入列表 a 中
    a = [sympify(arg) for arg in seq]
    # 获取序列长度 n
    n = len(a)
    # 若 n 小于 2，则直接返回列表 a
    if n < 2:
        return a

    # 如果 n 不是 2 的幂，则将 n 扩展为大于等于 n 的最小的 2 的幂
    if n&(n - 1):
        n = 2**n.bit_length()

    # 如果 a 的长度小于 n，则用零填充 a 到长度 n
    a += [S.Zero]*(n - len(a))
    # 初始化变量 h 为 2
    h = 2
    # 当 h 小于等于 n 时，执行以下循环
    while h <= n:
        hf = h // 2
        # 对每个分组中的元素执行 Walsh Hadamard 变换
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]
                a[i + j], a[i + j + hf] = u + v, u - v
        # 将 h 增倍
        h *= 2

    # 如果指定了逆变换，则对每个元素除以 n
    if inverse:
        a = [x/n for x in a]

    # 返回变换后的结果列表 a
    return a


# 定义一个函数 fwht，执行 Walsh Hadamard 变换，并使用 Hadamard 排序
def fwht(seq):
    r"""
    Performs the Walsh Hadamard Transform (**WHT**), and uses Hadamard
    ordering for the sequence.

    The sequence is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    seq : iterable
        The sequence on which WHT is to be applied.

    Examples
    ========

    >>> from sympy import fwht, ifwht
    >>> fwht([4, 2, 2, 0, 0, 2, -2, 0])
    [8, 0, 8, 0, 8, 8, 0, 0]
    >>> ifwht(_)
    [4, 2, 2, 0, 0, 2, -2, 0]

    >>> ifwht([19, -1, 11, -9, -7, 13, -15, 5])
    [2, 0, 4, 0, 3, 10, 0, 0]
    >>> fwht(_)
    [19, -1, 11, -9, -7, 13, -15, 5]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hadamard_transform
    .. [2] https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

    """

    # 调用 _walsh_hadamard_transform 函数执行 Walsh Hadamard 变换，并返回结果
    return _walsh_hadamard_transform(seq)


# 定义一个函数 ifwht，执行 Walsh Hadamard 变换的逆操作
def ifwht(seq):
    # 调用 _walsh_hadamard_transform 函数执行 Walsh Hadamard 变换的逆操作，并返回结果
    return _walsh_hadamard_transform(seq, inverse=True)

# 将 ifwht 函数的文档字符串设为与 fwht 函数相同的文档字符串
ifwht.__doc__ = fwht.__doc__


#----------------------------------------------------------------------------#
#                                                                            #
#                    Mobius Transform for Subset Lattice                     #
#                                                                            #
#----------------------------------------------------------------------------#

# 定义一个用于执行 Mobius 变换的内部函数 _mobius_transform
def _mobius_transform(seq, sgn, subset):
    r"""Utility function for performing Mobius Transform using
    Yate's Dynamic Programming method"""

    # 检查 seq 是否可迭代，若不是则抛出类型错误异常
    if not iterable(seq):
        raise TypeError("Expected a sequence of coefficients")

    # 将 seq 中的每个元素转换为符号对象，并放入列表 a 中
    a = [sympify(arg) for arg in seq]

    # 获取序列长度 n
    n = len(a)
    # 若 n 小于 2，则直接返回列表 a
    if n < 2:
        return a
    # 检查输入的整数 n 是否为 2 的幂次方，如果不是，则将 n 修改为大于等于 n 的最小的 2 的幂次方
    if n&(n - 1):
        n = 2**n.bit_length()
    
    # 如果数组 a 的长度小于 n，则在数组末尾填充 S.Zero 元素，使其长度达到 n
    a += [S.Zero]*(n - len(a))
    
    # 根据 subset 的值，执行不同的子集求和算法
    if subset:
        i = 1
        while i < n:
            for j in range(n):
                # 如果 j 和 i 的按位与结果为真，则将 a[j] 加上 sgn 乘以 a[j ^ i] 的值
                if j & i:
                    a[j] += sgn*a[j ^ i]
            i *= 2
    else:
        i = 1
        while i < n:
            for j in range(n):
                # 如果 j 和 i 的按位与结果为假，则跳过当前循环，继续下一次循环
                if j & i:
                    continue
                # 将 a[j] 加上 sgn 乘以 a[j ^ i] 的值
                a[j] += sgn*a[j ^ i]
            i *= 2
    
    # 返回处理后的数组 a
    return a
# 定义一个执行莫比乌斯变换的函数，针对子集格和序列的索引作为位掩码。

# 使用位掩码（索引）定义基于子集/超集的含义，需要将序列右侧自动填充零，确保序列大小为2的幂。

# 参数：
# seq : iterable
#     要应用莫比乌斯变换的序列。
# subset : bool, optional
#     指定是否通过枚举给定集合的子集或超集来应用莫比乌斯变换。

# 示例：
# >>> from sympy import symbols
# >>> from sympy import mobius_transform, inverse_mobius_transform
# >>> x, y, z = symbols('x y z')

# >>> mobius_transform([x, y, z])
# [x, x + y, x + z, x + y + z]
# >>> inverse_mobius_transform(_)
# [x, y, z, 0]

# >>> mobius_transform([x, y, z], subset=False)
# [x + y + z, y, z, 0]
# >>> inverse_mobius_transform(_, subset=False)
# [x, y, z, 0]

# >>> mobius_transform([1, 2, 3, 4])
# [1, 3, 4, 10]
# >>> inverse_mobius_transform(_)
# [1, 2, 3, 4]
# >>> mobius_transform([1, 2, 3, 4], subset=False)
# [10, 6, 7, 4]
# >>> inverse_mobius_transform(_, subset=False)
# [1, 2, 3, 4]

# 参考资料：
# .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_inversion_formula
# .. [2] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf
# .. [3] https://arxiv.org/pdf/1211.0189.pdf

def mobius_transform(seq, subset=True):
    # 调用内部函数 _mobius_transform 执行莫比乌斯变换，sgn=+1 表示正变换，subset 根据参数确定是子集还是超集变换
    return _mobius_transform(seq, sgn=+1, subset=subset)

def inverse_mobius_transform(seq, subset=True):
    # 调用内部函数 _mobius_transform 执行莫比乌斯变换，sgn=-1 表示反变换，subset 根据参数确定是子集还是超集变换
    return _mobius_transform(seq, sgn=-1, subset=subset)

# 将反变换函数的文档字符串设置为正变换函数的文档字符串
inverse_mobius_transform.__doc__ = mobius_transform.__doc__
```