# `D:\src\scipysrc\sympy\sympy\ntheory\bbp_pi.py`

```
# 这是一个使用BBP公式计算π的修正固定点实现。原始实现可以在以下网址找到：
# https://web.archive.org/web/20151116045029/http://en.literateprograms.org/Pi_with_the_BBP_formula_(Python)

'''
该部分的版权声明和许可协议授权任何获得此软件及相关文档的人免费使用、复制、修改、合并、发布、分发、再许可和/或出售软件副本，并允许被授权人员进行这些操作，但需满足以下条件：
- 上述版权声明和许可声明应包含在所有副本或实质性部分的软件中。
- 本软件按"原样"提供，不附带任何形式的明示或暗示的担保，包括但不限于对适销性、特定用途的适用性及非侵权性的担保。作者或版权持有人在任何情况下均不对任何索赔、损害或其他责任承担责任，无论是合同行为、侵权行为还是其他方式引起的、与软件或使用或其他交易相关的。
'''

# 导入需要的库
import math

# 计算用于确保返回的十六进制数字准确的工作精度位数
def calculate_precision(start, prec):
    # 使用数学运算计算所需的位数
    digits = int(math.log(start + prec) / math.log(16) + prec + 3)
    return digits

# 测试函数，用于检验精度是否满足要求
def test_precision():
    # 进行一系列测试以验证所需精度的正确性
    for i in range(0, 1000):
        for j in range(1, 1000):
            # 比较计算得到的十六进制数字和预先准备的数字列表
            a, b = pi_hex_digits(i, j), dig[i:i+j]
            if a != b:
                print('%s\n%s' % (a, b))

# 当添加量 `dt` 降至零时，退出用于评估级数是否收敛的 while 循环
# Import the `as_int` function from the sympy.utilities.misc module
from sympy.utilities.misc import as_int

# Define a function _series that calculates part of the BBP algorithm for hexadecimal digits of pi
def _series(j, n, prec=14):
    # Initialize sum variable s
    s = 0
    # Compute D based on function _dn with parameters n and prec
    D = _dn(n, prec)
    # Compute D4 as 4 times D
    D4 = 4 * D
    # Initialize d as j
    d = j
    # Loop from 0 to n (inclusive)
    for k in range(n + 1):
        # Compute next term in the left sum of the BBP algorithm
        s += (pow(16, n - k, d) << D4) // d
        # Increment d by 8
        d += 8

    # Initialize sum variable t for the right sum
    t = 0
    # Start k from n + 1
    k = n + 1
    # Compute e as D4 - 4
    e = D4 - 4  # 4*(D + n - k)
    # Initialize d as 8*k + j
    d = 8 * k + j
    # Infinite loop (break condition inside)
    while True:
        # Calculate dt as (1 << e) // d
        dt = (1 << e) // d
        # If dt is 0, break the loop
        if not dt:
            break
        # Add dt to t
        t += dt
        # Decrement e by 4
        e -= 4
        # Increment d by 8
        d += 8
    # Calculate total sum as s + t
    total = s + t

    # Return the computed total
    return total


# Define a function pi_hex_digits that returns hexadecimal digits of pi as a string
def pi_hex_digits(n, prec=14):
    """Returns a string containing ``prec`` (default 14) digits
    starting at the nth digit of pi in hex. Counting of digits
    starts at 0 and the decimal is not counted, so for n = 0 the
    returned value starts with 3; n = 1 corresponds to the first
    digit past the decimal point (which in hex is 2).

    Parameters
    ==========

    n : non-negative integer
    prec : non-negative integer. default = 14

    Returns
    =======

    str : Returns a string containing ``prec`` digits
          starting at the nth digit of pi in hex.
          If ``prec`` = 0, returns empty string.

    Raises
    ======

    ValueError
        If ``n`` < 0 or ``prec`` < 0.
        Or ``n`` or ``prec`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory.bbp_pi import pi_hex_digits
    >>> pi_hex_digits(0)
    '3243f6a8885a30'
    >>> pi_hex_digits(0, 3)
    '324'

    These are consistent with the following results

    >>> import math
    >>> hex(int(math.pi * 2**((14-1)*4)))
    '0x3243f6a8885a30'
    >>> hex(int(math.pi * 2**((3-1)*4)))
    '0x324'

    References
    ==========

    .. [1] http://www.numberworld.org/digits/Pi/
    """
    # Ensure n and prec are integers using as_int function
    n, prec = as_int(n), as_int(prec)
    # Raise ValueError if n or prec is negative
    if n < 0:
        raise ValueError('n cannot be negative')
    if prec < 0:
        raise ValueError('prec cannot be negative')
    # If prec is 0, return empty string
    if prec == 0:
        return ''

    # Define coefficients and indices arrays a and j
    n -= 1
    a = [4, 2, 1, 1]
    j = [1, 4, 5, 6]

    # Compute D using _dn function with parameters n and prec
    D = _dn(n, prec)
    # Compute x based on the series formula using _series function
    x = + (a[0]*_series(j[0], n, prec)
         - a[1]*_series(j[1], n, prec)
         - a[2]*_series(j[2], n, prec)
         - a[3]*_series(j[3], n, prec)) & (16**D - 1)

    # Format x as hexadecimal string with precision prec
    s = ("%0" + "%ix" % prec) % (x // 16**(D - prec))
    # Return the formatted hexadecimal string
    return s


# Define a function _dn that adjusts n based on precision
def _dn(n, prec):
    # Increment n by 1 because _series subtracts 1 from n
    n += 1

    # Placeholder comment (original code has commented-out assert statement)
    # 计算一个数 n 和一个精度 prec 的和的二进制位数，然后按照特定的公式计算结果
    return ((n + prec).bit_length() - 1) // 4 + prec + 3
```