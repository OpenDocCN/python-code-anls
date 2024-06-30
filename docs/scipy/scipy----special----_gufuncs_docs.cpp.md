# `D:\src\scipysrc\scipy\scipy\special\_gufuncs_docs.cpp`

```
# 定义字符串变量 lpn_doc，包含 Legendre 函数的描述和用法文档
const char *lpn_doc = R"(
    Legendre function of the first kind.

    Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to n (inclusive).

    See also special.legendre for polynomial class.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    )";

# 定义字符串变量 lpmn_doc，包含 Associated Legendre 函数的描述和用法文档
const char *lpmn_doc = R"(
    Sequence of associated Legendre functions of the first kind.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    This function takes a real argument ``z``. For complex arguments ``z``
    use clpmn instead.

    Parameters
    ----------
    z : array_like
        Input value.
    m : array_like
        Sign of the order m.

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    See Also
    --------
    clpmn: associated Legendre functions of the first kind for complex z

    Notes
    -----
    In the interval (-1, 1), Ferrer's function of the first kind is
    returned. The phase convention used for the intervals (1, inf)
    and (-inf, -1) is such that the result is always real.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.3

    )";

# 定义字符串变量 clpmn_doc，包含 Associated Legendre 函数的复数参数版本的描述和用法文档
const char *clpmn_doc = R"(
    Associated Legendre function of the first kind for complex arguments.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    z : array_like
        Input value.
    type : int
       takes values 2 or 3
       2: cut on the real axis ``|x| > 1``
       3: cut on the real axis ``-1 < x < 1`` (default)
    m : array_like
        Sign of the order m.

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders ``0..m`` and degrees ``0..n``
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders ``0..m`` and degrees ``0..n``

    See Also
    --------
    lpmn: associated Legendre functions of the first kind for real z

    Notes
    -----
    ```
    ```
)";
    # 默认情况下，即 `type=3`，根据 [1]_ 中的规定选择相应的相位约定，以使函数解析。切割线位于区间 (-1, 1) 内。
    # 一般来说，从切割线上方或下方接近会产生关于第一类Ferrer函数的相位因子（参见 `lpmn`）。

    # 对于 `type=2`，选择在 `|x| > 1` 处的切割线。在复平面上接近区间 (-1, 1) 内的实数值会产生第一类Ferrer函数。

    # 参考文献：
    # - [1] Zhang, Shanjie 和 Jin, Jianming. "Computation of Special Functions", John Wiley and Sons, 1996.
    #      https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    # - [2] NIST 数学函数数字图书馆
    #      https://dlmf.nist.gov/14.21
# 定义字符串常量 `lqn_doc`，包含Legendre第二类函数的文档说明
const char *lqn_doc = R"(
    Legendre function of the second kind.

    Compute sequence of Legendre functions of the second kind, Qn(z) and
    derivatives for all degrees from 0 to n (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    )";

# 定义字符串常量 `lqmn_doc`，包含关联Legendre第二类函数的文档说明
const char *lqmn_doc = R"(
    Sequence of associated Legendre functions of the second kind.

    Computes the associated Legendre function of the second kind of order m and
    degree n, ``Qmn(z)`` = :math:`Q_n^m(z)`, and its derivative, ``Qmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Qmn(z)`` and
    ``Qmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    z : array_like
        Input value.

    Returns
    -------
    Qmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Qmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    )";

# 定义字符串常量 `rctj_doc`，包含Ricatti-Bessel第一类函数的文档说明
const char *rctj_doc = R"(
    Compute Ricatti-Bessel function of the first kind and its derivative.

    The Ricatti-Bessel function of the first kind is defined as :math:`x
    j_n(x)`, where :math:`j_n` is the spherical Bessel function of the first
    kind of order :math:`n`.

    This function computes the value and first derivative of the
    Ricatti-Bessel function for all orders up to and including `n`.

    Parameters
    ----------
    x : ndarray
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(x), ..., jn(x)
    jnp : ndarray
        First derivative j0'(x), ..., jn'(x)

    Notes
    -----
    The computation is carried out via backward recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    )";

# 定义字符串常量 `rcty_doc`，包含Ricatti-Bessel第二类函数的文档说明
const char *rcty_doc = R"(
    Compute Ricatti-Bessel function of the second kind and its derivative.

    The Ricatti-Bessel function of the second kind is defined as :math:`x
    y_n(x)`, where :math:`y_n` is the spherical Bessel function of the second
    kind of order :math:`n`.

    This function computes the value and first derivative of the function for
    all orders up to and including `n`.

    Parameters
    # 定义文档字符串，描述函数的输入和输出
    ----------
    x : ndarray
        Argument at which to evaluate

    Returns
    -------
    yn : ndarray
        Value of y0(x), ..., yn(x)
    ynp : ndarray
        First derivative y0'(x), ..., yn'(x)

    Notes
    -----
    计算通过升序递推进行，使用关系 DLMF 10.51.1 [2]_。
    使用由 Shanjie Zhang 和 Jianming Jin 创建的 Fortran 程序的包装器 [1]_。

    References
    ----------
    参考文献，提供了详细的信息和链接
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    )";
# 定义一个常量指针，包含一个多行字符串文档
const char *sph_harm_all_doc = R"(
    Private function. This may be removed or modified at any time.
    )";
```