# `.\numpy\numpy\_core\code_generators\ufunc_docstrings.py`

```
"""
Docstrings for generated ufuncs

The syntax is designed to look like the function add_newdoc is being
called from numpy.lib, but in this file  add_newdoc puts the docstrings
in a dictionary. This dictionary is used in
numpy/_core/code_generators/generate_umath_doc.py to generate the docstrings
as a C #definitions for the ufuncs in numpy._core at the C level when the
ufuncs are created at compile time.

"""
import textwrap

# 创建一个空的字典，用于存储文档字符串
docdict = {}

# common parameter text to all ufuncs
# 定义参数的文本模板，供所有 ufuncs 共用
subst = {
    'PARAMS': textwrap.dedent("""
        out : ndarray, None, or tuple of ndarray and None, optional
            A location into which the result is stored. If provided, it must have
            a shape that the inputs broadcast to. If not provided or None,
            a freshly-allocated array is returned. A tuple (possible only as a
            keyword argument) must have length equal to the number of outputs.
        where : array_like, optional
            This condition is broadcast over the input. At locations where the
            condition is True, the `out` array will be set to the ufunc result.
            Elsewhere, the `out` array will retain its original value.
            Note that if an uninitialized `out` array is created via the default
            ``out=None``, locations within it where the condition is False will
            remain uninitialized.
        **kwargs
            For other keyword-only arguments, see the
            :ref:`ufunc docs <ufuncs.kwargs>`.
    """).strip(),
    'BROADCASTABLE_2': ("If ``x1.shape != x2.shape``, they must be "
                        "broadcastable to a common\n    shape (which becomes "
                        "the shape of the output)."),
    'OUT_SCALAR_1': "This is a scalar if `x` is a scalar.",
    'OUT_SCALAR_2': "This is a scalar if both `x1` and `x2` are scalars.",
}

# 定义一个函数，用于向 docdict 中添加新的文档字符串
def add_newdoc(place, name, doc):
    # 去除文档字符串的缩进并去除末尾空白
    doc = textwrap.dedent(doc).strip()

    # 定义需要跳过处理的 ufuncs 名称
    skip = (
        'matmul', 'vecdot',  # gufuncs do not use the OUT_SCALAR replacement strings
        'clip',  # clip has 3 inputs, which is not handled by this
    )
    
    # 如果名称不以下划线开头且不在跳过列表中
    if name[0] != '_' and name not in skip:
        # 检查文档字符串中是否包含特定的占位符，如果不包含则报错
        if '\nx :' in doc:
            assert '$OUT_SCALAR_1' in doc, "in {}".format(name)
        elif '\nx2 :' in doc or '\nx1, x2 :' in doc:
            assert '$OUT_SCALAR_2' in doc, "in {}".format(name)
        else:
            assert False, "Could not detect number of inputs in {}".format(name)

    # 替换文档字符串中的占位符
    for k, v in subst.items():
        doc = doc.replace('$' + k, v)

    # 将处理好的文档字符串存入 docdict 中，键为 place.name
    docdict['.'.join((place, name))] = doc


# 调用 add_newdoc 函数，向 docdict 中添加 'numpy._core.umath.absolute' 的文档字符串
add_newdoc('numpy._core.umath', 'absolute',
    """
    Calculate the absolute value element-wise.

    ``np.abs`` is a shorthand for this function.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    """
)
    absolute : ndarray
        An ndarray containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\\sqrt{ a^2 + b^2 }`.
        $OUT_SCALAR_1

    Examples
    --------
    >>> x = np.array([-1.2, 1.2])
    >>> np.absolute(x)
    array([ 1.2,  1.2])
    >>> np.absolute(1.2 + 1j)
    1.5620499351813308

    Plot the function over ``[-10, 10]``:

    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(start=-10, stop=10, num=101)
    创建一个包含从 -10 到 10 的 101 个数的数组，用于 x 值
    >>> plt.plot(x, np.absolute(x))
    绘制函数在 x 范围内的绝对值图像
    >>> plt.show()

    Plot the function over the complex plane:

    >>> xx = x + 1j * x[:, np.newaxis]
    创建一个复数数组 xx，其中每个元素为 x + 1j * x 的形式
    >>> plt.imshow(np.abs(xx), extent=[-10, 10, -10, 10], cmap='gray')
    显示复平面上函数的绝对值图像
    >>> plt.show()

    The `abs` function can be used as a shorthand for ``np.absolute`` on
    ndarrays.

    >>> x = np.array([-1.2, 1.2])
    使用 `abs` 函数作为 `np.absolute` 的简写，应用于 ndarray
    >>> abs(x)
    array([1.2, 1.2])
# 向numpy._core.umath模块中添加新的文档字符串，描述函数`add`
add_newdoc('numpy._core.umath', 'add',
    """
    Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.
        $BROADCASTABLE_2  # x1和x2可以广播
    $PARAMS  # 其它参数

    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        $OUT_SCALAR_2  # 返回值可以是ndarray或标量

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.

    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])

    The ``+`` operator can be used as a shorthand for ``np.add`` on ndarrays.

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> x1 + x2
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])
    """)

# 向numpy._core.umath模块中添加新的文档字符串，描述函数`arccos`
add_newdoc('numpy._core.umath', 'arccos',
    """
    Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if ``y = cos(x)``, then ``x = arccos(y)``.

    Parameters
    ----------
    x : array_like
        `x`-coordinate on the unit circle.
        For real arguments, the domain is [-1, 1].
    $PARAMS  # 其它参数

    Returns
    -------
    angle : ndarray
        The angle of the ray intersecting the unit circle at the given
        `x`-coordinate in radians [0, pi].
        $OUT_SCALAR_1  # 返回角度，单位是弧度，范围在[0, pi]

    See Also
    --------
    cos, arctan, arcsin, emath.arccos

    Notes
    -----
    `arccos` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that ``cos(z) = x``. The convention is to return
    the angle `z` whose real part lies in `[0, pi]`.

    For real-valued input data types, `arccos` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytic function that
    has branch cuts ``[-inf, -1]`` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse `cos` is also known as `acos` or cos^-1.

    References
    ----------
    M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
    10th printing, 1964, pp. 79.
    https://personal.math.ubc.ca/~cbm/aands/page_79.htm

    Examples
    --------
    We expect the arccos of 1 to be 0, and of -1 to be pi:

    >>> np.arccos([1, -1])
    array([ 0.        ,  3.14159265])

    Plot arccos:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-1, 1, num=100)
    >>> plt.plot(x, np.arccos(x))
    >>> plt.axis('tight')
    >>> plt.show()

    """)

# 向numpy._core.umath模块中添加新的文档字符串，描述函数`arccosh`
add_newdoc('numpy._core.umath', 'arccosh',
    """
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS  # 其它参数

    Returns
    -------
    angle : ndarray
        The inverse hyperbolic cosine of each element in `x`.

    """
)
    arccosh : ndarray
        Array of the same shape as `x`.
        $OUT_SCALAR_1

    See Also
    --------
    cosh, arcsinh, sinh, arctanh, tanh

    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in ``[-pi, pi]`` and the real part in
    ``[0, inf]``.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccosh` is a complex analytical function that
    has a branch cut `[-inf, 1]` and is continuous from above on it.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86.
           https://personal.math.ubc.ca/~cbm/aands/page_86.htm
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arccosh

    Examples
    --------
    >>> np.arccosh([np.e, 10.0])
    array([ 1.65745445,  2.99322285])
    >>> np.arccosh(1)
    0.0



    arccosh : ndarray
        返回与 `x` 相同形状的数组。
        $OUT_SCALAR_1

    See Also
    --------
    cosh, arcsinh, sinh, arctanh, tanh

    Notes
    -----
    `arccosh` 是一个多值函数：对于每个 `x` 存在无数个数 `z`，使得 `cosh(z) = x`。约定返回 `z`，其虚部位于 ``[-pi, pi]``，实部位于 ``[0, inf]``。

    对于实数输入类型，`arccosh` 总是返回实数输出。对于每个不能表达为实数或无穷大的值，它返回 ``nan`` 并设置 `invalid` 浮点错误标志。

    对于复数输入，`arccosh` 是一个复解析函数，具有分支切割 `[-inf, 1]`，在此处从上方连续。

    References
    ----------
    .. [1] M. Abramowitz 和 I.A. Stegun, "Handbook of Mathematical Functions",
           第10次印刷, 1964, pp. 86.
           https://personal.math.ubc.ca/~cbm/aands/page_86.htm
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arccosh

    Examples
    --------
    >>> np.arccosh([np.e, 10.0])
    array([ 1.65745445,  2.99322285])
    >>> np.arccosh(1)
    0.0
# 在 `numpy._core.umath` 模块中添加新的文档字符串，用于描述 `arcsin` 函数
add_newdoc('numpy._core.umath', 'arcsin',
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    x : array_like
        `y`-coordinate on the unit circle.
    $PARAMS  # 参数说明，具体参数的描述将在实际文档中填充

    Returns
    -------
    angle : ndarray
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.
        $OUT_SCALAR_1  # 返回值说明，具体返回值的描述将在实际文档中填充

    See Also
    --------
    sin, cos, arccos, tan, arctan, arctan2, emath.arcsin  # 相关函数链接，列出与本函数相关的其他函数

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arcsin` is a complex analytic function that
    has, by convention, the branch cuts [-inf, -1] and [1, inf]  and is
    continuous from above on the former and from below on the latter.

    The inverse sine is also known as `asin` or sin^{-1}.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    https://personal.math.ubc.ca/~cbm/aands/page_79.htm

    Examples
    --------
    >>> np.arcsin(1)     # pi/2
    1.5707963267948966
    >>> np.arcsin(-1)    # -pi/2
    -1.5707963267948966
    >>> np.arcsin(0)
    0.0

    """)

# 在 `numpy._core.umath` 模块中添加新的文档字符串，用于描述 `arcsinh` 函数
add_newdoc('numpy._core.umath', 'arcsinh',
    """
    Inverse hyperbolic sine element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS  # 参数说明，具体参数的描述将在实际文档中填充

    Returns
    -------
    out : ndarray or scalar
        Array of the same shape as `x`.
        $OUT_SCALAR_1  # 返回值说明，具体返回值的描述将在实际文档中填充

    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`. The convention is to return the
    `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    returns ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arcsinh` is a complex analytical function that
    has branch cuts `[1j, infj]` and `[-1j, -infj]` and is continuous from
    the right on the former and from the left on the latter.

    The inverse hyperbolic sine is also known as `asinh` or ``sinh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86.
           https://personal.math.ubc.ca/~cbm/aands/page_86.htm
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arcsinh

    Examples
    --------
    >>> np.arcsinh(np.array([np.e, 10.0]))

    """)
    array([ 1.72538256,  2.99822295])



# 创建一个包含两个元素的 NumPy 数组，元素分别为 1.72538256 和 2.99822295
add_newdoc('numpy._core.umath', 'arctan',
    """
    Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS  # Placeholder for additional parameter details

    Returns
    -------
    out : ndarray or scalar
        Array of angles in radians, in the range [-pi/2, pi/2].

    See Also
    --------
    arctan2 : Element-wise arctangent of x1/x2 with correct quadrant.
    angle : Argument of complex values.

    Notes
    -----
    `arctan` is a multi-valued function: for each `x` there are infinitely
    many numbers `z` such that tan(`z`) = `x`.  The convention is to return
    the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctan` is a complex analytic function that
    has [``1j, infj``] and [``-1j, -infj``] as branch cuts, and is continuous
    from the left on the former and from the right on the latter.

    The inverse tangent is also known as `atan` or tan^{-1}.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79.
    https://personal.math.ubc.ca/~cbm/aands/page_79.htm

    Examples
    --------
    We expect the arctan of 0 to be 0, and of 1 to be pi/4:

    >>> np.arctan([0, 1])
    array([ 0.        ,  0.78539816])

    >>> np.pi/4
    0.78539816339744828

    Plot arctan:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10, 10)
    >>> plt.plot(x, np.arctan(x))
    >>> plt.axis('tight')
    >>> plt.show()

    """
)

add_newdoc('numpy._core.umath', 'arctan2',
    """
    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that ``arctan2(x1, x2)`` is
    the signed angle in radians between the ray ending at the origin and
    passing through the point (1,0), and the ray ending at the origin and
    passing through the point (`x2`, `x1`).  (Note the role reversal: the
    "`y`-coordinate" is the first function parameter, the "`x`-coordinate"
    is the second.)  By IEEE convention, this function is defined for
    `x2` = +/-0 and for either or both of `x1` and `x2` = +/-inf (see
    Notes for specific values).

    This function is not defined for complex-valued arguments; for the
    so-called argument of complex values, use `angle`.

    Parameters
    ----------
    x1 : array_like, real-valued
        `y`-coordinates.
    x2 : array_like, real-valued
        `x`-coordinates.
        $BROADCASTABLE_2  # Placeholder for broadcasting rules
    $PARAMS  # Placeholder for additional parameter details

    Returns
    -------
    out : ndarray
        Array of angles in radians, in the range [-pi, pi].

    """
)
    angle : ndarray
        Array of angles in radians, in the range ``[-pi, pi]``.
        $OUT_SCALAR_2

    See Also
    --------
    arctan, tan, angle

    Notes
    -----
    *arctan2* is identical to the `atan2` function of the underlying
    C library.  The following special values are defined in the C
    standard: [1]_

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
     > 0   +/-inf +0 / +pi
     < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers, as are +inf
    and -inf.

    References
    ----------
    .. [1] ISO/IEC standard 9899:1999, "Programming language C."

    Examples
    --------
    Consider four points in different quadrants:

    >>> x = np.array([-1, +1, +1, -1])
    >>> y = np.array([-1, -1, +1, +1])
    >>> np.arctan2(y, x) * 180 / np.pi
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> np.arctan2([1., -1.], [0., 0.])
    array([ 1.57079633, -1.57079633])
    >>> np.arctan2([0., 0., np.inf], [+0., -0., np.inf])
    array([0.        , 3.14159265, 0.78539816])

    """
# 向 numpy._core.umath 模块添加新的文档字符串
add_newdoc('numpy._core.umath', '_arg',
    """
    DO NOT USE, ONLY FOR TESTING
    """)

# 向 numpy._core.umath 模块添加 'arctanh' 函数的文档字符串
add_newdoc('numpy._core.umath', 'arctanh',
    """
    Inverse hyperbolic tangent element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        Array of the same shape as `x`.
        $OUT_SCALAR_1

    See Also
    --------
    emath.arctanh

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that ``tanh(z) = x``. The convention is to return
    the `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctanh` is a complex analytical function
    that has branch cuts `[-1, -inf]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse hyperbolic tangent is also known as `atanh` or ``tanh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86.
           https://personal.math.ubc.ca/~cbm/aands/page_86.htm
    .. [2] Wikipedia, "Inverse hyperbolic function",
           https://en.wikipedia.org/wiki/Arctanh

    Examples
    --------
    >>> np.arctanh([0, -0.5])
    array([ 0.        , -0.54930614])

    """)

# 向 numpy._core.umath 模块添加 'bitwise_and' 函数的文档字符串
add_newdoc('numpy._core.umath', 'bitwise_and',
    """
    Compute the bit-wise AND of two arrays element-wise.

    Computes the bit-wise AND of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``&``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        Result.
        $OUT_SCALAR_2

    See Also
    --------
    logical_and
    bitwise_or
    bitwise_xor
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    The number 13 is represented by ``00001101``.  Likewise, 17 is
    represented by ``00010001``.  The bit-wise AND of 13 and 17 is
    therefore ``000000001``, or 1:

    >>> np.bitwise_and(13, 17)
    1

    >>> np.bitwise_and(14, 13)
    12
    >>> np.binary_repr(12)
    '1100'
    >>> np.bitwise_and([14,3], 13)
    array([12,  1])

    >>> np.bitwise_and([11,7], [4,25])
    array([0, 1])
    >>> np.bitwise_and(np.array([2,5,255]), np.array([3,14,16]))
    array([ 2,  4, 16])
    >>> np.bitwise_and([True, True], [False, True])
    array([False,  True])

    The ``&`` operator can be used as a shorthand for ``np.bitwise_and`` on
    ndarrays.

    >>> x1 = np.array([2, 5, 255])
    # 创建一个 NumPy 数组 x2，包含元素 [3, 14, 16]
    x2 = np.array([3, 14, 16])
    # 对数组 x1 和 x2 进行按位与操作，返回一个新的 NumPy 数组
    # 按位与操作会对数组中对应位置的元素进行按位与运算
    # 结果数组中的每个元素都是 x1 和 x2 对应位置元素按位与的结果
    x1 & x2
    # 返回结果为一个 NumPy 数组 [2, 4, 16]
    array([ 2,  4, 16])
# 添加新的文档字符串到指定模块和函数名的文档中
add_newdoc('numpy._core.umath', 'bitwise_or',
    """
    Compute the bit-wise OR of two arrays element-wise.

    Computes the bit-wise OR of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``|``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled.
        $BROADCASTABLE_2  # 指代两个数组可以进行广播操作
    $PARAMS  # 其他参数文档中定义的未指定参数

    Returns
    -------
    out : ndarray or scalar
        Result.
        $OUT_SCALAR_2  # 指代返回值可以是数组或标量

    See Also
    --------
    logical_or  # 相关函数：逻辑或
    bitwise_and  # 相关函数：按位与
    bitwise_xor  # 相关函数：按位异或
    binary_repr :  # 相关函数：返回输入数字的二进制表示作为字符串

    Examples
    --------
    The number 13 has the binary representation ``00001101``. Likewise,
    16 is represented by ``00010000``.  The bit-wise OR of 13 and 16 is
    then ``00011101``, or 29:

    >>> np.bitwise_or(13, 16)
    29
    >>> np.binary_repr(29)
    '11101'

    >>> np.bitwise_or(32, 2)
    34
    >>> np.bitwise_or([33, 4], 1)
    array([33,  5])
    >>> np.bitwise_or([33, 4], [1, 2])
    array([33,  6])

    >>> np.bitwise_or(np.array([2, 5, 255]), np.array([4, 4, 4]))
    array([  6,   5, 255])
    >>> np.array([2, 5, 255]) | np.array([4, 4, 4])
    array([  6,   5, 255])
    >>> np.bitwise_or(np.array([2, 5, 255, 2147483647], dtype=np.int32),
    ...               np.array([4, 4, 4, 2147483647], dtype=np.int32))
    array([         6,          5,        255, 2147483647])
    >>> np.bitwise_or([True, True], [False, True])
    array([ True,  True])

    The ``|`` operator can be used as a shorthand for ``np.bitwise_or`` on
    ndarrays.

    >>> x1 = np.array([2, 5, 255])
    >>> x2 = np.array([4, 4, 4])
    >>> x1 | x2
    array([  6,   5, 255])

    """)

# 添加新的文档字符串到指定模块和函数名的文档中
add_newdoc('numpy._core.umath', 'bitwise_xor',
    """
    Compute the bit-wise XOR of two arrays element-wise.

    Computes the bit-wise XOR of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``^``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled.
        $BROADCASTABLE_2  # 指代两个数组可以进行广播操作
    $PARAMS  # 其他参数文档中定义的未指定参数

    Returns
    -------
    out : ndarray or scalar
        Result.
        $OUT_SCALAR_2  # 指代返回值可以是数组或标量

    See Also
    --------
    logical_xor  # 相关函数：逻辑异或
    bitwise_and  # 相关函数：按位与
    bitwise_or  # 相关函数：按位或
    binary_repr :  # 相关函数：返回输入数字的二进制表示作为字符串

    Examples
    --------
    The number 13 is represented by ``00001101``. Likewise, 17 is
    represented by ``00010001``.  The bit-wise XOR of 13 and 17 is
    therefore ``00011100``, or 28:

    >>> np.bitwise_xor(13, 17)
    28
    >>> np.binary_repr(28)
    '11100'

    >>> np.bitwise_xor(31, 5)
    26
    >>> np.bitwise_xor([31,3], 5)
    array([26,  6])

    >>> np.bitwise_xor([31,3], [5,6])
    array([26,  5])
    >>> np.bitwise_xor([True, True], [False, True])
    array([ True, False])

    """
    The ``^`` operator can be used as a shorthand for ``np.bitwise_xor`` on
    ndarrays.

    >>> x1 = np.array([True, True])   # 创建一个包含布尔值的 NumPy 数组 x1
    >>> x2 = np.array([False, True])  # 创建另一个包含布尔值的 NumPy 数组 x2
    >>> x1 ^ x2                       # 对数组 x1 和 x2 执行按位异或操作
    array([ True, False])            # 返回按位异或操作后的结果数组，[True, False]
# 添加新的文档字符串到 'numpy._core.umath' 的 'ceil' 函数，描述其功能和用法
add_newdoc('numpy._core.umath', 'ceil',
    """
    Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    ``i >= x``.  It is often denoted as :math:`\\lceil x \\rceil`.

    Parameters
    ----------
    x : array_like
        Input data.
    $PARAMS

    Returns
    -------
    y : ndarray or scalar
        The ceiling of each element in `x`, with `float` dtype.
        $OUT_SCALAR_1

    See Also
    --------
    floor, trunc, rint, fix

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])

    """)

# 添加新的文档字符串到 'numpy._core.umath' 的 'trunc' 函数，描述其功能和用法
add_newdoc('numpy._core.umath', 'trunc',
    """
    Return the truncated value of the input, element-wise.

    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : array_like
        Input data.
    $PARAMS

    Returns
    -------
    y : ndarray or scalar
        The truncated value of each element in `x`.
        $OUT_SCALAR_1

    See Also
    --------
    ceil, floor, rint, fix

    Notes
    -----
    .. versionadded:: 1.3.0

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.trunc(a)
    array([-1., -1., -0.,  0.,  1.,  1.,  2.])

    """)

# 添加新的文档字符串到 'numpy._core.umath' 的 'conjugate' 函数，描述其功能和用法
add_newdoc('numpy._core.umath', 'conjugate',
    """
    Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    Parameters
    ----------
    x : array_like
        Input value.
    $PARAMS

    Returns
    -------
    y : ndarray
        The complex conjugate of `x`, with same dtype as `y`.
        $OUT_SCALAR_1

    Notes
    -----
    `conj` is an alias for `conjugate`:

    >>> np.conj is np.conjugate
    True

    Examples
    --------
    >>> np.conjugate(1+2j)
    (1-2j)

    >>> x = np.eye(2) + 1j * np.eye(2)
    >>> np.conjugate(x)
    array([[ 1.-1.j,  0.-0.j],
           [ 0.-0.j,  1.-1.j]])

    """)

# 添加新的文档字符串到 'numpy._core.umath' 的 'cos' 函数，描述其功能和用法
add_newdoc('numpy._core.umath', 'cos',
    """
    Cosine element-wise.

    Parameters
    ----------
    x : array_like
        Input array in radians.
    $PARAMS

    Returns
    -------
    y : ndarray
        The corresponding cosine values.
        $OUT_SCALAR_1

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972.

    Examples
    --------
    >>> np.cos(np.array([0, np.pi/2, np.pi]))
    array([  1.00000000e+00,   6.12303177e-17,  -1.00000000e+00])
    >>>
    >>> # Example of providing the optional output parameter
    >>> out1 = np.array([0], dtype='d')
    >>> out2 = np.cos([0.1], out1)

    """)
    >>> out2 is out1
    True
    >>>
    >>> # 检查 out2 是否与 out1 是同一个对象的示例
    >>> # 这里返回 True 表示 out2 和 out1 是同一个对象

    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> # 由于提供形状不匹配的 `out` 参数，导致 ValueError 的示例
    >>> # 在这个例子中，np.cos 函数尝试使用两个不同形状的数组进行计算，引发异常
    >>> np.cos(np.zeros((3,3)),np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: operands could not be broadcast together with shapes (3,3) (2,2)
# 向 numpy._core.umath 模块添加新的文档字符串和函数定义
add_newdoc('numpy._core.umath', 'cosh',
    """
    Hyperbolic cosine, element-wise.

    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        Output array of same shape as `x`.
        $OUT_SCALAR_1

    Examples
    --------
    >>> np.cosh(0)
    1.0

    The hyperbolic cosine describes the shape of a hanging cable:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-4, 4, 1000)
    >>> plt.plot(x, np.cosh(x))
    >>> plt.show()

    """)

# 向 numpy._core.umath 模块添加新的文档字符串和函数定义
add_newdoc('numpy._core.umath', 'degrees',
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like
        Input array in radians.
    $PARAMS

    Returns
    -------
    y : ndarray of floats
        The corresponding degree values; if `out` was supplied this is a
        reference to it.
        $OUT_SCALAR_1

    See Also
    --------
    rad2deg : equivalent function

    Examples
    --------
    Convert a radian array to degrees

    >>> rad = np.arange(12.)*np.pi/6
    >>> np.degrees(rad)
    array([   0.,   30.,   60.,   90.,  120.,  150.,  180.,  210.,  240.,
            270.,  300.,  330.])

    >>> out = np.zeros((rad.shape))
    >>> r = np.degrees(rad, out)
    >>> np.all(r == out)
    True

    """)

# 向 numpy._core.umath 模块添加新的文档字符串和函数定义
add_newdoc('numpy._core.umath', 'rad2deg',
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like
        Angle in radians.
    $PARAMS

    Returns
    -------
    y : ndarray
        The corresponding angle in degrees.
        $OUT_SCALAR_1

    See Also
    --------
    deg2rad : Convert angles from degrees to radians.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    .. versionadded:: 1.3.0

    rad2deg(x) is ``180 * x / pi``.

    Examples
    --------
    >>> np.rad2deg(np.pi/2)
    90.0

    """)

# 向 numpy._core.umath 模块添加新的文档字符串和函数定义
add_newdoc('numpy._core.umath', 'heaviside',
    """
    Compute the Heaviside step function.

    The Heaviside step function [1]_ is defined as::

                              0   if x1 < 0
        heaviside(x1, x2) =  x2   if x1 == 0
                              1   if x1 > 0

    where `x2` is often taken to be 0.5, but 0 and 1 are also sometimes used.

    Parameters
    ----------
    x1 : array_like
        Input values.
    x2 : array_like
        The value of the function when x1 is 0.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        The output array, element-wise Heaviside step function of `x1`.
        $OUT_SCALAR_2

    Notes
    -----
    .. versionadded:: 1.13.0

    References
    ----------
    .. [1] Wikipedia, "Heaviside step function",
           https://en.wikipedia.org/wiki/Heaviside_step_function

    Examples
    --------
    >>> np.heaviside([-1.5, 0, 2.0], 0.5)
    array([ 0. ,  0.5,  1. ])
    >>> np.heaviside([-1.5, 0, 2.0], 1)

    """)
    array([ 0.,  1.,  1.])
    """



    # 创建一个包含三个浮点数的NumPy数组，并没有对其进行赋值或命名
    array([ 0.,  1.,  1.])
    # 这是一个多行字符串的结束标记，用三个双引号包围
    """
# 添加新的文档字符串到numpy._core.umath模块，定义了divide函数的文档信息
add_newdoc('numpy._core.umath', 'divide',
    """
    Divide arguments element-wise.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.
        $BROADCASTABLE_2  # 标记此处支持数组广播
    $PARAMS  # 描述参数信息

    Returns
    -------
    y : ndarray or scalar
        The quotient ``x1/x2``, element-wise.
        $OUT_SCALAR_2  # 标记输出标量类型

    See Also
    --------
    seterr : Set whether to raise or warn on overflow, underflow and
             division by zero.

    Notes
    -----
    Equivalent to ``x1`` / ``x2`` in terms of array-broadcasting.

    The ``true_divide(x1, x2)`` function is an alias for
    ``divide(x1, x2)``.

    Examples
    --------
    >>> np.divide(2.0, 4.0)
    0.5
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.divide(x1, x2)
    array([[nan, 1. , 1. ],
           [inf, 4. , 2.5],
           [inf, 7. , 4. ]])

    The ``/`` operator can be used as a shorthand for ``np.divide`` on
    ndarrays.

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = 2 * np.ones(3)
    >>> x1 / x2
    array([[0. , 0.5, 1. ],
           [1.5, 2. , 2.5],
           [3. , 3.5, 4. ]])

    """)

# 添加新的文档字符串到numpy._core.umath模块，定义了equal函数的文档信息
add_newdoc('numpy._core.umath', 'equal',
    """
    Return (x1 == x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.
        $BROADCASTABLE_2  # 标记此处支持数组广播
    $PARAMS  # 描述参数信息

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        $OUT_SCALAR_2  # 标记输出标量类型

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less

    Examples
    --------
    >>> np.equal([0, 1, 3], np.arange(3))
    array([ True,  True, False])

    What is compared are values, not types. So an int (1) and an array of
    length one can evaluate as True:

    >>> np.equal(1, np.ones(1))
    array([ True])

    The ``==`` operator can be used as a shorthand for ``np.equal`` on
    ndarrays.

    >>> a = np.array([2, 4, 6])
    >>> b = np.array([2, 4, 2])
    >>> a == b
    array([ True,  True, False])

    """)

# 添加新的文档字符串到numpy._core.umath模块，定义了exp函数的文档信息
add_newdoc('numpy._core.umath', 'exp',
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : array_like
        Input values.
    $PARAMS  # 描述参数信息

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential of `x`.
        $OUT_SCALAR_1  # 标记输出标量类型

    See Also
    --------
    expm1 : Calculate ``exp(x) - 1`` for all elements in the array.
    exp2  : Calculate ``2**x`` for all elements in the array.

    Notes
    -----
    The irrational number ``e`` is also known as Euler's number.  It is
    approximately 2.718281, and is the base of the natural logarithm,
    ``ln`` (this means that, if :math:`x = \\ln y = \\log_e y`,
    then :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

    For complex arguments, ``x = a + ib``, we can write

    """)
    # Plot the magnitude and phase of ``exp(x)`` in the complex plane:
    >>> import matplotlib.pyplot as plt
    
    # Create a linearly spaced array of values from -2π to 2π with 100 points
    >>> x = np.linspace(-2*np.pi, 2*np.pi, 100)
    # Create a complex grid xx where each point is represented as a + ib
    >>> xx = x + 1j * x[:, np.newaxis]
    
    # Compute the exponential of each complex value in xx
    >>> out = np.exp(xx)
    
    # Create a subplot with 1 row and 2 columns, first subplot
    >>> plt.subplot(121)
    # Display the magnitude (absolute value) of out as an image plot
    >>> plt.imshow(np.abs(out),
    ...            extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi], cmap='gray')
    # Set the title of the subplot
    >>> plt.title('Magnitude of exp(x)')
    
    # Create a subplot with 1 row and 2 columns, second subplot
    >>> plt.subplot(122)
    # Display the phase (angle) of out as an image plot
    >>> plt.imshow(np.angle(out),
    ...            extent=[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi], cmap='hsv')
    # Set the title of the subplot
    >>> plt.title('Phase (angle) of exp(x)')
    # Display the plot
    >>> plt.show()
# 将函数 `add_newdoc` 用于模块 `numpy._core.umath`，添加新的文档字符串定义
add_newdoc('numpy._core.umath', 'exp2',
    """
    计算输入数组中所有元素 `p` 的 `2**p`。

    Parameters
    ----------
    x : array_like
        输入值数组.
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        每个元素对应的 2 的幂 `x`.
        $OUT_SCALAR_1

    See Also
    --------
    power

    Notes
    -----
    .. versionadded:: 1.3.0

    Examples
    --------
    >>> np.exp2([2, 3])
    array([ 4.,  8.])

    """)

# 将函数 `add_newdoc` 用于模块 `numpy._core.umath`，添加新的文档字符串定义
add_newdoc('numpy._core.umath', 'expm1',
    """
    计算数组中所有元素的 `exp(x) - 1`。

    Parameters
    ----------
    x : array_like
        输入值数组.
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        每个元素对应的指数减一: ``out = exp(x) - 1``.
        $OUT_SCALAR_1

    See Also
    --------
    log1p : ``log(1 + x)``, expm1 的逆操作.

    Notes
    -----
    对于小的 `x` 值，此函数提供比 ``exp(x) - 1`` 更高的精度。

    Examples
    --------
    对于 ``exp(1e-10) - 1`` 的真实值是 ``1.00000000005e-10``，有约 32 位有效数字。此示例展示了 expm1 在这种情况下的优越性。

    >>> np.expm1(1e-10)
    1.00000000005e-10
    >>> np.exp(1e-10) - 1
    1.000000082740371e-10

    """)

# 将函数 `add_newdoc` 用于模块 `numpy._core.umath`，添加新的文档字符串定义
add_newdoc('numpy._core.umath', 'fabs',
    """
    计算数组元素的绝对值。

    此函数返回 `x` 中数据的绝对值（正数）。复数值不受支持，使用 `absolute` 来获取复数数据的绝对值。

    Parameters
    ----------
    x : array_like
        需要计算绝对值的数值数组。如果 `x` 是标量，则返回的 `y` 也是标量.
    $PARAMS

    Returns
    -------
    y : ndarray or scalar
        `x` 的绝对值，返回值始终是浮点数.
        $OUT_SCALAR_1

    See Also
    --------
    absolute : 包括 `complex` 类型在内的绝对值计算.

    Examples
    --------
    >>> np.fabs(-1)
    1.0
    >>> np.fabs([-1.2, 1.2])
    array([ 1.2,  1.2])

    """)

# 将函数 `add_newdoc` 用于模块 `numpy._core.umath`，添加新的文档字符串定义
add_newdoc('numpy._core.umath', 'floor',
    """
    返回输入数据的向下取整结果，逐元素操作。

    标量 `x` 的向下取整是最大整数 `i`，使得 `i <= x`。通常表示为 :math:`\\lfloor x \\rfloor`。

    Parameters
    ----------
    x : array_like
        输入数据.
    $PARAMS

    Returns
    -------
    y : ndarray or scalar
        `x` 中每个元素的向下取整结果.
        $OUT_SCALAR_1

    See Also
    --------
    ceil, trunc, rint, fix

    Notes
    -----
    一些电子表格程序计算的是 "向零取整"，即 ``floor(-2.5) == -2``。而 NumPy 使用的是 `floor` 的定义，其中 ``floor(-2.5) == -3``。向零取整函数在 NumPy 中称为 ``fix``。

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)

    """)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])



# 创建一个包含浮点数的 NumPy 数组，数组元素依次为 -2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 2.0
# 添加新的文档字符串给 numpy._core.umath 模块的 floor_divide 函数
add_newdoc('numpy._core.umath', 'floor_divide',
    """
    返回输入的除法结果向下取整的最大整数。
    它等同于 Python 中的 ``//`` 运算符，配对使用 Python 的 ``%`` (`remainder`) 函数，
    使得 ``a = a % b + b * (a // b)`` 直至舍入误差。

    Parameters
    ----------
    x1 : array_like
        分子。
    x2 : array_like
        分母。
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : ndarray
        y = floor(`x1`/`x2`)
        $OUT_SCALAR_2

    See Also
    --------
    remainder : floor_divide 的余数补充。
    divmod : 同时执行整除和取余数。
    divide : 标准除法。
    floor : 向负无穷大取最近的整数。
    ceil : 向正无穷大取最近的整数。

    Examples
    --------
    >>> np.floor_divide(7,3)
    2
    >>> np.floor_divide([1., 2., 3., 4.], 2.5)
    array([ 0.,  0.,  1.,  1.])

    在 ndarrays 上，``//`` 运算符可以作为 ``np.floor_divide`` 的简写。

    >>> x1 = np.array([1., 2., 3., 4.])
    >>> x1 // 2.5
    array([0., 0., 1., 1.])

    """)

# 添加新的文档字符串给 numpy._core.umath 模块的 fmod 函数
add_newdoc('numpy._core.umath', 'fmod',
    """
    返回元素级别的除法余数。

    这是 C 库函数 fmod 在 NumPy 中的实现，余数与被除数 `x1` 的符号相同。
    它等同于 Matlab(TM) 中的 ``rem`` 函数，不应与 Python 模数运算符 ``x1 % x2`` 混淆。

    Parameters
    ----------
    x1 : array_like
        被除数。
    x2 : array_like
        除数。
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : array_like
        `x1` 除以 `x2` 的余数。
        $OUT_SCALAR_2

    See Also
    --------
    remainder : 等同于 Python ``%`` 运算符。
    divide

    Notes
    -----
    负被除数和除数的模数运算结果受约定约束。对于 `fmod`，结果的符号与被除数相同，
    而对于 `remainder`，结果的符号与除数相同。`fmod` 函数等同于 Matlab(TM) 的 ``rem`` 函数。

    Examples
    --------
    >>> np.fmod([-3, -2, -1, 1, 2, 3], 2)
    array([-1,  0, -1,  1,  0,  1])
    >>> np.remainder([-3, -2, -1, 1, 2, 3], 2)
    array([1, 0, 1, 1, 0, 1])

    >>> np.fmod([5, 3], [2, 2.])
    array([ 1.,  1.])
    >>> a = np.arange(-3, 3).reshape(3, 2)
    >>> a
    array([[-3, -2],
           [-1,  0],
           [ 1,  2]])
    >>> np.fmod(a, [2,2])
    array([[-1,  0],
           [-1,  0],
           [ 1,  0]])

    """)

# 添加新的文档字符串给 numpy._core.umath 模块的 greater 函数
add_newdoc('numpy._core.umath', 'greater',
    """
    返回元素级别的 (x1 > x2) 的真值。

    Parameters
    ----------
    x1, x2 : array_like
        输入数组。
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        # 定义函数输出，可以是一个数组或标量
        Output array, element-wise comparison of `x1` and `x2`.
        # 输出数组，对 `x1` 和 `x2` 进行逐元素比较
        Typically of type bool, unless ``dtype=object`` is passed.
        # 通常为布尔类型，除非传入 ``dtype=object`` 参数
        $OUT_SCALAR_2
        # 这里的 $OUT_SCALAR_2 是一个占位符，通常用于在文档中引用相关变量或标量


    See Also
    --------
    greater_equal, less, less_equal, equal, not_equal
    # 参见的其他函数，用于比较操作：greater_equal, less, less_equal, equal, not_equal

    Examples
    --------
    >>> np.greater([4,2],[2,2])
    array([ True, False])
    # 示例：使用 np.greater 对数组 [4, 2] 和 [2, 2] 进行比较，得到数组 [ True, False]

    The ``>`` operator can be used as a shorthand for ``np.greater`` on
    ndarrays.
    # ``>`` 运算符可以作为 np.greater 在 ndarray 上的简写形式使用
    >>> a = np.array([4, 2])
    >>> b = np.array([2, 2])
    >>> a > b
    array([ True, False])
    # 示例：使用 ``>`` 运算符对数组 a 和 b 进行比较，得到数组 [ True, False]

    """
    # 文档字符串结束
# 添加新的文档字符串到 numpy._core.umath 模块下的 greater_equal 函数
add_newdoc('numpy._core.umath', 'greater_equal',
    """
    Return the truth value of (x1 >= x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : bool or ndarray of bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        $OUT_SCALAR_2

    See Also
    --------
    greater, less, less_equal, equal, not_equal

    Examples
    --------
    >>> np.greater_equal([4, 2, 1], [2, 2, 2])
    array([ True, True, False])

    The ``>=`` operator can be used as a shorthand for ``np.greater_equal``
    on ndarrays.

    >>> a = np.array([4, 2, 1])
    >>> b = np.array([2, 2, 2])
    >>> a >= b
    array([ True,  True, False])

    """)

# 添加新的文档字符串到 numpy._core.umath 模块下的 hypot 函数
add_newdoc('numpy._core.umath', 'hypot',
    """
    Given the "legs" of a right triangle, return its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.
    (See Examples)

    Parameters
    ----------
    x1, x2 : array_like
        Leg of the triangle(s).
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    z : ndarray
        The hypotenuse of the triangle(s).
        $OUT_SCALAR_2

    Examples
    --------
    >>> np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    Example showing broadcast of scalar_like argument:

    >>> np.hypot(3*np.ones((3, 3)), [4])
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    """)

# 添加新的文档字符串到 numpy._core.umath 模块下的 invert 函数
add_newdoc('numpy._core.umath', 'invert',
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``~``.

    For signed integer inputs, the bit-wise NOT of the absolute value is
    returned. In a two's-complement system, this operation effectively flips
    all the bits, resulting in a representation that corresponds to the
    negative of the input plus one. This is the most common method of
    representing signed integers on computers [1]_. A N-bit two's-complement
    system can represent every integer in the range :math:`-2^{N-1}` to
    :math:`+2^{N-1}-1`.

    Parameters
    ----------
    x : array_like
        Only integer and boolean types are handled.
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        Result.
        $OUT_SCALAR_1

    See Also
    --------
    bitwise_and, bitwise_or, bitwise_xor
    logical_not
    binary_repr :
        Return the binary representation of the input number as a string.

    Notes
    -----
    ``numpy.bitwise_not`` is an alias for `invert`:
    """)
    # np.bitwise_not 是 np.invert 的别名，它们在功能上是等价的
    True
    
    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        https://en.wikipedia.org/wiki/Two's_complement
    
    Examples
    --------
    We've seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:
    
    # 使用 np.invert 函数对一个无符号 8 位整数（uint8）的数值 13 进行位取反操作
    >>> x = np.invert(np.array(13, dtype=np.uint8))
    # 输出取反后的结果
    >>> x
    242
    # 将结果以 8 位二进制形式显示
    >>> np.binary_repr(x, width=8)
    '11110010'
    
    The result depends on the bit-width:
    
    # 使用 np.invert 函数对一个无符号 16 位整数（uint16）的数值 13 进行位取反操作
    >>> x = np.invert(np.array(13, dtype=np.uint16))
    # 输出取反后的结果
    >>> x
    65522
    # 将结果以 16 位二进制形式显示
    >>> np.binary_repr(x, width=16)
    '1111111111110010'
    
    When using signed integer types, the result is the bit-wise NOT of
    the unsigned type, interpreted as a signed integer:
    
    # 使用有符号 8 位整数（int8）类型进行位取反操作
    >>> np.invert(np.array([13], dtype=np.int8))
    array([-14], dtype=int8)
    # 将结果 -14 以 8 位二进制形式显示
    >>> np.binary_repr(-14, width=8)
    
    Booleans are accepted as well:
    
    # 对布尔数组进行位取反操作
    >>> np.invert(np.array([True, False]))
    array([False,  True])
    
    The ``~`` operator can be used as a shorthand for ``np.invert`` on
    ndarrays.
    
    # ``~`` 操作符可以用作 ``np.invert`` 在 ndarray 上的简写
    >>> x1 = np.array([True, False])
    >>> ~x1
    array([False,  True])
# 添加新文档到 numpy._core.umath 模块，函数名为 'isfinite'
# 检测逐元素是否有限（既不是正无穷大也不是 NaN）
"""
Test element-wise for finiteness (not infinity and not Not a Number).

The result is returned as a boolean array.

Parameters
----------
x : array_like
    Input values.
$PARAMS

Returns
-------
y : ndarray, bool
    True where ``x`` is not positive infinity, negative infinity,
    or NaN; false otherwise.
    $OUT_SCALAR_1

See Also
--------
isinf, isneginf, isposinf, isnan

Notes
-----
Not a Number, positive infinity, and negative infinity are considered
to be non-finite.

NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
(IEEE 754). This means that Not a Number is not equivalent to infinity.
Also that positive infinity is not equivalent to negative infinity. But
infinity is equivalent to positive infinity.  Errors result if the
second argument is also supplied when `x` is a scalar input, or if
first and second arguments have different shapes.

Examples
--------
>>> np.isfinite(1)
True
>>> np.isfinite(0)
True
>>> np.isfinite(np.nan)
False
>>> np.isfinite(np.inf)
False
>>> np.isfinite(-np.inf)
False
>>> np.isfinite([np.log(-1.),1.,np.log(0)])
array([False,  True, False])

>>> x = np.array([-np.inf, 0., np.inf])
>>> y = np.array([2, 2, 2])
>>> np.isfinite(x, y)
array([0, 1, 0])
>>> y
array([0, 1, 0])
"""

# 添加新文档到 numpy._core.umath 模块，函数名为 'isinf'
# 检测逐元素是否为正无穷大或负无穷大
"""
Test element-wise for positive or negative infinity.

Returns a boolean array of the same shape as `x`, True where ``x ==
+/-inf``, otherwise False.

Parameters
----------
x : array_like
    Input values
$PARAMS

Returns
-------
y : bool (scalar) or boolean ndarray
    True where ``x`` is positive or negative infinity, false otherwise.
    $OUT_SCALAR_1

See Also
--------
isneginf, isposinf, isnan, isfinite

Notes
-----
NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
(IEEE 754).

Errors result if the second argument is supplied when the first
argument is a scalar, or if the first and second arguments have
different shapes.

Examples
--------
>>> np.isinf(np.inf)
True
>>> np.isinf(np.nan)
False
>>> np.isinf(-np.inf)
True
>>> np.isinf([np.inf, -np.inf, 1.0, np.nan])
array([ True,  True, False, False])

>>> x = np.array([-np.inf, 0., np.inf])
>>> y = np.array([2, 2, 2])
>>> np.isinf(x, y)
array([1, 0, 1])
>>> y
array([1, 0, 1])
"""

# 添加新文档到 numpy._core.umath 模块，函数名为 'isnan'
# 检测逐元素是否为 NaN，返回结果为布尔数组
"""
Test element-wise for NaN and return result as a boolean array.

Parameters
----------
x : array_like
    Input array.
$PARAMS

Returns
-------
y : ndarray or bool
    True where ``x`` is NaN, false otherwise.
    $OUT_SCALAR_1

See Also
    --------
    isinf, isneginf, isposinf, isfinite, isnat

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> np.isnan(np.nan)
    True                        # 检查 np.nan 是否为 NaN (Not a Number)，返回 True
    >>> np.isnan(np.inf)
    False                       # 检查 np.inf 是否为 NaN，返回 False
    >>> np.isnan([np.log(-1.),1.,np.log(0)])
    array([ True, False, False])  # 检查数组中每个元素是否为 NaN，返回布尔数组

    """
# 将新文档添加到numpy._core.umath模块，函数名为'isnat'
add_newdoc('numpy._core.umath', 'isnat',
    """
    Test element-wise for NaT (not a time) and return result as a boolean array.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    x : array_like
        Input array with datetime or timedelta data type.
    $PARAMS  # 参数的详细说明将在实际文档中替换

    Returns
    -------
    y : ndarray or bool
        True where ``x`` is NaT, false otherwise.
        $OUT_SCALAR_1  # 返回值的详细说明将在实际文档中替换

    See Also
    --------
    isnan, isinf, isneginf, isposinf, isfinite

    Examples
    --------
    >>> np.isnat(np.datetime64("NaT"))
    True
    >>> np.isnat(np.datetime64("2016-01-01"))
    False
    >>> np.isnat(np.array(["NaT", "2016-01-01"], dtype="datetime64[ns]"))
    array([ True, False])

    """)

# 将新文档添加到numpy._core.umath模块，函数名为'left_shift'
add_newdoc('numpy._core.umath', 'left_shift',
    """
    Shift the bits of an integer to the left.

    Bits are shifted to the left by appending `x2` 0s at the right of `x1`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x1` by ``2**x2``.

    Parameters
    ----------
    x1 : array_like of integer type
        Input values.
    x2 : array_like of integer type
        Number of zeros to append to `x1`. Has to be non-negative.
        $BROADCASTABLE_2  # 参数的详细说明将在实际文档中替换
    $PARAMS  # 参数的详细说明将在实际文档中替换

    Returns
    -------
    out : array of integer type
        Return `x1` with bits shifted `x2` times to the left.
        $OUT_SCALAR_2  # 返回值的详细说明将在实际文档中替换

    See Also
    --------
    right_shift : Shift the bits of an integer to the right.
    binary_repr : Return the binary representation of the input number
        as a string.

    Examples
    --------
    >>> np.binary_repr(5)
    '101'
    >>> np.left_shift(5, 2)
    20
    >>> np.binary_repr(20)
    '10100'

    >>> np.left_shift(5, [1,2,3])
    array([10, 20, 40])

    Note that the dtype of the second argument may change the dtype of the
    result and can lead to unexpected results in some cases (see
    :ref:`Casting Rules <ufuncs.casting>`):

    >>> a = np.left_shift(np.uint8(255), np.int64(1))  # Expect 254
    >>> print(a, type(a)) # Unexpected result due to upcasting
    510 <class 'numpy.int64'>
    >>> b = np.left_shift(np.uint8(255), np.uint8(1))
    >>> print(b, type(b))
    254 <class 'numpy.uint8'>

    The ``<<`` operator can be used as a shorthand for ``np.left_shift`` on
    ndarrays.

    >>> x1 = 5
    >>> x2 = np.array([1, 2, 3])
    >>> x1 << x2
    array([10, 20, 40])

    """)

# 将新文档添加到numpy._core.umath模块，函数名为'less'
add_newdoc('numpy._core.umath', 'less',
    """
    Return the truth value of (x1 < x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.
        $BROADCASTABLE_2  # 参数的详细说明将在实际文档中替换
    $PARAMS  # 参数的详细说明将在实际文档中替换

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        $OUT_SCALAR_2  # 返回值的详细说明将在实际文档中替换

    See Also
    --------
    greater, less_equal, greater_equal, equal, not_equal

    Examples
    --------
    >>> np.less([1, 2], [2, 2])

    """)
    array([ True, False])

    # 创建一个NumPy数组，包含布尔值True和False
    # 这个数组展示了在ndarray上使用“<”运算符作为np.less的缩写的效果

    >>> a = np.array([1, 2])
    >>> b = np.array([2, 2])
    >>> a < b
    # 比较数组a和数组b中的元素，返回一个布尔类型的数组，显示每个元素是否满足小于的条件
    array([ True, False])

    """
# 添加新的文档字符串给指定的numpy._core.umath函数'less_equal'
add_newdoc('numpy._core.umath', 'less_equal',
    """
    返回元素级别上的(x1 <= x2)的真值。

    参数
    ----------
    x1, x2 : array_like
        输入数组。
        $BROADCASTABLE_2
    $PARAMS

    返回
    -------
    out : ndarray 或者标量
        输出数组，元素级别上的 `x1` 和 `x2` 的比较结果。
        通常为布尔类型，除非传入 ``dtype=object``。
        $OUT_SCALAR_2

    另请参阅
    --------
    greater, less, greater_equal, equal, not_equal

    示例
    --------
    >>> np.less_equal([4, 2, 1], [2, 2, 2])
    array([False,  True,  True])

    运算符 ``<=`` 可以用作 ``np.less_equal`` 在 ndarray 上的简写。

    >>> a = np.array([4, 2, 1])
    >>> b = np.array([2, 2, 2])
    >>> a <= b
    array([False,  True,  True])

    """)

# 添加新的文档字符串给指定的numpy._core.umath函数'log'
add_newdoc('numpy._core.umath', 'log',
    """
    自然对数，元素级别。

    自然对数 `log` 是指数函数的反函数，因此 `log(exp(x)) = x`。自然对数的底数为 `e`。

    参数
    ----------
    x : array_like
        输入值。
    $PARAMS

    返回
    -------
    y : ndarray
        `x` 的自然对数，元素级别。
        $OUT_SCALAR_1

    另请参阅
    --------
    log10, log2, log1p, emath.log

    注意
    -----
    对数是多值函数：对于每个 `x`，都有无穷多个 `z` 满足 `exp(z) = x`。惯例上返回其虚部在 `(-pi, pi]` 之间的 `z`。

    对于实数输入数据类型，`log` 总是返回实数输出。对于每个不能表示为实数或无穷大的值，返回 `nan` 并设置 `invalid` 浮点错误标志。

    对于复数输入，`log` 是一个复解析函数，在其上具有一个分支割线 `[-inf, 0]`，在上方连续。`log` 处理浮点数的负零作为一个无穷小的负数，符合 C99 标准。

    在输入具有负实部和非常小的负复部（接近0）的情况下，结果接近 `-pi`，且确切地评估为 `-pi`。

    引用
    ----------
    .. [1] M. Abramowitz 和 I.A. Stegun, "Handbook of Mathematical Functions",
           第10版印刷, 1964, pp. 67.
           https://personal.math.ubc.ca/~cbm/aands/page_67.htm
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm

    示例
    --------
    >>> np.log([1, np.e, np.e**2, 0])
    array([  0.,   1.,   2., -inf])

    """)

# 添加新的文档字符串给指定的numpy._core.umath函数'log10'
add_newdoc('numpy._core.umath', 'log10',
    """
    返回输入数组的以10为底的对数，元素级别。

    参数
    ----------
    x : array_like
        输入值。
    $PARAMS

    返回
    -------
    y : ndarray
        `x` 的以10为底的对数，元素级别。当 x 为负时返回 NaN。
        $OUT_SCALAR_1


    """)
    # 查看 emath.log10 函数的相关文档和用法
    See Also
    --------
    emath.log10
    
    # 注意事项部分提供了对于对数函数 log10 的解释，尤其是对于复数输入和实数输入的处理规则
    Notes
    -----
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `10**z = x`. The convention is to return the
    `z` whose imaginary part lies in `(-pi, pi]`.
    
    For real-valued input data types, `log10` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    
    For complex-valued input, `log10` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it.
    `log10` handles the floating-point negative zero as an infinitesimal
    negative number, conforming to the C99 standard.
    
    In the cases where the input has a negative real part and a very small
    negative complex part (approaching 0), the result is so close to `-pi`
    that it evaluates to exactly `-pi`.
    
    # 参考文献部分引用了两个对数函数的标准参考资料
    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67.
           https://personal.math.ubc.ca/~cbm/aands/page_67.htm
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm
    
    # 示例部分展示了不同输入情况下的 `log10` 函数的输出结果
    Examples
    --------
    >>> np.log10([1e-15, -3.])
    array([-15.,  nan])
# 在 numpy._core.umath 模块中添加新的文档字符串，用于描述 log2 函数
add_newdoc('numpy._core.umath', 'log2',
    """
    Base-2 logarithm of `x`.

    Parameters
    ----------
    x : array_like
        Input values.
        $PARAMS

    Returns
    -------
    y : ndarray
        Base-2 logarithm of `x`.
        $OUT_SCALAR_1

    See Also
    --------
    log, log10, log1p, emath.log2

    Notes
    -----
    .. versionadded:: 1.3.0

    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `2**z = x`. The convention is to return the `z`
    whose imaginary part lies in `(-pi, pi]`.

    For real-valued input data types, `log2` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log2` is a complex analytical function that
    has a branch cut `[-inf, 0]` and is continuous from above on it. `log2`
    handles the floating-point negative zero as an infinitesimal negative
    number, conforming to the C99 standard.

    In the cases where the input has a negative real part and a very small
    negative complex part (approaching 0), the result is so close to `-pi`
    that it evaluates to exactly `-pi`.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 2**4])
    >>> np.log2(x)
    array([-inf,   0.,   1.,   4.])

    >>> xi = np.array([0+1.j, 1, 2+0.j, 4.j])
    >>> np.log2(xi)
    array([ 0.+2.26618007j,  0.+0.j        ,  1.+0.j        ,  2.+2.26618007j])

    """)

# 在 numpy._core.umath 模块中添加新的文档字符串，用于描述 logaddexp 函数
add_newdoc('numpy._core.umath', 'logaddexp',
    """
    Logarithm of the sum of exponentiations of the inputs.

    Calculates ``log(exp(x1) + exp(x2))``. This function is useful in
    statistics where the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers.  In such cases
    the logarithm of the calculated probability is stored. This function
    allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : array_like
        Input values.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    result : ndarray
        Logarithm of ``exp(x1) + exp(x2)``.
        $OUT_SCALAR_2

    See Also
    --------
    logaddexp2: Logarithm of the sum of exponentiations of inputs in base 2.

    Notes
    -----
    .. versionadded:: 1.3.0

    Examples
    --------
    >>> prob1 = np.log(1e-50)
    >>> prob2 = np.log(2.5e-50)
    >>> prob12 = np.logaddexp(prob1, prob2)
    >>> prob12
    -113.87649168120691
    >>> np.exp(prob12)
    3.5000000000000057e-50

    """)

# 在 numpy._core.umath 模块中添加新的文档字符串，用于描述 logaddexp2 函数
add_newdoc('numpy._core.umath', 'logaddexp2',
    """
    Logarithm of the sum of exponentiations of the inputs in base-2.

    Calculates ``log2(2**x1 + 2**x2)``. This function is useful in machine
    learning when the calculated probabilities of events may be so small as
    to exceed the range of normal floating point numbers.  In such cases
    the logarithm of the calculated probability is stored. This function
    allows adding probabilities stored in such a fashion.
    
    Parameters
    ----------
    x1, x2 : array_like
        Input values.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    result : ndarray
        Logarithm of ``2**x1 + 2**x2``.
        $OUT_SCALAR_3

    See Also
    --------
    logaddexp: Logarithm of the sum of exponentiations of the inputs.

    Notes
    -----
    .. versionadded:: 1.3.0

    Examples
    --------
    >>> prob1 = np.log2(1e-50)
    >>> prob2 = np.log2(2.5e-50)
    >>> prob12 = np.logaddexp2(prob1, prob2)
    >>> prob12
    -113.87649168120691
    >>> 2**prob12
    3.5000000000000057e-50

    """
    # 计算输入的两个值的指数和的对数，以2为底
    the base-2 logarithm of the calculated probability can be used instead.
    # 此函数允许添加以这种方式存储的概率。
    This function allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : array_like
        # 输入的值。
        Input values.
        # 可广播到二维的情况。
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    result : ndarray
        # ``2**x1 + 2**x2`` 的以2为底的对数。
        Base-2 logarithm of ``2**x1 + 2**x2``.
        # 输出是标量的情况。
        $OUT_SCALAR_2

    See Also
    --------
    logaddexp: Logarithm of the sum of exponentiations of the inputs.

    Notes
    -----
    # 版本新增功能：1.3.0
    .. versionadded:: 1.3.0

    Examples
    --------
    >>> prob1 = np.log2(1e-50)
    >>> prob2 = np.log2(2.5e-50)
    >>> prob12 = np.logaddexp2(prob1, prob2)
    >>> prob1, prob2, prob12
    (-166.09640474436813, -164.77447664948076, -164.28904982231052)
    >>> 2**prob12
    3.4999999999999914e-50

    """
# 添加新的文档字符串到指定的NumPy函数
add_newdoc('numpy._core.umath', 'log1p',
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Calculates ``log(1 + x)``.

    Parameters
    ----------
    x : array_like
        Input values.
        $PARAMS

    Returns
    -------
    y : ndarray
        Natural logarithm of `1 + x`, element-wise.
        $OUT_SCALAR_1

    See Also
    --------
    expm1 : ``exp(x) - 1``, the inverse of `log1p`.

    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.

    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.

    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `log1p` is a complex analytical function that
    has a branch cut `[-inf, -1]` and is continuous from above on it.
    `log1p` handles the floating-point negative zero as an infinitesimal
    negative number, conforming to the C99 standard.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67.
           https://personal.math.ubc.ca/~cbm/aands/page_67.htm
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm

    Examples
    --------
    >>> np.log1p(1e-99)
    1e-99
    >>> np.log(1 + 1e-99)
    0.0

    """)

# 添加新的文档字符串到指定的NumPy函数
add_newdoc('numpy._core.umath', 'logical_and',
    """
    Compute the truth value of x1 AND x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        $OUT_SCALAR_2

    See Also
    --------
    logical_or, logical_not, logical_xor
    bitwise_and

    Examples
    --------
    >>> np.logical_and(True, False)
    False
    >>> np.logical_and([True, False], [False, False])
    array([False, False])

    >>> x = np.arange(5)
    >>> np.logical_and(x>1, x<4)
    array([False, False,  True,  True, False])


    The ``&`` operator can be used as a shorthand for ``np.logical_and`` on
    boolean ndarrays.

    >>> a = np.array([True, False])
    >>> b = np.array([False, False])
    >>> a & b
    array([False, False])

    """)

# 添加新的文档字符串到指定的NumPy函数
add_newdoc('numpy._core.umath', 'logical_not',
    """
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : array_like
        Logical NOT is applied to the elements of `x`.
    $PARAMS

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical NOT operation applied to the elements
        of `x`.
        $OUT_SCALAR_3

    Examples
    --------
    >>> np.logical_not(True)
    False
    >>> np.logical_not([True, False, 0, 1])
    array([False,  True,  True, False])

    """)
    # y : bool or ndarray of bool
    # 表示结果变量 y，可以是单个布尔值或布尔值数组，与输入 x 具有相同的形状，表示对 x 中元素进行逻辑非操作的结果。
    # $OUT_SCALAR_1
    # 此处可能存在文档生成工具的占位符或注释标记，用于文档自动生成，实际使用时可能会被替换成相应内容。

    # See Also
    # --------
    # logical_and, logical_or, logical_xor
    # 与本函数相关的其他逻辑运算函数，包括逻辑与、逻辑或、逻辑异或。

    # Examples
    # --------
    # >>> np.logical_not(3)
    # False
    # 对标量 3 进行逻辑非运算，结果为 False。
    # >>> np.logical_not([True, False, 0, 1])
    # array([False,  True,  True, False])
    # 对布尔数组 [True, False, 0, 1] 中的每个元素进行逻辑非运算，返回对应的布尔值数组。

    # >>> x = np.arange(5)
    # >>> np.logical_not(x<3)
    # array([False, False, False,  True,  True])
    # 创建一个长度为 5 的数组 x，对 x<3 的结果进行逻辑非运算，返回结果数组。
# 向 numpy._core.umath 模块添加新的文档字符串，定义了 logical_or 函数
add_newdoc('numpy._core.umath', 'logical_or',
    """
    Compute the truth value of x1 OR x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical OR is applied to the elements of `x1` and `x2`.
        $BROADCASTABLE_2  # 描述参数可以广播到的形状
    $PARAMS  # 描述未列出的参数信息

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        $OUT_SCALAR_2  # 描述返回值的形状和类型

    See Also
    --------
    logical_and, logical_not, logical_xor  # 相关的逻辑运算函数
    bitwise_or  # 位运算的 OR 操作

    Examples
    --------
    >>> np.logical_or(True, False)
    True
    >>> np.logical_or([True, False], [False, False])
    array([ True, False])

    >>> x = np.arange(5)
    >>> np.logical_or(x < 1, x > 3)
    array([ True, False, False, False,  True])

    The ``|`` operator can be used as a shorthand for ``np.logical_or`` on
    boolean ndarrays.

    >>> a = np.array([True, False])
    >>> b = np.array([False, False])
    >>> a | b
    array([ True, False])

    """)

# 向 numpy._core.umath 模块添加新的文档字符串，定义了 logical_xor 函数
add_newdoc('numpy._core.umath', 'logical_xor',
    """
    Compute the truth value of x1 XOR x2, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical XOR is applied to the elements of `x1` and `x2`.
        $BROADCASTABLE_2  # 描述参数可以广播到的形状
    $PARAMS  # 描述未列出的参数信息

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        $OUT_SCALAR_2  # 描述返回值的形状和类型

    See Also
    --------
    logical_and, logical_or, logical_not, bitwise_xor  # 相关的逻辑运算函数和位运算的 XOR 操作

    Examples
    --------
    >>> np.logical_xor(True, False)
    True
    >>> np.logical_xor([True, True, False, False], [True, False, True, False])
    array([False,  True,  True, False])

    >>> x = np.arange(5)
    >>> np.logical_xor(x < 1, x > 3)
    array([ True, False, False, False,  True])

    Simple example showing support of broadcasting

    >>> np.logical_xor(0, np.eye(2))
    array([[ True, False],
           [False,  True]])

    """)

# 向 numpy._core.umath 模块添加新的文档字符串，定义了 maximum 函数
add_newdoc('numpy._core.umath', 'maximum',
    """
    Element-wise maximum of array elements.

    Compare two arrays and return a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared.
        $BROADCASTABLE_2  # 描述参数可以广播到的形状
    $PARAMS  # 描述未列出的参数信息

    Returns
    -------
    y : ndarray or scalar
        The maximum of `x1` and `x2`, element-wise.
        $OUT_SCALAR_2  # 描述返回值的形状和类型

    See Also
    --------
    minimum :  # 最小值函数的参考链接
        Element-wise minimum of two arrays, propagates NaNs.

    """)
    fmax :
        Element-wise maximum of two arrays, ignores NaNs.
    amax :
        The maximum value of an array along a given axis, propagates NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignores NaNs.

    fmin, amin, nanmin

    Notes
    -----
    The maximum is equivalent to ``np.where(x1 >= x2, x1, x2)`` when
    neither x1 nor x2 are nans, but it is faster and does proper
    broadcasting.

    Examples
    --------
    >>> np.maximum([2, 3, 4], [1, 5, 2])
    array([2, 5, 4])

    >>> np.maximum(np.eye(2), [0.5, 2]) # broadcasting
    array([[ 1. ,  2. ],
           [ 0.5,  2. ]])

    >>> np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
    array([nan, nan, nan])
    >>> np.maximum(np.inf, 1)
    inf

    """


注释：

    fmax :
        两个数组的逐元素最大值，忽略 NaN 值。
    amax :
        沿指定轴的数组最大值，传播 NaN 值。
    nanmax :
        沿指定轴的数组最大值，忽略 NaN 值。

    fmin, amin, nanmin :
        与上述相似，分别表示逐元素最小值、沿指定轴的最小值，以及忽略 NaN 的最小值。

    Notes
    -----
    maximum 函数的作用类似于 ``np.where(x1 >= x2, x1, x2)``，在 x1 和 x2 都不是 NaN 时，
    但它执行速度更快且进行适当的广播。

    Examples
    --------
    几个使用 maximum 函数的示例：
    - 比较两个数组的逐元素最大值，返回结果数组。
    - 使用广播功能，将单位矩阵与数组进行比较。
    - 处理包含 NaN 值的情况，返回 NaN 值的数组。
    - 处理与无穷大的比较，返回无穷大值。

    """
# 添加新文档（docstring）到numpy._core.umath中的'minimum'函数
add_newdoc('numpy._core.umath', 'minimum',
    """
    Element-wise minimum of array elements.

    Compare two arrays and return a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared.
        $BROADCASTABLE_2  # 标记参数x1和x2可以广播
    $PARAMS  # 标记包含其他参数说明的部分

    Returns
    -------
    y : ndarray or scalar
        The minimum of `x1` and `x2`, element-wise.
        $OUT_SCALAR_2  # 标记返回值是数组或标量

    See Also
    --------
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.
    amin :
        The minimum value of an array along a given axis, propagates NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignores NaNs.

    fmax, amax, nanmax

    Notes
    -----
    The minimum is equivalent to ``np.where(x1 <= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster and does proper
    broadcasting.

    Examples
    --------
    >>> np.minimum([2, 3, 4], [1, 5, 2])
    array([1, 3, 2])

    >>> np.minimum(np.eye(2), [0.5, 2]) # broadcasting
    array([[ 0.5,  0. ],
           [ 0. ,  1. ]])

    >>> np.minimum([np.nan, 0, np.nan],[0, np.nan, np.nan])
    array([nan, nan, nan])
    >>> np.minimum(-np.inf, 1)
    -inf

    """)

# 添加新文档（docstring）到numpy._core.umath中的'fmax'函数
add_newdoc('numpy._core.umath', 'fmax',
    """
    Element-wise maximum of array elements.

    Compare two arrays and return a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then the
    non-nan element is returned. If both elements are NaNs then the first
    is returned.  The latter distinction is important for complex NaNs,
    which are defined as at least one of the real or imaginary parts being
    a NaN. The net effect is that NaNs are ignored when possible.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared.
        $BROADCASTABLE_2  # 标记参数x1和x2可以广播
    $PARAMS  # 标记包含其他参数说明的部分

    Returns
    -------
    y : ndarray or scalar
        The maximum of `x1` and `x2`, element-wise.
        $OUT_SCALAR_2  # 标记返回值是数组或标量

    See Also
    --------
    fmin :
        Element-wise minimum of two arrays, ignores NaNs.
    maximum :
        Element-wise maximum of two arrays, propagates NaNs.
    amax :
        The maximum value of an array along a given axis, propagates NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignores NaNs.

    minimum, amin, nanmin

    Notes
    -----
    .. versionadded:: 1.3.0

    The fmax is equivalent to ``np.where(x1 >= x2, x1, x2)`` when neither
    """
    x1 nor x2 are NaNs, but it is faster and does proper broadcasting.
    # x1 和 x2 都不是 NaN 时，函数表现更快且执行正确的广播操作。

    Examples
    --------
    >>> np.fmax([2, 3, 4], [1, 5, 2])
    array([ 2.,  5.,  4.])
    # 对两个数组进行元素级比较，返回每个位置上的较大值组成的数组。

    >>> np.fmax(np.eye(2), [0.5, 2])
    array([[ 1. ,  2. ],
           [ 0.5,  2. ]])
    # 将单位矩阵与数组进行元素级比较，返回每个位置上的较大值组成的数组。

    >>> np.fmax([np.nan, 0, np.nan],[0, np.nan, np.nan])
    array([ 0.,  0., nan])
    # 对包含 NaN 的数组进行元素级比较，返回每个位置上的较大值组成的数组。
    
    """
# 添加新的文档字符串到指定的NumPy模块和函数名
add_newdoc('numpy._core.umath', 'fmin',
    """
    Element-wise minimum of array elements.
    
    Compare two arrays element-wise and return a new array containing the element-wise
    minima. If one of the elements being compared is NaN, then the non-NaN element is returned.
    If both elements are NaNs, then the first NaN element is returned, which is crucial for
    handling complex NaNs where either the real or imaginary parts are NaN. NaNs are effectively
    ignored whenever possible.
    
    Parameters
    ----------
    x1, x2 : array_like
        Arrays holding the elements to be compared.
        $BROADCASTABLE_2
    $PARAMS
    
    Returns
    -------
    y : ndarray or scalar
        Element-wise minimum of `x1` and `x2`.
        $OUT_SCALAR_2
    
    See Also
    --------
    fmax :
        Element-wise maximum of two arrays, ignoring NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating NaNs.
    amin :
        Minimum value of an array along a given axis, propagating NaNs.
    nanmin :
        Minimum value of an array along a given axis, ignoring NaNs.
    
    maximum, amax, nanmax
    
    Notes
    -----
    .. versionadded:: 1.3.0
    
    fmin is equivalent to ``np.where(x1 <= x2, x1, x2)`` when neither x1 nor x2 are NaNs,
    but it performs faster and handles broadcasting correctly.
    
    Examples
    --------
    >>> np.fmin([2, 3, 4], [1, 5, 2])
    array([1, 3, 2])
    
    >>> np.fmin(np.eye(2), [0.5, 2])
    array([[ 0.5,  0. ],
           [ 0. ,  1. ]])
    
    >>> np.fmin([np.nan, 0, np.nan],[0, np.nan, np.nan])
    array([ 0.,  0., nan])
    
    """)

# 添加新的文档字符串到指定的NumPy模块和函数名
add_newdoc('numpy._core.umath', 'clip',
    """
    Clip (limit) the values in an array.
    
    Given an interval, values outside the interval are clipped to the interval edges.
    For example, if an interval of ``[0, 1]`` is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.
    
    This function is equivalent to, but faster than ``np.minimum(np.maximum(a, a_min), a_max)``.
    
    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : array_like
        Minimum value.
    a_max : array_like
        Maximum value.
    out : ndarray, optional
        Array where clipped values are placed. It can be the input array for in-place clipping.
        `out` must be of the right shape to hold the output, and its type is preserved.
    $PARAMS
    
    See Also
    --------
    numpy.clip :
        Wrapper that makes the `a_min` and `a_max` arguments optional, dispatching to one of
        `~numpy._core.umath.clip`, `~numpy._core.umath.minimum`, and `~numpy._core.umath.maximum`.
    
    Returns
    -------
    clipped_array : ndarray
        Array with elements of `a`, but values < `a_min` are replaced with `a_min`,
        and values > `a_max` are replaced with `a_max`.
    """)

# 添加新的文档字符串到指定的NumPy模块和函数名
add_newdoc('numpy._core.umath', 'matmul',
    """
    Matrix product of two arrays.
    
    Parameters
    # 定义 matmul 函数，用于执行矩阵乘法操作
    ----------
    # x1, x2 : array_like
    #     输入的数组，不允许是标量。
    # out : ndarray, optional
    #     存储结果的位置。如果提供，则必须具有与签名 `(n,k),(k,m)->(n,m)` 相匹配的形状。
    #     如果未提供或为 None，则返回一个新分配的数组。
    # **kwargs
    #     其他关键字参数，参见 :ref:`ufunc docs <ufuncs.kwargs>`。
    #
    #     .. versionadded:: 1.16
    #        现在支持处理 ufunc 的 kwargs
    #
    # Returns
    # -------
    # y : ndarray
    #     输入的矩阵的乘积。
    #     当 x1 和 x2 都是 1-D 向量时，这是一个标量。
    #
    # Raises
    # ------
    # ValueError
    #     如果 `x1` 的最后一个维度与 `x2` 倒数第二个维度的大小不同。
    #
    #     如果传入的是标量值。
    #
    # See Also
    # --------
    # vdot : 复共轭点积。
    # tensordot : 在任意轴上进行求和。
    # einsum : Einstein 求和约定。
    # dot : 具有不同广播规则的备用矩阵乘积。
    #
    # Notes
    # -----
    #
    # 行为取决于以下方式的参数。
    #
    # - 如果两个参数都是 2-D，则它们像传统矩阵一样相乘。
    # - 如果任一参数是 N-D，N > 2，则将其视为驻留在最后两个索引中的矩阵堆栈，并相应广播。
    # - 如果第一个参数是 1-D，则通过在其维度之前加一个 1 来提升为矩阵。矩阵乘法后，去除前加的 1。
    # - 如果第二个参数是 1-D，则通过在其维度之后添加一个 1 来提升为矩阵。矩阵乘法后，去除后加的 1。
    #
    # ``matmul`` 与 ``dot`` 有两个重要的不同之处：
    #
    # - 不允许标量乘法，请使用 ``*`` 替代。
    # - 矩阵堆栈按照元素的方式一起广播，遵循签名 ``(n,k),(k,m)->(n,m)``：
    #
    #   >>> a = np.ones([9, 5, 7, 4])
    #   >>> c = np.ones([9, 5, 4, 3])
    #   >>> np.dot(a, c).shape
    #   (9, 5, 7, 9, 5, 3)
    #   >>> np.matmul(a, c).shape
    #   (9, 5, 7, 3)
    #   >>> # n 是 7, k 是 4, m 是 3
    #
    # ``matmul`` 函数实现了在 Python 3.5 引入的 ``@`` 操作符的语义，参见 :pep:`465`。
    #
    # 在可能的情况下，它使用了优化的 BLAS 库（参见 `numpy.linalg`）。
    #
    # Examples
    # --------
    #
    # 对于 2-D 数组，它是矩阵乘积：
    #
    # >>> a = np.array([[1, 0],
    # ...               [0, 1]])
    # >>> b = np.array([[4, 1],
    # ...               [2, 2]])
    # >>> np.matmul(a, b)
    # array([[4, 1],
    #        [2, 2]])
    #
    # 对于 2-D 与 1-D 混合，结果是通常的。
    #
    # >>> a = np.array([[1, 0],
    # ...               [0, 1]])
    # >>> b = np.array([1, 2])
    # >>> np.matmul(a, b)
    # array([1, 2])
    # >>> np.matmul(b, a)
    # array([1, 2])
    #
    # 对于数组堆栈，广播是常规的。
    #
    # >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    # 创建一个 3 维的 numpy 数组 a，形状为 (2, 2, 2 * 4)，并填充从 0 到 15 的数值
    a = np.arange(2 * 2 * 4).reshape((2, 4, 2))
    
    # 使用 np.matmul 计算 a 和 b 的矩阵乘积，返回结果的形状
    np.matmul(a,b).shape
    
    # 使用 np.matmul 计算 a 和 b 的矩阵乘积，并返回结果中第 (0, 1, 1) 位置的元素值
    np.matmul(a, b)[0, 1, 1]
    
    # 计算 a[0, 1, :] 和 b[0, :, 1] 的内积和
    sum(a[0, 1, :] * b[0 , :, 1])
    
    # 当两个复数向量作为参数时，使用 np.matmul 计算它们的矩阵乘积，返回复数形式的结果
    np.matmul([2j, 3j], [2j, 3j])
    
    # 当一个向量和一个标量作为参数时，抛出 ValueError 异常，因为标量不具备足够的维度
    np.matmul([1,2], 3)
    
    # 使用 @ 运算符作为 np.matmul 的简写，计算两个复数向量的矩阵乘积
    x1 = np.array([2j, 3j])
    x2 = np.array([2j, 3j])
    x1 @ x2
    
    # 此功能从 NumPy 1.10.0 版本开始添加支持
# 将新文档添加到 `numpy._core.umath` 模块，定义函数 `vecdot`
add_newdoc('numpy._core.umath', 'vecdot',
    """
    Vector dot product of two arrays.

    Let :math:`\\mathbf{a}` be a vector in `x1` and :math:`\\mathbf{b}` be
    a corresponding vector in `x2`. The dot product is defined as:

    .. math::
       \\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=0}^{n-1} \\overline{a_i}b_i

    where the sum is over the last dimension (unless `axis` is specified) and
    where :math:`\\overline{a_i}` denotes the complex conjugate if :math:`a_i`
    is complex and the identity otherwise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays, scalars not allowed.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the broadcasted shape of `x1` and `x2` with the last axis
        removed. If not provided or None, a freshly-allocated array is used.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The vector dot product of the inputs.
        This is a scalar only when both x1, x2 are 1-d vectors.

    Raises
    ------
    ValueError
        If the last dimension of `x1` is not the same size as
        the last dimension of `x2`.

        If a scalar value is passed in.

    See Also
    --------
    vdot : same but flattens arguments first
    einsum : Einstein summation convention.

    Examples
    --------
    Get the projected size along a given normal for an array of vectors.

    >>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
    >>> n = np.array([0., 0.6, 0.8])
    >>> np.vecdot(v, n)
    array([ 3.,  8., 10.])

    .. versionadded:: 2.0.0
    """)

# 将新文档添加到 `numpy._core.umath` 模块，定义函数 `modf`
add_newdoc('numpy._core.umath', 'modf',
    """
    Return the fractional and integral parts of an array, element-wise.

    The fractional and integral parts are negative if the given number is
    negative.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    y1 : ndarray
        Fractional part of `x`.
        $OUT_SCALAR_1
    y2 : ndarray
        Integral part of `x`.
        $OUT_SCALAR_1

    Notes
    -----
    For integer input the return values are floats.

    See Also
    --------
    divmod : ``divmod(x, 1)`` is equivalent to ``modf`` with the return values
             switched, except it always has a positive remainder.

    Examples
    --------
    >>> np.modf([0, 3.5])
    (array([ 0. ,  0.5]), array([ 0.,  3.]))
    >>> np.modf(-0.5)
    (-0.5, -0)

    """)

# 将新文档添加到 `numpy._core.umath` 模块，定义函数 `multiply`
add_newdoc('numpy._core.umath', 'multiply',
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays to be multiplied.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : ndarray
        The product of `x1` and `x2`, element-wise.
        $OUT_SCALAR_2

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array broadcasting.
    """)
    Examples
    --------
    >>> np.multiply(2.0, 4.0)
    8.0
    # 对两个数进行乘法运算，返回结果 8.0
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.multiply(x1, x2)
    array([[  0.,   1.,   4.],
           [  0.,   4.,  10.],
           [  0.,   7.,  16.]])
    # 创建一个 3x3 的二维数组 x1 和一个长度为 3 的一维数组 x2，
    # 对它们进行对应位置的乘法运算，返回一个新的二维数组
    
    The ``*`` operator can be used as a shorthand for ``np.multiply`` on
    ndarrays.
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> x1 * x2
    array([[  0.,   1.,   4.],
           [  0.,   4.,  10.],
           [  0.,   7.,  16.]])
    # 使用 ``*`` 操作符可以作为 np.multiply 的简写形式，适用于 ndarray
# 添加新的文档字符串到指定的 numpy 模块中的函数 'negative'
add_newdoc('numpy._core.umath', 'negative',
    """
    Numerical negative, element-wise.

    Parameters
    ----------
    x : array_like or scalar
        Input array.
    $PARAMS

    Returns
    -------
    y : ndarray or scalar
        Returned array or scalar: `y = -x`.
        $OUT_SCALAR_1

    Examples
    --------
    >>> np.negative([1.,-1.])
    array([-1.,  1.])

    The unary ``-`` operator can be used as a shorthand for ``np.negative`` on
    ndarrays.

    >>> x1 = np.array(([1., -1.]))
    >>> -x1
    array([-1.,  1.])

    """)

# 添加新的文档字符串到指定的 numpy 模块中的函数 'positive'
add_newdoc('numpy._core.umath', 'positive',
    """
    Numerical positive, element-wise.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    x : array_like or scalar
        Input array.

    Returns
    -------
    y : ndarray or scalar
        Returned array or scalar: `y = +x`.
        $OUT_SCALAR_1

    Notes
    -----
    Equivalent to `x.copy()`, but only defined for types that support
    arithmetic.

    Examples
    --------

    >>> x1 = np.array(([1., -1.]))
    >>> np.positive(x1)
    array([ 1., -1.])

    The unary ``+`` operator can be used as a shorthand for ``np.positive`` on
    ndarrays.

    >>> x1 = np.array(([1., -1.]))
    >>> +x1
    array([ 1., -1.])

    """)

# 添加新的文档字符串到指定的 numpy 模块中的函数 'not_equal'
add_newdoc('numpy._core.umath', 'not_equal',
    """
    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        $OUT_SCALAR_2

    See Also
    --------
    equal, greater, greater_equal, less, less_equal

    Examples
    --------
    >>> np.not_equal([1.,2.], [1., 3.])
    array([False,  True])
    >>> np.not_equal([1, 2], [[1, 3],[1, 4]])
    array([[False,  True],
           [False,  True]])

    The ``!=`` operator can be used as a shorthand for ``np.not_equal`` on
    ndarrays.

    >>> a = np.array([1., 2.])
    >>> b = np.array([1., 3.])
    >>> a != b
    array([False,  True])


    """)

# 添加新的文档字符串到指定的 numpy 模块中的函数 '_ones_like'
add_newdoc('numpy._core.umath', '_ones_like',
    """
    This function used to be the numpy.ones_like, but now a specific
    function for that has been written for consistency with the other
    *_like functions. It is only used internally in a limited fashion now.

    See Also
    --------
    ones_like

    """)

# 添加新的文档字符串到指定的 numpy 模块中的函数 'power'
add_newdoc('numpy._core.umath', 'power',
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in
    `x2`.  `x1` and `x2` must be broadcastable to the same shape.

    An integer type raised to a negative integer power will raise a
    ``ValueError``.

    Negative values raised to a non-integral value will return ``nan``.
    To get complex results, cast the input to complex, or specify the
    """)
    # 返回 `x1` 中每个元素以 `x2` 中对应元素为指数的幂运算结果，支持广播
    # 如果 `dtype` 设置为 `complex`，则返回复数结果（参见下面的示例）

    Parameters
    ----------
    x1 : array_like
        底数数组。
    x2 : array_like
        指数数组。
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : ndarray
        数组 `x1` 中每个元素的指数为 `x2` 中对应元素的幂运算结果。
        $OUT_SCALAR_2

    See Also
    --------
    float_power : 将整数提升为浮点数的幂函数

    Examples
    --------
    对数组中的每个元素进行立方运算。

    >>> x1 = np.arange(6)
    >>> x1
    [0, 1, 2, 3, 4, 5]
    >>> np.power(x1, 3)
    array([  0,   1,   8,  27,  64, 125])

    将不同指数应用于不同的底数。

    >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    >>> np.power(x1, x2)
    array([  0.,   1.,   8.,  27.,  16.,   5.])

    广播效果示例。

    >>> x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> x2
    array([[1, 2, 3, 3, 2, 1],
           [1, 2, 3, 3, 2, 1]])
    >>> np.power(x1, x2)
    array([[ 0,  1,  8, 27, 16,  5],
           [ 0,  1,  8, 27, 16,  5]])

    在 ndarray 上，可以使用 `**` 操作符作为 `np.power` 的简写。

    >>> x2 = np.array([1, 2, 3, 3, 2, 1])
    >>> x1 = np.arange(6)
    >>> x1 ** x2
    array([ 0,  1,  8, 27, 16,  5])

    将负数提升至非整数指数将导致结果为 `nan`（并生成警告）。

    >>> x3 = np.array([-1.0, -4.0])
    >>> with np.errstate(invalid='ignore'):
    ...     p = np.power(x3, 1.5)
    ...
    >>> p
    array([nan, nan])

    要获取复数结果，需要设置参数 `dtype=complex`。

    >>> np.power(x3, 1.5, dtype=complex)
    array([-1.83697020e-16-1.j, -1.46957616e-15-8.j])
# 在 numpy._core.umath 模块中添加新文档或修改现有文档，函数名为 float_power
add_newdoc('numpy._core.umath', 'float_power',
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in `x2`.
    `x1` and `x2` must be broadcastable to the same shape. This differs from
    the power function in that integers, float16, and float32  are promoted to
    floats with a minimum precision of float64 so that the result is always
    inexact.  The intent is that the function will return a usable result for
    negative powers and seldom overflow for positive powers.

    Negative values raised to a non-integral value will return ``nan``.
    To get complex results, cast the input to complex, or specify the
    ``dtype`` to be ``complex`` (see the example below).

    .. versionadded:: 1.12.0

    Parameters
    ----------
    x1 : array_like
        The bases.
    x2 : array_like
        The exponents.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : ndarray
        The bases in `x1` raised to the exponents in `x2`.
        $OUT_SCALAR_2

    See Also
    --------
    power : power function that preserves type

    Examples
    --------
    Cube each element in a list.

    >>> x1 = range(6)
    >>> x1
    [0, 1, 2, 3, 4, 5]
    >>> np.float_power(x1, 3)
    array([   0.,    1.,    8.,   27.,   64.,  125.])

    Raise the bases to different exponents.

    >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    >>> np.float_power(x1, x2)
    array([  0.,   1.,   8.,  27.,  16.,   5.])

    The effect of broadcasting.

    >>> x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> x2
    array([[1, 2, 3, 3, 2, 1],
           [1, 2, 3, 3, 2, 1]])
    >>> np.float_power(x1, x2)
    array([[  0.,   1.,   8.,  27.,  16.,   5.],
           [  0.,   1.,   8.,  27.,  16.,   5.]])

    Negative values raised to a non-integral value will result in ``nan``
    (and a warning will be generated).

    >>> x3 = np.array([-1, -4])
    >>> with np.errstate(invalid='ignore'):
    ...     p = np.float_power(x3, 1.5)
    ...
    >>> p
    array([nan, nan])

    To get complex results, give the argument ``dtype=complex``.

    >>> np.float_power(x3, 1.5, dtype=complex)
    array([-1.83697020e-16-1.j, -1.46957616e-15-8.j])

    """)

# 在 numpy._core.umath 模块中添加新文档或修改现有文档，函数名为 radians
add_newdoc('numpy._core.umath', 'radians',
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like
        Input array in degrees.
    $PARAMS

    Returns
    -------
    y : ndarray
        The corresponding radian values.
        $OUT_SCALAR_1

    See Also
    --------
    deg2rad : equivalent function

    Examples
    --------
    Convert a degree array to radians

    >>> deg = np.arange(12.) * 30.
    >>> np.radians(deg)
    array([ 0.        ,  0.52359878,  1.04719755,  1.57079633,  2.0943951 ,
            2.61799388,  3.14159265,  3.66519143,  4.1887902 ,  4.71238898,
            5.23598776,  5.75958653])

    >>> out = np.zeros((deg.shape))

    """)
    # 使用 NumPy 的 radians 函数将角度数组 deg 转换为弧度，并将结果存储到数组 out 中
    ret = np.radians(deg, out)
    # 检查返回的数组 ret 是否与数组 out 是同一个对象（即它们在内存中的地址是否相同）
    ret is out
    # 返回 True，表明 ret 和 out 是同一个数组对象
    True
# 添加新的文档字符串到指定的 NumPy 函数
add_newdoc('numpy._core.umath', 'deg2rad',
    """
    将角度从度转换为弧度。

    Parameters
    ----------
    x : array_like
        角度值（以度为单位）。
    $PARAMS

    Returns
    -------
    y : ndarray
        对应的弧度值。
        $OUT_SCALAR_1

    See Also
    --------
    rad2deg : 将弧度转换为角度。
    unwrap : 通过包装方式去除角度中的大跳变。

    Notes
    -----
    .. versionadded:: 1.3.0

    ``deg2rad(x)`` 的计算结果为 ``x * pi / 180``。

    Examples
    --------
    >>> np.deg2rad(180)
    3.1415926535897931

    """)

# 添加新的文档字符串到指定的 NumPy 函数
add_newdoc('numpy._core.umath', 'reciprocal',
    """
    返回参数的倒数，逐元素计算。

    计算 ``1/x``。

    Parameters
    ----------
    x : array_like
        输入数组。
    $PARAMS

    Returns
    -------
    y : ndarray
        返回的数组。
        $OUT_SCALAR_1

    Notes
    -----
    .. note::
        此函数不适用于整数。

    对于绝对值大于1的整数参数，结果总是0，这是由于 Python 处理整数除法的方式。对于整数0，结果是溢出。

    Examples
    --------
    >>> np.reciprocal(2.)
    0.5
    >>> np.reciprocal([1, 2., 3.33])
    array([ 1.       ,  0.5      ,  0.3003003])

    """)

# 添加新的文档字符串到指定的 NumPy 函数
add_newdoc('numpy._core.umath', 'remainder',
    """
    返回元素级别的除法余数。

    计算与 `floor_divide` 函数互补的余数。等效于 Python 的模运算符 ``x1 % x2``，并且具有与除数 `x2` 相同的符号。
    MATLAB 中等效于 ``np.remainder`` 的函数是 ``mod``。

    .. warning::

        这与以下内容不应混淆：

        * Python 3.7 的 `math.remainder` 和 C 的 ``remainder``，计算 IEEE 余数，是 ``round(x1 / x2)`` 的补数。
        * MATLAB 的 ``rem`` 函数或 C 的 ``%`` 运算符，是 ``int(x1 / x2)`` 的补数。

    Parameters
    ----------
    x1 : array_like
        被除数数组。
    x2 : array_like
        除数数组。
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    y : ndarray
        除法商 ``floor_divide(x1, x2)`` 的元素级别余数。
        $OUT_SCALAR_2

    See Also
    --------
    floor_divide : Python 中的 ``//`` 运算符的等效。
    divmod : 同时执行 floor 除法和余数计算。
    fmod : MATLAB 中的 ``rem`` 函数的等效。
    divide, floor

    Notes
    -----
    当 `x2` 为0且 `x1` 和 `x2` 均为（数组的）整数时，返回0。
    ``mod`` 是 ``remainder`` 的别名。

    Examples
    --------
    >>> np.remainder([4, 7], [2, 3])
    array([0, 1])
    >>> np.remainder(np.arange(7), 5)
    array([0, 1, 2, 3, 4, 0, 1])

    ``%`` 运算符可用作 ndarray 上 ``np.remainder`` 的简写。

    >>> x1 = np.arange(7)
    >>> x1 % 5

    """)
    array([0, 1, 2, 3, 4, 0, 1])
    """
    创建一个名为 `array` 的 NumPy 数组，包含整数元素 0, 1, 2, 3, 4, 0, 1。
    """
# 添加新的文档字符串到 numpy._core.umath 模块下的 divmod 函数
add_newdoc('numpy._core.umath', 'divmod',
    """
    Return element-wise quotient and remainder simultaneously.

    .. versionadded:: 1.13.0

    ``np.divmod(x, y)`` is equivalent to ``(x // y, x % y)``, but faster
    because it avoids redundant work. It is used to implement the Python
    built-in function ``divmod`` on NumPy arrays.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out1 : ndarray
        Element-wise quotient resulting from floor division.
        $OUT_SCALAR_2
    out2 : ndarray
        Element-wise remainder from floor division.
        $OUT_SCALAR_2

    See Also
    --------
    floor_divide : Equivalent to Python's ``//`` operator.
    remainder : Equivalent to Python's ``%`` operator.
    modf : Equivalent to ``divmod(x, 1)`` for positive ``x`` with the return
           values switched.

    Examples
    --------
    >>> np.divmod(np.arange(5), 3)
    (array([0, 0, 0, 1, 1]), array([0, 1, 2, 0, 1]))

    The `divmod` function can be used as a shorthand for ``np.divmod`` on
    ndarrays.

    >>> x = np.arange(5)
    >>> divmod(x, 3)
    (array([0, 0, 0, 1, 1]), array([0, 1, 2, 0, 1]))

    """)

# 添加新的文档字符串到 numpy._core.umath 模块下的 right_shift 函数
add_newdoc('numpy._core.umath', 'right_shift',
    """
    Shift the bits of an integer to the right.

    Bits are shifted to the right `x2`.  Because the internal
    representation of numbers is in binary format, this operation is
    equivalent to dividing `x1` by ``2**x2``.

    Parameters
    ----------
    x1 : array_like, int
        Input values.
    x2 : array_like, int
        Number of bits to remove at the right of `x1`.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray, int
        Return `x1` with bits shifted `x2` times to the right.
        $OUT_SCALAR_2

    See Also
    --------
    left_shift : Shift the bits of an integer to the left.
    binary_repr : Return the binary representation of the input number
        as a string.

    Examples
    --------
    >>> np.binary_repr(10)
    '1010'
    >>> np.right_shift(10, 1)
    5
    >>> np.binary_repr(5)
    '101'

    >>> np.right_shift(10, [1,2,3])
    array([5, 2, 1])

    The ``>>`` operator can be used as a shorthand for ``np.right_shift`` on
    ndarrays.

    >>> x1 = 10
    >>> x2 = np.array([1,2,3])
    >>> x1 >> x2
    array([5, 2, 1])

    """)

# 添加新的文档字符串到 numpy._core.umath 模块下的 rint 函数
add_newdoc('numpy._core.umath', 'rint',
    """
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        Output array is same shape and type as `x`.
        $OUT_SCALAR_1

    See Also
    --------
    fix, ceil, floor, trunc

    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,

    """)
    -0.5 and 0.5 round to 0.0, etc.
    # 在 numpy 中，-0.5 和 0.5 等数字会被四舍五入到最接近的整数，此外还有类似的情况。

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    # 创建一个 numpy 数组 a，包含了一组浮点数
    >>> np.rint(a)
    # 使用 numpy 的 rint 函数对数组 a 中的每个元素进行四舍五入
    array([-2., -2., -0.,  0.,  2.,  2.,  2.])
    # 返回结果是一个新的 numpy 数组，包含了每个元素四舍五入后的值

    """
# 将新的文档条目添加到numpy._core.umath模块中，名称为'sign'
add_newdoc('numpy._core.umath', 'sign',
    """
    Returns an element-wise indication of the sign of a number.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.  nan
    is returned for nan inputs.

    For complex inputs, the `sign` function returns ``x / abs(x)``, the
    generalization of the above (and ``0 if x==0``).

    .. versionchanged:: 2.0.0
        Definition of complex sign changed to follow the Array API standard.

    Parameters
    ----------
    x : array_like
        Input values.
    $PARAMS

    Returns
    -------
    y : ndarray
        The sign of `x`.
        $OUT_SCALAR_1

    Notes
    -----
    There is more than one definition of sign in common use for complex
    numbers.  The definition used here is equivalent to :math:`x/\\sqrt{x*x}`
    which is different from a common alternative, :math:`x/|x|`.

    Examples
    --------
    >>> np.sign([-5., 4.5])
    array([-1.,  1.])
    >>> np.sign(0)
    0
    >>> np.sign([3-4j, 8j])
    array([0.6-0.8j, 0. +1.j ])

    """)

# 将新的文档条目添加到numpy._core.umath模块中，名称为'signbit'
add_newdoc('numpy._core.umath', 'signbit',
    """
    Returns element-wise True where signbit is set (less than zero).

    Parameters
    ----------
    x : array_like
        The input value(s).
    $PARAMS

    Returns
    -------
    result : ndarray of bool
        Output array, or reference to `out` if that was supplied.
        $OUT_SCALAR_1

    Examples
    --------
    >>> np.signbit(-1.2)
    True
    >>> np.signbit(np.array([1, -2.3, 2.1]))
    array([False,  True, False])

    """)

# 将新的文档条目添加到numpy._core.umath模块中，名称为'copysign'
add_newdoc('numpy._core.umath', 'copysign',
    """
    Change the sign of x1 to that of x2, element-wise.

    If `x2` is a scalar, its sign will be copied to all elements of `x1`.

    Parameters
    ----------
    x1 : array_like
        Values to change the sign of.
    x2 : array_like
        The sign of `x2` is copied to `x1`.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        The values of `x1` with the sign of `x2`.
        $OUT_SCALAR_2

    Examples
    --------
    >>> np.copysign(1.3, -1)
    -1.3
    >>> 1/np.copysign(0, 1)
    inf
    >>> 1/np.copysign(0, -1)
    -inf

    >>> np.copysign([-1, 0, 1], -1.1)
    array([-1., -0., -1.])
    >>> np.copysign([-1, 0, 1], np.arange(3)-1)
    array([-1.,  0.,  1.])

    """)

# 将新的文档条目添加到numpy._core.umath模块中，名称为'nextafter'
add_newdoc('numpy._core.umath', 'nextafter',
    """
    Return the next floating-point value after x1 towards x2, element-wise.

    Parameters
    ----------
    x1 : array_like
        Values to find the next representable value of.
    x2 : array_like
        The direction where to look for the next representable value of `x1`.
        $BROADCASTABLE_2
    $PARAMS

    Returns
    -------
    out : ndarray or scalar
        The next representable values of `x1` in the direction of `x2`.
        $OUT_SCALAR_2

    Examples
    --------
    >>> eps = np.finfo(np.float64).eps
    >>> np.nextafter(1, 2) == eps + 1
    True

    """)
    # 调用 NumPy 中的 nextafter 函数，用于获取接近指定数字的下一个浮点数
    np.nextafter([1, 2], [2, 1])
    # 比较 nextafter 返回的结果是否等于指定的值数组 [eps + 1, 2 - eps]
    np.nextafter([1, 2], [2, 1]) == [eps + 1, 2 - eps]
    # 返回一个布尔数组，指示每个对应位置的元素是否相等
    array([ True,  True])
# 添加新的文档字符串到numpy._core.umath模块中，定义spacing函数
add_newdoc('numpy._core.umath', 'spacing',
    """
    Return the distance between x and the nearest adjacent number.

    Parameters
    ----------
    x : array_like
        Values to find the spacing of.
    $PARAMS  # 参数详细信息将在实际使用时被替换

    Returns
    -------
    out : ndarray or scalar
        The spacing of values of `x`.
        $OUT_SCALAR_1  # 返回值的具体描述将在实际使用时被替换

    Notes
    -----
    It can be considered as a generalization of EPS:
    ``spacing(np.float64(1)) == np.finfo(np.float64).eps``, and there
    should not be any representable number between ``x + spacing(x)`` and
    x for any finite x.

    Spacing of +- inf and NaN is NaN.

    Examples
    --------
    >>> np.spacing(1) == np.finfo(np.float64).eps
    True

    """)

# 添加新的文档字符串到numpy._core.umath模块中，定义sin函数
add_newdoc('numpy._core.umath', 'sin',
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x : array_like
        Angle, in radians (:math:`2 \\pi` rad equals 360 degrees).
    $PARAMS  # 参数详细信息将在实际使用时被替换

    Returns
    -------
    y : array_like
        The sine of each element of x.
        $OUT_SCALAR_1  # 返回值的具体描述将在实际使用时被替换

    See Also
    --------
    arcsin, sinh, cos

    Notes
    -----
    The sine is one of the fundamental functions of trigonometry (the
    mathematical study of triangles).  Consider a circle of radius 1
    centered on the origin.  A ray comes in from the :math:`+x` axis, makes
    an angle at the origin (measured counter-clockwise from that axis), and
    departs from the origin.  The :math:`y` coordinate of the outgoing
    ray's intersection with the unit circle is the sine of that angle.  It
    ranges from -1 for :math:`x=3\\pi / 2` to +1 for :math:`\\pi / 2.`  The
    function has zeroes where the angle is a multiple of :math:`\\pi`.
    Sines of angles between :math:`\\pi` and :math:`2\\pi` are negative.
    The numerous properties of the sine and related functions are included
    in any standard trigonometry text.

    Examples
    --------
    Print sine of one angle:

    >>> np.sin(np.pi/2.)
    1.0

    Print sines of an array of angles given in degrees:

    >>> np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180. )
    array([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])

    Plot the sine function:

    >>> import matplotlib.pylab as plt
    >>> x = np.linspace(-np.pi, np.pi, 201)
    >>> plt.plot(x, np.sin(x))
    >>> plt.xlabel('Angle [rad]')
    >>> plt.ylabel('sin(x)')
    >>> plt.axis('tight')
    >>> plt.show()

    """)

# 添加新的文档字符串到numpy._core.umath模块中，定义sinh函数
add_newdoc('numpy._core.umath', 'sinh',
    """
    Hyperbolic sine, element-wise.

    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or
    ``-1j * np.sin(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS  # 参数详细信息将在实际使用时被替换

    Returns
    -------
    y : ndarray
        The corresponding hyperbolic sine values.
        $OUT_SCALAR_1  # 返回值的具体描述将在实际使用时被替换

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------

    """)
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972, pg. 83.

    Examples
    --------
    >>> np.sinh(0)
    0.0
    >>> np.sinh(np.pi*1j/2)
    1j
    >>> np.sinh(np.pi*1j) # (exact value is 0)
    1.2246063538223773e-016j
    >>> # Discrepancy due to vagaries of floating point arithmetic.

    >>> # Example of providing the optional output parameter
    >>> out1 = np.array([0], dtype='d')
    >>> out2 = np.sinh([0.1], out1)
    >>> out2 is out1
    True

    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> np.sinh(np.zeros((3,3)),np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: operands could not be broadcast together with shapes (3,3) (2,2)
# 在 numpy._core.umath 中添加新的文档字符串和函数 'sqrt'
add_newdoc('numpy._core.umath', 'sqrt',
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    x : array_like
        The values whose square-roots are required.
    $PARAMS  # 描述参数的占位符

    Returns
    -------
    y : ndarray
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`.  If any element in `x` is
        complex, a complex array is returned (and the square-roots of
        negative reals are calculated).  If all of the elements in `x`
        are real, so is `y`, with negative elements returning ``nan``.
        If `out` was provided, `y` is a reference to it.
        $OUT_SCALAR_1  # 描述返回值的占位符

    See Also
    --------
    emath.sqrt
        A version which returns complex numbers when given negative reals.
        Note that 0.0 and -0.0 are handled differently for complex inputs.

    Notes
    -----
    *sqrt* has--consistent with common convention--as its branch cut the
    real "interval" [`-inf`, 0), and is continuous from above on it.
    A branch cut is a curve in the complex plane across which a given
    complex function fails to be continuous.

    Examples
    --------
    >>> np.sqrt([1,4,9])
    array([ 1.,  2.,  3.])

    >>> np.sqrt([4, -1, -3+4J])
    array([ 2.+0.j,  0.+1.j,  1.+2.j])

    >>> np.sqrt([4, -1, np.inf])
    array([ 2., nan, inf])

    """)

# 在 numpy._core.umath 中添加新的文档字符串和函数 'cbrt'
add_newdoc('numpy._core.umath', 'cbrt',
    """
    Return the cube-root of an array, element-wise.

    .. versionadded:: 1.10.0  # 描述函数的版本增加信息

    Parameters
    ----------
    x : array_like
        The values whose cube-roots are required.
    $PARAMS  # 描述参数的占位符

    Returns
    -------
    y : ndarray
        An array of the same shape as `x`, containing the
        cube root of each element in `x`.
        If `out` was provided, `y` is a reference to it.
        $OUT_SCALAR_1  # 描述返回值的占位符

    Examples
    --------
    >>> np.cbrt([1,8,27])
    array([ 1.,  2.,  3.])

    """)

# 在 numpy._core.umath 中添加新的文档字符串和函数 'square'
add_newdoc('numpy._core.umath', 'square',
    """
    Return the element-wise square of the input.

    Parameters
    ----------
    x : array_like
        Input data.
    $PARAMS  # 描述参数的占位符

    Returns
    -------
    out : ndarray or scalar
        Element-wise `x*x`, of the same shape and dtype as `x`.
        $OUT_SCALAR_1  # 描述返回值的占位符

    See Also
    --------
    numpy.linalg.matrix_power
    sqrt
    power

    Examples
    --------
    >>> np.square([-1j, 1])
    array([-1.-0.j,  1.+0.j])

    """)

# 在 numpy._core.umath 中添加新的文档字符串和函数 'subtract'
add_newdoc('numpy._core.umath', 'subtract',
    """
    Subtract arguments, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be subtracted from each other.
        $BROADCASTABLE_2  # 描述参数的占位符
    $PARAMS  # 描述参数的占位符

    Returns
    -------
    y : ndarray
        The difference of `x1` and `x2`, element-wise.
        $OUT_SCALAR_2  # 描述返回值的占位符

    Notes
    -----
    Equivalent to ``x1 - x2`` in terms of array broadcasting.

    Examples
    --------
    >>> np.subtract(1.0, 4.0)
    -3.0

    >>> x1 = np.arange(9.0).reshape((3, 3))

    """)
    # 创建一个包含三个元素的 NumPy 数组，元素值为 [0.0, 1.0, 2.0]
    >>> x2 = np.arange(3.0)
    
    # 使用 NumPy 的 subtract 函数计算两个数组的差，结果是一个二维数组
    # 第一个数组 x1 是一个 3x3 的矩阵，包含 0 到 8 的整数，reshape 后为 (3, 3)
    # 第二个数组 x2 是一个包含三个元素的一维数组 [0.0, 1.0, 2.0]
    >>> np.subtract(x1, x2)
    array([[ 0.,  0.,  0.],
           [ 3.,  3.,  3.],
           [ 6.,  6.,  6.]])
    
    # 在 NumPy 中，操作符 '-' 可以作为 np.subtract 的缩写，用于对数组进行减法运算
    # 这里展示了使用操作符 '-' 计算数组之间的减法，结果与上面使用 np.subtract 的结果相同
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    # 创建一个 3x3 的数组 x1，包含 0 到 8 的浮点数，reshape 后形状为 (3, 3)
    >>> x2 = np.arange(3.0)
    # 创建一个包含三个元素的一维数组 x2，元素值为 [0.0, 1.0, 2.0]
    
    >>> x1 - x2
    # 对数组 x1 和 x2 进行减法操作，得到一个二维数组
    array([[0., 0., 0.],
           [3., 3., 3.],
           [6., 6., 6.]])
    
    # 这里再次展示了使用操作符 '-' 进行数组减法，得到的结果与上面使用 np.subtract 的结果一致
# 添加新的文档字符串给 'numpy._core.umath' 中的 'tan' 函数
add_newdoc('numpy._core.umath', 'tan',
    """
    Compute tangent element-wise.

    Equivalent to ``np.sin(x)/np.cos(x)`` element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    y : ndarray
        The corresponding tangent values.
        $OUT_SCALAR_1

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972.

    Examples
    --------
    >>> from math import pi
    >>> np.tan(np.array([-pi,pi/2,pi]))
    array([  1.22460635e-16,   1.63317787e+16,  -1.22460635e-16])
    >>>
    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = np.array([0], dtype='d')
    >>> out2 = np.cos([0.1], out1)
    >>> out2 is out1
    True
    >>>
    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> np.cos(np.zeros((3,3)),np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: operands could not be broadcast together with shapes (3,3) (2,2)

    """)

# 添加新的文档字符串给 'numpy._core.umath' 中的 'tanh' 函数
add_newdoc('numpy._core.umath', 'tanh',
    """
    Compute hyperbolic tangent element-wise.

    Equivalent to ``np.sinh(x)/np.cosh(x)`` or ``-1j * np.tan(1j*x)``.

    Parameters
    ----------
    x : array_like
        Input array.
    $PARAMS

    Returns
    -------
    y : ndarray
        The corresponding hyperbolic tangent values.
        $OUT_SCALAR_1

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
           New York, NY: Dover, 1972, pg. 83.
           https://personal.math.ubc.ca/~cbm/aands/page_83.htm

    .. [2] Wikipedia, "Hyperbolic function",
           https://en.wikipedia.org/wiki/Hyperbolic_function

    Examples
    --------
    >>> np.tanh((0, np.pi*1j, np.pi*1j/2))
    array([ 0. +0.00000000e+00j,  0. -1.22460635e-16j,  0. +1.63317787e+16j])

    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = np.array([0], dtype='d')
    >>> out2 = np.tanh([0.1], out1)
    >>> out2 is out1
    True

    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> np.tanh(np.zeros((3,3)),np.zeros((2,2)))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: operands could not be broadcast together with shapes (3,3) (2,2)

    """)

# 添加新的文档字符串给 'numpy._core.umath' 中的 'frexp' 函数
add_newdoc('numpy._core.umath', 'frexp',
    """
    Decompose the elements of x into mantissa and twos exponent.
    
    """)
    Returns (`mantissa`, `exponent`), where ``x = mantissa * 2**exponent``.
    The mantissa lies in the open interval(-1, 1), while the twos
    exponent is a signed integer.

    Parameters
    ----------
    x : array_like
        Array of numbers to be decomposed.
    out1 : ndarray, optional
        Output array for the mantissa. Must have the same shape as `x`.
    out2 : ndarray, optional
        Output array for the exponent. Must have the same shape as `x`.
    $PARAMS

    Returns
    -------
    mantissa : ndarray
        Floating values between -1 and 1.
        $OUT_SCALAR_1
    exponent : ndarray
        Integer exponents of 2.
        $OUT_SCALAR_1

    See Also
    --------
    ldexp : Compute ``y = x1 * 2**x2``, the inverse of `frexp`.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.

    Examples
    --------
    >>> x = np.arange(9)
    >>> y1, y2 = np.frexp(x)
    >>> y1
    array([ 0.   ,  0.5  ,  0.5  ,  0.75 ,  0.5  ,  0.625,  0.75 ,  0.875,
            0.5  ])
    >>> y2
    array([0, 1, 2, 2, 3, 3, 3, 3, 4])
    >>> y1 * 2**y2
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])

    """
# 添加新的文档字符串给 'numpy._core.umath' 模块的 'ldexp' 函数
add_newdoc('numpy._core.umath', 'ldexp',
    """
    返回 x1 * 2**x2，逐元素计算。

    使用尾数 `x1` 和指数 `x2` 构造浮点数 ``x1 * 2**x2``。

    参数
    ----------
    x1 : array_like
        乘数数组。
    x2 : array_like, int
        指数数组。
        $BROADCASTABLE_2
    $PARAMS

    返回
    -------
    y : ndarray 或标量
        ``x1 * 2**x2`` 的结果。
        $OUT_SCALAR_2

    参见
    --------
    frexp : 从 ``x = y1 * 2**y2`` 返回 (y1, y2)，是 `ldexp` 的逆操作。

    注意
    -----
    不支持复杂数数据类型，否则会引发 TypeError。

    `ldexp` 可以作为 `frexp` 的逆操作，如果单独使用，使用表达式 ``x1 * 2**x2`` 更加清晰。

    示例
    --------
    >>> np.ldexp(5, np.arange(4))
    array([ 5., 10., 20., 40.], dtype=float16)

    >>> x = np.arange(6)
    >>> np.ldexp(*np.frexp(x))
    array([ 0.,  1.,  2.,  3.,  4.,  5.])

    """)

# 添加新的文档字符串给 'numpy._core.umath' 模块的 'gcd' 函数
add_newdoc('numpy._core.umath', 'gcd',
    """
    返回 ``|x1|`` 和 ``|x2|`` 的最大公约数。

    参数
    ----------
    x1, x2 : array_like, int
        值的数组。
        $BROADCASTABLE_2

    返回
    -------
    y : ndarray 或标量
        输入绝对值的最大公约数。
        $OUT_SCALAR_2

    参见
    --------
    lcm : 最小公倍数

    示例
    --------
    >>> np.gcd(12, 20)
    4
    >>> np.gcd.reduce([15, 25, 35])
    5
    >>> np.gcd(np.arange(6), 20)
    array([20,  1,  2,  1,  4,  5])

    """)

# 添加新的文档字符串给 'numpy._core.umath' 模块的 'lcm' 函数
add_newdoc('numpy._core.umath', 'lcm',
    """
    返回 ``|x1|`` 和 ``|x2|`` 的最小公倍数。

    参数
    ----------
    x1, x2 : array_like, int
        值的数组。
        $BROADCASTABLE_2

    返回
    -------
    y : ndarray 或标量
        输入绝对值的最小公倍数。
        $OUT_SCALAR_2

    参见
    --------
    gcd : 最大公约数

    示例
    --------
    >>> np.lcm(12, 20)
    60
    >>> np.lcm.reduce([3, 12, 20])
    60
    >>> np.lcm.reduce([40, 12, 20])
    120
    >>> np.lcm(np.arange(6), 20)
    array([ 0, 20, 20, 60, 20, 20])

    """)

# 添加新的文档字符串给 'numpy._core.umath' 模块的 'bitwise_count' 函数
add_newdoc('numpy._core.umath', 'bitwise_count',
    """
    计算输入数组 ``x`` 绝对值中 1 的位数。
    类似于内置的 `int.bit_count` 或 C++ 中的 ``popcount``。

    参数
    ----------
    x : array_like, unsigned int
        输入数组。
    $PARAMS

    返回
    -------
    y : ndarray
        输入中 1 的位数。
        对所有整数类型返回 uint8
        $OUT_SCALAR_1

    参考
    ----------
    .. [1] https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    .. [2] Wikipedia, "Hamming weight",
           https://en.wikipedia.org/wiki/Hamming_weight


    """)
    .. [3] http://aggregate.ee.engr.uky.edu/MAGIC/#Population%20Count%20(Ones%20Count)

    Examples
    --------
    >>> np.bitwise_count(1023)
    # 对给定的整数 1023 进行位计数，返回其中包含的二进制位为 1 的个数，结果为 10
    10
    >>> a = np.array([2**i - 1 for i in range(16)])
    # 创建一个包含 16 个元素的 NumPy 数组 a，其中每个元素都是 2 的幂次减 1 的值
    >>> np.bitwise_count(a)
    # 对数组 a 中的每个元素进行位计数，返回包含每个元素二进制位为 1 的个数的数组，数据类型为 uint8
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
          dtype=uint8)

    """
# 给指定模块和函数添加新的文档字符串
add_newdoc('numpy._core.umath', 'str_len',
    """
    返回每个元素的长度。对于字节串，长度是以字节计算；对于Unicode串，是以Unicode代码点计算。

    Parameters
    ----------
    x : array_like
        输入数组，其元素类型为``StringDType``、``bytes_``或``str_``
        $PARAMS

    Returns
    -------
    y : ndarray
        整数类型的输出数组
        $OUT_SCALAR_1

    See Also
    --------
    len

    Examples
    --------
    >>> a = np.array(['Grace Hopper Conference', 'Open Source Day'])
    >>> np.strings.str_len(a)
    array([23, 15])
    >>> a = np.array(['\u0420', '\u043e'])
    >>> np.strings.str_len(a)
    array([1, 1])
    >>> a = np.array([['hello', 'world'], ['\u0420', '\u043e']])
    >>> np.strings.str_len(a)
    array([[5, 5], [1, 1]])

    """)

# 给指定模块和函数添加新的文档字符串
add_newdoc('numpy._core.umath', 'isalpha',
    """
    如果解释为字符串的数据中所有字符都是字母并且至少有一个字符，则为每个元素返回True，否则返回False。

    对于字节串（即``bytes``），字母字符是序列
    b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'中的字节值。
    对于Unicode串，字母字符是Unicode字符数据库中定义的“Letter”。

    Parameters
    ----------
    x : array_like
        输入数组，其元素类型为``StringDType``、``bytes_``或``str_``
        $PARAMS

    Returns
    -------
    y : ndarray
        布尔类型的输出数组
        $OUT_SCALAR_1

    See Also
    --------
    str.isalpha

    """)

# 给指定模块和函数添加新的文档字符串
add_newdoc('numpy._core.umath', 'isdigit',
    """
    如果字符串中所有字符都是数字并且至少有一个字符，则为每个元素返回True，否则返回False。

    对于字节串，数字是序列
    b'0123456789'中的字节值。
    对于Unicode串，数字包括十进制字符和需要特殊处理的数字，如兼容上标数字。
    这也包括不能用于十进制数的数字，如Kharosthi数字。

    Parameters
    ----------
    x : array_like
        输入数组，其元素类型为``StringDType``、``bytes_``或``str_``
        $PARAMS

    Returns
    -------
    y : ndarray
        布尔类型的输出数组
        $OUT_SCALAR_1

    See Also
    --------
    str.isdigit

    Examples
    --------
    >>> a = np.array(['a', 'b', '0'])
    >>> np.strings.isdigit(a)
    array([False, False,  True])
    >>> a = np.array([['a', 'b', '0'], ['c', '1', '2']])
    >>> np.strings.isdigit(a)
    array([[False, False,  True], [False,  True,  True]])

    """)

# 给指定模块和函数添加新的文档字符串
add_newdoc('numpy._core.umath', 'isspace',
    r"""
    如果字符串中只有空白字符并且至少有一个字符，则为每个元素返回True，否则返回False。

    对于字节串，空白字符是序列
    b' \t\n\r\x0b\f'中的字符。
    对于Unicode串，字符是
    whitespace, if, in the Unicode character database, its general
    category is Zs (“Separator, space”), or its bidirectional class
    is one of WS, B, or S.


# 检查字符是否是空白字符，根据 Unicode 字符数据库的定义，如果其一般类别为 Zs（“分隔符，空格”），或其双向类别为 WS、B 或 S 中的一种。



    Parameters
    ----------
    x : array_like, with ``StringDType``, ``bytes_``, or ``str_`` dtype


# 参数
# ------
# x : array_like
#     输入数组，元素类型可以是 ``StringDType``、``bytes_`` 或 ``str_``。



    $PARAMS


# $PARAMS
# （未提供足够的上下文，可能是占位符或参数的具体描述）



    Returns
    -------
    y : ndarray
        Output array of bools
        $OUT_SCALAR_1


# 返回
# ------
# y : ndarray
#     布尔类型的输出数组。
#     $OUT_SCALAR_1
# （未提供足够的上下文，可能是占位符或返回值的具体描述）



    See Also
    --------
    str.isspace


# 参见
# ------
# str.isspace
# （指向相关的字符串方法，用于检查字符串是否只包含空白字符）



    """)




# """
# （多余的字符串结尾标记，可能是输入文档中的错误）
# 向 numpy._core.umath 模块添加新文档，定义 isalnum 函数
add_newdoc('numpy._core.umath', 'isalnum',
    """
    Returns true for each element if all characters in the string are
    alphanumeric and there is at least one character, false otherwise.

    Parameters
    ----------
    x : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings.
        $PARAMS

    Returns
    -------
    out : ndarray
        Output array of bools
        $OUT_SCALAR_1

    See Also
    --------
    str.isalnum

    Examples
    --------
    >>> a = np.array(['a', '1', 'a1', '(', ''])
    >>> np.strings.isalnum(a)
    array([ True,  True,  True, False, False])
    
    """)

# 向 numpy._core.umath 模块添加新文档，定义 islower 函数
add_newdoc('numpy._core.umath', 'islower',
    """
    Returns true for each element if all cased characters in the
    string are lowercase and there is at least one cased character,
    false otherwise.

    Parameters
    ----------
    x : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings.
        $PARAMS

    Returns
    -------
    out : ndarray
        Output array of bools
        $OUT_SCALAR_1

    See Also
    --------
    str.islower

    Examples
    --------
    >>> np.strings.islower("GHC")
    array(False)
    >>> np.strings.islower("ghc")
    array(True)

    """)

# 向 numpy._core.umath 模块添加新文档，定义 isupper 函数
add_newdoc('numpy._core.umath', 'isupper',
    """
    Return true for each element if all cased characters in the
    string are uppercase and there is at least one character, false
    otherwise.

    Parameters
    ----------
    x : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings.
        $PARAMS

    Returns
    -------
    out : ndarray
        Output array of bools
        $OUT_SCALAR_1

    See Also
    --------
    str.isupper

    Examples
    --------
    >>> np.strings.isupper("GHC")
    array(True)     
    >>> a = np.array(["hello", "HELLO", "Hello"])
    >>> np.strings.isupper(a)
    array([False,  True, False]) 

    """)

# 向 numpy._core.umath 模块添加新文档，定义 istitle 函数
add_newdoc('numpy._core.umath', 'istitle',
    """
    Returns true for each element if the element is a titlecased
    string and there is at least one character, false otherwise.

    Parameters
    ----------
    x : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings.
        $PARAMS

    Returns
    -------
    out : ndarray
        Output array of bools
        $OUT_SCALAR_1

    See Also
    --------
    str.istitle

    Examples
    --------
    >>> np.strings.istitle("Numpy Is Great")
    array(True)

    >>> np.strings.istitle("Numpy is great")
    array(False)
    
    """)

# 向 numpy._core.umath 模块添加新文档，定义 isdecimal 函数
add_newdoc('numpy._core.umath', 'isdecimal',
    """
    For each element, return True if there are only decimal
    characters in the element.

    Decimal characters include digit characters, and all characters
    that can be used to form decimal-radix numbers,
    e.g. ``U+0660, ARABIC-INDIC DIGIT ZERO``.

    Parameters
    ----------
    x : array_like, with ``StringDType`` or ``str_`` dtype
        Input array of strings.
        $PARAMS

    Returns
    -------
    y : ndarray
        Output array of bools
        $OUT_SCALAR_1

    See Also
    --------

    """)
    str.isdecimal



    # 返回一个布尔值，指示字符串是否只包含十进制字符
    str.isdecimal



    Examples
    --------
    >>> np.strings.isdecimal(['12345', '4.99', '123ABC', ''])
    array([ True, False, False, False])

    """



    # 示例用法：使用 np.strings.isdecimal() 函数检查字符串数组中每个字符串是否只包含十进制字符
    Examples
    --------
    >>> np.strings.isdecimal(['12345', '4.99', '123ABC', ''])
    array([ True, False, False, False])

    """
# 添加新的文档字符串到 numpy._core.umath 模块中的 isnumeric 函数
add_newdoc('numpy._core.umath', 'isnumeric',
    """
    对于每个元素，如果元素中仅包含数字字符，则返回 True。

    数字字符包括数字字符本身，以及具有 Unicode 数字值属性的所有字符，
    例如 ``U+2155, VULGAR FRACTION ONE FIFTH``。

    Parameters
    ----------
    x : array_like，具有 ``StringDType`` 或 ``str_`` dtype
    $PARAMS

    Returns
    -------
    y : ndarray
        布尔类型的输出数组
        $OUT_SCALAR_1

    See Also
    --------
    str.isnumeric

    Examples
    --------
    >>> np.strings.isnumeric(['123', '123abc', '9.0', '1/4', 'VIII'])
    array([ True, False, False, False, False])

    """)

# 添加新的文档字符串到 numpy._core.umath 模块中的 find 函数
add_newdoc('numpy._core.umath', 'find',
    """
    对于每个元素，在字符串中返回子字符串 `x2` 第一次出现的最低索引，
    使得 `x2` 包含在范围 [`x3`, `x4`] 中。

    Parameters
    ----------
    x1 : array-like，具有 ``StringDType``，``bytes_`` 或 ``str_`` dtype

    x2 : array-like，具有 ``StringDType``，``bytes_`` 或 ``str_`` dtype

    x3 : array_like，具有 ``int_`` dtype

    x4 : array_like，具有 ``int_`` dtype
        $PARAMS

    `x3` 和 `x4` 被解释为切片表示法。

    Returns
    -------
    y : ndarray
        整数类型的输出数组
        $OUT_SCALAR_2

    See Also
    --------
    str.find

    Examples
    --------
    >>> a = np.array(["NumPy is a Python library"])
    >>> np.strings.find(a, "Python", 0, None)
    array([11])

    """)

# 添加新的文档字符串到 numpy._core.umath 模块中的 rfind 函数
add_newdoc('numpy._core.umath', 'rfind',
    """
    对于每个元素，在字符串中返回子字符串 `x2` 最后一次出现的最高索引，
    使得 `x2` 包含在范围 [`x3`, `x4`] 中。

    Parameters
    ----------
    x1 : array-like，具有 ``StringDType``，``bytes_`` 或 ``str_`` dtype

    x2 : array-like，具有 ``StringDType``，``bytes_`` 或 ``str_`` dtype

    x3 : array_like，具有 ``int_`` dtype

    x4 : array_like，具有 ``int_`` dtype
        $PARAMS

    `x3` 和 `x4` 被解释为切片表示法。

    Returns
    -------
    y : ndarray
        整数类型的输出数组
        $OUT_SCALAR_2

    See Also
    --------
    str.rfind

    """)

# 添加新的文档字符串到 numpy._core.umath 模块中的 count 函数
add_newdoc('numpy._core.umath', 'count',
    """
    返回数组中子字符串 `x2` 在范围 [`x3`, `x4`] 中的非重叠出现次数。

    Parameters
    ----------
    x1 : array-like，具有 ``StringDType``，``bytes_`` 或 ``str_`` dtype

    x2 : array-like，具有 ``StringDType``，``bytes_`` 或 ``str_`` dtype
       要搜索的子字符串。

    x3 : array_like，具有 ``int_`` dtype

    x4 : array_like，具有 ``int_`` dtype
        $PARAMS

    `x3` 和 `x4` 被解释为切片表示法。

    Returns
    -------
    y : ndarray
        整数类型的输出数组
        $OUT_SCALAR_2

    See Also
    --------
    str.count

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    """)
    # 创建一个包含字符串数组的 NumPy 数组，每个字符串最长为 7 个字符
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    # 使用 np.strings.count 函数统计每个字符串中出现 'A' 的次数
    >>> np.strings.count(c, 'A')
    # 返回一个包含每个字符串中 'A' 的出现次数的 NumPy 数组
    array([3, 1, 1])
    # 使用 np.strings.count 函数统计每个字符串中出现 'aA' 的次数（大小写敏感）
    >>> np.strings.count(c, 'aA')
    # 返回一个包含每个字符串中 'aA' 的出现次数的 NumPy 数组
    array([3, 1, 0])
    # 使用 np.strings.count 函数统计每个字符串中在指定范围内（从 start 到 end-1）出现 'A' 的次数
    >>> np.strings.count(c, 'A', start=1, end=4)
    # 返回一个包含每个字符串在指定范围内出现 'A' 的次数的 NumPy 数组
    array([2, 1, 1])
    # 使用 np.strings.count 函数统计每个字符串中在指定范围内（从 start 到 end-1）出现 'A' 的次数
    >>> np.strings.count(c, 'A', start=1, end=3)
    # 返回一个包含每个字符串在指定范围内出现 'A' 的次数的 NumPy 数组
    array([1, 0, 0])
# 将新文档添加到numpy._core.umath模块中的index函数
add_newdoc('numpy._core.umath', 'index',
    """
    类似于`find`，但在子字符串未找到时引发ValueError异常。

    Parameters
    ----------
    x1 : array_like，具有`StringDType`，`bytes_`或`unicode_` dtype

    x2 : array_like，具有`StringDType`，`bytes_`或`unicode_` dtype

    x3, x4 : array_like，具有任何整数dtype
        要查找的范围，按切片表示法解释。
        $PARAMS

    Returns
    -------
    out : ndarray
        输出整数数组。如果未找到`x2`，则引发ValueError异常。
        $OUT_SCALAR_2

    See Also
    --------
    find, str.find

    Examples
    --------
    >>> a = np.array(["Computer Science"])
    >>> np.strings.index(a, "Science")
    array([9])

    """)

# 将新文档添加到numpy._core.umath模块中的rindex函数
add_newdoc('numpy._core.umath', 'rindex',
    """
    类似于`rfind`，但在子字符串未找到时引发ValueError异常。

    Parameters
    ----------
    x1 : array_like，具有`StringDType`，`bytes_`或`unicode_` dtype

    x2 : array_like，具有`StringDType`，`bytes_`或`unicode_` dtype

    x3, x4 : array_like，具有任何整数dtype
        要查找的范围，按切片表示法解释。
        $PARAMS

    Returns
    -------
    out : ndarray
        输出整数数组。如果未找到`x2`，则引发ValueError异常。
        $OUT_SCALAR_2

    See Also
    --------
    rfind, str.rfind

    Examples
    --------
    >>> a = np.array(["Computer Science"])
    >>> np.strings.rindex(a, "Science")
    array([9])

    """)

# 将新文档添加到numpy._core.umath模块中的_replace函数
add_newdoc('numpy._core.umath', '_replace',
    """
    `replace`的ufunc实现。此内部函数由带有设置`out`的`replace`调用，
    以便知道结果字符串缓冲区的大小。
    """)

# 将新文档添加到numpy._core.umath模块中的startswith函数
add_newdoc('numpy._core.umath', 'startswith',
    """
    返回布尔数组，其中在`x1`中的字符串元素以`x2`开头时为`True`，否则为`False`。

    Parameters
    ----------
    x1 : array-like，具有`StringDType`，`bytes_`或`str_` dtype

    x2 : array-like，具有`StringDType`，`bytes_`或`str_` dtype

    x3 : array_like，具有`int_` dtype

    x4 : array_like，具有`int_` dtype
        $PARAMS
        使用`x3`，从该位置开始比较。使用`x4`，在该位置停止比较。

    Returns
    -------
    out : ndarray
        输出布尔数组
        $OUT_SCALAR_2

    See Also
    --------
    str.startswith

    """)

# 将新文档添加到numpy._core.umath模块中的endswith函数
add_newdoc('numpy._core.umath', 'endswith',
    """
    返回布尔数组，其中在`x1`中的字符串元素以`x2`结尾时为`True`，否则为`False`。

    Parameters
    ----------
    x1 : array-like，具有`StringDType`，`bytes_`或`str_` dtype

    x2 : array-like，具有`StringDType`，`bytes_`或`str_` dtype

    x3 : array_like，具有`int_` dtype
    x4 : array_like, with ``int_`` dtype
        # 参数 x4：数组类型，要求为整数类型
        # $PARAMS
        # 与 x3 一起使用时，从该位置开始测试。与 x4 一起使用时，
        # 在该位置停止比较。

    Returns
    -------
    out : ndarray
        # 返回值：布尔数组
        # $OUT_SCALAR_2

    See Also
    --------
    str.endswith
        # 参见：str.endswith 函数

    Examples
    --------
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.endswith(s, 'ar')
    array([False,  True])
    >>> np.strings.endswith(s, 'a', start=1, end=2)
    array([False,  True])

    """
# 添加新文档到 'numpy._core.umath' 模块中的 '_strip_chars' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_strip_chars', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_lstrip_chars' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_lstrip_chars', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_rstrip_chars' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_rstrip_chars', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_strip_whitespace' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_strip_whitespace', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_lstrip_whitespace' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_lstrip_whitespace', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_rstrip_whitespace' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_rstrip_whitespace', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_expandtabs_length' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_expandtabs_length', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_expandtabs' 函数，文档内容为空字符串
add_newdoc('numpy._core.umath', '_expandtabs', '')

# 添加新文档到 'numpy._core.umath' 模块中的 '_center' 函数，提供详细的文档说明
add_newdoc('numpy._core.umath', '_center',
    """
    Return a copy of `x1` with its elements centered in a string of
    length `x2`.

    Parameters
    ----------
    x1 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings to be centered.
    x2 : array_like, with any integer dtype
        The length of the resulting strings, unless `width < str_len(a)`.
    x3 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        The padding character to use.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types.

    See Also
    --------
    str.center : Equivalent Python built-in method.

    Examples
    --------
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'])
    >>> np.strings.center(c, width=9)
    array(['   a1b2  ', '   1b2a  ', '   b2a1  ', '   2a1b  '], dtype='<U9')
    >>> np.strings.center(c, width=9, fillchar='*')
    array(['***a1b2**', '***1b2a**', '***b2a1**', '***2a1b**'], dtype='<U9')
    >>> np.strings.center(c, width=1)
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')
    """
)

# 添加新文档到 'numpy._core.umath' 模块中的 '_ljust' 函数，提供详细的文档说明
add_newdoc('numpy._core.umath', '_ljust',
    """
    Return an array with the elements of `x1` left-justified in a
    string of length `x2`.

    Parameters
    ----------
    x1 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings to be left-justified.
    x2 : array_like, with any integer dtype
        The length of the resulting strings, unless `width < str_len(a)`.
    x3 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        The padding character to use.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type.

    See Also
    --------
    str.ljust : Equivalent Python built-in method.

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.ljust(c, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.ljust(c, width=9)
    array(['aAaAaA   ', '  aA     ', 'abBABba  '], dtype='<U9')
    """
)

# 添加新文档到 'numpy._core.umath' 模块中的 '_rjust' 函数，提供详细的文档说明
add_newdoc('numpy._core.umath', '_rjust',
    """
    Return an array with the elements of `x1` right-justified in a
    string of length `x2`.

    Parameters
    ----------
    x1 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array of strings to be right-justified.
    x2 : array_like, with any integer dtype
        The length of the resulting strings, unless `width < str_len(a)`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type.
    """
)
    x3 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        The padding character to use.
        $PARAMS

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type
        $OUT_SCALAR_2

    See Also
    --------
    str.rjust

    Examples
    --------
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rjust(a, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.rjust(a, width=9)
    array(['   aAaAaA', '     aA  ', '  abBABba'], dtype='<U9')


注释：


# x3参数：array_like，其元素可以是StringDType、bytes_或str_类型的数组
# 使用的填充字符
# $PARAMS

# 返回值
# -------
# out : ndarray
#     输出的ndarray数组，其元素类型为StringDType、bytes_或str_，取决于输入类型
# $OUT_SCALAR_2

# 参见
# --------
# str.rjust

# 示例
# --------
# 创建一个包含字符串的numpy数组a
# >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
# 对数组a中的每个字符串右对齐，宽度为3
# >>> np.strings.rjust(a, width=3)
# 返回一个数组，包含右对齐后的结果，元素类型为'<U7'
# array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
# 对数组a中的每个字符串右对齐，宽度为9
# >>> np.strings.rjust(a, width=9)
# 返回一个数组，包含右对齐后的结果，元素类型为'<U9'
# array(['   aAaAaA', '     aA  ', '  abBABba'], dtype='<U9')
# 将新文档添加到numpy._core.umath模块中，函数名为'_zfill'
add_newdoc('numpy._core.umath', '_zfill',
    """
    Return the numeric string left-filled with zeros. A leading
    sign prefix (``+``/``-``) is handled by inserting the padding
    after the sign character rather than before.

    Parameters
    ----------
    x1 : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        输入数组，可以是``StringDType``、``bytes_``或``str_``类型

    x2 : array_like, with any integer dtype
        要在`x1`中的元素左侧填充的字符串宽度
        $PARAMS

    Returns
    -------
    out : ndarray
        输出数组，类型为``StringDType``、``bytes_``或``str_``，取决于输入类型
        $OUT_SCALAR_2

    See Also
    --------
    str.zfill

    Examples
    --------
    >>> np.strings.zfill(['1', '-1', '+1'], 3)
    array(['001', '-01', '+01'], dtype='<U3')

    """)

# 将新文档添加到numpy._core.umath模块中，函数名为'_partition_index'
add_newdoc('numpy._core.umath', '_partition_index',
    """
    Partition each element in ``x1`` around ``x2``, at precomputed
    index ``x3``.

    For each element in ``x1``, split the element at the first
    occurrence of ``x2`` at location ``x3``, and return a 3-tuple
    containing the part before the separator, the separator itself,
    and the part after the separator. If the separator is not found,
    the first item of the tuple will contain the whole string, and
    the second and third ones will be the empty string.

    Parameters
    ----------
    x1 : array-like, with ``bytes_``, or ``str_`` dtype
        输入数组

    x2 : array-like, with ``bytes_``, or ``str_`` dtype
        用于在``x1``中的每个字符串元素周围分割的分隔符

    x3 : array-like, with any integer dtype
        分隔符的预先计算索引（<0表示分隔符不存在）

    Returns
    -------
    out : 3-tuple:
        - 包含分隔符之前部分的``bytes_``或``str_`` dtype数组
        - 包含分隔符的``bytes_``或``str_`` dtype数组
        - 包含分隔符之后部分的``bytes_``或``str_`` dtype数组

    See Also
    --------
    str.partition

    Examples
    --------
    此ufunc最容易通过``np.strings.partition``使用，在计算索引后调用它：

    >>> x = np.array(["Numpy is nice!"])
    >>> np.strings.partition(x, " ")
    (array(['Numpy'], dtype='<U5'),
     array([' '], dtype='<U1'),
     array(['is nice!'], dtype='<U8'))

    """)

# 将新文档添加到numpy._core.umath模块中，函数名为'_rpartition_index'
add_newdoc('numpy._core.umath', '_rpartition_index',
    """
    Partition each element in ``x1`` around the right-most separator,
    ``x2``, at precomputed index ``x3``.

    For each element in ``x1``, split the element at the last
    occurrence of ``x2`` at location ``x3``, and return a 3-tuple
    containing the part before the separator, the separator itself,
    and the part after the separator. If the separator is not found,
    the third item of the tuple will contain the whole string, and
    the first and second ones will be the empty string.

    Parameters
    ----------
    x1 : array-like, with ``bytes_``, or ``str_`` dtype
        输入数组

    x2 : array-like, with ``bytes_``, or ``str_`` dtype
        最右边分隔符，用于在``x1``中的每个元素周围分割

    x3 : array-like, with any integer dtype
        分隔符的预先计算索引（<0表示分隔符不存在）

    Returns
    -------
    out : 3-tuple:
        - 包含分隔符之前部分的``bytes_``或``str_`` dtype数组
        - 包含分隔符的``bytes_``或``str_`` dtype数组
        - 包含分隔符之后部分的``bytes_``或``str_`` dtype数组

    """)
    x1 : array-like, with ``bytes_``, or ``str_`` dtype
        输入数组，可以是具有 ``bytes_`` 或 ``str_`` 数据类型的数组
    x2 : array-like, with ``bytes_``, or ``str_`` dtype
        用于分割 ``x1`` 中每个字符串元素的分隔符数组
    x3 : array-like, with any integer dtype
        分隔符的索引数组（<0 表示分隔符不存在）

    Returns
    -------
    out : 3-tuple:
        返回一个包含三个元素的元组:
        - 具有 ``bytes_`` 或 ``str_`` 数据类型的数组，包含分隔符之前的部分
        - 具有 ``bytes_`` 或 ``str_`` 数据类型的数组，包含分隔符本身
        - 具有 ``bytes_`` 或 ``str_`` 数据类型的数组，包含分隔符之后的部分

    See Also
    --------
    str.rpartition
        参考 Python 字符串方法 rpartition 的用法

    Examples
    --------
    通过 ``np.strings.rpartition`` 最方便地使用这个函数，它在计算索引后会调用这个函数：

    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rpartition(a, 'A')
    (array(['aAaAa', '  a', 'abB'], dtype='<U5'),
     array(['A', 'A', 'A'], dtype='<U1'),
     array(['', '  ', 'Bba'], dtype='<U3'))

    """
# 将新的文档添加到指定的 NumPy 模块和函数
add_newdoc('numpy._core.umath', '_partition',
    """
    Partition each element in ``x1`` around ``x2``.
    
    For each element in ``x1``, split the element at the first
    occurrence of ``x2`` and return a 3-tuple containing the part before
    the separator, the separator itself, and the part after the
    separator. If the separator is not found, the first item of the
    tuple will contain the whole string, and the second and third ones
    will be the empty string.
    
    Parameters
    ----------
    x1 : array-like, with ``StringDType`` dtype
        Input array
    x2 : array-like, with ``StringDType`` dtype
        Separator to split each string element in ``x1``.
    
    Returns
    -------
    out : 3-tuple:
        - ``StringDType`` array with the part before the separator
        - ``StringDType`` array with the separator
        - ``StringDType`` array with the part after the separator
    
    See Also
    --------
    str.partition
    
    Examples
    --------
    The ufunc is used most easily via ``np.strings.partition``,
    which calls it under the hood::
    
    >>> x = np.array(["Numpy is nice!"], dtype="T")
    >>> np.strings.partition(x, " ")
    (array(['Numpy'], dtype=StringDType()),
     array([' '], dtype=StringDType()),
     array(['is nice!'], dtype=StringDType()))
    
    """)

# 将新的文档添加到指定的 NumPy 模块和函数
add_newdoc('numpy._core.umath', '_rpartition',
    """
    Partition each element in ``x1`` around the right-most separator,
    ``x2``.
    
    For each element in ``x1``, split the element at the last
    occurrence of ``x2`` at location ``x3``, and return a 3-tuple
    containing the part before the separator, the separator itself,
    and the part after the separator. If the separator is not found,
    the third item of the tuple will contain the whole string, and
    the first and second ones will be the empty string.
    
    Parameters
    ----------
    x1 : array-like, with ``StringDType`` dtype
        Input array
    x2 : array-like, with ``StringDType`` dtype
        Separator to split each string element in ``x1``.
    
    Returns
    -------
    out : 3-tuple:
        - ``StringDType`` array with the part before the separator
        - ``StringDType`` array with the separator
        - ``StringDType`` array with the part after the separator
    
    See Also
    --------
    str.rpartition
    
    Examples
    --------
    The ufunc is used most easily via ``np.strings.rpartition``,
    which calls it after calculating the indices::
    
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'], dtype="T")
    >>> np.strings.rpartition(a, 'A')
    (array(['aAaAa', '  a', 'abB'], dtype=StringDType()),
     array(['A', 'A', 'A'], dtype=StringDType()),
     array(['', '  ', 'Bba'], dtype=StringDType()))
    
    """)
```