# `numpy-ml\numpy_ml\neural_nets\utils\utils.py`

```
import numpy as np

#######################################################################
#                           Training Utils                            #
#######################################################################

# 定义一个函数用于生成训练数据的小批量索引
def minibatch(X, batchsize=256, shuffle=True):
    """
    Compute the minibatch indices for a training dataset.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)`
        The dataset to divide into minibatches. Assumes the first dimension
        represents the number of training examples.
    batchsize : int
        The desired size of each minibatch. Note, however, that if ``X.shape[0] %
        batchsize > 0`` then the final batch will contain fewer than batchsize
        entries. Default is 256.
    shuffle : bool
        Whether to shuffle the entries in the dataset before dividing into
        minibatches. Default is True.

    Returns
    -------
    mb_generator : generator
        A generator which yields the indices into X for each batch
    n_batches: int
        The number of batches
    """
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize : (i + 1) * batchsize]

    return mb_generator(), n_batches


#######################################################################
#                            Padding Utils                            #
#######################################################################

# 计算二维卷积时的填充维度
def calc_pad_dims_2D(X_shape, out_dim, kernel_shape, stride, dilation=0):
    """
    Compute the padding necessary to ensure that convolving `X` with a 2D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with dimension
    `out_dim`.

    Parameters
    ----------
    X_shape : tuple of `(n_ex, in_rows, in_cols, in_ch)`
        输入体积的维度。对 `in_rows` 和 `in_cols` 进行填充。
    out_dim : tuple of `(out_rows, out_cols)`
        应用卷积后输出示例的期望维度。
    kernel_shape : 2-tuple
        2D 卷积核的维度。
    stride : int
        卷积核的步幅。
    dilation : int
        卷积核元素之间插入的像素数。默认为 0。

    Returns
    -------
    padding_dims : 4-tuple
        `X` 的填充维度。组织形式为 (左，右，上，下)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(out_dim, tuple):
        raise ValueError("`out_dim` must be of type tuple")

    if not isinstance(kernel_shape, tuple):
        raise ValueError("`kernel_shape` must be of type tuple")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fr, fc = kernel_shape
    out_rows, out_cols = out_dim
    n_ex, in_rows, in_cols, in_ch = X_shape

    # 根据膨胀因子更新有效滤波器形状
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    pr = int((stride * (out_rows - 1) + _fr - in_rows) / 2)
    pc = int((stride * (out_cols - 1) + _fc - in_cols) / 2)

    out_rows1 = int(1 + (in_rows + 2 * pr - _fr) / stride)
    out_cols1 = int(1 + (in_cols + 2 * pc - _fc) / stride)

    # 向右/向下添加不对称填充像素
    pr1, pr2 = pr, pr
    if out_rows1 == out_rows - 1:
        pr1, pr2 = pr, pr + 1
    elif out_rows1 != out_rows:
        raise AssertionError

    pc1, pc2 = pc, pc
    if out_cols1 == out_cols - 1:
        pc1, pc2 = pc, pc + 1
    elif out_cols1 != out_cols:
        raise AssertionError
    # 检查是否有任何一个元素小于0，如果是则抛出数值错误异常
    if any(np.array([pr1, pr2, pc1, pc2]) < 0):
        raise ValueError(
            "Padding cannot be less than 0. Got: {}".format((pr1, pr2, pc1, pc2))
        )
    # 返回元组 (pr1, pr2, pc1, pc2)
    return (pr1, pr2, pc1, pc2)
def calc_pad_dims_1D(X_shape, l_out, kernel_width, stride, dilation=0, causal=False):
    """
    Compute the padding necessary to ensure that convolving `X` with a 1D kernel
    of shape `kernel_shape` and stride `stride` produces outputs with length
    `l_out`.

    Parameters
    ----------
    X_shape : tuple of `(n_ex, l_in, in_ch)`
        Dimensions of the input volume. Padding is applied on either side of
        `l_in`.
    l_out : int
        The desired length an output example after applying the convolution.
    kernel_width : int
        The width of the 1D convolution kernel.
    stride : int
        The stride for the convolution kernel.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.
    causal : bool
        Whether to compute the padding dims for a regular or causal
        convolution. If causal, padding is added only to the left side of the
        sequence. Default is False.

    Returns
    -------
    padding_dims : 2-tuple
        Padding dims for X. Organized as (left, right)
    """
    if not isinstance(X_shape, tuple):
        raise ValueError("`X_shape` must be of type tuple")

    if not isinstance(l_out, int):
        raise ValueError("`l_out` must be of type int")

    if not isinstance(kernel_width, int):
        raise ValueError("`kernel_width` must be of type int")

    if not isinstance(stride, int):
        raise ValueError("`stride` must be of type int")

    d = dilation
    fw = kernel_width
    n_ex, l_in, in_ch = X_shape

    # update effective filter shape based on dilation factor
    _fw = fw * (d + 1) - d
    total_pad = int((stride * (l_out - 1) + _fw - l_in))

    if not causal:
        pw = total_pad // 2
        l_out1 = int(1 + (l_in + 2 * pw - _fw) / stride)

        # add asymmetric padding pixels to right / bottom
        pw1, pw2 = pw, pw
        if l_out1 == l_out - 1:
            pw1, pw2 = pw, pw + 1
        elif l_out1 != l_out:
            raise AssertionError
    # 如果是因果卷积，只在序列的左侧填充
    if causal:
        pw1, pw2 = total_pad, 0
        # 计算输出序列的长度
        l_out1 = int(1 + (l_in + total_pad - _fw) / stride)
        # 断言输出序列的长度与给定的长度相同
        assert l_out1 == l_out

    # 如果填充值中有任何一个小于0，抛出数值错误异常
    if any(np.array([pw1, pw2]) < 0):
        raise ValueError("Padding cannot be less than 0. Got: {}".format((pw1, pw2)))
    # 返回填充值元组
    return (pw1, pw2)
def pad1D(X, pad, kernel_width=None, stride=None, dilation=0):
    """
    Zero-pad a 3D input volume `X` along the second dimension.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
        Input volume. Padding is applied to `l_in`.
    pad : tuple, int, or {'same', 'causal'}
        The padding amount. If 'same', add padding to ensure that the output
        length of a 1D convolution with a kernel of `kernel_shape` and stride
        `stride` is the same as the input length.  If 'causal' compute padding
        such that the output both has the same length as the input AND
        ``output[t]`` does not depend on ``input[t + 1:]``. If 2-tuple,
        specifies the number of padding columns to add on each side of the
        sequence.
    kernel_width : int
        The dimension of the 2D convolution kernel. Only relevant if p='same'
        or 'causal'. Default is None.
    stride : int
        The stride for the convolution kernel. Only relevant if p='same' or
        'causal'. Default is None.
    dilation : int
        The dilation of the convolution kernel. Only relevant if p='same' or
        'causal'. Default is None.

    Returns
    -------
    X_pad : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, padded_seq, in_channels)`
        The padded output volume
    p : 2-tuple
        The number of 0-padded columns added to the (left, right) of the sequences
        in `X`.
    """
    # 将 pad 参数赋值给 p
    p = pad
    # 如果 p 是整数，则转换为元组
    if isinstance(p, int):
        p = (p, p)

    # 如果 p 是元组
    if isinstance(p, tuple):
        # 对输入 X 进行零填充，沿第二维度填充
        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # 计算 'same' 或 'causal' 卷积的正确填充维度
    # 如果填充方式为"same"或"causal"，并且存在卷积核宽度和步长
    if p in ["same", "causal"] and kernel_width and stride:
        # 判断是否为因果卷积
        causal = p == "causal"
        # 计算填充维度
        p = calc_pad_dims_1D(
            X.shape, X.shape[1], kernel_width, stride, causal=causal, dilation=dilation
        )
        # 对输入数据进行一维填充
        X_pad, p = pad1D(X, p)

    # 返回填充后的数据和填充维度
    return X_pad, p
# 在二维输入体积 `X` 的第二和第三维度上进行零填充

def pad2D(X, pad, kernel_shape=None, stride=None, dilation=0):
    """
    Zero-pad a 4D input volume `X` along the second and third dimensions.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume. Padding is applied to `in_rows` and `in_cols`.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        has the same dimensions as the input.  If 2-tuple, specifies the number
        of padding rows and colums to add *on both sides* of the rows/columns
        in `X`. If 4-tuple, specifies the number of rows/columns to add to the
        top, bottom, left, and right of the input volume.
    kernel_shape : 2-tuple
        The dimension of the 2D convolution kernel. Only relevant if p='same'.
        Default is None.
    stride : int
        The stride for the convolution kernel. Only relevant if p='same'.
        Default is None.
    dilation : int
        The dilation of the convolution kernel. Only relevant if p='same'.
        Default is 0.

    Returns
    -------
    X_pad : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, padded_in_rows, padded_in_cols, in_channels)`
        The padded output volume.
    p : 4-tuple
        The number of 0-padded rows added to the (top, bottom, left, right) of
        `X`.
    """
    p = pad
    # 如果 `p` 是整数，则转换为四元组
    if isinstance(p, int):
        p = (p, p, p, p)

    # 如果 `p` 是元组
    if isinstance(p, tuple):
        # 如果元组长度为2，则扩展为四元组
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        # 对输入体积 `X` 进行零填充
        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # 计算 'same' 卷积的正确填充维度
    # 如果填充方式为"same"且卷积核形状和步长都不为空
    if p == "same" and kernel_shape and stride is not None:
        # 计算二维卷积的填充维度
        p = calc_pad_dims_2D(
            X.shape, X.shape[1:3], kernel_shape, stride, dilation=dilation
        )
        # 对输入数据进行二维填充
        X_pad, p = pad2D(X, p)
    # 返回填充后的输入数据和填充维度
    return X_pad, p
def dilate(X, d):
    """
    Dilate the 4D volume `X` by `d`.

    Notes
    -----
    For a visual depiction of a dilated convolution, see [1].

    References
    ----------
    .. [1] Dumoulin & Visin (2016). "A guide to convolution arithmetic for deep
       learning." https://arxiv.org/pdf/1603.07285v1.pdf

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume.
    d : int
        The number of 0-rows to insert between each adjacent row + column in `X`.

    Returns
    -------
    Xd : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The dilated array where

        .. math::

            \\text{out_rows}  &=  \\text{in_rows} + d(\\text{in_rows} - 1) \\\\
            \\text{out_cols}  &=  \\text{in_cols} + d (\\text{in_cols} - 1)
    """
    # 获取输入体积的形状信息
    n_ex, in_rows, in_cols, n_in = X.shape
    # 生成行索引，重复插入0的行数
    r_ix = np.repeat(np.arange(1, in_rows), d)
    # 生成列索引，重复插入0的列数
    c_ix = np.repeat(np.arange(1, in_cols), d)
    # 在行方向插入0行
    Xd = np.insert(X, r_ix, 0, axis=1)
    # 在列方向插入0列
    Xd = np.insert(Xd, c_ix, 0, axis=2)
    # 返回扩张后的数组
    return Xd


#######################################################################
#                     Convolution Arithmetic                          #
#######################################################################


def calc_fan(weight_shape):
    """
    Compute the fan-in and fan-out for a weight matrix/volume.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be
        `in_ch`, `out_ch`.

    Returns
    -------
    fan_in : int
        The number of input units in the weight tensor
    fan_out : int
        The number of output units in the weight tensor
    """
    # 如果权重矩阵维度为2
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    # 如果权重形状的长度为3或4，则进行以下操作
    elif len(weight_shape) in [3, 4]:
        # 获取输入通道数和输出通道数
        in_ch, out_ch = weight_shape[-2:]
        # 计算卷积核大小
        kernel_size = np.prod(weight_shape[:-2])
        # 计算输入和输出的神经元数量
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    # 如果权重形状的长度不是3或4，则引发值错误异常
    else:
        raise ValueError("Unrecognized weight dimension: {}".format(weight_shape))
    # 返回输入神经元数量和输出神经元数量
    return fan_in, fan_out
# 计算给定卷积的输出体积的维度

def calc_conv_out_dims(X_shape, W_shape, stride=1, pad=0, dilation=0):
    """
    Compute the dimension of the output volume for the specified convolution.

    Parameters
    ----------
    X_shape : 3-tuple or 4-tuple
        The dimensions of the input volume to the convolution. If 3-tuple,
        entries are expected to be (`n_ex`, `in_length`, `in_ch`). If 4-tuple,
        entries are expected to be (`n_ex`, `in_rows`, `in_cols`, `in_ch`).
    weight_shape : 3-tuple or 4-tuple
        The dimensions of the weight volume for the convolution. If 3-tuple,
        entries are expected to be (`f_len`, `in_ch`, `out_ch`). If 4-tuple,
        entries are expected to be (`fr`, `fc`, `in_ch`, `out_ch`).
    pad : tuple, int, or {'same', 'causal'}
        The padding amount. If 'same', add padding to ensure that the output
        length of a 1D convolution with a kernel of `kernel_shape` and stride
        `stride` is the same as the input length.  If 'causal' compute padding
        such that the output both has the same length as the input AND
        ``output[t]`` does not depend on ``input[t + 1:]``. If 2-tuple, specifies the
        number of padding columns to add on each side of the sequence. Default
        is 0.
    stride : int
        The stride for the convolution kernel. Default is 1.
    dilation : int
        The dilation of the convolution kernel. Default is 0.

    Returns
    -------
    out_dims : 3-tuple or 4-tuple
        The dimensions of the output volume. If 3-tuple, entries are (`n_ex`,
        `out_length`, `out_ch`). If 4-tuple, entries are (`n_ex`, `out_rows`,
        `out_cols`, `out_ch`).
    """
    
    # 创建一个与输入形状相同的零矩阵，用于计算卷积输出维度
    dummy = np.zeros(X_shape)
    
    # 将输入参数中的stride、pad、dilation分别赋值给s、p、d
    s, p, d = stride, pad, dilation
    # 如果输入数据的维度为3
    if len(X_shape) == 3:
        # 对输入数据进行一维填充
        _, p = pad1D(dummy, p)
        # 获取填充后的两个维度值
        pw1, pw2 = p
        # 获取权重矩阵的形状信息
        fw, in_ch, out_ch = W_shape
        # 获取输入数据的批量大小、长度和通道数
        n_ex, in_length, in_ch = X_shape

        # 调整有效滤波器大小以考虑膨胀
        _fw = fw * (d + 1) - d
        # 计算输出长度
        out_length = (in_length + pw1 + pw2 - _fw) // s + 1
        # 定义输出维度
        out_dims = (n_ex, out_length, out_ch)

    # 如果输入数据的维度为4
    elif len(X_shape) == 4:
        # 对输入数据进行二维填充
        _, p = pad2D(dummy, p)
        # 获取填充后的四个维度值
        pr1, pr2, pc1, pc2 = p
        # 获取权重矩阵的形状信息
        fr, fc, in_ch, out_ch = W_shape
        # 获取输入数据的批量大小、行数、列数和通道数
        n_ex, in_rows, in_cols, in_ch = X_shape

        # 调整有效滤波器大小以考虑膨胀
        _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d
        # 计算输出行数和列数
        out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
        out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1
        # 定义输出维度
        out_dims = (n_ex, out_rows, out_cols, out_ch)
    else:
        # 抛出异常，表示无法识别的输入维度数量
        raise ValueError("Unrecognized number of input dims: {}".format(len(X_shape)))
    # 返回输出维度
    return out_dims
# 定义一个函数，用于计算在 im2col 函数中列向量化之前的 X 矩阵的索引
def _im2col_indices(X_shape, fr, fc, p, s, d=0):
    """
    Helper function that computes indices into X in prep for columnization in
    :func:`im2col`.

    Code extended from Andrej Karpathy's `im2col.py`
    """
    pr1, pr2, pc1, pc2 = p
    n_ex, n_in, in_rows, in_cols = X_shape

    # 调整有效滤波器大小以考虑膨胀
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    out_rows = (in_rows + pr1 + pr2 - _fr) // s + 1
    out_cols = (in_cols + pc1 + pc2 - _fc) // s + 1

    if any([out_rows <= 0, out_cols <= 0]):
        raise ValueError(
            "Dimension mismatch during convolution: "
            "out_rows = {}, out_cols = {}".format(out_rows, out_cols)
        )

    # i1/j1 : row/col templates
    # i0/j0 : n. copies (len) and offsets (values) for row/col templates
    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, n_in) * (d + 1)
    i1 = s * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * n_in) * (d + 1)
    j1 = s * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (fr * fc * n_in, out_height * out_width)
    # j.shape = (fr * fc * n_in, out_height * out_width)
    # k.shape = (fr * fc * n_in, 1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(n_in), fr * fc).reshape(-1, 1)
    return k, i, j
    # 输入体积（未填充）的 numpy 数组，形状为 `(n_ex, in_rows, in_cols, in_ch)`
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume (not padded).
    # 包含 `(kernel_rows, kernel_cols, in_ch, out_ch)` 的 4 元组
    W_shape: 4-tuple containing `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The dimensions of the weights/kernels in the present convolutional
        layer.
    # 填充量。如果为 'same'，则添加填充以确保使用 `kernel_shape` 和 `stride` 进行 2D 卷积的输出体积与输入体积具有相同的维度。
    # 如果为 2 元组，则指定要在 X 的行和列的两侧添加的填充行数和列数。如果为 4 元组，则指定要添加到输入体积的顶部、底部、左侧和右侧的行数/列数。
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in X. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    # 每个卷积核的步幅
    stride : int
        The stride of each convolution kernel
    # 插入在卷积核元素之间的像素数。默认为 0。
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    # 重塑后的输入体积，其中：

    # Q = kernel_rows * kernel_cols * n_in
    # Z = n_ex * out_rows * out_cols
    X_col : :py:class:`ndarray <numpy.ndarray>` of shape (Q, Z)
        The reshaped input volume where where:

        .. math::

            Q  &=  \\text{kernel_rows} \\times \\text{kernel_cols} \\times \\text{n_in} \\\\
            Z  &=  \\text{n_ex} \\times \\text{out_rows} \\times \\text{out_cols}
    """
    # 解包 W_shape
    fr, fc, n_in, n_out = W_shape
    # 解包步幅、填充、扩张
    s, p, d = stride, pad, dilation
    # 解包 X 的形状
    n_ex, in_rows, in_cols, n_in = X.shape

    # 对输入进行零填充
    X_pad, p = pad2D(X, p, W_shape[:2], stride=s, dilation=d)
    pr1, pr2, pc1, pc2 = p

    # 重新排列以使通道成为第一维
    X_pad = X_pad.transpose(0, 3, 1, 2)

    # 获取 im2col 的索引
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, p, s, d)

    X_col = X_pad[:, k, i, j]
    X_col = X_col.transpose(1, 2, 0).reshape(fr * fc * n_in, -1)
    return X_col, p
def col2im(X_col, X_shape, W_shape, pad, stride, dilation=0):
    """
    Take columns of a 2D matrix and rearrange them into the blocks/windows of
    a 4D image volume.

    Notes
    -----
    A NumPy reimagining of MATLAB's ``col2im`` 'sliding' function.

    Code extended from Andrej Karpathy's ``im2col.py``.

    Parameters
    ----------
    X_col : :py:class:`ndarray <numpy.ndarray>` of shape `(Q, Z)`
        The columnized version of `X` (assumed to include padding)
    X_shape : 4-tuple containing `(n_ex, in_rows, in_cols, in_ch)`
        The original dimensions of `X` (not including padding)
    W_shape: 4-tuple containing `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The dimensions of the weights in the present convolutional layer
    pad : 4-tuple of `(left, right, up, down)`
        Number of zero-padding rows/cols to add to `X`
    stride : int
        The stride of each convolution kernel
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    img : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        The reshaped `X_col` input matrix
    """
    # 检查 pad 是否为 4 元组
    if not (isinstance(pad, tuple) and len(pad) == 4):
        raise TypeError("pad must be a 4-tuple, but got: {}".format(pad))

    # 获取 stride 和 dilation
    s, d = stride, dilation
    pr1, pr2, pc1, pc2 = pad
    fr, fc, n_in, n_out = W_shape
    n_ex, in_rows, in_cols, n_in = X_shape

    # 创建一个填充后的 X 矩阵
    X_pad = np.zeros((n_ex, n_in, in_rows + pr1 + pr2, in_cols + pc1 + pc2))
    # 获取 im2col 的索引
    k, i, j = _im2col_indices((n_ex, n_in, in_rows, in_cols), fr, fc, pad, s, d)

    # 重塑 X_col
    X_col_reshaped = X_col.reshape(n_in * fr * fc, -1, n_ex)
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)

    # 在 X_pad 上进行加法操作
    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    # 更新 pr2 和 pc2
    pr2 = None if pr2 == 0 else -pr2
    pc2 = None if pc2 == 0 else -pc2
    # 返回填充后的 X 矩阵
    return X_pad[:, :, pr1:pr2, pc1:pc2]
#                             Convolution                             #
#######################################################################

# 定义一个二维卷积函数，用于计算输入 `X` 与一组卷积核 `W` 的卷积（实际上是互相关）
def conv2D(X, W, stride, pad, dilation=0):
    """
    A faster (but more memory intensive) implementation of the 2D "convolution"
    (technically, cross-correlation) of input `X` with a collection of kernels in
    `W`.

    Notes
    -----
    Relies on the :func:`im2col` function to perform the convolution as a single
    matrix multiplication.

    For a helpful diagram, see Pete Warden's 2015 blogpost [1].

    References
    ----------
    .. [1] Warden (2015). "Why GEMM is at the heart of deep learning,"
       https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume (unpadded).
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_rows, kernel_cols, in_ch, out_ch)`
        A volume of convolution weights/kernels for a given layer.
    stride : int
        The stride of each convolution kernel.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in `X`. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Z : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The covolution of `X` with `W`.
    """
    # 将步长和膨胀设为变量s和d
    s, d = stride, dilation
    # 调用pad2D函数计算二维卷积的填充情况，返回填充后的输入和输出尺寸
    _, p = pad2D(X, pad, W.shape[:2], s, dilation=dilation)
    # 将输入的参数解包为四个变量：pr1, pr2, pc1, pc2
    pr1, pr2, pc1, pc2 = p
    # 解包卷积核的形状信息：fr, fc, in_ch, out_ch
    fr, fc, in_ch, out_ch = W.shape
    # 解包输入数据的形状信息：n_ex, in_rows, in_cols, in_ch
    n_ex, in_rows, in_cols, in_ch = X.shape

    # 根据膨胀因子更新有效滤波器形状
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    # 计算卷积输出的维度
    out_rows = int((in_rows + pr1 + pr2 - _fr) / s + 1)
    out_cols = int((in_cols + pc1 + pc2 - _fc) / s + 1)

    # 将输入数据和卷积核转换为适当的二维矩阵，并计算它们的乘积
    X_col, _ = im2col(X, W.shape, p, s, d)
    W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)

    # 计算卷积结果并重塑为指定形状
    Z = (W_col @ X_col).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

    # 返回卷积结果
    return Z
# 定义一个一维卷积函数，用于对输入 X 和卷积核集合 W 进行卷积操作
def conv1D(X, W, stride, pad, dilation=0):
    """
    A faster (but more memory intensive) implementation of a 1D "convolution"
    (technically, cross-correlation) of input `X` with a collection of kernels in
    `W`.

    Notes
    -----
    Relies on the :func:`im2col` function to perform the convolution as a single
    matrix multiplication.

    For a helpful diagram, see Pete Warden's 2015 blogpost [1].

    References
    ----------
    .. [1] Warden (2015). "Why GEMM is at the heart of deep learning,"
       https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
        Input volume (unpadded)
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_width, in_ch, out_ch)`
        A volume of convolution weights/kernels for a given layer
    stride : int
        The stride of each convolution kernel
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 1D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding colums to add *on both sides*
        of the columns in X.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Z : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_out, out_ch)`
        The convolution of X with W.
    """
    # 计算需要添加的填充数量
    _, p = pad1D(X, pad, W.shape[0], stride, dilation=dilation)

    # 为 X 添加一个行维度，以便使用 im2col/col2im
    X2D = np.expand_dims(X, axis=1)
    W2D = np.expand_dims(W, axis=0)
    p2D = (0, 0, p[0], p[1])
    # 调用二维卷积函数 conv2D 进行卷积操作
    Z2D = conv2D(X2D, W2D, stride, p2D, dilation)

    # 去除行维度，返回结果
    return np.squeeze(Z2D, axis=1)


def deconv2D_naive(X, W, stride, pad, dilation=0):
    """
    # 对输入体积 `X` 进行“反卷积”（更准确地说是转置卷积），考虑步长、填充和膨胀
    # 注意
    # 与使用卷积矩阵的转置不同，这种方法使用直接卷积并进行零填充，概念上简单直观，但计算效率低下
    # 详细解释请参见 [1]
    # 参考资料
    # [1] Dumoulin & Visin (2016). "A guide to convolution arithmetic for deep learning." https://arxiv.org/pdf/1603.07285v1.pdf
    # 参数
    # X : 形状为 `(n_ex, in_rows, in_cols, in_ch)` 的 :py:class:`ndarray <numpy.ndarray>`
    #     输入体积（未填充）
    # W: 形状为 `(kernel_rows, kernel_cols, in_ch, out_ch)` 的 :py:class:`ndarray <numpy.ndarray>`
    #     给定层的卷积权重/卷积核体积
    # stride : int
    #     每个卷积核的步长
    # pad : tuple, int, 或 'same'
    #     填充量。如果为 'same'，则添加填充以确保具有 `kernel_shape` 和步长 `stride` 的 2D 卷积的输出体积与输入体积具有相同的维度。
    #     如果为 2-元组，则指定要在 `X` 的行和列的 *两侧* 添加的填充行和列数。如果为 4-元组，则指定要添加到输入体积的顶部、底部、左侧和右侧的行/列数。
    # dilation : int
    #     插入在卷积核元素之间的像素数。默认为 0
    # 返回
    # Y : 形状为 `(n_ex, out_rows, out_cols, n_out)` 的 :py:class:`ndarray <numpy.ndarray>`
    #     使用步长 `s` 和膨胀 `d` 对（填充的）输入体积 `X` 与 `W` 进行反卷积后的结果
    """
    # 如果步长大于 1
    if stride > 1:
        # 对 X 进行膨胀操作，膨胀次数为 stride - 1
        X = dilate(X, stride - 1)
        # 将步长设为 1
        stride = 1

    # 对输入进行填充
    # X_pad 为填充后的输入，p 为填充的参数
    X_pad, p = pad2D(X, pad, W.shape[:2], stride=stride, dilation=dilation)
    # 获取输入张量 X_pad 的扩展维度、行数、列数和通道数
    n_ex, in_rows, in_cols, n_in = X_pad.shape
    # 获取权重矩阵 W 的滤波器行数、列数、输入通道数和输出通道数
    fr, fc, n_in, n_out = W.shape
    # 设置步长 s 和膨胀因子 d
    s, d = stride, dilation
    # 设置填充参数 pr1, pr2, pc1, pc2
    pr1, pr2, pc1, pc2 = p

    # 根据膨胀因子更新有效滤波器形状
    _fr, _fc = fr * (d + 1) - d, fc * (d + 1) - d

    # 计算反卷积输出维度
    out_rows = s * (in_rows - 1) - pr1 - pr2 + _fr
    out_cols = s * (in_cols - 1) - pc1 - pc2 + _fc
    out_dim = (out_rows, out_cols)

    # 添加额外填充以达到目标输出维度
    _p = calc_pad_dims_2D(X_pad.shape, out_dim, W.shape[:2], s, d)
    X_pad, pad = pad2D(X_pad, _p, W.shape[:2], stride=s, dilation=dilation)

    # 使用翻转的权重矩阵执行前向卷积（注意设置 pad 为 0，因为我们已经添加了填充）
    Z = conv2D(X_pad, np.rot90(W, 2), s, 0, d)

    # 如果 pr2 为 0，则将其设置为 None；否则将其设置为 -pr2
    pr2 = None if pr2 == 0 else -pr2
    # 如果 pc2 为 0，则将其设置为 None；否则将其设置为 -pc2
    pc2 = None if pc2 == 0 else -pc2
    # 返回 Z 的切片，根据 pr1, pr2, pc1, pc2 进行切片
    return Z[:, pr1:pr2, pc1:pc2, :]
# 定义一个使用朴素方法实现的二维“卷积”（实际上是交叉相关）的函数
def conv2D_naive(X, W, stride, pad, dilation=0):
    """
    A slow but more straightforward implementation of a 2D "convolution"
    (technically, cross-correlation) of input `X` with a collection of kernels `W`.

    Notes
    -----
    This implementation uses ``for`` loops and direct indexing to perform the
    convolution. As a result, it is slower than the vectorized :func:`conv2D`
    function that relies on the :func:`col2im` and :func:`im2col`
    transformations.

    Parameters
    ----------
    X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
        Input volume.
    W: :py:class:`ndarray <numpy.ndarray>` of shape `(kernel_rows, kernel_cols, in_ch, out_ch)`
        The volume of convolution weights/kernels.
    stride : int
        The stride of each convolution kernel.
    pad : tuple, int, or 'same'
        The padding amount. If 'same', add padding to ensure that the output of
        a 2D convolution with a kernel of `kernel_shape` and stride `stride`
        produces an output volume of the same dimensions as the input.  If
        2-tuple, specifies the number of padding rows and colums to add *on both
        sides* of the rows/columns in `X`. If 4-tuple, specifies the number of
        rows/columns to add to the top, bottom, left, and right of the input
        volume.
    dilation : int
        Number of pixels inserted between kernel elements. Default is 0.

    Returns
    -------
    Z : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        The covolution of `X` with `W`.
    """
    # 将 stride 和 dilation 参数赋值给 s 和 d
    s, d = stride, dilation
    # 对输入 X 进行二维填充，得到填充后的输入 X_pad 和填充元组 p
    X_pad, p = pad2D(X, pad, W.shape[:2], stride=s, dilation=d)

    # 解包填充元组 p，得到填充的行数和列数
    pr1, pr2, pc1, pc2 = p
    # 获取卷积核的形状信息
    fr, fc, in_ch, out_ch = W.shape
    # 获取输入 X 的形状信息
    n_ex, in_rows, in_cols, in_ch = X.shape

    # 根据膨胀因子更新有效滤波器形状
    fr, fc = fr * (d + 1) - d, fc * (d + 1) - d

    # 计算输出的行数
    out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
    # 计算输出列数
    out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

    # 创建一个全零数组，用于存储卷积操作的结果
    Z = np.zeros((n_ex, out_rows, out_cols, out_ch))
    # 遍历每个样本
    for m in range(n_ex):
        # 遍历输出通道
        for c in range(out_ch):
            # 遍历输出行
            for i in range(out_rows):
                # 遍历输出列
                for j in range(out_cols):
                    # 计算窗口的起始和结束位置
                    i0, i1 = i * s, (i * s) + fr
                    j0, j1 = j * s, (j * s) + fc

                    # 从输入数据中提取窗口数据
                    window = X_pad[m, i0 : i1 : (d + 1), j0 : j1 : (d + 1), :]
                    # 执行卷积操作并将结果存储到输出数组中
                    Z[m, i, j, c] = np.sum(window * W[:, :, :, c])
    # 返回卷积操作的结果数组
    return Z
# 权重初始化函数：He uniform 初始化策略
def he_uniform(weight_shape):
    """
    Initializes network weights `W` with using the He uniform initialization
    strategy.

    Notes
    -----
    The He uniform initializations trategy initializes the weights in `W` using
    draws from Uniform(-b, b) where

    .. math::

        b = \sqrt{\\frac{6}{\\text{fan_in}}}

    Developed for deep networks with ReLU nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    # 计算输入和输出的神经元数量
    fan_in, fan_out = calc_fan(weight_shape)
    # 计算 b 值
    b = np.sqrt(6 / fan_in)
    # 从 Uniform(-b, b) 中随机初始化权重
    return np.random.uniform(-b, b, size=weight_shape)


# 权重初始化函数：He normal 初始化策略
def he_normal(weight_shape):
    """
    Initialize network weights `W` using the He normal initialization strategy.

    Notes
    -----
    The He normal initialization strategy initializes the weights in `W` using
    draws from TruncatedNormal(0, b) where the variance `b` is

    .. math::

        b = \\frac{2}{\\text{fan_in}}

    He normal initialization was originally developed for deep networks with
    :class:`~numpy_ml.neural_nets.activations.ReLU` nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    # 计算输入和输出的神经元数量
    fan_in, fan_out = calc_fan(weight_shape)
    # 计算标准差
    std = np.sqrt(2 / fan_in)
    # 从 TruncatedNormal(0, b) 中随机初始化权重
    return truncated_normal(0, std, weight_shape)


# 权重初始化函数：Glorot uniform 初始化策略
def glorot_uniform(weight_shape, gain=1.0):
    """
    Initialize network weights `W` using the Glorot uniform initialization
    strategy.

    Notes
    -----
    The Glorot uniform initialization strategy initializes weights using draws
    from ``Uniform(-b, b)`` where:

    .. math::

        b = \\text{gain} \sqrt{\\frac{6}{\\text{fan_in} + \\text{fan_out}}}

    The motivation for Glorot uniform initialization is to choose weights to
    ensure that the variance of the layer outputs are approximately equal to
    the variance of its inputs.

    This initialization strategy was primarily developed for deep networks with
    tanh and logistic sigmoid nonlinearities.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume.

    Returns
    -------
    W : :py:class:`ndarray <numpy.ndarray>` of shape `weight_shape`
        The initialized weights.
    """
    # 计算权重矩阵/体积的维度
    fan_in, fan_out = calc_fan(weight_shape)
    # 计算 b 的值
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    # 返回从 Uniform(-b, b) 中随机初始化的权重
    return np.random.uniform(-b, b, size=weight_shape)
# 使用 Glorot 正态初始化策略初始化网络权重 `W`
def glorot_normal(weight_shape, gain=1.0):
    # 计算权重矩阵/体的维度
    fan_in, fan_out = calc_fan(weight_shape)
    # 计算标准差
    std = gain * np.sqrt(2 / (fan_in + fan_out))
    # 返回从截断正态分布中抽取的值
    return truncated_normal(0, std, weight_shape)

# 通过拒绝抽样生成从截断正态分布中抽取的值
def truncated_normal(mean, std, out_shape):
    # 拒绝抽样方案从具有均值 `mean` 和标准差 `std` 的正态分布中抽取样本，并重新抽样任何值超过 `mean` 两个标准差的值
    pass
    # 从参数为 `mean` 和 `std` 的截断正态分布中抽取形状为 `out_shape` 的样本
    samples = np.random.normal(loc=mean, scale=std, size=out_shape)
    # 创建一个逻辑数组，标记样本是否超出均值加减两倍标准差的范围
    reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    # 当仍有样本超出范围时，重新抽取这些样本
    while any(reject.flatten()):
        # 重新从参数为 `mean` 和 `std` 的正态分布中抽取超出范围的样本数量的样本
        resamples = np.random.normal(loc=mean, scale=std, size=reject.sum())
        # 将重新抽取的样本替换原来超出范围的样本
        samples[reject] = resamples
        # 更新标记数组，检查是否仍有样本超出范围
        reject = np.logical_or(samples >= mean + 2 * std, samples <= mean - 2 * std)
    # 返回处理后的样本数组
    return samples
```