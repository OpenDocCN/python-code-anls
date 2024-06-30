# `D:\src\scipysrc\scipy\scipy\signal\_max_len_seq.py`

```
# 导入 numpy 库，简称为 np，用于数值计算
import numpy as np

# 从模块 _max_len_seq_inner 中导入函数 _max_len_seq_inner
from ._max_len_seq_inner import _max_len_seq_inner

# 定义 __all__ 变量，用于模块的导入控制，只导出 max_len_seq 函数
__all__ = ['max_len_seq']

# 这是一个预定义的字典，包含了不同位数的线性移位寄存器的反馈多项式 taps
# 用于 max_len_seq 函数中产生最大长度序列
_mls_taps = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
             9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
             14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
             18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
             23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
             27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
             31: [28], 32: [31, 30, 10]}

def max_len_seq(nbits, state=None, length=None, taps=None):
    """
    Maximum length sequence (MLS) generator.

    Parameters
    ----------
    nbits : int
        使用的比特数。生成的序列长度为 ``(2**nbits) - 1``。生成长序列（例如 nbits > 16）可能需要很长时间。
    state : array_like, optional
        如果是数组，必须是长度为 ``nbits`` 的二进制（bool）表示。如果为 None，将使用全为1的种子，产生可重复的表示。
        如果 ``state`` 全为零，则会引发错误，因为这是无效的。默认值为 None。
    length : int, optional
        要计算的样本数。如果为 None，则计算整个长度 ``(2**nbits) - 1``。
    taps : array_like, optional
        要使用的多项式反馈 taps（例如 ``[7, 6, 1]`` 用于 8 位序列）。如果为 None，则将自动选择 taps（最多为 ``nbits == 32``）。

    Returns
    -------
    seq : array
        结果的 MLS 序列，由 0 和 1 组成。
    state : array
        移位寄存器的最终状态。

    Notes
    -----
    MLS 生成的算法通常描述在：

        https://en.wikipedia.org/wiki/Maximum_length_sequence

    taps 的默认值特别来自于以下网站中每个 ``nbits`` 的第一个选项：

        https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm

    .. versionadded:: 0.15.0

    Examples
    --------
    MLS 使用二进制约定：

    >>> from scipy.signal import max_len_seq
    >>> max_len_seq(4)[0]
    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=int8)

    MLS 具有白色频谱（除了直流分量）：

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from numpy.fft import fft, ifft, fftshift, fftfreq
    >>> seq = max_len_seq(6)[0]*2-1  # +1 和 -1
    >>> spec = fft(seq)
    >>> N = len(seq)
    >>> plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    MLS 的循环自相关是一个冲激函数：

    """
    # 这里是函数的具体实现，根据参数生成最大长度序列
    pass  # pass 用于占位，表示函数体暂未实现，具体逻辑需要根据文档进行编写
    # 计算给定频谱的逆傅里叶变换，并取实部得到自相关结果
    acorrcirc = ifft(spec * np.conj(spec)).real
    
    # 绘制图形，展示线性最大长度序列（MLS）的自相关近似脉冲响应
    plt.figure()
    
    # 绘制自相关结果的图像，使用fftshift对结果进行中心化处理
    plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
    
    # 设置图像的边距
    plt.margins(0.1, 0.1)
    
    # 添加网格线
    plt.grid(True)
    
    # 显示图形
    plt.show()

    # 计算序列seq的全序列自相关
    acorr = np.correlate(seq, seq, 'full')
    
    # 绘制图形，展示序列的全序列自相关
    plt.figure()
    
    # 绘制全序列自相关的图像
    plt.plot(np.arange(-N+1, N), acorr, '.-')
    
    # 设置图像的边距
    plt.margins(0.1, 0.1)
    
    # 添加网格线
    plt.grid(True)
    
    # 显示图形
    plt.show()

    """
    # 确定taps数组的数据类型，根据系统的指针大小选择np.int32或np.int64
    taps_dtype = np.int32 if np.intp().itemsize == 4 else np.int64
    
    # 如果taps为None，则选择与nbits对应的预设值
    if taps is None:
        if nbits not in _mls_taps:
            # 如果nbits不在已知的MLS taps范围内，则抛出错误
            known_taps = np.array(list(_mls_taps.keys()))
            raise ValueError(f'nbits must be between {known_taps.min()} and '
                             f'{known_taps.max()} if taps is None')
        # 从预设值中选择taps数组
        taps = np.array(_mls_taps[nbits], taps_dtype)
    else:
        # 将taps转换为唯一值的数组，并反转顺序
        taps = np.unique(np.array(taps, taps_dtype))[::-1]
        # 检查taps数组的有效性，必须为非负数且在0到nbits之间
        if np.any(taps < 0) or np.any(taps > nbits) or taps.size < 1:
            raise ValueError('taps must be non-empty with values between '
                             'zero and nbits (inclusive)')
        taps = np.array(taps)  # 在Cython和Pythran中需要这一步

    # 最大可能的状态值
    n_max = (2**nbits) - 1
    
    # 如果未指定length，则使用n_max作为默认长度
    if length is None:
        length = n_max
    else:
        length = int(length)
        # 长度必须为非负数
        if length < 0:
            raise ValueError('length must be greater than or equal to 0')

    # 如果未指定state，则创建一个全为1的状态数组
    if state is None:
        state = np.ones(nbits, dtype=np.int8, order='c')
    else:
        # 将state转换为布尔值数组，并确保是1和0
        state = np.array(state, dtype=bool, order='c').astype(np.int8)
    
    # 检查state的维度和大小是否符合要求
    if state.ndim != 1 or state.size != nbits:
        raise ValueError('state must be a 1-D array of size nbits')
    
    # 确保state数组不全为0
    if np.all(state == 0):
        raise ValueError('state must not be all zeros')

    # 创建一个空的序列seq，用于存储最大长度序列的输出
    seq = np.empty(length, dtype=np.int8, order='c')
    
    # 调用内部函数_max_len_seq_inner生成最大长度序列，并返回更新后的state
    state = _max_len_seq_inner(taps, state, nbits, length, seq)
    
    # 返回生成的序列seq和最终状态state
    return seq, state
```