# `D:\src\scipysrc\scipy\scipy\signal\_max_len_seq_inner.pyx`

```
# Author: Eric Larson
# 2014

# 导入必要的库
import numpy as np
# 导入 C 版本的 numpy 库
cimport numpy as np
# 导入 Cython 库
cimport cython

# 调用 numpy 库的 import_array 函数
np.import_array()

# 定义一个 Cython 函数，用于实现 max_len_seq 的快速内循环
@cython.cdivision(True)  # faster modulo
@cython.boundscheck(False)  # designed to stay within bounds
@cython.wraparound(False)  # we don't use negative indexing
def _max_len_seq_inner(const Py_ssize_t[::1] taps,
                       np.int8_t[::1] state,
                       Py_ssize_t nbits, Py_ssize_t length,
                       np.int8_t[::1] seq):
    # Here we compute MLS using a shift register, indexed using a ring buffer
    # technique (faster than using something like np.roll to shift)
    # 计算最大长度序列 (MLS) 使用移位寄存器，并使用环形缓冲区索引技术进行索引
    cdef Py_ssize_t n_taps = taps.shape[0]
    cdef Py_ssize_t idx = 0
    cdef np.int8_t feedback
    cdef Py_ssize_t i
    # 循环生成序列
    for i in range(length):
        feedback = state[idx]
        seq[i] = feedback
        # 计算反馈值
        for ti in range(n_taps):
            feedback ^= state[(taps[ti] + idx) % nbits]
        state[idx] = feedback
        idx = (idx + 1) % nbits
    # 将状态数组进行滚动，以便下一次运行时当 idx==0 时，状态处于正确位置
    return np.roll(state, -idx, axis=0)
```