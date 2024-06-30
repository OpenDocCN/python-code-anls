# `D:\src\scipysrc\scipy\scipy\signal\tests\test_max_len_seq.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose, assert_array_equal  # 导入 NumPy 测试模块中的数组比较函数
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 断言，起别名为 assert_raises

from numpy.fft import fft, ifft  # 导入 NumPy 中的 FFT 相关函数

from scipy.signal import max_len_seq  # 导入 SciPy 中的 max_len_seq 函数，用于生成最大长度序列


class TestMLS:  # 定义测试类 TestMLS

    def test_mls_inputs(self):
        # 测试当状态全部为零时抛出 ValueError 异常
        assert_raises(ValueError, max_len_seq, 10, state=np.zeros(10))
        # 测试状态数组大小不匹配时抛出 ValueError 异常
        assert_raises(ValueError, max_len_seq, 10, state=np.ones(3))
        # 测试长度参数为负数时抛出 ValueError 异常
        assert_raises(ValueError, max_len_seq, 10, length=-1)
        # 测试长度为零时返回空数组
        assert_array_equal(max_len_seq(10, length=0)[0], [])
        # 测试未知的 taps 参数值时抛出 ValueError 异常
        assert_raises(ValueError, max_len_seq, 64)
        # 测试 taps 参数包含非法值时抛出 ValueError 异常
        assert_raises(ValueError, max_len_seq, 10, taps=[-1, 1])
    def test_mls_output(self):
        # 定义一些备用的工作中的 taps
        alt_taps = {2: [1], 3: [2], 4: [3], 5: [4, 3, 2], 6: [5, 4, 1], 7: [4],
                    8: [7, 5, 3]}
        
        # 假设其他比特位级别也可以工作，测试更高阶次的太慢了...
        for nbits in range(2, 8):
            for state in [None, np.round(np.random.rand(nbits))]:
                for taps in [None, alt_taps[nbits]]:
                    if state is not None and np.all(state == 0):
                        state[0] = 1  # 不能全为零
                    # 调用 max_len_seq 函数获取最大长度序列及相关状态
                    orig_m = max_len_seq(nbits, state=state,
                                         taps=taps)[0]
                    # 转换成 +/- 1 表示形式
                    m = 2. * orig_m - 1.
                    
                    # 首先，确保得到的序列全是 1 或者 -1
                    err_msg = "mls had non binary terms"
                    assert_array_equal(np.abs(m), np.ones_like(m),
                                       err_msg=err_msg)
                    
                    # 通过循环交叉相关测试，这相当于在频域中一个信号和其共轭的乘积
                    tester = np.real(ifft(fft(m) * np.conj(fft(m))))
                    out_len = 2**nbits - 1
                    # 冲激响应幅度等于测试长度
                    err_msg = "mls impulse has incorrect value"
                    assert_allclose(tester[0], out_len, err_msg=err_msg)
                    
                    # 稳态为 -1
                    err_msg = "mls steady-state has incorrect value"
                    assert_allclose(tester[1:], np.full(out_len - 1, -1),
                                    err_msg=err_msg)
                    
                    # 使用几种选项进行拆分测试
                    for n in (1, 2**(nbits - 1)):
                        m1, s1 = max_len_seq(nbits, state=state, taps=taps,
                                             length=n)
                        m2, s2 = max_len_seq(nbits, state=s1, taps=taps,
                                             length=1)
                        m3, s3 = max_len_seq(nbits, state=s2, taps=taps,
                                             length=out_len - n - 1)
                        new_m = np.concatenate((m1, m2, m3))
                        # 检查是否与原始序列一致
                        assert_array_equal(orig_m, new_m)
```