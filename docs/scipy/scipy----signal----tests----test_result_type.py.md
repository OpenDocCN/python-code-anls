# `D:\src\scipysrc\scipy\scipy\signal\tests\test_result_type.py`

```
# 导入必要的库和模块
import numpy as np
from numpy.testing import assert_

# 导入信号处理相关函数
from scipy.signal import (decimate,      # 信号下采样函数
                          lfilter_zi,    # IIR 滤波器初始状态函数
                          lfiltic,       # FIR 滤波器初始状态函数
                          sos2tf,        # SOS（second-order sections）转换为传递函数形式函数
                          sosfilt_zi     # SOS 滤波器初始状态函数
                          )

# 测试信号下采样函数
def test_decimate():
    # 创建一个长度为 32 的浮点型全 1 数组
    ones_f32 = np.ones(32, dtype=np.float32)
    # 验证下采样后数组的数据类型为浮点型
    assert_(decimate(ones_f32, 2).dtype == np.float32)

    # 创建一个长度为 32 的整型全 1 数组
    ones_i64 = np.ones(32, dtype=np.int64)
    # 验证下采样后数组的数据类型为双精度浮点型
    assert_(decimate(ones_i64, 2).dtype == np.float64)
    

# 测试 IIR 滤波器初始状态函数
def test_lfilter_zi():
    # 设置 IIR 滤波器的分子系数和分母系数，数据类型为浮点型
    b_f32 = np.array([1, 2, 3], dtype=np.float32)
    a_f32 = np.array([4, 5, 6], dtype=np.float32)
    # 验证返回的初始状态的数据类型为浮点型
    assert_(lfilter_zi(b_f32, a_f32).dtype == np.float32)


# 测试 FIR 滤波器初始状态函数
def test_lfiltic():
    # 设置 FIR 滤波器的分子系数和分母系数，数据类型为浮点型
    b_f32 = np.array([1, 2, 3], dtype=np.float32)
    a_f32 = np.array([4, 5, 6], dtype=np.float32)
    # 创建一个长度为 32 的浮点型全 1 数组
    x_f32 = np.ones(32, dtype=np.float32)
    
    # 将系数数组转换为双精度浮点型
    b_f64 = b_f32.astype(np.float64)
    a_f64 = a_f32.astype(np.float64)
    x_f64 = x_f32.astype(np.float64)

    # 验证返回的初始状态的数据类型为双精度浮点型
    assert_(lfiltic(b_f64, a_f32, x_f32).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f64, x_f32).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f32, x_f64).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f32, x_f32, x_f64).dtype == np.float64)


# 测试 SOS 转换为传递函数形式函数
def test_sos2tf():
    # 设置 SOS 系数数组，数据类型为浮点型
    sos_f32 = np.array([[4, 5, 6, 1, 2, 3]], dtype=np.float32)
    # 调用函数计算传递函数形式的分子系数和分母系数
    b, a = sos2tf(sos_f32)
    # 验证返回的分子系数和分母系数的数据类型为浮点型
    assert_(b.dtype == np.float32)
    assert_(a.dtype == np.float32)


# 测试 SOS 滤波器初始状态函数
def test_sosfilt_zi():
    # 设置 SOS 系数数组，数据类型为浮点型
    sos_f32 = np.array([[4, 5, 6, 1, 2, 3]], dtype=np.float32)
    # 验证返回的初始状态的数据类型为浮点型
    assert_(sosfilt_zi(sos_f32).dtype == np.float32)
```