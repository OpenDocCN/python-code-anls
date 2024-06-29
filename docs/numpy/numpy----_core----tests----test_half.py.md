# `.\numpy\numpy\_core\tests\test_half.py`

```
# 导入 platform 模块，用于获取平台信息
import platform
# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 导入 numpy 库，并分别导入其中的 uint16, float16, float32, float64 数据类型
import numpy as np
from numpy import uint16, float16, float32, float64
# 导入 numpy.testing 模块中的 assert_, assert_equal, _OLD_PROMOTION, IS_WASM 方法
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM

# 定义一个函数 assert_raises_fpe，用于验证是否引发浮点异常
def assert_raises_fpe(strmatch, callable, *args, **kwargs):
    try:
        # 调用给定的可调用对象，传入参数
        callable(*args, **kwargs)
    except FloatingPointError as exc:
        # 检查异常消息中是否包含指定的字符串
        assert_(str(exc).find(strmatch) >= 0,
                "Did not raise floating point %s error" % strmatch)
    else:
        # 如果没有引发预期的异常，抛出断言错误
        assert_(False,
                "Did not raise floating point %s error" % strmatch)

# 定义一个测试类 TestHalf
class TestHalf:
    # 在每个测试方法执行前调用，设置测试环境
    def setup_method(self):
        # 创建一个包含所有可能的 float16 值的数组
        self.all_f16 = np.arange(0x10000, dtype=uint16)
        self.all_f16.dtype = float16

        # 如果硬件支持，NaN 值可能会引发无效的浮点异常，这里忽略无效操作错误
        with np.errstate(invalid='ignore'):
            # 创建包含所有可能的 float16 值的数组，并转换为 float32 和 float64 类型
            self.all_f32 = np.array(self.all_f16, dtype=float32)
            self.all_f64 = np.array(self.all_f16, dtype=float64)

        # 创建一个包含所有非 NaN 值的 float16 数组，并按顺序排列
        self.nonan_f16 = np.concatenate(
                                (np.arange(0xfc00, 0x7fff, -1, dtype=uint16),
                                 np.arange(0x0000, 0x7c01, 1, dtype=uint16)))
        self.nonan_f16.dtype = float16
        # 转换非 NaN 值数组为 float32 和 float64 类型
        self.nonan_f32 = np.array(self.nonan_f16, dtype=float32)
        self.nonan_f64 = np.array(self.nonan_f16, dtype=float64)

        # 创建一个包含所有有限 float16 值的数组，并按顺序排列
        self.finite_f16 = self.nonan_f16[1:-1]
        self.finite_f32 = self.nonan_f32[1:-1]
        self.finite_f64 = self.nonan_f64[1:-1]
    def test_half_conversions(self):
        """Checks that all 16-bit values survive conversion
           to/from 32-bit and 64-bit float"""
        # Because the underlying routines preserve the NaN bits, every
        # value is preserved when converting to/from other floats.

        # Convert from float32 back to float16
        with np.errstate(invalid='ignore'):
            # 创建一个包含所有 float32 值的数组 b，将其转换为 float16 类型
            b = np.array(self.all_f32, dtype=float16)
        # 避免由于 Q/SNaNs 的位差而测试 NaNs
        b_nn = b == b
        # 断言转换后的 float16 数组与原始 float16 数组的 uint16 表示相等
        assert_equal(self.all_f16[b_nn].view(dtype=uint16),
                     b[b_nn].view(dtype=uint16))

        # Convert from float64 back to float16
        with np.errstate(invalid='ignore'):
            # 创建一个包含所有 float64 值的数组 b，将其转换为 float16 类型
            b = np.array(self.all_f64, dtype=float16)
        # 避免由于 Q/SNaNs 的位差而测试 NaNs
        b_nn = b == b
        # 断言转换后的 float16 数组与原始 float16 数组的 uint16 表示相等
        assert_equal(self.all_f16[b_nn].view(dtype=uint16),
                     b[b_nn].view(dtype=uint16))

        # Convert float16 to longdouble and back
        # This doesn't necessarily preserve the extra NaN bits,
        # so exclude NaNs.
        # 创建一个不含 NaN 的 float16 数组，并将其转换为 longdouble 类型的数组 a_ld
        a_ld = np.array(self.nonan_f16, dtype=np.longdouble)
        # 将 longdouble 数组 a_ld 转换回 float16 类型的数组 b
        b = np.array(a_ld, dtype=float16)
        # 断言转换后的 float16 数组与原始 float16 数组的 uint16 表示相等
        assert_equal(self.nonan_f16.view(dtype=uint16),
                     b.view(dtype=uint16))

        # Check the range for which all integers can be represented
        # 创建一个包含所有整数的 int 数组 i_int，将其转换为 float16 类型的数组 i_f16
        i_int = np.arange(-2048, 2049)
        i_f16 = np.array(i_int, dtype=float16)
        # 将 float16 数组 i_f16 转换回 int 类型的数组 j
        j = np.array(i_f16, dtype=int)
        # 断言转换后的 int 数组 j 与原始 int 数组 i_int 相等
        assert_equal(i_int, j)
    # 定义一个测试方法，用于测试半精度浮点数转换的四舍五入行为
    def test_half_conversion_rounding(self, float_t, shift, offset):
        # 假设在转换过程中使用了四舍六入五成双的方式
        max_pattern = np.float16(np.finfo(np.float16).max).view(np.uint16)

        # 测试所有（正数）有限数字，denormalized（非正规化）情况最为有趣
        f16s_patterns = np.arange(0, max_pattern+1, dtype=np.uint16)
        f16s_float = f16s_patterns.view(np.float16).astype(float_t)

        # 将值向上或向下移动半个比特位（或者不移动）
        if shift == "up":
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[1:]
        elif shift == "down":
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[:-1]
        else:
            f16s_float = f16s_float[1:-1]

        # 将浮点数增加一个最小值：
        if offset == "up":
            f16s_float = np.nextafter(f16s_float, float_t(np.inf))
        elif offset == "down":
            f16s_float = np.nextafter(f16s_float, float_t(-np.inf))

        # 转换回 float16 并获取其比特模式：
        res_patterns = f16s_float.astype(np.float16).view(np.uint16)

        # 上述计算尝试使用原始值或 float16 值之间的精确中点。
        # 然后进一步以尽可能小的偏移量对它们进行偏移。
        # 如果没有偏移发生，将需要“四舍六入五成双”的逻辑，一个任意小的偏移应该始终导致正常的上下舍入。

        # 计算预期模式：
        cmp_patterns = f16s_patterns[1:-1].copy()

        if shift == "down" and offset != "up":
            shift_pattern = -1
        elif shift == "up" and offset != "down":
            shift_pattern = 1
        else:
            # 不能进行偏移，要么偏移为 None，因此所有舍入将回到原始状态，
            # 要么偏移过大，也会减少偏移量。
            shift_pattern = 0

        # 如果发生舍入，是正常舍入还是“四舍六入五成双”？
        if offset is None:
            # 发生四舍六入五成双，仅修改非偶数，转换为允许 + (-1)
            cmp_patterns[0::2].view(np.int16)[...] += shift_pattern
        else:
            cmp_patterns.view(np.int16)[...] += shift_pattern

        # 断言结果模式与预期模式相等
        assert_equal(res_patterns, cmp_patterns)

    # 使用 pytest 的参数化装饰器，定义了多个参数组合进行测试
    @pytest.mark.parametrize(["float_t", "uint_t", "bits"],
                             [(np.float32, np.uint32, 23),
                              (np.float64, np.uint64, 52)])
    # 测试半精度浮点数到整数和回合偶数的转换
    def test_half_conversion_denormal_round_even(self, float_t, uint_t, bits):
        # 测试在决定是否进行偶数舍入时是否考虑所有位（即末位不丢失）。
        # 参见 gh-12721。最小的非规格化数可能会丢失最多位：
        smallest_value = np.uint16(1).view(np.float16).astype(float_t)
        assert smallest_value == 2**-24

        # 根据偶数舍入规则将会舍入为零：
        rounded_to_zero = smallest_value / float_t(2)
        assert rounded_to_zero.astype(np.float16) == 0

        # 对于 float_t 来说，尾数将全部为 0，测试不应丢失这些较低位：
        for i in range(bits):
            # 稍微增加值应使其向上舍入：
            larger_pattern = rounded_to_zero.view(uint_t) | uint_t(1 << i)
            larger_value = larger_pattern.view(float_t)
            assert larger_value.astype(np.float16) == smallest_value

    # 测试 NaN 和 Inf
    def test_nans_infs(self):
        with np.errstate(all='ignore'):
            # 检查一些 ufuncs
            assert_equal(np.isnan(self.all_f16), np.isnan(self.all_f32))
            assert_equal(np.isinf(self.all_f16), np.isinf(self.all_f32))
            assert_equal(np.isfinite(self.all_f16), np.isfinite(self.all_f32))
            assert_equal(np.signbit(self.all_f16), np.signbit(self.all_f32))
            assert_equal(np.spacing(float16(65504)), np.inf)

            # 检查所有值与 NaN 的比较
            nan = float16(np.nan)

            assert_(not (self.all_f16 == nan).any())
            assert_(not (nan == self.all_f16).any())

            assert_((self.all_f16 != nan).all())
            assert_((nan != self.all_f16).all())

            assert_(not (self.all_f16 < nan).any())
            assert_(not (nan < self.all_f16).any())

            assert_(not (self.all_f16 <= nan).any())
            assert_(not (nan <= self.all_f16).any())

            assert_(not (self.all_f16 > nan).any())
            assert_(not (nan > self.all_f16).any())

            assert_(not (self.all_f16 >= nan).any())
            assert_(not (nan >= self.all_f16).any())
    def test_half_values(self):
        """Confirms a small number of known half values"""
        # 创建包含一些已知半精度浮点数的 NumPy 数组
        a = np.array([1.0, -1.0,
                      2.0, -2.0,
                      0.0999755859375, 0.333251953125,  # 1/10, 1/3
                      65504, -65504,           # Maximum magnitude
                      2.0**(-14), -2.0**(-14),  # Minimum normal
                      2.0**(-24), -2.0**(-24),  # Minimum subnormal
                      0, -1/1e1000,            # Signed zeros
                      np.inf, -np.inf])
        # 创建用于存储期望半精度浮点数表示的 NumPy 数组，并将其作为 uint16 解释
        b = np.array([0x3c00, 0xbc00,
                      0x4000, 0xc000,
                      0x2e66, 0x3555,
                      0x7bff, 0xfbff,
                      0x0400, 0x8400,
                      0x0001, 0x8001,
                      0x0000, 0x8000,
                      0x7c00, 0xfc00], dtype=np.uint16)
        # 将 b 数组的数据类型更改为 float16
        b.dtype = np.float16
        # 使用 assert_equal 断言两个数组 a 和 b 相等
        assert_equal(a, b)

    def test_half_rounding(self):
        """Checks that rounding when converting to half is correct"""
        # 创建包含各种浮点数的 NumPy 数组，用于测试半精度浮点数转换时的舍入行为
        a = np.array([2.0**-25 + 2.0**-35,  # Rounds to minimum subnormal
                      2.0**-25,       # Underflows to zero (nearest even mode)
                      2.0**-26,       # Underflows to zero
                      1.0+2.0**-11 + 2.0**-16,  # rounds to 1.0+2**(-10)
                      1.0+2.0**-11,   # rounds to 1.0 (nearest even mode)
                      1.0+2.0**-12,   # rounds to 1.0
                      65519,          # rounds to 65504
                      65520],         # rounds to inf
                      dtype=np.float64)
        # 期望的半精度浮点数舍入结果
        rounded = [2.0**-24,
                   0.0,
                   0.0,
                   1.0+2.0**(-10),
                   1.0,
                   1.0,
                   65504,
                   np.inf]

        # 检查从 float64 到 float16 的舍入是否正确
        with np.errstate(over="ignore"):
            # 使用 float16 类型将 a 数组转换为 b 数组
            b = np.array(a, dtype=np.float16)
        # 使用 assert_equal 断言转换后的数组 b 与期望的舍入结果 rounded 相等
        assert_equal(b, rounded)

        # 检查从 float32 到 float16 的舍入是否正确
        a = np.array(a, dtype=np.float32)
        with np.errstate(over="ignore"):
            # 使用 float16 类型将 a 数组转换为 b 数组
            b = np.array(a, dtype=np.float16)
        # 使用 assert_equal 断言转换后的数组 b 与期望的舍入结果 rounded 相等
        assert_equal(b, rounded)
    def test_half_correctness(self):
        """Take every finite float16, and check the casting functions with
           a manual conversion."""

        # Create an array of all finite float16s
        a_bits = self.finite_f16.view(dtype=uint16)

        # Convert to 64-bit float manually
        # 计算符号位
        a_sgn = (-1.0)**((a_bits & 0x8000) >> 15)
        # 计算指数位
        a_exp = np.array((a_bits & 0x7c00) >> 10, dtype=np.int32) - 15
        # 计算尾数位
        a_man = (a_bits & 0x03ff) * 2.0**(-10)
        # 对于标准化浮点数，隐含位为1
        a_man[a_exp != -15] += 1
        # 对于非标准化浮点数，指数为-14
        a_exp[a_exp == -15] = -14

        # 计算手动转换后的64位浮点数
        a_manual = a_sgn * a_man * 2.0**a_exp

        # 找到第一个不相等的浮点数
        a32_fail = np.nonzero(self.finite_f32 != a_manual)[0]
        if len(a32_fail) != 0:
            bad_index = a32_fail[0]
            # 断言第一个不相等的是半精度浮点数，显示具体数值
            assert_equal(self.finite_f32, a_manual,
                 "First non-equal is half value 0x%x -> %g != %g" %
                            (a_bits[bad_index],
                             self.finite_f32[bad_index],
                             a_manual[bad_index]))

        # 找到第一个不相等的双精度浮点数
        a64_fail = np.nonzero(self.finite_f64 != a_manual)[0]
        if len(a64_fail) != 0:
            bad_index = a64_fail[0]
            # 断言第一个不相等的是半精度浮点数，显示具体数值
            assert_equal(self.finite_f64, a_manual,
                 "First non-equal is half value 0x%x -> %g != %g" %
                            (a_bits[bad_index],
                             self.finite_f64[bad_index],
                             a_manual[bad_index]))

    def test_half_ordering(self):
        """Make sure comparisons are working right"""

        # All non-NaN float16 values in reverse order
        a = self.nonan_f16[::-1].copy()

        # 32-bit float copy
        b = np.array(a, dtype=float32)

        # Should sort the same
        # 对a和b进行排序，期望结果相同
        a.sort()
        b.sort()
        assert_equal(a, b)

        # Comparisons should work
        # 比较运算应该正确工作
        assert_((a[:-1] <= a[1:]).all())
        assert_(not (a[:-1] > a[1:]).any())
        assert_((a[1:] >= a[:-1]).all())
        assert_(not (a[1:] < a[:-1]).any())
        # All != except for +/-0
        # 所有元素应该都不相等，除了+/-0
        assert_equal(np.nonzero(a[:-1] < a[1:])[0].size, a.size-2)
        assert_equal(np.nonzero(a[1:] > a[:-1])[0].size, a.size-2)
    def test_half_funcs(self):
        """Test the various ArrFuncs"""

        # 使用 np.arange 创建一个长度为 10 的 float16 类型的数组，并断言与 dtype 为 float32 的相同
        assert_equal(np.arange(10, dtype=float16),
                     np.arange(10, dtype=float32))

        # 使用 np.zeros 创建长度为 5 的 float16 类型的数组，并将所有元素填充为 1
        a = np.zeros((5,), dtype=float16)
        a.fill(1)
        # 断言数组 a 与长度为 5、元素全部为 float16 类型的数组相等
        assert_equal(a, np.ones((5,), dtype=float16))

        # 创建包含特定元素的 float16 类型数组 a
        a = np.array([0, 0, -1, -1/1e20, 0, 2.0**-24, 7.629e-6], dtype=float16)
        # 断言数组 a 中非零元素的索引数组与预期相同
        assert_equal(a.nonzero()[0],
                     [2, 5, 6])
        # 字节交换数组 a，并根据当前平台设置其字节顺序
        a = a.byteswap()
        a = a.view(a.dtype.newbyteorder())
        # 再次断言数组 a 中非零元素的索引数组与预期相同
        assert_equal(a.nonzero()[0],
                     [2, 5, 6])

        # 使用 np.arange 创建一个从 0 到 10（不包括）的步长为 0.5 的 float16 类型数组 a
        a = np.arange(0, 10, 0.5, dtype=float16)
        # 创建一个全为 1 的 float16 类型数组 b，长度为 20
        b = np.ones((20,), dtype=float16)
        # 断言数组 a 与 b 的点积结果为 95
        assert_equal(np.dot(a, b),
                     95)

        # 创建包含特定元素的 float16 类型数组 a
        a = np.array([0, -np.inf, -2, 0.5, 12.55, 7.3, 2.1, 12.4], dtype=float16)
        # 断言数组 a 的最大值索引为 4
        assert_equal(a.argmax(),
                     4)
        # 创建包含特定元素的 float16 类型数组 a
        a = np.array([0, -np.inf, -2, np.inf, 12.55, np.nan, 2.1, 12.4], dtype=float16)
        # 断言数组 a 的最大值索引为 5
        assert_equal(a.argmax(),
                     5)

        # 使用 np.arange 创建一个长度为 10 的 float16 类型数组 a
        a = np.arange(10, dtype=float16)
        # 遍历数组 a 的每个元素，并断言每个元素与其索引相等
        for i in range(10):
            assert_equal(a.item(i), i)
    # 定义一个测试方法，用于测试 np.spacing 和 np.nextafter 的功能
    def test_spacing_nextafter(self):
        """Test np.spacing and np.nextafter"""
        
        # 创建一个包含所有非负有限数的数组，数据类型为 uint16
        a = np.arange(0x7c00, dtype=uint16)
        
        # 创建一个包含正无穷大的 float16 类型数组
        hinf = np.array((np.inf,), dtype=float16)
        
        # 创建一个包含 NaN 的 float16 类型数组
        hnan = np.array((np.nan,), dtype=float16)
        
        # 将 uint16 类型数组 a 视图转换为 float16 类型数组 a_f16
        a_f16 = a.view(dtype=float16)

        # 断言验证 np.spacing 对 a_f16 中除最后一个元素外的元素计算的正确性
        assert_equal(np.spacing(a_f16[:-1]), a_f16[1:] - a_f16[:-1])

        # 断言验证 np.nextafter 对 a_f16 中除最后一个元素外的元素向正无穷大方向计算的正确性
        assert_equal(np.nextafter(a_f16[:-1], hinf), a_f16[1:])
        
        # 断言验证 np.nextafter 对第一个元素向负无穷大方向计算的正确性
        assert_equal(np.nextafter(a_f16[0], -hinf), -a_f16[1])
        
        # 断言验证 np.nextafter 对 a_f16 中除第一个元素外的元素向负无穷大方向计算的正确性
        assert_equal(np.nextafter(a_f16[1:], -hinf), a_f16[:-1])

        # 断言验证 np.nextafter 对正无穷大 hinf 向 a_f16 中各元素计算的正确性
        assert_equal(np.nextafter(hinf, a_f16), a_f16[-1])
        
        # 断言验证 np.nextafter 对负无穷大 -hinf 向 a_f16 中各元素计算的正确性
        assert_equal(np.nextafter(-hinf, a_f16), -a_f16[-1])

        # 断言验证 np.nextafter 对正无穷大 hinf 与自身计算的正确性
        assert_equal(np.nextafter(hinf, hinf), hinf)
        
        # 断言验证 np.nextafter 对正无穷大 hinf 向负无穷大 -hinf 计算的正确性
        assert_equal(np.nextafter(hinf, -hinf), a_f16[-1])
        
        # 断言验证 np.nextafter 对负无穷大 -hinf 向正无穷大 hinf 计算的正确性
        assert_equal(np.nextafter(-hinf, hinf), -a_f16[-1])
        
        # 断言验证 np.nextafter 对负无穷大 -hinf 与自身计算的正确性
        assert_equal(np.nextafter(-hinf, -hinf), -hinf)

        # 断言验证 np.nextafter 对 a_f16 向 hnan 计算的正确性
        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        
        # 断言验证 np.nextafter 对 hnan 向 a_f16 计算的正确性
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])

        # 断言验证 np.nextafter 对 hnan 与自身计算的正确性
        assert_equal(np.nextafter(hnan, hnan), hnan)
        
        # 断言验证 np.nextafter 对正无穷大 hinf 向 hnan 计算的正确性
        assert_equal(np.nextafter(hinf, hnan), hnan)
        
        # 断言验证 np.nextafter 对 hnan 向正无穷大 hinf 计算的正确性
        assert_equal(np.nextafter(hnan, hinf), hnan)

        # 将数组 a 中的元素切换为负数
        a |= 0x8000

        # 断言验证 np.spacing 对第一个元素计算的正确性
        assert_equal(np.spacing(a_f16[0]), np.spacing(a_f16[1]))
        
        # 断言验证 np.spacing 对 a_f16 中除第一个元素外的元素计算的正确性
        assert_equal(np.spacing(a_f16[1:]), a_f16[:-1] - a_f16[1:])

        # 断言验证 np.nextafter 对第一个元素向正无穷大 hinf 计算的正确性
        assert_equal(np.nextafter(a_f16[0], hinf), -a_f16[1])
        
        # 断言验证 np.nextafter 对 a_f16 中除最后一个元素外的元素向正无穷大 hinf 计算的正确性
        assert_equal(np.nextafter(a_f16[1:], hinf), a_f16[:-1])
        
        # 断言验证 np.nextafter 对 a_f16 中除第一个元素外的元素向负无穷大 -hinf 计算的正确性
        assert_equal(np.nextafter(a_f16[:-1], -hinf), a_f16[1:])

        # 断言验证 np.nextafter 对正无穷大 hinf 向 a_f16 计算的正确性
        assert_equal(np.nextafter(hinf, a_f16), -a_f16[-1])
        
        # 断言验证 np.nextafter 对负无穷大 -hinf 向 a_f16 计算的正确性
        assert_equal(np.nextafter(-hinf, a_f16), a_f16[-1])

        # 断言验证 np.nextafter 对 a_f16 向 hnan 计算的正确性
        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        
        # 断言验证 np.nextafter 对 hnan 向 a_f16 计算的正确性
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])
    # 定义一个测试方法，用于验证半精度数据在与其他数据类型进行计算时的强制转换
    def test_half_coercion(self, weak_promotion):
        """Test that half gets coerced properly with the other types"""
        # 创建一个包含单个元素的 numpy 数组，数据类型为 float16
        a16 = np.array((1,), dtype=float16)
        # 创建一个包含单个元素的 numpy 数组，数据类型为 float32
        a32 = np.array((1,), dtype=float32)
        # 创建一个 float16 类型的变量，赋值为 1
        b16 = float16(1)
        # 创建一个 float32 类型的变量，赋值为 1
        b32 = float32(1)

        # 断言：a16 的平方的数据类型应为 float16
        assert np.power(a16, 2).dtype == float16
        # 断言：a16 的 2.0 次方的数据类型应为 float16
        assert np.power(a16, 2.0).dtype == float16
        # 断言：a16 的 b16 次方的数据类型应为 float16
        assert np.power(a16, b16).dtype == float16
        # 根据 weak_promotion 变量决定的预期数据类型，断言：a16 的 b32 次方的数据类型
        expected_dt = float32 if weak_promotion else float16
        assert np.power(a16, b32).dtype == expected_dt
        # 断言：a16 的 a16 次方的数据类型应为 float16
        assert np.power(a16, a16).dtype == float16
        # 断言：a16 的 a32 次方的数据类型应为 float32
        assert np.power(a16, a32).dtype == float32

        # 根据 weak_promotion 变量决定的预期数据类型，断言：b16 的 2 次方的数据类型
        expected_dt = float16 if weak_promotion else float64
        assert np.power(b16, 2).dtype == expected_dt
        # 根据 weak_promotion 变量决定的预期数据类型，断言：b16 的 2.0 次方的数据类型
        assert np.power(b16, 2.0).dtype == expected_dt
        # 断言：b16 的 b16 次方的数据类型应为 float16
        assert np.power(b16, b16).dtype, float16
        # 断言：b16 的 b32 次方的数据类型应为 float32
        assert np.power(b16, b32).dtype, float32
        # 断言：b16 的 a16 次方的数据类型应为 float16
        assert np.power(b16, a16).dtype, float16
        # 断言：b16 的 a32 次方的数据类型应为 float32
        assert np.power(b16, a32).dtype, float32

        # 断言：a32 的 a16 次方的数据类型应为 float32
        assert np.power(a32, a16).dtype == float32
        # 断言：a32 的 b16 次方的数据类型应为 float32
        assert np.power(a32, b16).dtype == float32
        # 根据 weak_promotion 变量决定的预期数据类型，断言：b32 的 a16 次方的数据类型
        expected_dt = float32 if weak_promotion else float16
        assert np.power(b32, a16).dtype == expected_dt
        # 断言：b32 的 b16 次方的数据类型应为 float32
        assert np.power(b32, b16).dtype == float32

    # 使用 pytest 的装饰器标记此测试为条件跳过，如果机器架构为 "armv5tel"，则跳过并注明原因为 "See gh-413."
    @pytest.mark.skipif(platform.machine() == "armv5tel",
                        reason="See gh-413.")
    # 使用 pytest 的装饰器标记此测试为条件跳过，如果运行环境为 WASM，则跳过并注明原因为 "fp exceptions don't work in wasm."
    @pytest.mark.skipif(IS_WASM,
                        reason="fp exceptions don't work in wasm.")
    # 定义一个测试方法，用于验证 half 类型是否兼容 __array_interface__
    def test_half_array_interface(self):
        """Test that half is compatible with __array_interface__"""
        # 定义一个空类 Dummy
        class Dummy:
            pass

        # 创建一个包含单个元素的 numpy 数组，数据类型为 float16
        a = np.ones((1,), dtype=float16)
        # 创建一个 Dummy 类的实例 b
        b = Dummy()
        # 将 b 的 __array_interface__ 设置为 a 的 __array_interface__
        b.__array_interface__ = a.__array_interface__
        # 创建一个 numpy 数组 c，其值与 b 相同
        c = np.array(b)
        # 断言：c 的数据类型应为 float16
        assert_(c.dtype == float16)
        # 断言：a 与 c 应相等
        assert_equal(a, c)
```