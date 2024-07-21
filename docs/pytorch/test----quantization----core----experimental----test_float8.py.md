# `.\pytorch\test\quantization\core\experimental\test_float8.py`

```
# Owner(s): ["oncall: quantization"]

# 引入单元测试相关模块
import unittest

# 引入 torch 库及其所需的测试相关模块
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_WINDOWS,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

# 定义一组 torch.float8 数据类型列表
FLOAT8_DTYPES = [
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
]

# 定义 CUDA 下的 torch.float8 数据类型列表
CUDA_FLOAT8_DTYPES = [
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

# 定义每种 torch.float8 数据类型对应的尾数位数
MANTISSA_BITS = {
    torch.float8_e5m2: 2,
    torch.float8_e5m2fnuz: 2,
    torch.float8_e4m3fn: 3,
    torch.float8_e4m3fnuz: 3,
}

# 定义每种 torch.float8 数据类型对应的最小指数值
MINEXP = {
    torch.float8_e5m2: -14,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fn: -6,
    torch.float8_e4m3fnuz: -7,
}

# 定义每种 torch.float8 数据类型对应的特殊数值及其表示形式
SPECIAL_NUMBERS = {
    torch.float8_e5m2: [
        ("01111100", float("inf"), "inf"),
        ("11111100", -1.0 * float("inf"), "neg_inf"),
        ("01111101", float("nan"), "nan"),
        ("11111101", float("nan"), "nan"),
        ("01111110", float("nan"), "nan"),
        ("11111110", float("nan"), "nan"),
        ("01111111", float("nan"), "nan"),
        ("11111111", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("10000000", -0.0, "neg_zero"),
        ("01111011", 57344.0, "max_normal"),
        ("11111011", -57344.0, "neg_max_normal"),
        ("00000100", 2**-14, "min_normal"),
        ("10000100", -1 * (2**-14), "neg_min_normal"),
        ("00000011", 0.75 * (2**-14), "max_subnorm"),
        ("10000011", -0.75 * (2**-14), "neg_max_subnorm"),
        ("00000001", 2**-16, "min_subnorm"),
        ("10000001", -1 * (2**-16), "neg_min_subnorm"),
    ],
    torch.float8_e5m2fnuz: [
        ("10000000", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("00000000", -0.0, "neg_zero"),
        ("01111111", 57344.0, "max_normal"),
        ("11111111", -57344.0, "neg_max_normal"),
        ("00000100", 2**-15, "min_normal"),
        ("10000100", -1 * (2**-15), "neg_min_normal"),
        ("00000011", 0.75 * (2**-15), "max_subnorm"),
        ("10000011", -0.75 * (2**-15), "neg_max_subnorm"),
        ("00000001", 0.25 * (2**-15), "min_subnorm"),
        ("10000001", -0.25 * (2**-15), "neg_min_subnorm"),
    ],
    torch.float8_e4m3fn: [
        ("01111111", float("nan"), "nan"),
        ("11111111", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("10000000", -0.0, "neg_zero"),
        ("01111110", 448.0, "max_normal"),
        ("11111110", -448.0, "neg_max_normal"),
        ("00001000", 2**-6, "min_normal"),
        ("10001000", -1 * (2**-6), "neg_min_normal"),
        ("00000111", 0.875 * (2**-6), "max_subnorm"),
        ("10000111", -0.875 * (2**-6), "neg_max_subnorm"),
        ("00000001", 2**-9, "min_subnorm"),
        ("10000001", -1 * (2**-9), "neg_min_subnorm"),
    ],
    torch.float8_e4m3fnuz: [
        # 定义一个包含多个元组的列表，每个元组包含三个元素：二进制表示、浮点数值、描述字符串
        ("10000000", float("nan"), "nan"),              # 表示 NaN
        ("00000000", 0.0, "zero"),                      # 表示正零
        ("00000000", -0.0, "neg_zero"),                 # 表示负零
        ("01111111", 240.0, "max_normal"),              # 表示最大正规数
        ("11111111", -240.0, "neg_max_normal"),         # 表示最大负规范数
        ("00001000", 2**-7, "min_normal"),              # 表示最小正规数
        ("10001000", -1 * (2**-7), "neg_min_normal"),   # 表示最小负规范数
        ("00000111", 0.875 * (2**-7), "max_subnorm"),   # 表示最大次规范数
        ("10000111", -0.875 * (2**-7), "neg_max_subnorm"),  # 表示最大负次规范数
        ("00000001", 0.125 * (2**-7), "min_subnorm"),   # 表示最小次规范数
        ("10000001", -0.125 * (2**-7), "neg_min_subnorm"),  # 表示最小负次规范数
    ],
}

FLOAT8_DTYPES_WITH_INF = [torch.float8_e5m2]

# 定义一个变量，包含了具有无穷大的 float8 数据类型
def simulate_fp8_precision(input, variant):
    """Round input (as float32) to the given float8 datatype variant."""

    # 常量定义
    dtype = torch.float32  # 使用 torch 的 float32 类型
    int_type = torch.int32  # 使用 torch 的 int32 类型
    mbits = MANTISSA_BITS[variant]  # 从 MANTISSA_BITS 字典中获取指定 variant 的小数部分位数
    minexp = MINEXP[variant]  # 从 MINEXP 字典中获取指定 variant 的最小指数值

    input = input.to(dtype)  # 将输入转换为 float32 类型

    # 提取位字段组成部分
    signs = torch.sign(input)  # 提取输入的符号
    input_int = torch.abs(input).view(int_type)  # 获取输入的绝对值，并以 int_type 类型视图查看

    exponent_bits = (input_int & 0x7F800000) >> 23  # 提取指数位
    mantissa_bits = input_int & 0x007FFFFF  # 提取尾数位

    exponent_base = exponent_bits - 0x7F  # 计算指数的基数

    # 添加隐含的前导 1 到尾数，即创建 1.mmmmmmmm
    f32_is_normal = exponent_bits != 0
    mantissa_val_base = f32_is_normal * 0x00800000 + mantissa_bits

    # 将尾数位移以匹配最小指数 - 在更低精度的数据类型中，非规格化数保持正常
    denormal_bits = torch.maximum(
        minexp - exponent_base, torch.tensor(0, dtype=int_type)
    )
    mantissa_val = mantissa_val_base >> denormal_bits
    exponent = exponent_base + denormal_bits

    # 四舍五入尾数
    last_unrounded_bit = 1 << (23 - mbits)
    rounding_mask = last_unrounded_bit - 1
    mantissa_val_rounded = (mantissa_val + (rounding_mask >> 1)) & ~rounding_mask

    # 四舍五入到最近的偶数
    ties = (mantissa_val & rounding_mask) == (last_unrounded_bit >> 1)
    is_odd = (mantissa_val_rounded & last_unrounded_bit) != 0
    mantissa_val_rounded += (ties & is_odd) * last_unrounded_bit

    # 重新组合尾数和指数
    vals = (mantissa_val_rounded * 2.0 ** (-23 + exponent)).to(dtype)

    # 用适当的 inf/NaN 替换溢出值（不进行饱和）
    have_inf = variant in FLOAT8_DTYPES_WITH_INF
    vals[vals > torch.finfo(variant).max] = torch.inf if have_inf else torch.nan

    return vals * signs

# 定义一系列浮点8位数据类型的测试用例
ROUND_TRIP_TEST_CASES = (
    # 一个通用的“soak test”。
    subtest(
        lambda dtype, device: torch.rand((100, 100), device=device)
        * torch.finfo(dtype).max,
        name="soak",
    ),
    # 一个低于较低精度类型中最小正常值的范围，以确保正确将其舍入到该类型中最接近的次正规数。
    subtest(
        lambda dtype, device: torch.rand(1000, device=device)
        * 2
        * torch.finfo(dtype).smallest_normal,
        name="subnormals",
    ),
    # 一系列整数，以施加四舍五入到最接近的偶数。
    subtest(
        lambda dtype, device: torch.arange(
            int(torch.finfo(dtype).max), dtype=torch.int, device=device
        ),
        name="rte",
    ),
    # 在最大值附近的一系列值。
    subtest(
        lambda dtype, device: torch.finfo(dtype).max
        + (torch.finfo(dtype).eps * torch.finfo(dtype).max)
        * torch.arange(-3, 3, 0.25, device=device),
        name="extremes",
    ),
)

# 定义一个用于浮点8位数据类型的测试类
class TestFloat8Dtype(TestCase):
    """
    Sanity test for zeros comparison
    """

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    # 装饰器：在CUDA环境下使用指定的float8数据类型
    def test_creation_with_zeros(self, dtype, device):
        """Sanity test, round-trip casting of zeros."""
        # 创建一个大小为8的零张量，使用指定的float数据类型和设备
        x = torch.zeros(8, dtype=torch.float, device=device)
        # 创建一个大小为8的零张量，使用指定的dtype和设备
        x8 = torch.zeros(8, dtype=dtype, device=device)
        # 断言两个张量是否相等，进行round-trip转换测试，比较时绝对误差和相对误差均为0
        self.assertEqual(x, x8.float(), atol=0, rtol=0)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @parametrize("get_input", ROUND_TRIP_TEST_CASES)
    # 装饰器：使用指定的float8数据类型和CUDA环境下的float8数据类型，参数化测试用例
    def test_cast_round_trip(self, dtype, get_input, device):
        """Numerical test of float8 conversion, by performing a round-trip cast
        to the float8 dtype and back to float32, comparing against simulated
        lower precision."""
        # 获取测试输入数据
        x = get_input(dtype, device)
        # 将输入数据和其相反数拼接起来
        x = torch.cat((x, -x))
        # 将输入数据转换为指定的dtype
        x8 = x.to(dtype)
        # 使用模拟的float8精度进行转换
        x8_simulated = simulate_fp8_precision(x, dtype)
        # 断言模拟的float8精度转换结果与实际转换结果在float32精度下是否相等
        self.assertEqual(x8_simulated, x8.float())

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    # 装饰器：使用指定的float8数据类型和CUDA环境下的float8数据类型
    def test_special_numbers(self, dtype, device):
        """Test special numbers."""

        def compare_binary_with_decimal(binary, decimal, number_name, dtype, device):
            # 将二进制字符串转换为整数
            bits_int = int(binary, 2)
            # 创建一个uint8张量，数据为bits_int，使用指定的dtype和设备
            tensor_int = torch.tensor([bits_int], dtype=torch.uint8, device=device)
            # 将uint8张量视图转换为指定的float8张量
            tensor_fp8 = tensor_int.view(dtype)
            if number_name == "nan":
                # 断言float8张量是否为NaN
                assert tensor_fp8.isnan()
            else:
                # 将float8张量转换为float32张量
                tensor_fp32 = tensor_fp8.float()
                # 创建一个参考的float32张量，数据为decimal，使用指定的dtype和设备
                ref_tensor_fp32 = torch.tensor(
                    [decimal], dtype=torch.float, device=device
                )
                # 断言float32张量和参考的float32张量是否相等，比较时绝对误差和相对误差均为0
                self.assertEqual(tensor_fp32, ref_tensor_fp32, atol=0, rtol=0)

        # 遍历指定dtype下的特殊数值
        for number in SPECIAL_NUMBERS[dtype]:
            # 比较二进制和十进制数值
            compare_binary_with_decimal(*number, dtype, device)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    # 装饰器：使用指定的float8数据类型和CUDA环境下的float8数据类型
    def test_type_promotion_fails(self, dtype, device):
        """Test that float8 is not promoted to higher precision Float Type."""
        # 遍历其他数据类型
        for other_dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]:
            # 创建一个大小为8的随机张量，使用指定的dtype和设备
            x = torch.randn(8, device=device).to(dtype)
            # 创建一个大小为8的随机张量，使用指定的其他dtype和设备
            y = torch.randn(8, device=device).to(other_dtype)
            # 断言加法操作是否引发RuntimeError，错误信息为"Promotion for Float8 Types is not supported"
            with self.assertRaisesRegex(
                RuntimeError, "Promotion for Float8 Types is not supported"
            ):
                x + y

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    # 装饰器：使用指定的float8数据类型和CUDA环境下的float8数据类型
    def test_empty(self, dtype, device):
        # 使用确定性保护，检查确定性算法是否启用
        with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
            # 对于每个使用确定性算法的情况
            for use_deterministic in (True, False):
                # 使用或不使用确定性算法
                torch.use_deterministic_algorithms(use_deterministic)
                # 创建一个空的4x4张量，使用指定的dtype和设备
                x = torch.empty(4, 4, device=device, dtype=dtype)
# 在全局作用域中实例化 TestFloat8Dtype 类的设备类型测试
instantiate_device_type_tests(TestFloat8Dtype, globals())

# 定义 TestFloat8DtypeCPUOnly 类，继承自 TestCase
class TestFloat8DtypeCPUOnly(TestCase):

    """
    Test of mul implementation

    注意：目前仅支持 CPU，因为要在 CUDA 上添加它需要添加另一个 C++ dtype 宏，而目前还没有未缩放的 float8 乘法的使用案例，似乎不值得这样做。
    """

    # 使用 @dtypes 装饰器，参数为 CUDA_FLOAT8_DTYPES 中的数据类型
    @dtypes(*CUDA_FLOAT8_DTYPES)
    def test_mul(self, dtype):
        shape = (10, 10)
        # 生成随机张量 a，并模拟成 fp8 精度
        a = torch.randn(shape)
        a8_simulated = simulate_fp8_precision(a, dtype)
        # 将张量 a 转换为指定数据类型 dtype
        a8 = a.to(dtype)
        # 生成随机张量 b，并模拟成 fp8 精度
        b = torch.randn(shape)
        b8_simulated = simulate_fp8_precision(b, dtype)
        # 将张量 b 转换为指定数据类型 dtype
        b8 = b.to(dtype)
        # 计算 fp8 精度下的张量乘法
        mul8 = a8 * b8
        # 计算模拟的 fp8 精度下的张量乘法，并转换为指定数据类型 dtype
        mul8_simulated = (a8_simulated * b8_simulated).to(dtype)
        # 断言 fp8 精度下的乘法结果是否相等
        self.assertEqual(mul8, mul8_simulated)

    # 使用 @unittest.skipIf 装饰器，当 IS_WINDOWS 为真时跳过该测试（因为 torch.compile 在 Windows 上尚不支持）
    # 同时使用 @dtypes 装饰器，参数为 CUDA_FLOAT8_DTYPES 中的数据类型
    def test_pt2_traceable_aot_eager(self, dtype):
        # 定义一个编译函数 f，使用后端 "aot_eager"，并生成完整的计算图
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            # 将输入张量 x 转换为指定数据类型 dtype
            x = x.to(dtype)
            # 张量 x 转换为 float 类型
            x = x.float()
            return x

        # 生成一个随机张量 x，并要求计算梯度
        x = torch.randn(1).requires_grad_()
        # 对 f(x) 求和，并进行反向传播
        f(x).sum().backward()

# 在全局作用域中实例化 TestFloat8DtypeCPUOnly 类的设备类型测试，仅适用于 CPU
instantiate_device_type_tests(TestFloat8DtypeCPUOnly, globals(), only_for="cpu")

# 如果运行时为主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```