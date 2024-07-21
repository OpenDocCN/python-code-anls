# `.\pytorch\test\inductor\test_fp8.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的模块和库
import functools
import unittest

import torch
from torch import Tensor
from torch._inductor import utils  # 导入模块内部的 utils
from torch._inductor.test_case import run_tests, TestCase  # 导入测试运行和测试用例类
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, SM90OrLater  # 导入 CUDA 相关的测试常量
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入参数化测试实例化装饰器
    parametrize,  # 导入参数化装饰器
    TEST_WITH_ROCM,  # ROCm 测试标记
)
from torch.testing._internal.inductor_utils import HAS_CUDA  # 导入 CUDA 检查工具

# 设置矩阵乘法的精度为高精度浮点数
torch.set_float32_matmul_precision("high")

# FP8 支持消息提示
f8_msg = "FP8 is only supported on H100+ and sm_89 and MI300+ devices"

# 定义 e4m3/e5m2 的常量极限值
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
E4M3FNUZ_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
E5M2FNUZ_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max


def _to_fp8_saturated(x: Tensor, float8_dtype: torch.dtype) -> Tensor:
    # 将输入张量 x 转换为饱和的 float8 类型张量
    # 在 PyTorch 中，默认情况下，转换到 `float8_e4m3fn` 和 `e5m2` 类型时不会饱和。
    # 在此上下文中，我们需要进行饱和处理。
    # 一个常见的情况是，当张量的历史最大值是 `amax1`，当前的最大值是 `amax2` 时，我们希望进行饱和处理。
    # 这在使用延迟缩放时很常见。
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif float8_dtype == torch.float8_e5m2:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    elif float8_dtype == torch.float8_e4m3fnuz:
        x = x.clamp(min=-1 * E4M3FNUZ_MAX_POS, max=E4M3FNUZ_MAX_POS)
    elif float8_dtype == torch.float8_e5m2fnuz:
        x = x.clamp(min=-1 * E5M2FNUZ_MAX_POS, max=E5M2FNUZ_MAX_POS)
    else:
        raise TypeError(f"Unsupported float8_dtype: {float8_dtype}")
    return x.to(float8_dtype)


# 实例化参数化测试
@instantiate_parametrized_tests
class TestFP8Types(TestCase):
    # 如果平台不支持 FP8，则跳过测试，并给出相应的提示消息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    # 如果在 ROCm 环境下，则跳过测试，并给出相应的提示消息
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported yet")
    # 参数化测试装饰器，指定测试的数据类型为 torch.float16 和 torch.bfloat16
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    # 定义一个测试函数，用于测试在指定数据类型下的矩阵乘法操作
    def test_eager_fallback(self, dtype: torch.dtype):
        # 定义权重的形状
        weight_shape = (32, 16)

        # 根据当前环境选择适当的 torch.float8 类型
        e4m3_type = (
            torch.float8_e4m3fn if torch.version.hip is None else torch.float8_e4m3fnuz
        )

        # 定义未包装的 fp8 矩阵乘法函数
        def fp8_matmul_unwrapped(x):
            # 定义在 CUDA 设备上的输入缩放因子
            a_scale = torch.Tensor([1.0]).to(device="cuda")
            b_scale = torch.Tensor([1.0]).to(device="cuda")
            output_scale = None
            # 随机生成输入偏置
            input_bias = torch.rand(32, device="cuda", dtype=dtype)
            # 随机生成权重，并转换为指定的 e4m3 类型
            weight = torch.rand(*weight_shape, device="cuda", dtype=dtype).T.to(
                e4m3_type
            )
            # 计算输入缩放因子的倒数
            a_inverse_scale = 1 / a_scale
            b_inverse_scale = 1 / b_scale
            # 调用 torch._scaled_mm 函数执行矩阵乘法
            output, updated_amax = torch._scaled_mm(
                x,
                weight,
                bias=input_bias,
                out_dtype=dtype,
                scale_a=a_inverse_scale,
                scale_b=b_inverse_scale,
                scale_result=output_scale,
            )
            return output

        # 使用 torch.compile 编译 fp8_matmul_unwrapped 函数，使用 'inductor' 后端，并启用动态功能
        compiled_fp8_matmul = torch.compile(
            fp8_matmul_unwrapped, backend="inductor", dynamic=True
        )

        # 定义输入张量的形状
        x_shape = (16, 16)
        # 随机生成输入张量，并转换为 e4m3_type 类型
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(e4m3_type)
        # 调用编译后的函数计算 y_fp8
        y_fp8 = compiled_fp8_matmul(x)

        # 重新定义输入张量的形状
        x_shape = (15, 16)
        # 随机生成输入张量，并转换为 e4m3_type 类型
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(e4m3_type)
        # 再次调用编译后的函数计算 y_fp8
        y_fp8 = compiled_fp8_matmul(x)

    # 在不支持 FP8 的平台上跳过测试，显示相应的消息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    # 参数化测试函数，测试不同数据类型和形状的有效类型转换
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    @parametrize(
        "dst_types",
        [(torch.float8_e4m3fn, torch.float8_e5m2)]
        if torch.version.hip is None
        else [(torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)],
    )
    def test_valid_cast(self, dtype: torch.dtype, shape: str, dst_types: tuple):
        # 解包目标类型元组
        e4m3, e5m2 = dst_types

        # 定义类型转换函数 fp8_cast
        def fp8_cast(x):
            # 将输入张量 x 转换为 e4m3 类型，再转换回原始数据类型 dtype
            y0 = x.to(dtype=e4m3).to(dtype)
            # 将输入张量 x 转换为 e5m2 类型，再转换回原始数据类型 dtype
            y1 = x.to(dtype=e5m2).to(dtype)
            return y0, y1

        # 使用 torch.compile 编译 fp8_cast 函数，使用 'inductor' 后端，并启用动态功能
        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        # 解析并转换形状字符串为整数列表
        shape = [int(dim) for dim in shape.split(",")]
        # 随机生成指定形状的输入张量，并使用 CUDA 设备和指定的数据类型 dtype
        x = torch.rand(*shape, device="cuda", dtype=dtype)
        # 调用编译后的函数计算 y0_fp8 和 y1_fp8
        y0_fp8, y1_fp8 = compiled_fp8_cast(x)

        # 使用 torch.testing.assert_close 检查 y0_fp8 和 y1_fp8 是否与输入张量 x 接近
        torch.testing.assert_close(y0_fp8, x, rtol=5e-1, atol=5e-1)
        torch.testing.assert_close(y1_fp8, x, rtol=5e-1, atol=5e-1)

    # 在不支持 FP8 的平台上跳过测试，显示相应的消息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    # 定义一个测试方法，用于测试不良类型转换的情况
    def test_bad_cast(self):
        # 定义一个函数，用于将输入 x 转换为指定的数据类型 dtype
        def fp8_cast(x, dtype):
            return x.to(dtype=dtype)

        # 使用 torch.compile 编译 fp8_cast 函数，使用 "inductor" 后端进行动态编译
        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        # 定义输入张量的形状
        x_shape = (16, 16, 16)

        # 断言捕获异常，检查是否抛出指定异常类型和消息
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            # 创建在 CUDA 设备上的随机张量 x，并将其转换为 torch.float8_e4m3fn 数据类型
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e4m3fn)
            # 使用编译后的函数 compiled_fp8_cast 进行类型转换
            y = compiled_fp8_cast(x, torch.float8_e5m2)

        # 再次进行异常断言，检查不支持的类型转换异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            # 创建在 CUDA 设备上的随机张量 x，并将其转换为 torch.float8_e5m2 数据类型
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e5m2)
            # 使用编译后的函数 compiled_fp8_cast 进行类型转换
            y = compiled_fp8_cast(x, torch.float8_e4m3fn)

    # 如果平台不支持 FP8，跳过此测试，并输出相应的消息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    # 参数化测试，测试不同的源数据类型
    @parametrize("src_dtype", (torch.float16, torch.bfloat16, torch.float))
    # 参数化测试，测试不同的目标数据类型，根据 torch.version.hip 是否为 None 来选择不同的目标类型
    @parametrize(
        "dst_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
    # 参数化测试，测试不同的张量形状
    @parametrize("shape", ("16,16,16", "4,2048,4096"))
    def test_to_fp8_saturated(
        self, src_dtype: torch.dtype, dst_dtype: torch.dtype, shape: str
    ):
        # 定义一个函数，将输入张量 x 转换为指定数据类型 dtype 的饱和 FP8 格式
        def fp8_saturated(x, dtype):
            return _to_fp8_saturated(x, dtype)

        # 使用 torch.compile 编译 fp8_saturated 函数，使用 "inductor" 后端进行动态编译
        compiled_fp8_cast = torch.compile(
            fp8_saturated, backend="inductor", dynamic=True
        )

        # 将字符串形状转换为整数列表
        shape = [int(dim) for dim in shape.split(",")]

        # 在 CUDA 设备上创建一个随机张量 x，指定数据类型为 src_dtype
        x = torch.rand(*shape, device="cuda", dtype=src_dtype)

        # 使用编译后的函数 compiled_fp8_cast 将 x 转换为目标数据类型 dst_dtype
        y_compiled = compiled_fp8_cast(x, dst_dtype)

        # 调用原始的 fp8_saturated 函数将 x 转换为目标数据类型 dst_dtype
        y = fp8_saturated(x, dst_dtype)

        # 使用 torch.testing.assert_close 方法比较编译后和原始方法的结果，以验证其接近程度
        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=5e-1, atol=5e-1)

    # 如果测试运行在 ROCm 平台上，跳过此测试，因为 ROCm 存在精度问题
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm fails with accuracy issue")
    # 如果当前设备不支持 SM90 或更高版本，跳过此测试
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    # 参数化测试，测试不同的 float8 数据类型
    @parametrize(
        "float8_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
    # 参数化测试，测试不同的张量形状
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_amax_fp8_quant(self, float8_dtype: torch.dtype, shape: str):
        # 将 shape 字符串按逗号分隔并转换为整数列表
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        # 定义函数 amax_fp8，计算输入张量的最大绝对值，然后进行量化操作
        def amax_fp8(x: Tensor, scale: Tensor):
            y = torch.amax(torch.abs(x))
            y_scaled = y.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(y_scaled, float8_dtype)
            return bits_fp8

        # 编译 amax_fp8 函数以进行量化操作，使用 "inductor" 后端
        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        # 创建输入张量 x，随机初始化，数据类型为 torch.half，在 CUDA 设备上
        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        # 创建比例因子 scale 张量，数值为 0.2，在 CUDA 设备上，数据类型为 torch.float
        scale = torch.tensor(0.2, device="cuda", dtype=torch.float)

        # 使用编译后的函数计算量化结果 y_compiled
        y_compiled = compiled_amax_fp8_quant(x, scale)
        # 使用非编译版本的函数计算量化结果 y
        y = amax_fp8(x, scale)

        # 断言编译版本和非编译版本的结果非常接近，使用相对和绝对容差
        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize(
        "float8_dtype",
        # 根据条件选择不同的 float8_dtype 参数
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_amax_along_with_fp8_quant(self, float8_dtype: torch.dtype, shape: str):
        # 将 shape 字符串按逗号分隔并转换为整数列表
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        # 定义函数 amax_fp8，计算输入张量的最大绝对值，然后进行量化操作，同时填充 amax_buffer
        def amax_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            amax_buffer.fill_(torch.amax(torch.abs(x)))
            x_scaled = x.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        # 编译 amax_fp8 函数以进行量化操作，使用 "inductor" 后端
        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        # 创建输入张量 x，随机初始化，数据类型为 torch.half，在 CUDA 设备上
        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        # 创建比例因子 scale 张量，数值为 1.0，在 CUDA 设备上，数据类型为 torch.float
        scale = torch.tensor(1.0, device="cuda", dtype=torch.float)

        # 创建用于编译版本的 amax_buffer，数据类型为 torch.half
        amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
        # 使用编译后的函数计算量化结果 y_compiled，并填充 amax_buffer_compiled
        y_compiled = compiled_amax_fp8_quant(x, scale, amax_buffer_compiled)
        
        # 创建用于非编译版本的 amax_buffer，数据类型为 torch.half
        amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
        # 使用非编译版本的函数计算量化结果 y，并填充 amax_buffer
        y = amax_fp8(x, scale, amax_buffer)

        # 断言编译版本和非编译版本的结果非常接近，使用相对和绝对容差
        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        # 断言编译版本和非编译版本的 amax_buffer 结果非常接近，使用相对和绝对容差
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm fails with accuracy issue")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize(
        "float8_dtype",
        # 根据条件选择不同的 float8_dtype 参数
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
    @parametrize("amax_keep_dim", (True, False))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    # 测试函数：测试使用 FP8 量化的 LayerNorm
    def test_layernorm_fp8_quant(
        self, float8_dtype: torch.dtype, amax_keep_dim: bool, shape: str
    ):
        # 将 shape 字符串解析为整数列表
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        # 定义 FP8 版本的 LayerNorm 函数
        def ln_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            # 执行 LayerNorm 操作，将输入 x 转换为 float 类型
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],  # LayerNorm 的维度为 hidden_size
                weight=None,    # 不使用权重
                bias=None,      # 不使用偏置
                eps=1e-05,      # epsilon 值设为 1e-05
            )
            # 计算 x 的绝对值的最大值，并将结果存入 amax_buffer 中
            amax_buffer.fill_(
                torch.amax(torch.abs(x), keepdim=amax_keep_dim).reshape(-1)[0]
            )
            # 将 x 缩放到指定的 scale
            x_scaled = x * scale
            # 将缩放后的 x 转换为 FP8 格式
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        # 编译 ln_fp8 函数，使用 "inductor" 后端
        compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

        # 生成输入张量 x，形状为 (batch_size, sequence_length, hidden_size)
        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        # 创建缩放因子 scale 张量，存储在 CUDA 设备上，数据类型为 torch.float
        scale = torch.tensor(0.2, device="cuda", dtype=torch.float)

        # 创建编译后的 amax_buffer，数据类型为 torch.half
        amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
        # 使用编译后的 ln_fp8 函数计算输出 y_compiled
        y_compiled = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        
        # 创建非编译的 amax_buffer，数据类型为 torch.half
        amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
        # 使用普通的 ln_fp8 函数计算输出 y
        y = ln_fp8(x, scale, amax_buffer)

        # 断言编译前后结果 y_compiled 和 y 在一定误差范围内相等
        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        # 断言编译前后的 amax_buffer 在一定误差范围内相等
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )
        ):
            # 将形状参数解析为整数列表，包括批量大小、序列长度和隐藏层大小
            shape = [int(dim) for dim in shape.split(",")]
            batch_size, sequence_length, hidden_size = shape

            # 定义层归一化函数 ln
            def ln(x: Tensor):
                # 使用 PyTorch 提供的层归一化函数对输入 x 进行归一化处理
                x = torch.nn.functional.layer_norm(
                    x.to(dtype=torch.float),
                    [hidden_size],
                    weight=None,
                    bias=None,
                    eps=1e-05,
                )
                return x

            # 定义 fp8 格式的层归一化函数 ln_fp8
            def ln_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
                # 使用 PyTorch 提供的层归一化函数对输入 x 进行归一化处理
                x = torch.nn.functional.layer_norm(
                    x.to(dtype=torch.float),
                    [hidden_size],
                    weight=None,
                    bias=None,
                    eps=1e-05,
                )
                # 计算 x 的绝对值最大值，并存入 amax_buffer
                amax = torch.amax(torch.abs(x), keepdim=keepdim)
                amax_buffer.view_as(amax).copy_(amax)
                # 将 x 按比例缩放为 scale，并转换为 fp8 格式
                x_scaled = x * scale
                bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
                return bits_fp8

            # 使用 Inductor 后端编译 ln_fp8 函数
            compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

            # 创建一个随机张量 x，指定在 CUDA 上进行计算，并且数据类型为半精度浮点数
            x_shape = (batch_size, sequence_length, hidden_size)
            x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
            # 创建一个在 CUDA 上的张量 scale，数据类型为单精度浮点数，值为 0.2
            scale = torch.tensor(0.2, device="cuda", dtype=torch.float)

            # 创建用于存储最大值的缓冲区张量，数据类型为半精度浮点数
            amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
            amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
            # 调用编译后的 ln_fp8 函数，并测量其性能
            _ = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
            compiled_latency = utils.do_bench_using_profiling(
                functools.partial(compiled_ln_fp8_quant, x, scale, amax_buffer_compiled)
            )
            # 调用非编译的 ln_fp8 函数，并测量其性能
            eager_latency = utils.do_bench_using_profiling(
                functools.partial(ln_fp8, x, scale, amax_buffer)
            )

            # 使用 Inductor 后端编译 ln 函数
            compiled_ln = torch.compile(ln, backend="inductor")
            _ = compiled_ln(x)
            # 测量编译后 ln 函数的性能
            ln_latency = utils.do_bench_using_profiling(functools.partial(compiled_ln, x))

            # 输出配置信息和基准测试结果
            print(
                f"Config: {float8_dtype=}, {shape=}, {keepdim=}. "
                f"Benchmark results: Inductor: {compiled_latency}ms, Eager: {eager_latency}ms, "
                f"LN only Inductor: {ln_latency}ms."
            )
# 如果当前脚本作为主程序运行（而非被导入到其他模块中），则执行以下代码块
if __name__ == "__main__":
    # 检查是否有 CUDA 支持的条件（通常是一个布尔值或类似的条件）
    if HAS_CUDA:
        # 如果有 CUDA 支持，则运行测试函数
        run_tests()
```