# `.\pytorch\test\test_matmul_cuda.py`

```py
# Owner(s): ["module: linear algebra"]

# 导入单元测试模块
import unittest
# 导入 product 函数，用于生成迭代器元组的笛卡尔积
from itertools import product
# 导入 functools 模块的 partial 函数，用于创建 partial 对象
from functools import partial
# 导入 Optional 类型提示
from typing import Optional

# 导入 PyTorch 库
import torch

# 导入 PyTorch 的量化相关模块
from torch.quantization._quantized_conversions import (
    pack_int4_to_int8,
    quantized_weight_reorder_for_mixed_dtypes_linear_cutlass,
)

# 导入 PyTorch 测试相关模块
from torch.testing import make_tensor
# 导入 PyTorch 内部的 CUDA 相关模块
from torch.testing._internal.common_cuda import (
    SM53OrLater,
    _get_torch_cuda_version,
    PLATFORM_SUPPORTS_FP8
)
# 导入 PyTorch 内部的设备类型相关模块
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    tol as xtol,
    toleranceOverride,
)

# 导入 PyTorch 内部的通用工具模块
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_JETSON,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfRocmVersionLessThan,
    TEST_WITH_ROCM,
    skipIfRocm,
    TestCase,
)

# 检查当前 CUDA 设备是否支持 Ampere 架构 (SM8X)
_IS_SM8X = False
if torch.cuda.is_available():
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8

# 防止意外地更改默认的张量数据类型为 float32
assert torch.get_default_dtype() is torch.float32

# 标记测试类为仅在 ARM64 架构下跳过执行
@unittest.skipIf(IS_ARM64, "Issue with numpy version on arm")
class TestMatmulCuda(TestCase):
    def setUp(self):
        # 设置测试前的准备工作
        super(self.__class__, self).setUp()
        # 禁用 CUDA 矩阵乘法运算中的 TF32 模式
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        # 测试结束后的清理工作
        torch.backends.cuda.matmul.allow_tf32 = True
        super(self.__class__, self).tearDown()
    # 定义一个方法，使用 cuBLAS 执行矩阵乘法和加法操作，用于测试 CUDA 和 CPU 计算结果的一致性
    def cublas_addmm(self, size: int, dtype: torch.dtype, reduced_precision: bool = False):
        #
        # 检查通过比较 CUDA 调用 torch.addmm 和 CPU 调用的结果之间的偏差，来检测 cuBLAS 的严重不准确性
        #

        # 获取维度参数
        n, m, p = (size + 1, size, size + 2)

        # 禁用减少精度的操作，以避免某些核心功能检查失败
        orig_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        orig_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision

        # 在 CPU 上生成随机张量（种子在 common_utils.py 导入时设置）
        # （不使用 numpy 是因为它不支持 bfloat16）
        make_arg = partial(make_tensor, dtype=dtype, device="cpu")
        m_beta = make_arg(1)
        m_input = make_arg((n, p))
        m_1 = make_arg((n, m))
        m_2 = make_arg((m, p))

        # *(B)FLOAT16 特殊处理*
        # 因为后端不在 CPU 上支持 float16 的张量化，
        # 而 bfloat16 可能会导致精度问题，
        # 所以在这些情况下转换为 float32
        # （但对于其他类型，如 float32 和 int*，保持不变）
        if dtype == torch.float16 or dtype == torch.bfloat16:
            m_beta = m_beta.to(dtype=torch.float32)
            m_input = m_input.to(dtype=torch.float32)
            m_1 = m_1.to(dtype=torch.float32)
            m_2 = m_2.to(dtype=torch.float32)

        # 获取 CPU 计算结果
        res_cpu = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())

        # *(B)FLOAT16 特殊处理*
        # 转换回 (b)float16
        if dtype == torch.float16 or dtype == torch.bfloat16:
            m_beta = m_beta.to(dtype=dtype)
            m_input = m_input.to(dtype=dtype)
            m_1 = m_1.to(dtype=dtype)
            m_2 = m_2.to(dtype=dtype)
            res_cpu = res_cpu.to(dtype=dtype)

        # 将参数张量移到 CUDA 上
        m_beta = m_beta.to("cuda")
        m_input = m_input.to("cuda")
        m_1 = m_1.to("cuda")
        m_2 = m_2.to("cuda")

        # 获取 CUDA 计算结果
        res_cuda = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())

        # 将结果移到 CPU 上进行比较
        res_cuda = res_cuda.to("cpu")

        # 比较两者结果是否一致
        self.assertEqual(res_cpu, res_cuda)

        # 恢复原始的 cuBLAS 减少精度设置
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig_bf16
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16
    # 设置在不同数据类型和参数大小下进行的测试的容差覆盖
    @toleranceOverride({torch.float16: xtol(atol=1e-1, rtol=1e-1),
                        torch.bfloat16: xtol(atol=1e-1, rtol=1e-1),
                        torch.float32: xtol(atol=1e-1, rtol=1e-1)})
    # 指定数据类型和参数大小进行参数化测试
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("size", [100, 1000, 10000])
    # 测试使用 cuBLAS 的 addmm 函数，关闭减少精度选项
    def test_cublas_addmm(self, size: int, dtype: torch.dtype):
        self.cublas_addmm(size, dtype, False)

    # 仅在 CUDA 下运行的测试
    @onlyCUDA
    # 如果 ROCm 版本低于 5.2，则跳过测试
    @skipIfRocmVersionLessThan((5, 2))
    # 导入 'xtol' 作为 'tol'，以避免上述代码中的别名问题
    @toleranceOverride({torch.float16: xtol(atol=7e-1, rtol=2e-1),
                        torch.bfloat16: xtol(atol=1e1, rtol=2e-1)})
    # 指定减少精度测试的数据类型和参数大小进行参数化测试
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("size", [100, 1000, 10000])
    # 测试使用 cuBLAS 的 addmm 函数，开启减少精度选项
    def test_cublas_addmm_reduced_precision(self, size: int, dtype: torch.dtype):
        self.cublas_addmm(size, dtype, True)

    # 仅在 CUDA 下运行的测试
    @onlyCUDA
    # 设置在不同数据类型下进行的测试的容差覆盖
    @toleranceOverride({torch.float16: xtol(atol=1e-3, rtol=2e-3)})
    # 指定数据类型进行参数化测试
    @dtypes(torch.float16)
    # 测试使用 cuBLAS 的 addmm 函数，验证对齐性
    def test_cublas_addmm_alignment(self, dtype):
        device = 'cuda'
        # 扰动 X、A 或 B 的对齐方式
        for idx in range(0, 3):
            for offset in range(1, 3):
                offsets = [0, 0, 0]
                offsets[idx] = offset
                x_offset, a_offset, b_offset = offsets
                # 创建随机张量 A，保证需要梯度，指定数据类型和设备
                A = torch.rand((5120 * 2560 + a_offset), requires_grad=True, dtype=dtype, device=device)
                A = A[a_offset:].reshape(5120, 2560)
                # 创建随机张量 X，保证需要梯度，指定数据类型和设备
                X = torch.rand((26 * 2560 + x_offset), requires_grad=True, dtype=dtype, device=device)
                X = X[x_offset:].reshape(26, 1, 2560)
                # 创建随机张量 B，保证需要梯度，指定数据类型和设备
                B = torch.rand((5120 + b_offset), requires_grad=True, dtype=dtype, device=device)
                B = B[b_offset:].reshape(5120)
                # 使用线性函数计算输出
                out = torch.nn.functional.linear(X, A, B)
                # 断言输出与矩阵乘积的和相等
                self.assertEqual(out, torch.matmul(X, A.transpose(1, 0)) + B)

    # 仅在 CUDA 下运行的测试
    @onlyCUDA
    # 如果运行在 Jetson 平台上，由于尺寸过大，跳过测试
    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    # 设置在不同数据类型下进行的测试的容差覆盖
    @toleranceOverride({torch.float32: xtol(atol=1e-5, rtol=1.1e-5)})
    # 指定数据类型进行参数化测试，条件为 ROCm 或 SM53 及更高版本
    @dtypes(*([torch.float32, torch.float16] +
              [torch.bfloat16] if TEST_WITH_ROCM or SM53OrLater else []))
    # 使用参数化的方式设置批大小和矩阵尺寸
    @parametrize(
        "batch_size, N, M, P",
        [(2, 100, 100, 100),
         (2, 1000, 1000, 1000),
         (1, 10000, 1000, 10000),
         (1, 10000, 10000, 10000)],
        # 自定义测试名称的生成函数
        name_fn=lambda batch_size, N, M, P: f"{batch_size}_{N}_{M}_{P}",
    )
    # 如果运行在 ROCm 平台上，跳过测试
    @skipIfRocm
    # 定义一个测试方法，用于测试在给定设备上使用不同大小的输入进行 cublas baddbmm 操作
    def test_cublas_baddbmm_large_input(self, device, batch_size, N, M, P, dtype):
        # 如果数据类型是 torch.float16 或 torch.bfloat16，则将 CPU 数据类型设置为 torch.float32
        cpu_dtype = dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            cpu_dtype = torch.float32

        # 在指定设备上生成随机数据 M1, M2, A
        M1 = torch.rand((N, M), device=device, dtype=dtype)
        M2 = torch.rand((M, P), device=device, dtype=dtype)
        A = torch.rand((N, P), device=device, dtype=dtype)

        # 定义一个函数将张量转换为 CPU 上的指定数据类型
        def _convert_to_cpu(t):
            return t.to(device='cpu', dtype=cpu_dtype)
        # 将 M1, M2, A 转换为 CPU 上的对应数据类型
        M1_cpu, M2_cpu, A_cpu = map(_convert_to_cpu, [M1, M2, A])

        # 使用 functional.linear 执行线性运算，生成 CPU 和 GPU 上的结果并进行比较
        out1_cpu = torch.nn.functional.linear(M1_cpu, M2_cpu.t(), A_cpu).to(dtype=dtype)
        out1_gpu = torch.nn.functional.linear(M1, M2.t(), A).cpu()
        self.assertEqual(out1_cpu, out1_gpu)

        # 如果 N == M == P，则进行乘以单位矩阵的测试
        if N == M and M == P:
            # 在指定设备上生成单位矩阵 M2_eye
            M2_eye = torch.eye(N, device=device, dtype=dtype)
            # 使用 functional.linear 执行乘以单位矩阵的运算，并进行比较
            out1_eye_gpu = torch.nn.functional.linear(M1, M2_eye.t(), torch.zeros_like(A))
            self.assertEqual(M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu())

        # 定义一个函数将张量扩展到指定批次大小
        def _expand_to_batch(t: torch.Tensor):
            return t.expand((batch_size, ) + t.size())
        # 扩展输入张量 M1, M2, A 到指定批次大小
        alpha, beta = 1.0, 1.0
        M1, M2, A, M1_cpu, M2_cpu, A_cpu = map(_expand_to_batch, [M1, M2, A, M1_cpu, M2_cpu, A_cpu])

        # 使用 torch.baddbmm 执行批量矩阵乘加运算，生成 CPU 和 GPU 上的结果并进行比较
        out2_cpu = torch.baddbmm(A_cpu, M1_cpu, M2_cpu, beta=beta, alpha=alpha).to(dtype=dtype)
        out2_gpu = torch.baddbmm(A, M1, M2, beta=beta, alpha=alpha).cpu()
        self.assertEqual(out2_cpu, out2_gpu)

        # 如果 N == M == P，则进行乘以单位矩阵的测试
        if N == M and M == P:
            # 在指定设备上生成扩展的单位矩阵 M2_eye
            M2_eye = torch.eye(N, device=device, dtype=dtype).expand(batch_size, N, N)
            # 使用 torch.baddbmm 执行乘以扩展单位矩阵的运算，并进行比较
            out2_eye_gpu = torch.baddbmm(torch.zeros_like(A), M1, M2_eye, beta=beta, alpha=alpha)
            self.assertEqual(M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu())

        # 比较 out1_gpu 和 out2_gpu 的结果
        self.assertEqual(out1_gpu, out2_gpu[0])
f8_msg = "FP8 is only supported on H100+ and sm_89 and MI300+ devices"
# 定义消息字符串，指出FP8仅支持特定设备

if torch.version.hip:
    # 如果是使用 HIP 版本的 Torch
    e4m3_type = torch.float8_e4m3fnuz
    e5m2_type = torch.float8_e5m2fnuz
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max
else:
    # 如果不是使用 HIP 版本的 Torch
    e4m3_type = torch.float8_e4m3fn
    e5m2_type = torch.float8_e5m2
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max

# 避免在计算比例时除以零
EPS = 1e-12

def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """ Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: the float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    # 创建一个与输入张量同样大小的张量 scale，数据类型为 float32
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        # 如果 float8 数据类型是 e4m3_type，计算对应的 scale
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        # 如果 float8 数据类型是 e5m2_type，计算对应的 scale
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        # 如果 float8 数据类型不是支持的类型，抛出异常
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # 如果原始数据类型是 float16，确保 scale 在 float16 能表示的范围内
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)

    # 将计算得到的 scale 复制到 scale 张量中并返回
    scale.copy_(res)
    return scale

def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype, dim=None):
    """ Calculate the scale factor for a tensor based on its maximum absolute value.
    Args:
        x: The input tensor.
        float8_dtype: the float8 dtype.
        dim: Optional dimension along which to compute the maximum.
    """
    if dim is None:
        # 如果未指定维度，计算张量 x 的绝对值的最大值
        amax = torch.max(torch.abs(x))
    else:
        # 如果指定了维度，计算沿指定维度的绝对值的最大值
        amax = torch.max(torch.abs(x), dim=dim).values

    # 调用 amax_to_scale 函数计算并返回对应的 scale
    return amax_to_scale(amax, float8_dtype, x.dtype)

def mm_float8_emulated(x, x_scale, y, y_scale, out_dtype) -> torch.Tensor:
    """ Emulated matrix multiplication for float8 tensors.
    Args:
        x: Input tensor x.
        x_scale: Scale factor for tensor x.
        y: Input tensor y.
        y_scale: Scale factor for tensor y.
        out_dtype: Output data type.
    """
    # 将输入张量 x 和 y 转换为 float32，并按比例缩放
    x_fp32 = x.to(torch.float) / x_scale
    y_fp32 = y.to(torch.float) / y_scale
    # 执行 float32 精度下的矩阵乘法
    out_fp32 = torch.mm(x_fp32, y_fp32)

    # 将结果转换为指定的输出数据类型并返回
    return out_fp32.to(out_dtype)

def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ Unwrapped version of addmm operation for float8 tensors.
    Args:
        a_data: Input tensor a.
        a_scale: Scale factor for tensor a.
        b_data: Input tensor b.
        b_scale: Scale factor for tensor b.
        output_dtype: Output data type.
        output_scale: Scale factor for output tensor.
        bias: Optional bias tensor.
    """
    # 计算 a 和 b 张量的倒数作为调整因子
    a_inverse_scale = a_scale.reciprocal()
    b_inverse_scale = b_scale.reciprocal()
    if output_dtype == torch.float32 and bias is not None:
        # 当输出数据类型为 float32 且有偏置时，不支持 _scaled_mm 操作
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
        )
        # 添加偏置并返回结果
        output += bias
        return output
    else:
        # 调用 _scaled_mm 执行矩阵乘法操作
        output = torch._scaled_mm(
            a_data,
            b_data,
            bias=bias,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
        )
        return output
    # 返回函数的输出结果
    return output
# 定义一个函数 mm_float8，接受四个张量 a, b, a_scale, b_scale，一个输出的数据类型 output_dtype 和一个可选的输出缩放 output_scale
def mm_float8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,  # 输出数据类型
    output_scale: Optional[torch.Tensor] = None,  # 预先计算的输出缩放，可选
) -> torch.Tensor:
    # 调用 addmm_float8_unwrapped 函数，传入参数 a, a_scale, b, b_scale, output_dtype, output_scale，并返回结果
    return addmm_float8_unwrapped(
        a, a_scale, b, b_scale, output_dtype, output_scale
    )

# 定义一个函数 to_fp8_saturated，接受一个张量 x 和一个 fp8 数据类型 fp8_dtype
def to_fp8_saturated(
    x: torch.Tensor,
    fp8_dtype: torch.dtype
):
    # 如果 fp8_dtype 是 e4m3_type 类型
    if fp8_dtype == e4m3_type:
        # 对张量 x 进行截断，限制在 -E4M3_MAX_POS 到 E4M3_MAX_POS 之间
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    # 如果 fp8_dtype 是 e5m2_type 类型
    elif fp8_dtype == e5m2_type:
        # 对张量 x 进行截断，限制在 -E5M2_MAX_POS 到 E5M2_MAX_POS 之间
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        # 抛出异常，表示不支持的 fp8_dtype 类型
        raise ValueError(f"to_fp8_saturated(): Unsupported fp8_dtype: {fp8_dtype}")

    # 将张量 x 转换为 fp8_dtype 类型并返回
    return x.to(fp8_dtype)

# 使用 unittest 的 skipIf 装饰器，检查 CUDA 是否可用，若不可用则跳过测试
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
# 定义一个测试类 TestFP8MatmulCuda，继承自 TestCase
class TestFP8MatmulCuda(TestCase):

    # 使用 unittest 的 skipIf 装饰器，检查是否支持 fp8，若不支持则跳过测试，显示 f8_msg 提示信息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    # 定义一个测试方法 _test_tautological_mm，接受参数 device（设备类型，默认为 "cuda"）、x_dtype（输入张量的数据类型，默认为 e4m3_type）、y_dtype（输出张量的数据类型，默认为 e4m3_type）、out_dtype（输出张量的数据类型，默认为 None）、size（张量的大小，默认为 16）
    def _test_tautological_mm(self, device: str = "cuda",
                              x_dtype: torch.dtype = e4m3_type,
                              y_dtype: torch.dtype = e4m3_type,
                              out_dtype: Optional[torch.dtype] = None,
                              size: int = 16) -> None:
        # 在设备上生成一个随机张量 x_fp8，数据类型为 x_dtype，大小为 size × size
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        # 在设备上生成一个单位矩阵 y_fp8，数据类型为 y_dtype，并转置
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        # 计算浮点张量的矩阵乘积 out_fp32
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        # 创建一个设备上的缩放因子 scale_a 和 scale_b，值为 1.0
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        # 使用 torch._scaled_mm 函数进行浮点 8 位乘法，传入 x_fp8, y_fp8, scale_a, scale_b，并指定输出数据类型为 out_dtype
        out_fp8 = torch._scaled_mm(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        # 如果指定了 out_dtype，则断言 out_fp8 的数据类型与指定的 out_dtype 相等
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        # 断言浮点张量的矩阵乘积 out_fp32 与 out_fp8 转换为浮点类型后的结果相等
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    # 使用 unittest 的 skipIf 装饰器，检查是否支持 fp8，若不支持则跳过测试，显示 f8_msg 提示信息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    # 定义一个测试方法 test_float8_basics，接受参数 device
    def test_float8_basics(self, device):
        # 调用 _test_tautological_mm 方法，测试 e4m3_type 类型的输入和输出张量，大小为 16
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)
        # 如果当前不是 hipblaslt 版本，则测试 e4m3_type 和 e5m2_type 类型的输入张量，大小分别为 32 和 48
        if torch.version.hip is None:
            self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
            self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)
        # 根据 NVIDIA 文档，e5m2_type 类型的 8F_E5M2 矩阵乘法不支持，测试此时应抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, e5m2_type, e5m2_type)

        # 测试大小为 64 的输入张量，并指定输出数据类型为 torch.float16
        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        # 测试大小为 96 的输入张量，并指定输出数据类型为 torch.float32
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        # 如果当前不是 hipblaslt 版本，则测试大小为 80 的输入张量，并指定输出数据类型为 torch.bfloat16
        if torch.version.hip is None:
            self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)
        # 测试指定了输出数据类型为 e5m2_type 时应抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)
    def test_float8_scale(self, device) -> None:
        # 设置张量的大小为16x16
        size = (16, 16)
        # 创建一个填充了0.5的张量x，设备为指定的设备，数据类型为e4m3_type
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # 如果使用hipblaslt，则y的数据类型为e4m3_type，否则为e5m2_type
        # hipblaslt尚不支持混合e4m3_type输入
        y_type = e4m3_type if torch.version.hip else e5m2_type
        # 创建一个填充了0.5的张量y，并转置，设备为指定的设备，数据类型为y_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        # 创建设备为指定设备的标量张量scale_a和scale_b
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        # 使用torch._scaled_mm计算x和y的乘积，使用scale_a和scale_b进行缩放
        out_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        # 断言out_fp8转换为torch.float后与填充值为4的张量相等
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        # 使用torch._scaled_mm再次计算x和y的乘积，使用scale_a和scale_b进行缩放
        out_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        # 断言两次计算结果out_fp8和out_fp8_s相等
        self.assertEqual(out_fp8, out_fp8_s)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated(self, base_dtype):
        # 设定随机种子为42
        torch.manual_seed(42)
        # 输入数据类型为e4m3_type，输出数据类型为base_dtype
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        # 创建设备为cuda的base_dtype类型的随机张量x和转置后的随机张量y
        x = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        y = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        # 将x和y按照input_dtype转换为缩放张量，并转换为float类型
        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        # 将x和y乘以它们的缩放因子，然后转换为饱和的fp8格式
        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # 计算实际的F8乘积out_scaled_mm
        out_scaled_mm = mm_float8(
            x_fp8,
            y_fp8,
            a_scale=x_scale,
            b_scale=y_scale,
            output_dtype=output_dtype
        )

        # 计算模拟的F8乘积out_emulated
        out_emulated = mm_float8_emulated(
            x_fp8,
            x_scale,
            y_fp8,
            y_scale,
            output_dtype
        )

        # 如果输出数据类型与base_dtype不同，则转换输出类型为compare_type
        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        # 根据base_dtype选择不同的误差容限atol和rtol
        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        # 使用torch.testing.assert_close断言out_scaled_mm和out_emulated的接近程度
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    # 测试函数，用于测试改变步幅的 scaled_mm 函数
    def test_scaled_mm_change_stride(self, base_dtype):
        # 设置随机种子
        torch.manual_seed(42)
        # 输入数据类型为 e4m3_type
        input_dtype = e4m3_type
        # 输出数据类型为 base_dtype
        output_dtype = base_dtype
        # 用指定步幅创建一个空的张量 x，存储在 GPU 上，数据类型为 base_dtype
        x = torch.empty_strided((16, 16), (16, 1), device="cuda", dtype=base_dtype)
        # 用指定步幅创建一个空的张量 y，存储在 GPU 上，数据类型为 base_dtype
        y = torch.empty_strided((16, 32), (1, 64), device="cuda", dtype=base_dtype)

        # 将张量 x 转换为 scale，并转换为 float 类型
        x_scale = tensor_to_scale(x, input_dtype).float()
        # 将张量 y 转换为 scale，并转换为 float 类型
        y_scale = tensor_to_scale(y, input_dtype).float()

        # 将 x 乘以 x_scale 并转换为饱和的 fp8 格式
        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        # 将 y 乘以 y_scale 并转换为饱和的 fp8 格式
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # 计算实际的 float8 矩阵乘法
        out_scaled_mm = mm_float8(
            x_fp8,
            y_fp8,
            a_scale=x_scale,
            b_scale=y_scale,
            output_dtype=output_dtype
        )

        # 计算模拟的 float8 矩阵乘法
        out_emulated = mm_float8_emulated(
            x_fp8,
            x_scale,
            y_fp8,
            y_scale,
            output_dtype
        )

        # 如果输出数据类型与 base_dtype 不同，则转换结果为 compare_type
        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            # 对 out_scaled_mm 进行标准化处理
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            # 对 out_emulated 进行标准化处理
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        # 如果 base_dtype 是 torch.bfloat16 或 torch.float16，则设置容差为 7e-2，否则为 3e-3
        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        # 断言 out_scaled_mm 与 out_emulated 的近似程度
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    # 如果平台支持 float8，执行下面的测试函数；否则跳过并输出 f8_msg
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_float8_bias(self, device) -> None:
        # 定义三个维度 k、l、m
        (k, l, m) = (16, 48, 32)
        # 创建全为 1 的张量 x，存储在指定设备上，并转换为 e4m3_type 数据类型
        x = torch.ones((k, l), device=device).to(e4m3_type)
        # 创建全为 0.25 的张量 y，存储在指定设备上，数据类型为 e4m3_type，并进行转置
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        # 创建全为 4.0 的偏置张量 bias，存储在指定设备上，数据类型为 torch.half
        bias = torch.full((m,), 4.0, device=device, dtype=torch.half)
        # 创建值为 1.0 的标量张量 scale_a 和 scale_b，存储在指定设备上
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        
        # 执行 torch._scaled_mm 函数，计算 x 和 y 的乘法并输出 fp8 结果 out_fp8
        out_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        # 执行 torch._scaled_mm 函数，计算 x 和 y 的乘法，并加上偏置 bias，输出 fp8 结果 outb_fp8
        outb_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, bias=bias)
        
        # 将 out_fp8 转换为 float32 数据类型，并赋给 out_fp32
        out_fp32 = out_fp8.to(torch.float32)
        # 将 outb_fp8 转换为 float32 数据类型，并赋给 outb_fp32
        outb_fp32 = outb_fp8.to(torch.float32)
        
        # 计算 out_fp32 和 outb_fp32 之间的差值，并取绝对值
        difference = torch.abs(out_fp32 - outb_fp32)
        
        # 断言差值与张量全为 4.0 的相等性
        self.assertEqual(difference, torch.tensor(4.0, device=device).expand_as(out_fp32))
    # 测试在不可分割的前导维度上的矩阵乘法，使用给定的设备和偏置参数
    def test_non_divisible_leading_dim(self, device, bias: bool) -> None:
        # 创建一个形状为 (17, 16) 的随机张量 x，位于指定的设备上，并转换到指定的数据类型 e4m3_type
        x = torch.rand((17, 16), device=device).to(e4m3_type)
        # 创建一个形状为 (16, 16) 的随机张量 y，位于指定的设备上，并转置
        y = torch.rand((16, 16), device=device).to(e4m3_type).t()
        # 创建一个值为 1.0 的标量张量 scale_a，位于指定的设备上
        scale_a = torch.tensor(1.0, device=device)
        # 创建一个值为 1.0 的标量张量 scale_b，位于指定的设备上
        scale_b = torch.tensor(1.0, device=device)
        # 如果需要偏置，则创建一个形状为 (16,) 的随机张量 input_bias，位于指定的设备上，并转换为半精度类型
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.half)
        # 调用 torch._scaled_mm 执行矩阵乘法，接收返回值但不使用
        _ = torch._scaled_mm(x, y, scale_a, scale_b, bias=input_bias)

    # 在不支持 FP8 的平台上跳过测试，并给出相应的信息消息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_float8_bias_relu_edgecase(self, device) -> None:
        # 定义三个维度的大小 k, l, m
        (k, l, m) = (16, 48, 32)
        # 创建一个形状为 (k, l) 的全零张量 x，位于指定的设备上，并转换到指定的数据类型 e4m3_type
        x = torch.full((k, l), 0.0, device=device).to(e4m3_type)
        # 创建一个形状为 (m, l) 的全一张量 y，位于指定的设备上，并转置
        y = torch.full((m, l), 1.0, device=device, dtype=e4m3_type).t()
        # 创建一个形状为 (m,) 的全负三张量 bias，位于指定的设备上，并转换为半精度类型
        bias = torch.full((m,), -3.0, device=device, dtype=torch.half)
        # 创建一个值为 1.0 的标量张量 scale_a，位于指定的设备上
        scale_a = torch.tensor(1.0, device=device)
        # 创建一个值为 1.0 的标量张量 scale_b，位于指定的设备上
        scale_b = torch.tensor(1.0, device=device)
        # 调用 torch._scaled_mm 执行矩阵乘法，返回 FP8 精度的输出
        outb_fp8 = torch._scaled_mm(x, y, scale_a, scale_b, bias=bias)
        # 将 FP8 输出转换为单精度浮点数
        outb_fp32 = outb_fp8.to(torch.float32)
        # 使用断言确保 outb_fp32 与形状相同且值均为 -3.0 的张量相等
        self.assertEqual(outb_fp32, torch.tensor(-3.0, device=device).expand_as(outb_fp32))

    # 在不支持 FP8 的平台上跳过测试，并给出相应的信息消息
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_float32_output_errors_with_bias(self, device) -> None:
        # 定义三个维度的大小 k, l, m
        (k, l, m) = (16, 48, 32)
        # 创建一个形状为 (k, l) 的随机张量 x，位于指定的设备上，并转换到指定的数据类型 e4m3_type
        x = torch.rand((k, l), device=device).to(e4m3_type)
        # 创建一个形状为 (m, l) 的全 0.25 张量 y，位于指定的设备上，并转置
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        # 创建一个值为 1.0 的标量张量 scale_a，位于指定的设备上
        scale_a = torch.tensor(1.0, device=device)
        # 创建一个值为 1.0 的标量张量 scale_b，位于指定的设备上
        scale_b = torch.tensor(1.0, device=device)
        # 创建一个形状为 (m,) 的全 4.0 张量 bias，位于指定的设备上，并转换为 Bfloat16 类型
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        # 使用断言捕获运行时错误，确保当 out_dtype 设置为 Float32 时，不支持使用偏置
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: torch._scaled_mm(x, y, scale_a, scale_b, bias=bias, out_dtype=torch.float32),
        )

    # 如果平台支持 FP8，则跳过测试，给出相应的信息消息
    @unittest.skipIf(PLATFORM_SUPPORTS_FP8,
                     "This test is only for devices with compute capability < 8.9")
    def test_error_message_fp8_pre_sm89(self, device) -> None:
        # 定义三个维度的大小 k, l, m
        (k, l, m) = (16, 48, 32)
        # 创建一个形状为 (k, l) 的随机张量 x，位于指定的设备上，并转换到指定的数据类型 e4m3_type
        x = torch.rand((k, l), device=device).to(e4m3_type)
        # 创建一个形状为 (m, l) 的随机张量 y，位于指定的设备上，并转换到指定的数据类型 e4m3_type，并转置
        y = torch.rand((m, l), device=device).to(e4m3_type).t()
        # 创建一个值为 1.0 的标量张量 scale_a，位于指定的设备上
        scale_a = torch.tensor(1.0, device=device)
        # 创建一个值为 1.0 的标量张量 scale_b，位于指定的设备上
        scale_b = torch.tensor(1.0, device=device)
        # 使用断言捕获运行时错误，确保在不支持的设备上调用 torch._scaled_mm 时会显示正确的错误信息
        self.assertRaisesRegex(
            RuntimeError,
            r"torch\.\_scaled\_mm is only supported on CUDA devices with compute capability \>\= 9\.0 or 8\.9, or ROCm MI300\+",
            lambda: torch._scaled_mm(x, y, scale_a, scale_b, out_dtype=torch.float32),
        )
    def test_float8_scale_fast_accum(self, device) -> None:
        size = (16, 16)
        # 创建一个大小为 (16, 16) 的张量 x，每个元素值为 0.5，使用指定的设备和数据类型 e4m3_type
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # 如果使用的是 hipblaslt，则 y 的数据类型为 e4m3_type，否则为 e5m2_type
        y_type = e4m3_type if torch.version.hip else e5m2_type
        # 创建一个大小为 (16, 16) 的张量 y，每个元素值为 0.5，转置后使用指定的设备和数据类型 y_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        # 创建一个在指定设备上的标量张量 scale_a，值为 1.5
        scale_a = torch.tensor(1.5, device=device)
        # 创建一个在指定设备上的标量张量 scale_b，值为 0.66
        scale_b = torch.tensor(0.66, device=device)
        # 执行 torch._scaled_mm 操作，使用 x, y 进行矩阵乘法，并进行缩放操作，启用快速累加
        out_fp8 = torch._scaled_mm(x, y, scale_a, scale_b, use_fast_accum=True)
        # 断言输出的 out_fp8 转换为 float 后与大小为 (16, 16) 的全为 4 的张量相等
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        # 再次执行 torch._scaled_mm 操作，使用 x, y 进行矩阵乘法，使用 scale_a 和 scale_b 进行缩放，启用快速累加
        out_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, use_fast_accum=True)
        # 断言两次计算的 out_fp8 相等
        self.assertEqual(out_fp8, out_fp8_s)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or IS_WINDOWS, f8_msg)
    @skipIfRocm()
    @parametrize("use_fast_accum", [True, False])
    def test_float8_rowwise_scaling_sanity(self, device, use_fast_accum: bool) -> None:
        M, K, N = (1024, 512, 2048)
        fill_value = 0.5
        # 创建一个大小为 (M, K) 的张量 x，每个元素值为 fill_value，使用指定的设备
        x = torch.full((M, K), fill_value, device=device)
        # 创建一个大小为 (N, K) 的张量 y，每个元素值为 fill_value，使用指定的设备
        y = torch.full((N, K), fill_value, device=device)

        # 创建一个大小为 x.shape[0] 的张量 x_scales，每个元素值为 1，使用指定的设备和数据类型 torch.float32
        x_scales = torch.ones(x.shape[0], device=device, dtype=torch.float32)
        # 创建一个大小为 y.shape[0] 的张量 y_scales，每个元素值为 1，使用指定的设备和数据类型 torch.float32
        y_scales = torch.ones(y.shape[0], device=device, dtype=torch.float32)

        # 将张量 x 转换为指定的数据类型 torch.float8_e4m3fn，并赋给 x_fp8
        x_fp8 = x.to(torch.float8_e4m3fn)
        # 将张量 y 转换为指定的数据类型 torch.float8_e4m3fn 后进行转置，并赋给 y_fp8
        y_fp8 = y.to(torch.float8_e4m3fn).t()

        # 执行 torch._scaled_mm 操作，使用 x_fp8, y_fp8 进行矩阵乘法，并进行缩放操作，输出数据类型为 torch.bfloat16，启用快速累加
        out_fp8 = torch._scaled_mm(
            x_fp8,
            y_fp8,
            scale_a=x_scales,
            scale_b=y_scales,
            out_dtype=torch.bfloat16,
            use_fast_accum=use_fast_accum,
        )
        # 断言输出的 out_fp8 转换为 torch.float32 后与大小为 (M, N) 的张量，每个元素为 K * (fill_value**2)，相等
        self.assertEqual(
            out_fp8.to(torch.float32), torch.full((M, N), K * (fill_value**2), device=device)
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or IS_WINDOWS, f8_msg)
    @skipIfRocm()
    # 测试函数，用于测试 float8 错误消息处理功能
    def test_float8_error_messages(self, device) -> None:
        # 定义矩阵维度 M, K, N
        M, K, N = (1024, 512, 2048)
        # 填充值设为 0.5
        fill_value = 0.5
        # 创建填充值为 0.5 的 MxK 大小的张量 x，并移动到指定设备上
        x = torch.full((M, K), fill_value, device=device)
        # 创建填充值为 0.5 的 NxK 大小的张量 y，并移动到指定设备上
        y = torch.full((N, K), fill_value, device=device)

        # 将张量 x 转换为 float8_e4m3fn 类型的张量 x_fp8
        x_fp8 = x.to(torch.float8_e4m3fn)
        # 将张量 y 转换为 float8_e4m3fn 类型的张量，并进行转置，得到 y_fp8
        y_fp8 = y.to(torch.float8_e4m3fn).t()

        # 断言捕获 RuntimeError 异常，验证行级缩放时的错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "For row-wise scaling, scale_a must be size 1024 but got 1 and scale_b must be size 2048 but got 2",
        ):
            # 调用 torch._scaled_mm 函数，传入 x_fp8, y_fp8 和非法的 scale_a, scale_b 参数
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((), device="cuda"),
                scale_b=torch.ones((2), device="cuda"),
                out_dtype=torch.bfloat16,
            )

        # 断言捕获 RuntimeError 异常，验证行级缩放时的错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "For row-wise scaling, scale_b must have size 2048 but got 2049.",
        ):
            # 调用 torch._scaled_mm 函数，传入 x_fp8, y_fp8 和非法的 scale_a, scale_b 参数
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((M), device="cuda"),
                scale_b=torch.ones((N + 1), device="cuda"),
                out_dtype=torch.bfloat16,
            )

        # 断言捕获 RuntimeError 异常，验证 scale_a 和 scale_b 需要是一维张量的错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "Both scale_a and scale_b must be 1-dimensional tensors",
        ):
            # 调用 torch._scaled_mm 函数，传入 x_fp8, y_fp8 和非法的 scale_a, scale_b 参数
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((M), device="cuda"),
                scale_b=torch.ones((N, N), device="cuda"),
                out_dtype=torch.bfloat16,
            )

        # 断言捕获 RuntimeError 异常，验证 scale_a 和 scale_b 需要是连续张量的错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "Both scale_a and scale_b must be contiguous.",
        ):
            # 调用 torch._scaled_mm 函数，传入 x_fp8, y_fp8 和非法的 scale_a, scale_b 参数
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((M), device="cuda"),
                scale_b=torch.ones((N * 2), device="cuda")[::2],
                out_dtype=torch.bfloat16,
            )

        # 断言捕获 RuntimeError 异常，验证第二个输入必须是 float8_e4m3fn 类型的错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "For row-wise scaling the second input is required to be a float8_e4m3fn dtype.",
        ):
            # 调用 torch._scaled_mm 函数，传入 x_fp8, 非法的 y_fp8 和其它参数
            torch._scaled_mm(
                x_fp8,
                y_fp8.to(torch.float8_e5m2),
                scale_a=torch.ones((M), device="cuda"),
                scale_b=torch.ones((N), device="cuda"),
                out_dtype=torch.bfloat16,
            )

    # 如果不支持 FP8 或者在 Windows 平台上，则跳过测试
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or IS_WINDOWS, f8_msg)
    # 跳过在 ROCm 上的测试
    @skipIfRocm()
    # 参数化测试，base_dtype 只有一个取值：torch.bfloat16
    @parametrize("base_dtype", [torch.bfloat16])
    # 定义测试函数，比较缩放乘积与模拟行向量乘积的结果
    def test_scaled_mm_vs_emulated_row_wise(self, base_dtype):
        # 设置随机种子以确保结果可重复
        torch.manual_seed(42)
        # 输入数据类型为 e4m3_type
        input_dtype = e4m3_type
        # 输出数据类型为 base_dtype
        output_dtype = base_dtype

        # 创建随机张量 x 和 y，设备为 CUDA，数据类型为 base_dtype
        x = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        y = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        # 计算 x 的缩放因子并转换为浮点型
        x_scales = tensor_to_scale(x, input_dtype, dim=1).float()
        # 计算 y 的缩放因子并转换为浮点型
        y_scales = tensor_to_scale(y, input_dtype, dim=0).float()

        # 将 x 缩放并转换为饱和的浮点 8 位表示
        x_fp8 = to_fp8_saturated(x * x_scales[:, None], e4m3_type)
        # 将 y 缩放并转换为饱和的浮点 8 位表示
        y_fp8 = to_fp8_saturated(y * y_scales[None, :], e4m3_type)

        # 计算实际的浮点 8 位矩阵乘积
        out_scaled_mm = mm_float8(
            x_fp8, y_fp8, a_scale=x_scales, b_scale=y_scales, output_dtype=output_dtype
        )

        # 计算模拟的浮点 8 位矩阵乘积
        out_emulated = mm_float8_emulated(
            x_fp8, x_scales[:, None], y_fp8, y_scales[None, :], output_dtype
        )

        # 根据数据类型选择不同的绝对误差和相对误差容差
        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3

        # 断言实际浮点 8 位乘积与模拟浮点 8 位乘积的近似程度
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)
# 使用 unittest 模块中的装饰器跳过测试，如果在 ROCm 平台下，因为 ROCm 不支持 CUTLASS
# 跳过测试，如果在 Windows 平台下，因为 Windows 不支持 CUTLASS 扩展
# 跳过测试，如果不是在 SM 8.x 上，因为混合数据类型的线性运算仅在 SM 8.x 上支持
@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
@unittest.skipIf(IS_WINDOWS, "Windows doesn't support CUTLASS extensions")
@unittest.skipIf(not _IS_SM8X, "mixed dtypes linear only supported on SM 8.x")
# 定义一个测试类 TestMixedDtypesLinearCuda，继承自 TestCase 类
class TestMixedDtypesLinearCuda(TestCase):
    # 使用 dtypes 装饰器，声明测试使用的数据类型为 torch.float16 和 torch.bfloat16
    @dtypes(torch.float16, torch.bfloat16)
# 实例化设备类型的测试，应用于 TestMatmulCuda 类，全局范围内，除了 "cpu" 设备类型之外
instantiate_device_type_tests(TestMatmulCuda, globals(), except_for="cpu")
# 实例化设备类型的测试，应用于 TestFP8MatmulCuda 类，全局范围内，除了 "cpu" 设备类型之外
instantiate_device_type_tests(TestFP8MatmulCuda, globals(), except_for="cpu")
# 实例化设备类型的测试，应用于 TestMixedDtypesLinearCuda 类，全局范围内，除了 "cpu" 设备类型之外
instantiate_device_type_tests(TestMixedDtypesLinearCuda, globals(), except_for="cpu")

# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 启用 TestCase 类中的默认数据类型检查功能
    TestCase._default_dtype_check_enabled = True
    # 运行所有的测试
    run_tests()
```