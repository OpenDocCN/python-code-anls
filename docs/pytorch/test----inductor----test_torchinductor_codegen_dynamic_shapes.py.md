# `.\pytorch\test\inductor\test_torchinductor_codegen_dynamic_shapes.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的模块和库
import importlib  # 导入 importlib 库，用于动态导入模块
import os  # 导入 os 模块，提供操作系统相关功能
import sys  # 导入 sys 模块，提供系统相关的参数和功能
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 深度学习框架
from torch._inductor.compile_fx import compile_fx  # 从 torch._inductor.compile_fx 模块中导入 compile_fx 函数
from torch._inductor.test_case import TestCase  # 从 torch._inductor.test_case 模块中导入 TestCase 类
from torch.testing._internal.common_utils import (  # 从 torch.testing._internal.common_utils 导入多个变量和函数
    IS_CI,  # 是否在 CI 环境中
    IS_WINDOWS,  # 是否在 Windows 环境中
    TEST_WITH_ASAN,  # 是否使用 AddressSanitizer 进行测试
    TEST_WITH_ROCM,  # 是否在 ROCm 环境中进行测试
)
from torch.testing._internal.inductor_utils import (  # 从 torch.testing._internal.inductor_utils 导入多个函数和变量
    _check_has_dynamic_shape,  # 检查生成的 C++/Triton 代码中是否具有动态形状的函数
    GPU_TYPE,  # GPU 类型
    HAS_CPU,  # 是否有 CPU
    HAS_GPU,  # 是否有 GPU
)

# 如果在 Windows 并且在 CI 环境中，输出错误信息并退出
if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_codegen_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# 动态导入 filelock 模块
importlib.import_module("filelock")

# 将 test/ 目录下的辅助文件添加到 sys.path 中，使其可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # 从 inductor.test_torchinductor 导入多个函数和类
    CommonTemplate,  # 通用模板类
    copy_tests,  # 复制测试函数
    run_and_get_cpp_code,  # 运行并获取生成的 C++ 代码函数
    run_and_get_triton_code,  # 运行并获取 Triton 代码函数
    TestFailure,  # 测试失败类
)
from inductor.test_torchinductor_dynamic_shapes import (  # 从 inductor.test_torchinductor_dynamic_shapes 导入多个函数和变量
    make_dynamic_cls,  # 创建动态类函数
    test_failures as dynamic_shapes_test_failures,  # 动态形状测试失败
)


# 检查生成的 C++/Triton 代码中的模式，以查看是否具有动态形状
def check_codegen(
    self: TestCase,  # 测试用例对象
    model,  # 模型对象
    example_inputs,  # 示例输入数据
    kwargs=None,  # 额外参数
    *,
    is_cpp_code: bool,  # 是否为 C++ 代码
):
    kwargs = kwargs or {}  # 如果 kwargs 为 None，则设为空字典

    if is_cpp_code is False:  # 如果不是 C++ 代码
        if hasattr(model, "to"):  # 如果模型具有 to 方法
            model = model.to(device=GPU_TYPE)  # 将模型移动到指定设备

        def copy_fn(x):  # 定义复制函数，用于保留输入数据在设备上的步幅
            if not isinstance(x, torch.Tensor):  # 如果 x 不是 Tensor 对象
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device=GPU_TYPE, dtype=x.dtype
            ).copy_(x)

        example_inputs = tuple(copy_fn(x) for x in example_inputs)  # 复制示例输入数据

    torch._dynamo.reset()  # 重置动态编译器
    torch._inductor.codecache.FxGraphCache.clear()  # 清除 FX 图缓存
    torch._inductor.metrics.reset()  # 重置指标

    called = False  # 初始化 called 变量为 False

    def compile_fx_wrapper(model_, example_inputs_):  # 编译 FX 的包装器函数
        nonlocal called  # 使用 nonlocal 声明调用外部变量
        called = True  # 设置 called 为 True
        return compile_fx(model_, example_inputs_)  # 调用编译 FX 函数

    def run(*ex, **kwargs):  # 运行模型的函数
        return model(*ex, **kwargs)

    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=True)(run)  # 使用动态编译优化运行函数

    if is_cpp_code:  # 如果是 C++ 代码
        _, code = run_and_get_cpp_code(run, *example_inputs, **kwargs)  # 运行并获取生成的 C++ 代码
        _check_has_dynamic_shape(self, code)  # 检查 C++ 代码是否具有动态形状
    else:
        code = run_and_get_triton_code(run, *example_inputs, **kwargs)  # 运行并获取 Triton 代码
        self.assertTrue("def triton" in code, f"Failed to find triton kernel\n{code}")  # 断言是否找到 Triton 核心代码

    assert called, "Ran graph without calling compile_fx"  # 断言是否调用了 compile_fx 函数

    torch._dynamo.reset()  # 重置动态编译器


# 默认情况下标记为 xfail 的测试失败字典，设置 is_skip=True 可跳过
test_failures = {
    #
    # Failed to find dynamic for loop variable (no kernels generated)
    #
    "test_fft_real_input_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True  # 标记 test_fft_real_input_dynamic_shapes 为跳过测试
    ),
    # 下面是一系列测试用例，每个测试用例都用一个字符串作为键，对应一个 TestFailure 对象作为值
    "test_fft_real_input_real_output_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),
    "test_to_device_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),
    
    #
    # 下面的注释指出了无法找到动态循环变量的失败情况
    #
    "test_arange1_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange2_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange3_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange4_dynamic_shapes": TestFailure(("cpu",)),
    "test_arange6_dynamic_shapes": TestFailure(("cpu",)),
    "test_clamp_type_promotion_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv2d_channels_last_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv3d_channels_last_dynamic_shapes": TestFailure(("cpu",)),
    "test_expand_dynamic_shapes": TestFailure(("cpu",)),
    "test_full_boolean_dynamic_shapes": TestFailure(("cpu",)),
    "test_glu_dynamic_shapes": TestFailure(("cpu",)),
    "test_isinf2_dynamic_shapes": TestFailure(("cpu",)),
    "test_linspace1_dynamic_shapes": TestFailure(("cpu",)),
    "test_masked_scatter_dynamic_shapes": TestFailure(("cpu",)),
    "test_stack_dynamic_shapes": TestFailure(("cpu",)),
    "test_tensor2_dynamic_shapes": TestFailure(("cpu",)),
    "test_tensor3_dynamic_shapes": TestFailure(("cpu",)),
    "test_to_device_constant_dynamic_shapes": TestFailure("cpu"),
    "test_upsample_nearest2d_backward_dynamic_shapes": TestFailure(("cpu",)),
    "test_views3_dynamic_shapes": TestFailure(("cpu",)),
    "test_views4_dynamic_shapes": TestFailure(("cpu",)),
    "test_zeros_dynamic_shapes": TestFailure(("cpu",)),
    "test_uint_dynamic_shapes": TestFailure(("cpu",)),
    "test_issue102546_dynamic_shapes": TestFailure(("cpu",)),
    "test_repeat_as_strided_dynamic_shapes": TestFailure(("cpu",)),
    
    #
    # 下面的注释指出了找不到 for 循环或 Triton 内核的失败情况
    #
    "test_complex_fallback_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_adaptive_avg_pool2d2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_adaptive_max_pool2d2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_fractional_max_pool2d2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_argmax_to_float_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_avg_pool2d7_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_avg_pool2d_backward4_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_avg_pool3d_backward4_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_baddbmm_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_bmm2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_both_scalars_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_compar_dynamic_shapes": TestFailure(("cpu",)),
    "test_const_int32_to_float_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    "test_conv2d_backward_channels_last_dynamic_shapes": TestFailure(("cpu",)),
    "test_conv_backward_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试卷积反向传播在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_conv_functional_bn_fuse_dynamic_shapes": TestFailure(("cpu",), is_skip=True),
    # 定义一个测试用例，测试卷积功能与批量归一化融合在动态形状下的行为，仅在 CPU 上执行，并且标记为跳过测试

    "test_convolution2_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试二维卷积在动态形状下的行为，支持的设备包括 CPU

    "test_cumprod_zero_dim_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试零维累积乘积在动态形状下的行为，支持的设备包括 CPU

    "test_cumsum_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试累积求和在动态形状下的行为，支持的设备包括 CPU

    "test_cumsum_no_mask_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试没有掩码的累积求和在动态形状下的行为，支持的设备包括 CPU

    "test_cumsum_zero_dim_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试零维累积求和在动态形状下的行为，支持的设备包括 CPU

    "test_div8_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试除以8操作在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_embedding_bag_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试嵌入包在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_empty1_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试空张量1在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_empty2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试空张量2在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_empty_strided_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试空步幅张量在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_bucketize_dynamic_shapes": TestFailure("cpu"),
    # 定义一个测试用例，测试桶化操作在动态形状下的行为，支持的设备包括 CPU

    "test_bucketize_default_kwargs_dynamic_shapes": TestFailure("cpu"),
    # 定义一个测试用例，测试具有默认关键字参数的桶化操作在动态形状下的行为，支持的设备包括 CPU

    "test_bucketize_int_dynamic_shapes": TestFailure("cpu"),
    # 定义一个测试用例，测试整数类型的桶化操作在动态形状下的行为，支持的设备包括 CPU

    "test_like_rands_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试类似随机数张量在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_linspace2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试线性空间2在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_linspace3_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试线性空间3在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_logcumsumexp_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试对数累积求和指数在动态形状下的行为，支持的设备包括 CPU

    "test_logcumsumexp_zero_dim_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试零维对数累积求和指数在动态形状下的行为，支持的设备包括 CPU

    "test_max_pool2d6_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试最大池化2d6在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_max_pool2d8_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试最大池化2d8在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_max_pool2d_with_indices_backward5_dynamic_shapes": TestFailure(
        ("cpu", "cuda")
    ),
    # 定义一个测试用例，测试带有索引的最大池化反向传播5在动态形状下的行为，支持的设备包括 CPU、CUDA

    "test_max_pool2d_with_indices_backward6_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu")
    ),
    # 定义一个测试用例，测试带有索引的最大池化反向传播6在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_misaligned_address_issue1_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试地址不对齐问题1在动态形状下的行为，支持的设备包括 CPU

    "test_mm_views_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试视图乘法在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_new_empty_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试新空张量在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_new_empty_strided_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试新空步幅张量在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_new_ones_dynamic_shapes": TestFailure(("cpu",)),
    # 定义一个测试用例，测试新全1张量在动态形状下的行为，支持的设备包括 CPU

    "test_permute2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 定义一个测试用例，测试排列2在动态形状下的行为，支持的设备包括 CPU、CUDA、XPU

    "test_pointwise_airy_ai_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 定义一个测试用例，测试点对点艾里函数Ai在动态形状下的行为，支持的设备包括 CUDA、XPU

    "test_pointwise_digamma_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 定义一个测试用例，测试点对点Ψ函数在动态形状下的行为，支持的设备包括 CUDA、XPU

    "test_pointwise_gammainc_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 定义一个测试用例，测试点对点不完全伽马函数在动态形状下的行为，
    # 创建一个名为 "test_pointwise_psi_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_psi_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_pointwise_scaled_modified_bessel_k0_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_scaled_modified_bessel_k0_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_scaled_modified_bessel_k1_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_scaled_modified_bessel_k1_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_spherical_bessel_j0_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_spherical_bessel_j0_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_pointwise_zeta_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_zeta_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_pointwise_chebyshev_polynomial_t_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_chebyshev_polynomial_t_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_chebyshev_polynomial_u_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_chebyshev_polynomial_u_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_chebyshev_polynomial_v_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_chebyshev_polynomial_v_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_chebyshev_polynomial_w_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_chebyshev_polynomial_w_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_shifted_chebyshev_polynomial_t_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_shifted_chebyshev_polynomial_t_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_shifted_chebyshev_polynomial_u_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_shifted_chebyshev_polynomial_u_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_shifted_chebyshev_polynomial_v_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_shifted_chebyshev_polynomial_v_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_shifted_chebyshev_polynomial_w_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_shifted_chebyshev_polynomial_w_dynamic_shapes": TestFailure(
        ("cuda", "xpu")
    ),
    # 创建一个名为 "test_pointwise_hermite_polynomial_h_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_hermite_polynomial_h_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_pointwise_hermite_polynomial_he_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_hermite_polynomial_he_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_pointwise_laguerre_polynomial_l_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_laguerre_polynomial_l_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_pointwise_legendre_polynomial_p_dynamic_shapes" 的测试失败对象，指定在 CUDA 和 XPU 上执行失败
    "test_pointwise_legendre_polynomial_p_dynamic_shapes": TestFailure(("cuda", "xpu")),
    # 创建一个名为 "test_polar_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败，并且标记为跳过
    "test_polar_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),
    # 创建一个名为 "test_randn_generator_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_randn_generator_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_randn_like_empty_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_randn_like_empty_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_single_elem_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_single_elem_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_single_elem_indirect_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_single_elem_indirect_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_sort_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_sort_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_sort_stable_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_sort_stable_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_sort_transpose_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_sort_transpose_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_split_cumsum_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_split_cumsum_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_split_cumsum_low_prec_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_split_cumsum_low_prec_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_split_cumprod_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_split_cumprod_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_split_cumprod_low_prec_dynamic_shapes" 的测试失败对象，指定在 CPU 上执行失败
    "test_split_cumprod_low_prec_dynamic_shapes": TestFailure(("cpu",)),
    # 创建一个名为 "test_split_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_split_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_topk_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_topk_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_unbind_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_unbind_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_views5_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_views5_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_view_detach_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_view_detach_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个名为 "test_view_on_aliased_dynamic_shapes" 的测试失败对象，指定在 CPU、CUDA 和 XPU 上执行失败
    "test_view_on_aliased_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),
    # 创建一个字典，其中包含测试名称和对应的 TestFailure 对象，用于测试失败的记录
    {
        "test_linear_float64_dynamic_shapes": TestFailure("cpu"),  # 测试线性浮点数动态形状，使用 CPU
        "test_adaptive_avg_pool_with_output_size_0_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu")  # 测试自适应平均池化输出大小为0的动态形状，支持 CPU、CUDA、XPU
        ),
        "test_zero_element_mutation_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),  # 测试零元素突变动态形状，支持 CPU、CUDA、XPU
        "test_custom_op_3_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),  # 测试自定义操作3的动态形状，支持 CPU、CUDA、XPU
        "test_custom_op_fixed_layout_sequential_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu")  # 测试固定布局顺序的自定义操作的动态形状，支持 CPU、CUDA、XPU
        ),
        "test_cat_uint8_dynamic_shapes": TestFailure(
            ("cpu",)  # 测试 uint8 输入上的 cat 操作，在 CPU 上使用 aten 回退
        ),
        #
        # 未使用 'common' 或直接调用 'assertEqual' 的测试：
        #
        "test_arange5_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 测试 arange5 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        "test_cat_inplace_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试 inplace cat 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_cat_of_loops_and_extern_kernel_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试循环和外部内核的 cat 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        # 需要启用动态形状的 CL
        "test_scaled_dot_product_efficient_attention_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试缩放点积效率注意力的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_dropout_deterministic_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试确定性 dropout 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_dropout_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 测试 dropout 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        "test_dtype_mismatch_issue_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试数据类型不匹配问题的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_forced_buffer_realize_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试强制缓冲区实现的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_tmp_not_defined_issue3_dynamic_shapes": TestFailure(("cpu",), is_skip=True),  # 测试 tmp 未定义问题3的动态形状，支持 CPU，标记为跳过
        "test_gather2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 测试 gather2 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        "test_inplace_add_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试 inplace add 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_inplace_mixed_dtype_ops_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试混合数据类型 inplace 操作的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_input_mutation1_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试输入突变1的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_input_mutation2_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试输入突变2的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_input_mutation3_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试输入突变3的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_input_mutation4_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试输入突变4的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_kernel_names_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试内核名称的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_lerp_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 测试 lerp 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        "test_linear_buffer_reuse_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试线性缓冲区重用的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_list_clearing_dynamic_shapes": TestFailure(
            ("cpu", "cuda", "xpu"), is_skip=True  # 测试清空列表的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        ),
        "test_dropout2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 测试 dropout2 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
        "test_dropout3_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 测试 dropout3 的动态形状，支持 CPU、CUDA、XPU，标记为跳过
    }
    "test_masked_fill_promotion_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_min_max_reduction_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_multi_gpu_recompile_on_index_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_output_strides_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_pow3_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu"), is_skip=True),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_profiler_mark_wrapper_call_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_rand_like_deterministic_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_repeat_interleave_2_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),  # 标记用于测试的函数失败案例：测试在动态形状下执行，支持的设备为CPU、CUDA和XPU。

    "test_slice_mutation2_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_strided_inputs_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_transposed_propagates_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_require_stride_expanded_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_unspec_inputs_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_zero_dim_reductions_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    "test_sdpa_dynamic_shapes": TestFailure(("cpu",), is_skip=True),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU。

    "test_sdpa_unaligned_mask_dynamic_shapes": TestFailure(("cpu",), is_skip=True),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU。

    #
    # The following tests do not support dynamic shapes yet:
    #

    "test_cudnn_rnn_dynamic_shapes": TestFailure(("cuda",)),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CUDA。

    # test_roi_align uses torchvision, which doesn't work with dynamic shapes
    "test_roi_align_dynamic_shapes": TestFailure(("cpu", "cuda", "xpu")),  # 标记用于测试的函数失败案例：测试在动态形状下执行，支持的设备为CPU、CUDA和XPU。

    "test_aliased_buffer_reuse_dynamic_shapes": TestFailure(("cpu",)),  # 标记用于测试的函数失败案例：测试在动态形状下执行，支持的设备为CPU。

    # The input of this case has only 1 elements
    "test_mutations_loop_fusion_dynamic_shapes": TestFailure(
        ("cpu", "cuda", "xpu"), is_skip=True
    ),  # 标记用于测试的函数失败案例：测试在动态形状下执行时跳过，支持的设备为CPU、CUDA和XPU。

    # Refinement means we don't actually generate dynamic shapes (but only on
    # cpu apparently?!)
    "test_nonzero_unbacked_refinement_dynamic_shapes": TestFailure(("cpu",)),  # 标记用于测试的函数失败案例：测试在动态形状下执行，支持的设备为CPU。

    **dynamic_shapes_test_failures,
}

`
# 判断是否需要在测试中使用 ROCm
if TEST_WITH_ROCM:
    # 更新测试失败字典，添加多个测试用例及其预期失败的设备
    test_failures.update(
        {
            "test_split_cumsum_dynamic_shapes": TestFailure(("cpu", "cuda")),
            "test_split_cumsum_low_prec_dynamic_shapes": TestFailure(("cpu", "cuda")),
            "test_split_cumprod_dynamic_shapes": TestFailure(("cpu", "cuda")),
            "test_split_cumprod_low_prec_dynamic_shapes": TestFailure(("cpu", "cuda")),
        }
    )

# 使用 make_dynamic_cls 函数创建一个新的类，继承自 CommonTemplate，指定属性为 "_expected_failure_codegen_dynamic"
DynamicShapesCodegenCommonTemplate = make_dynamic_cls(
    CommonTemplate, xfail_prop="_expected_failure_codegen_dynamic"
)

# 判断是否有 CPU 支持
if HAS_CPU:

    # 定义一个测试类，继承自 TestCase，表示 CPU 设备的动态形状代码生成测试
    class DynamicShapesCodegenCpuTests(TestCase):
        maxDiff = None  # 设置最大差异为 None，表示不限制输出差异
        device = "cpu"  # 设置测试设备为 CPU

        # 定义一个通用测试方法
        def common(self: TestCase, model, example_inputs, kwargs=None, **_rest):
            # 调用 check_codegen 函数进行代码生成检查，设置 is_cpp_code 为 True
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=True,
            )

    # 将测试模板拷贝到 CPU 测试类中
    copy_tests(
        DynamicShapesCodegenCommonTemplate,
        DynamicShapesCodegenCpuTests,
        "cpu",
        test_failures,
    )

# 判断是否有 GPU 支持且不使用 ASAN
if HAS_GPU and not TEST_WITH_ASAN:

    # 定义一个测试类，继承自 TestCase，表示 GPU 设备的动态形状代码生成测试
    class DynamicShapesCodegenGPUTests(TestCase):
        maxDiff = None  # 设置最大差异为 None，表示不限制输出差异
        device = GPU_TYPE  # 设置测试设备为 GPU 类型

        # 定义一个通用测试方法
        def common(self: TestCase, model, example_inputs, kwargs=None, **_rest):
            # 调用 check_codegen 函数进行代码生成检查，设置 is_cpp_code 为 False
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=False,
            )

    # 将测试模板拷贝到 GPU 测试类中
    copy_tests(
        DynamicShapesCodegenCommonTemplate,
        DynamicShapesCodegenGPUTests,
        GPU_TYPE,
        test_failures,
    )

# 如果脚本作为主程序执行
if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # 判断是否有 CPU 或 GPU 支持，并运行测试，需文件锁支持
    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
```