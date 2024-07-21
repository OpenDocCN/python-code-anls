# `.\pytorch\test\inductor\test_cutlass_backend.py`

```py
# 导入所需的库和模块
# 作者：["module: inductor"]
import logging  # 导入日志记录模块
import os  # 导入操作系统相关功能的模块
import unittest  # 导入单元测试框架
from typing import Callable, List, Optional  # 导入类型提示相关模块

import torch  # 导入PyTorch深度学习框架
from torch._dynamo.utils import counters  # 导入计数器工具
from torch._inductor import config  # 导入激励器配置相关模块
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller  # 导入CUDA模板调用器
from torch._inductor.codegen.cuda.cutlass_utils import get_max_alignment  # 导入CUDA相关工具函数
from torch._inductor.ir import ChoiceCaller, FixedLayout  # 导入激励器中间表示和布局相关模块
from torch._inductor.select_algorithm import NoValidChoicesError  # 导入选择算法异常类
from torch._inductor.test_case import run_tests, TestCase  # 导入测试运行和测试案例类
from torch._inductor.utils import fresh_inductor_cache  # 导入激励器缓存刷新工具函数
from torch.testing._internal.common_cuda import SM75OrLater, SM80OrLater, SM90OrLater  # 导入CUDA测试相关条件
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入参数化测试实例化函数
    parametrize,  # 导入参数化装饰器
)

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 导入测试工具中的CPU和CUDA判断标志

# 设置浮点数矩阵乘法的精度为高精度
torch.set_float32_matmul_precision("high")

# 如果有CUDA支持，则设置CUDA内存分配器的参数
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

# 设置Cutlass目录路径为相对路径
_CUTLASS_DIR = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass/")

# 设置日志记录器
log = logging.getLogger(__name__)

# 根据当前环境设置CUDA标志，排除HIP版本的Torch
HAS_CUDA = HAS_CUDA and not torch.version.hip
SM75OrLater = SM75OrLater and not torch.version.hip
SM80OrLater = SM80OrLater and not torch.version.hip
SM90OrLater = SM90OrLater and not torch.version.hip


def _get_path_without_sccache() -> str:
    """
    获取不包含sccache的PATH环境变量。
    """
    # 获取当前的PATH环境变量，并排除包含"/opt/cache/bin"路径的部分
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


@instantiate_parametrized_tests
class TestCutlassBackend(TestCase):
    def setUp(self):
        """
        在调用父类setUp()之前，禁用自动缓存刷新机制。
        这是为了解决自动调优过程中与持久子进程的交互问题。
        """
        # 保存旧的禁用缓存刷新的环境变量
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )
        try:
            # 设置环境变量禁用缓存刷新
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            # 调用父类的setUp()方法
            super().setUp()
        finally:
            # 恢复旧的禁用缓存刷新的环境变量设置
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = old_disable_fresh_cache_envvar
        # 设置随机数种子为1234
        torch.random.manual_seed(1234)

    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    # 定义一个测试函数，用于验证 Cutlass GEMM 阈值的工作方式
    def test_max_autotune_cutlass_threshold(self):
        """
        Make sure Cutlass GEMM threshold works as intended.
        """

        # 如果当前使用的是 HIP 版本的 Torch，则直接返回，不进行测试
        if torch.version.hip:
            return

        # 设置 CUDA 后端矩阵乘法的允许 FP16 降低精度减少
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义一个矩阵乘法函数 mm(a, b)，返回结果
        def mm(a, b):
            return a @ b

        # 创建两个随机初始化的 CUDA 半精度矩阵 a 和 b
        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(10, 100).cuda().half()

        # 使用 config.patch 上下文管理器设置一系列 CUDA 编译和自动调优的参数
        with config.patch(
            {
                "max_autotune": True,  # 启用最大自动调优
                "autotune_in_subproc": True,  # 在子进程中进行自动调优
                "max_autotune_gemm_backends": "CUTLASS,ATen",  # 指定自动调优使用的后端：Cutlass 和 ATen
                "compile_threads": 4,  # 编译线程数
                "cuda.cutlass_backend_min_gemm_size": 100000,  # Cutlass 后端的最小 GEMM 大小阈值
                "cuda.cutlass_dir": _CUTLASS_DIR,  # Cutlass 库的路径
                "cuda.cutlass_max_profiling_configs": 2,  # 最大的 Cutlass 预配置数量
            }
        ):
            # 导入 CUDA 模板调用器
            from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

            # 使用 unittest.mock.patch 临时替换 autotune_select_algorithm 函数
            with mock.patch(
                "torch._inductor.select_algorithm.autotune_select_algorithm"
            ) as mocked_select_algorithm:
                # 编译矩阵乘法函数 mm(a, b)，并关闭动态编译
                Y_compiled = torch.compile(mm, dynamic=False)(a, b)
                # 直接调用矩阵乘法函数 mm(a, b)，获得结果 Y
                Y = mm(a, b)
                # 获取 autotune_select_algorithm 函数的第一个参数 passed_choice_callers
                passed_choice_callers: List[ChoiceCaller] = mocked_select_algorithm[0][
                    1
                ]
                # 断言 passed_choice_callers 列表中的所有元素都是 ChoiceCaller 实例
                assert all(
                    isinstance(cc, ChoiceCaller) for cc in passed_choice_callers
                ), "Argument 1 to autotune_select_algorithm should be a list of ChoiceCaller instances"
                # 断言所有传递给 autotune_select_algorithm 的 ChoiceCaller 实例不包含 CUDATemplateCaller
                assert all(
                    not isinstance(cc, CUDATemplateCaller)
                    for cc in passed_choice_callers
                ), "Cutlass Kernels should have been filtered, GEMM size is too small"
            # 使用 torch.testing.assert_close 检查 Y_compiled 和 Y 的近似相等性
            torch.testing.assert_close(Y_compiled, Y)

    # 使用 unittest.skipIf 装饰器，如果不满足 SM75OrLater 条件，则跳过测试
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    # 使用 unittest.skipIf 装饰器，如果运行在 fbcode 环境中，则跳过测试
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # 使用 unittest.mock.patch.dict 修改 os.environ 字典，设置 PATH 环境变量为 _get_path_without_sccache() 返回的路径
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_precompile(self):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        # 如果是在 HIP 环境下，则直接返回，不执行后续代码
        if torch.version.hip:
            return

        # 禁止 CUDA 矩阵乘法运算中的 FP16 降精度优化
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义矩阵乘法函数 mm
        def mm(a, b):
            return a @ b

        # 生成随机数据矩阵 a 和 b，并在 CUDA 上运行，使用半精度浮点数
        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(10, 100).cuda().half()

        # 使用配置 patch，设置自动调优、在子进程中自动调优、选择 GEMM 后端为 CUTLASS、Triton、ATen、编译线程数为 4
        # 设置 CUTLASS 相关目录和最大性能配置数
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS,Triton,ATen",
                "compile_threads": 4,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            # 编译 mm 函数，并执行矩阵乘法，动态模式关闭
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            # 直接执行 mm 函数的矩阵乘法
            Y = mm(a, b)
            # 断言编译后的结果 Y_compiled 和直接执行的结果 Y 在数值上的接近性
            torch.testing.assert_close(Y_compiled, Y)

    # TODO: Enable dynamic test cases when dynamic support is added.
    # 根据条件跳过测试用例，条件为不支持 SM75 或者处于 FBCode 环境中
    @unittest.skipIf(not SM75OrLater, "need sm_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # 参数化测试函数，dynamic 参数为 False 和 True 两种情况，max_autotune_gemm_backends 参数为 "CUTLASS" 和 "ATen,Triton,CUTLASS" 两种情况
    @parametrize("dynamic", (False, True))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    # 使用 mock.patch.dict 修改环境变量，设置 PATH 为不包含 sccache 的路径
    def test_max_autotune_cutlass_backend_regular_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        # 如果 max_autotune_gemm_backends 为 "CUTLASS" 并且处于 HIP 环境下，则直接返回，不执行后续代码
        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        # 禁止 CUDA 矩阵乘法运算中的 FP16 降精度优化
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义矩阵乘法函数 mm
        def mm(a, b):
            return a @ b

        # 生成随机数据矩阵 a 和 b，并在 CUDA 上运行，使用半精度浮点数
        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(16, 128).cuda().half()

        # 使用配置 patch，设置自动调优、不在子进程中自动调优、根据 max_autotune_gemm_backends 选择对应后端
        # 设置 CUTLASS 相关目录和最大性能配置数
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            # 编译 mm 函数，并执行矩阵乘法，dynamic 模式由参数决定
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            # 直接执行 mm 函数的矩阵乘法
            Y = mm(a, b)
            # 断言编译后的结果 Y_compiled 和直接执行的结果 Y 在数值上的接近性
            torch.testing.assert_close(Y_compiled, Y)

    # 根据条件跳过测试用例，条件为不支持 SM90 或者处于 FBCode 环境中
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # 使用 mock.patch.dict 修改环境变量，设置 PATH 为不包含 sccache 的路径
    def test_max_autotune_cutlass_backend_regular_mm_streamk(
        self, dynamic: bool = False, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        # 函数内容同上，略过重复注释
        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b):
            return a @ b

        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(16, 128).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        确保在子进程中自动调整矩阵乘法时不会崩溃。
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # 禁用 CUDA 中的 FP16 减少精度优化

        def mm(a, b):
            return a @ b
        # 定义矩阵乘法函数 mm，计算两个张量的矩阵乘积

        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(16, 128).cuda().half()
        # 生成随机张量 a 和 b，转移到 CUDA 设备并使用半精度表示

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.cutlass_op_allowlist_regex": "stream_k",  # only stream-k GEMM Kernels
            }
        ):
            # 使用指定配置进行上下文管理，用于自动调整矩阵乘法算法

            for M, K, N in (
                (128, 16, 128),
                (1024, 256, 1024),
                (
                    16384,
                    1024,
                    16384,
                ),
                (
                    16384,
                    1408,
                    16384,
                ),
            ):
                # 针对不同的矩阵大小 M, K, N 进行迭代测试

                a = torch.randn(M, K).cuda().half()
                b = torch.randn(K, N).cuda().half()
                # 重新生成随机张量 a 和 b，转移到 CUDA 设备并使用半精度表示

                Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
                Y = mm(a, b)
                # 使用动态编译执行矩阵乘法 mm，计算 Y_compiled 和 Y

                # 由于涉及的矩阵乘法规模巨大，我们需要放宽数值限制。
                # 许多小的加法差异会累积起来。
                torch.testing.assert_close(Y_compiled, Y, atol=0.01, rtol=0.01)
                # 断言 Y_compiled 和 Y 在指定的数值容差范围内相等
    ):
        # 如果需要使用混合精度，则允许 CUDA 矩阵乘法使用降低的 FP16 精度
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            mixed_precision
        )

        # 注意：可用的操作还取决于形状的对齐
        # 如果这些形状不都至少对齐到8个元素，
        # 可能没有可用的 Cutlass 3.x 操作允许融合
        if batch_size is None:
            # 生成一个大小为256x32的随机张量，并移至CUDA设备
            a = torch.randn(256, 32).cuda()
            b = torch.randn(32, 256).cuda()
        else:
            # 根据给定的批大小生成随机张量，并移至CUDA设备
            a = torch.randn(batch_size, 256, 32).cuda()
            b = torch.randn(batch_size, 32, 256).cuda()
        if fp16:
            # 如果指定使用FP16精度，则将张量转换为半精度浮点数
            a = a.half()
            b = b.half()

        # 使用指定的配置上下文进行设置
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 4,
                "cuda.version": "12.2",  # 需要启用我们需要的内核
            }
        ):
            # 将 cuda_epilogue_fusion_counter 设置为零
            counters["inductor"]["cuda_epilogue_fusion_counter"] = 0
            # 使用动态编译方式编译矩阵乘法，并执行计算
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            # 直接执行矩阵乘法计算
            Y = mm(a, b)
            # 检查实际的融合计数是否与预期的一致
            actual_count = counters["inductor"]["cuda_epilogue_fusion_counter"]
            assert (
                actual_count == expected_fuse_count
            ), f"Expected fuse count of {expected_fuse_count} but got {actual_count}"
            # 使用指定的容差检查编译后的结果与直接计算结果的一致性
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not SM90OrLater, "需要 sm_90")
    @unittest.skipIf(torch.version.hip, "不支持 HIP")
    @unittest.skipIf(config.is_fbcode(), "fbcode 需要不同的 CUTLASS 路径设置")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16(self):
        def mm(a, b):
            return (a @ b) * 3.0

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 函数进行测试
        # 使用单精度浮点数而不是混合精度，预期的融合计数为零
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "需要 sm_90")
    @unittest.skipIf(torch.version.hip, "不支持 HIP")
    @unittest.skipIf(config.is_fbcode(), "fbcode 需要不同的 CUTLASS 路径设置")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.0

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 函数进行测试
        # 使用混合精度，预期的融合计数为零
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "需要 sm_90")
    @unittest.skipIf(torch.version.hip, "不支持 HIP")
    @unittest.skipIf(config.is_fbcode(), "fbcode 需要不同的 CUTLASS 路径设置")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 方法，测试混合精度为 False，使用 FP16，期望融合计数为 0，传入自定义的矩阵乘法函数 mm
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 方法，测试混合精度为 True，使用 FP16，期望融合计数为 0，传入自定义的矩阵乘法函数 mm
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 方法，测试混合精度为 False，使用 FP16，期望融合计数为 0，传入自定义的矩阵乘法函数 mm
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=False, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 方法，测试混合精度为 True，使用 FP16，期望融合计数为 0，传入自定义的矩阵乘法函数 mm
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_relu6_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.clamp(torch.nn.functional.relu(a @ b), max=6.0)

        # 调用 _test_max_autotune_cutlass_backend_epilogue_fusion 方法，测试混合精度为 True，使用 FP16，期望融合计数为 0，传入自定义的矩阵乘法函数 mm
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(torch.version.hip, "HIP not supported")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    def test_max_autotune_cutlass_backend_no_fusion_dtype_mismatch(self):
        def mm(a, b):
            # 不应该融合，因为输出的数据类型与矩阵乘法的数据类型不同
            return (a @ b).to(torch.float32) * 0.00001

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    def test_max_autotune_cutlass_backend_simple_bmm(self):
        def bmm(a, b):
            return torch.bmm(a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(  # 测试 bmm
            mixed_precision=False,
            fp16=True,
            expected_fuse_count=0,
            mm=bmm,
            batch_size=10,
        )

    @unittest.skipIf(not SM90OrLater, "需要 SM_90")
    @unittest.skipIf(torch.version.hip, "不支持 HIP")
    @unittest.skipIf(config.is_fbcode(), "fbcode 需要不同的 CUTLASS 路径设置")
    def test_max_autotune_cutlass_backend_shape_dependent_normalization_fusion(self):
        def mm(a, b):
            return (a @ b) / b.size(1)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            mixed_precision=True, fp16=True, expected_fuse_count=0, mm=mm
        )

    # TODO: 当动态支持被添加时启用动态测试用例
    @unittest.skipIf(not SM75OrLater, "需要 SM_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode 需要不同的 CUTLASS 路径设置")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_mm_bias(
        self, dynamic: bool = False, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        """
        确保在子进程中自动调优矩阵乘法（mm）不会崩溃。
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b, bias):
            return torch.nn.functional.linear(a, b, bias)

        a = torch.randn(2048, 4096).cuda().half()
        bias = torch.randn(2048).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            Y = mm(a, a, bias)
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, a, bias)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-1, rtol=1e-1)

    @unittest.skipIf(not SM75OrLater, "需要 SM_75")
    @unittest.skipIf(config.is_fbcode(), "fbcode 需要不同的 CUTLASS 路径设置")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm(
        self, dynamic, max_autotune_gemm_backends
    ):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        # 禁止使用 FP16 降低精度的功能
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义 addmm 函数，使用 torch.addmm 执行矩阵乘加运算
        def addmm(x, a, b, alpha, beta):
            return torch.addmm(x, a, b, alpha=alpha, beta=beta)

        # 定义 compare_results 函数，用于比较预期结果和编译函数的结果
        def compare_results(
            m: int, k: int, n: int, alpha: float, beta: float, x_shape: List[int]
        ) -> None:
            # 创建并在 GPU 上分配随机数据张量 x，数据类型为半精度浮点数
            x = torch.randn(x_shape).cuda().half()
            # 创建并在 GPU 上分配随机数据张量 a，数据类型为半精度浮点数
            a = torch.randn(m, k).cuda().half()
            # 创建并在 GPU 上分配随机数据张量 b，数据类型为半精度浮点数
            b = torch.randn(k, n).cuda().half()
            # 使用 addmm 函数计算预期的结果 y_expected
            y_expected = addmm(x, a, b, alpha, beta)

            # 编译 addmm 函数，并使用动态参数进行编译
            compiled_fn = torch.compile(addmm, dynamic=dynamic)
            # 使用编译后的函数计算结果 y
            y = compiled_fn(x, a, b, alpha, beta)
            # 使用 torch.testing.assert_close 检验 y 和 y_expected 的接近程度
            torch.testing.assert_close(y, y_expected)

        # 使用 config.patch 上下文管理器设置 CUDA 自动调优相关的配置
        with config.patch(
            {
                "max_autotune": True,
                # 针对此示例，部分 Cutlass 内核在 IMA 下会失败，导致不可恢复的 CUDA 错误，需要在子进程中进行调优
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 4,
                "cuda.cutlass_op_allowlist_regex": "",
                "cuda.cutlass_op_denylist_regex": "pingpong",  # Pingpong 内核可能导致数值问题
            }
        ):
            # 不进行广播操作，运行 compare_results 函数
            compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 2048])
            # 第一维度进行广播操作，运行 compare_results 函数
            compare_results(4096, 25728, 2048, 2.0, 0.4, [2048])
            # 最后一维度进行广播操作，运行 compare_results 函数
            compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 1])

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "CUTLASS,ATen"))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_int_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        if "CUTLASS" in max_autotune_gemm_backends.upper() and torch.version.hip:
            return

        def mm(a, b):
            return torch._int_mm(a, b)

        # CUTLASS only supports row-major/column-major combination of
        # layouts for this operation, thus the transpose of tensor b
        # (on the other side, Triton at the moment doesn't support
        # this combination, so it's excluded from the test).  Also,
        # for CUTLASS alignment requirements, number of columns in
        # both tensors has to be divisible by 16.

        # 生成随机整数张量 a，形状为 (100, 16)，数据类型为 torch.int8，在 GPU 上分配内存
        a = torch.randint(0, 5, (100, 16), dtype=torch.int8).cuda()
        # 生成随机整数张量 b，形状为 (32, 16)，数据类型为 torch.int8，在 GPU 上分配内存，并对其进行转置
        b = torch.randint(0, 5, (32, 16), dtype=torch.int8).cuda().T

        # 使用 config.patch 上下文管理器设置一系列配置项，用于测试自动调优的最大参数
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            # 编译并执行 mm 函数，使用动态计算（如果支持），传入张量 a 和转置后的张量 b
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            # 直接执行 mm 函数，传入张量 a 和转置后的张量 b
            Y = mm(a, b)
            # 使用 torch.testing.assert_close 检查 Y_compiled 和 Y 的近似相等性
            torch.testing.assert_close(Y_compiled, Y)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "CUTLASS,Triton,ATen"))
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    # 测试函数，测试最大自动调优下 CUTLASS 后端混合矩阵乘法的行为
    def test_max_autotune_cutlass_backend_mixed_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        确保在子进程中自动调整矩阵乘法操作不会崩溃。
        """

        if max_autotune_gemm_backends == "CUTLASS" and torch.version.hip:
            return

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b):
            return torch.mm(a, b.to(torch.half))

        # CUTLASS only supports row-major/column-major combination of
        # layouts for this operation, thus the transpose of tensor b.
        # Also, for CUTLASS alignment requirements, number of columns
        # of the first tensor has to be divisible by 16.
        # CUTLASS 只支持此操作的行主/列主布局组合，因此需要对张量 b 进行转置。
        # 另外，对于 CUTLASS 的对齐要求，第一个张量的列数必须是 16 的倍数。
        a = torch.randn(100, 16).cuda().half()
        b = torch.randint(0, 5, (100, 16), dtype=torch.int8).cuda().T

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_dir": _CUTLASS_DIR,
                "cuda.cutlass_max_profiling_configs": 2,
                "use_mixed_mm": True,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_op_denylist(
        self,
    ):
        # 定义一个函数，使用 torch.addmm 实现矩阵乘法加法运算
        def my_addmm(x, a, b, alpha, beta):
            return torch.addmm(x, a, b, alpha=beta, beta=alpha)

        # 生成一个大小为 (128, 128) 的随机张量，将其移至 GPU 并使用 half 精度
        x = torch.randn((128, 128)).cuda().half()
        # 生成一个大小为 (128, 128) 的随机张量 a，将其移至 GPU 并使用 half 精度
        a = torch.randn(128, 128).cuda().half()
        # 生成一个大小为 (128, 128) 的随机张量 b，将其移至 GPU 并使用 half 精度
        b = torch.randn(128, 128).cuda().half()

        # 定义一个函数，用于在未选择算法时引发 NoValidChoicesError 异常
        def select_no_algorithm(*args, **kwargs):
            raise NoValidChoicesError

        # 使用 fresh_inductor_cache 上下文管理器
        with fresh_inductor_cache():
            # 使用 config.patch 上下文管理器来配置测试参数
            with config.patch(
                {
                    "max_autotune": True,
                    # 在此示例中，某些 Cutlass 内核在使用 IMA 时会失败，导致不可恢复的 CUDA 错误
                    # 因此需要在此处调整子进程中的自动调整
                    "autotune_in_subproc": False,
                    "max_autotune_gemm_backends": "CUTLASS,ATen",
                    "cuda.cutlass_dir": _CUTLASS_DIR,
                    "cuda.cutlass_max_profiling_configs": 2,
                    "cuda.cutlass_op_allowlist_regex": "",
                    "cuda.cutlass_op_denylist_regex": "pingpong",  # Pingpong 内核可能导致数值问题
                }
            ):
                # 使用 mock.patch 上下文管理器来模拟选择算法函数
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
                    # 编译 my_addmm 函数，禁用动态模式
                    torch.compile(my_addmm, dynamic=False)(x, a, b, 1.0, 2.0)
                    # 检查选择算法函数的调用参数
                    args, kwargs = sa.call_args
                    op_name, choices, _, __ = args
                    # 断言 op_name 应为 "addmm"
                    assert op_name == "addmm"
                    cuda_template_count = 0
                    # 遍历选择列表，检查是否有 CUDATemplateCaller 实例
                    for choice in choices:
                        if isinstance(choice, CUDATemplateCaller):
                            choice_info = choice.info_dict()
                            # 断言所有 pingpong 内核已被过滤
                            assert (
                                "pingpong" not in choice_info["op_conf_name"]
                            ), "All pingpong Kernels should have been filtered"
                            cuda_template_count += 1
                    # 断言至少存在一个 CUDATemplateCaller 实例
                    assert cuda_template_count > 0, "No CUDATemplateCaller choices"

    # 如果不支持 SM90 或更高版本，则跳过测试
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    # 如果运行环境为 fbcode，则需要不同的 CUTLASS 路径设置，因此跳过测试
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # 使用 unittest.mock.patch.dict 来修改 PATH 环境变量，移除 sccache 路径
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    # 定义测试函数 test_cutlass_backend_op_allowlist
    def test_cutlass_backend_op_allowlist(
        self,
        ):
            # 定义一个函数 addmm，用于执行 torch 中的矩阵乘法加法操作
            def addmm(x, a, b, alpha, beta):
                return torch.addmm(x, a, b, alpha=alpha, beta=beta)

            # 生成一个大小为 (128, 128) 的随机张量并移到 GPU 上，并使用半精度数据类型
            x = torch.randn((128, 128)).cuda().half()
            # 生成一个大小为 (128, 128) 的随机张量并移到 GPU 上，并使用半精度数据类型
            a = torch.randn(128, 128).cuda().half()
            # 生成一个大小为 (128, 128) 的随机张量并移到 GPU 上，并使用半精度数据类型
            b = torch.randn(128, 128).cuda().half()

            # 定义一个函数 select_no_algorithm，用于在自动调优失败时引发 NoValidChoicesError 异常
            def select_no_algorithm(*args, **kwargs):
                raise NoValidChoicesError

            # 在使用 fresh_inductor_cache 上下文管理器的环境中
            with fresh_inductor_cache():
                # 在使用 config.patch 上下文管理器的环境中，设置一些 CUDA 相关的配置项
                with config.patch(
                    {
                        "max_autotune": True,
                        # 在此示例中，一些 Cutlass 内核会因为 IMA 而导致不可恢复的 CUDA 错误，
                        # 因此我们需要在此处调整子进程中的自动调优设置。
                        "autotune_in_subproc": False,
                        "max_autotune_gemm_backends": "CUTLASS,ATen",
                        "cuda.cutlass_dir": _CUTLASS_DIR,
                        "cuda.cutlass_max_profiling_configs": 2,
                        "cuda.cutlass_op_allowlist_regex": "pingpong",
                        "cuda.cutlass_op_denylist_regex": None,  # Pingpong Kernels 可能导致数值问题
                    }
                ):
                    # 在使用 mock.patch 上下文管理器的环境中，替换 torch._inductor.kernel.mm.autotune_select_algorithm 函数
                    with mock.patch(
                        "torch._inductor.kernel.mm.autotune_select_algorithm",
                        wraps=select_no_algorithm,
                    ) as sa:
                        # 调用 torch.compile 函数，对 addmm 函数进行编译，禁用动态模式
                        torch.compile(addmm, dynamic=False)(x, a, b, 1.0, 1.0)
                        # 获取 sa 被调用时的参数
                        args, kwargs = sa.call_args
                        # 解包参数
                        op_name, choices, _, __ = args
                        # 断言操作名称为 "addmm"
                        assert op_name == "addmm"
                        cuda_template_count = 0
                        # 遍历 choices 中的每个选项
                        for choice in choices:
                            # 检查选项是否为 CUDATemplateCaller 类型
                            if isinstance(choice, CUDATemplateCaller):
                                # 获取选择的信息字典
                                choice_info = choice.info_dict()
                                # 断言选择的操作配置名称中包含 "pingpong"
                                assert (
                                    "pingpong" in choice_info["op_conf_name"]
                                ), "Only pingpong Kernels should have been allowed"
                                cuda_template_count += 1
                        # 断言至少有一个 CUDATemplateCaller 选择
                        assert cuda_template_count > 0, "No CUDATemplateCaller choices"

    @unittest.skipIf(not SM80OrLater, "need sm_90")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CUTLASS path setup")
    # 使用 unittest.mock.patch.dict 设置 os.environ 的 PATH 环境变量
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    # 定义一个测试函数，用于测试 get_max_alignment 函数的不同情况
    def test_get_max_alignment(self):
        # 创建一个 FixedLayout 对象 l4，表示一个固定布局，包含 size、stride 等参数
        l4 = FixedLayout("cpu", torch.half, size=(1, 2, 4), stride=(0, 4, 1))
        # 调用 get_max_alignment 函数计算 l4 的最大对齐要求并赋值给 m4
        m4 = get_max_alignment(l4)
        # 断言 m4 的值应为 4，用于简单连续情况下的测试
        self.assertEqual(
            m4, 4, "Wrong max alignment. Should have been 4. (simple, contiguous case)"
        )

        # 创建另一个 FixedLayout 对象 l4_2，表示另一种固定布局情况
        l4_2 = FixedLayout("cpu", torch.half, size=(1, 4, 2), stride=(0, 1, 4))
        # 调用 get_max_alignment 函数计算 l4_2 的最大对齐要求并赋值给 m4_2
        m4_2 = get_max_alignment(l4_2)
        # 断言 m4_2 的值应为 4，测试未正确处理步长的情况
        self.assertEqual(
            m4_2,
            4,
            "Wrong max alignment. Should have been 4. Did not deal with strides correctly",
        )

        # 创建另一个 FixedLayout 对象 l1，表示另一种固定布局情况
        l1 = FixedLayout("cpu", torch.half, size=(2, 4, 2), stride=(23, 1, 4))
        # 调用 get_max_alignment 函数计算 l1 的最大对齐要求并赋值给 m1
        m1 = get_max_alignment(l1)
        # 断言 m1 的值应为 1，测试未正确考虑步长情况
        self.assertEqual(
            m1,
            1,
            "Wrong max alignment. Should have been 1. Did not take stride into account correctly",
        )

        # 创建另一个 FixedLayout 对象 l2，表示另一种固定布局情况，包括偏移量
        l2 = FixedLayout("cpu", torch.half, size=(1, 2, 4), stride=(0, 4, 1), offset=6)
        # 调用 get_max_alignment 函数计算 l2 的最大对齐要求并赋值给 m2
        m2 = get_max_alignment(l2)
        # 断言 m2 的值应为 2，由于选择了偏移量的影响
        self.assertEqual(
            m2, 2, "Wrong max alignment. Should have been 2. (due to choice of offset)"
        )

        # 创建另一个 FixedLayout 对象 l8，表示另一种固定布局情况，包括更复杂的参数设置
        l8 = FixedLayout(
            "cpu", torch.half, size=(2, 2, 8), stride=(32, 8, 1), offset=24
        )
        # 调用 get_max_alignment 函数计算 l8 的最大对齐要求并赋值给 m8
        m8 = get_max_alignment(l8)
        # 断言 m8 的值应为 8，测试复杂布局的最大对齐要求
        self.assertEqual(m8, 8, "Wrong max alignment. Should have been 8.")

        # 创建另一个 FixedLayout 对象 l4，这次是用不同的数据类型
        l4 = FixedLayout(
            "cpu", torch.float32, size=(2, 2, 8), stride=(32, 8, 1), offset=24
        )
        # 调用 get_max_alignment 函数计算 l4 的最大对齐要求并赋值给 m4
        m4 = get_max_alignment(l4)
        # 断言 m4 的值应为 4，由于使用了 float32 数据类型
        self.assertEqual(
            m4, 4, "Wrong max alignment. Should have been 4 (due to float32 dtype )."
        )
if __name__ == "__main__":
    # 如果当前脚本被直接执行而非被导入，则执行以下代码块
    from torch._inductor.utils import is_big_gpu

    # 设置环境以便在持续集成环境中工作
    if HAS_CUDA and HAS_CPU and is_big_gpu(0):
        # 如果系统拥有CUDA、CPU，并且检测到GPU是大型GPU，则运行测试
        run_tests()
```