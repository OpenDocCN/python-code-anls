# `.\pytorch\test\inductor\test_ck_backend.py`

```
# Owner(s): ["module: inductor"]
# 导入所需的模块和库
import logging
import os
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

# 设置全局的浮点数矩阵乘法精度为高精度
torch.set_float32_matmul_precision("high")

# 如果支持 CUDA，则设置 CUDA 内存分配器设置为不可扩展段
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)


def _get_path_without_sccache() -> str:
    """
    获取不包含 sccache 的 PATH 环境变量。
    """
    # 获取当前环境的 PATH 变量，并去除包含 "/opt/cache/bin" 的路径
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    # 返回过滤后的 PATH 环境变量字符串
    return ":".join(path_envs)


# 使用 parametrize 装饰器实例化参数化测试
@instantiate_parametrized_tests
class TestCKBackend(TestCase):
    def setUp(self):
        """
        设置测试环境。
        """
        # 在调用父类 setUp() 之前，禁用自动缓存刷新机制
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )

        # 设置随机种子以确保可复现性
        torch.random.manual_seed(1234)

        try:
            # 尝试导入 ck4inductor 库，并设置其路径
            import ck4inductor

            self.ck_dir = os.path.dirname(ck4inductor.__file__)
            os.environ["TORCHINDUCTOR_CK_DIR"] = self.ck_dir
        except ImportError as e:
            # 如果导入失败，则跳过测试并抛出相应异常
            raise unittest.SkipTest("Composable Kernel library not installed") from e

        try:
            # 设置环境变量，禁用自动缓存刷新
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            # 调用父类的 setUp() 方法初始化测试环境
            super().setUp()
        finally:
            # 恢复原来的自动缓存刷新设置
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = old_disable_fresh_cache_envvar

    # 如果不是 ROCm 版本，则跳过测试
    @unittest.skipIf(not torch.version.hip, "ROCM only")
    # 如果是在 fbcode 环境下运行，则跳过测试
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CK path setup")
    # 使用 mock.patch.dict 装饰器设置 PATH 环境变量
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    # 使用 parametrize 装饰器指定参数化测试的参数
    @parametrize("max_autotune_gemm_backends", ("CK", "ATen,Triton,CK"))
    def test_max_autotune_precompile(self, max_autotune_gemm_backends):
        """
        Make sure autotuning mm in subprocesses doesn't crash.
        """

        # 禁用 FP16 减少精度
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义矩阵乘法函数
        def mm(a, b):
            return a @ b

        # 定义张量的选项，使用 CUDA 设备和 BFloat16 数据类型
        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        # 创建随机张量 a 和 b
        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        # 断言检查 config 是否包含 "rocm"
        assert "rocm" in dir(config)

        # 使用 config.patch 上下文管理器设置多个配置项
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "compile_threads": 2,
                "rocm.n_max_profiling_configs": 2,
                "rocm.ck_dir": self.ck_dir,
            }
        ):
            # 使用 torch.compile 编译 mm 函数，禁用动态模式
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            # 执行未编译的 mm 函数
            Y = mm(a, b)
            # 断言检查 Y_compiled 和 Y 的结果是否接近
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.skipIf(config.is_fbcode(), "fbcode requires different CK path setup")
    @unittest.mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("max_autotune_gemm_backends", ("CK", "ATen,Triton,CK"))
    def test_max_autotune_precompile_preselected(self, max_autotune_gemm_backends):
        """
        End to end test for picking preselected ck instances
        """

        # 禁用 FP16 减少精度
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义矩阵乘法函数
        def mm(a, b):
            return a @ b

        # 定义张量的选项，使用 CUDA 设备和 Float16 数据类型
        tensor_options = {"device": "cuda", "dtype": torch.float16}

        # 创建随机张量 a 和 b，同时对 b 进行转置
        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(2048, 256, **tensor_options).transpose(0, 1)

        # 断言检查 config 是否包含 "rocm"
        assert "rocm" in dir(config)

        # 使用 config.patch 上下文管理器设置多个配置项
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "compile_threads": 12,
                "rocm.ck_dir": self.ck_dir,
                "rocm.use_preselected_instances": True,
            }
        ):
            # 使用 torch.compile 编译 mm 函数，禁用动态模式
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            # 执行未编译的 mm 函数
            Y = mm(a, b)
            # 断言检查 Y_compiled 和 Y 的结果是否接近
            torch.testing.assert_close(Y_compiled, Y)
    def test_max_autotune_precompile_non_contiguous(self, max_autotune_gemm_backends):
        """
        Make sure the ck template can work with non-contiguous inputs
        """

        # 禁止 CUDA 运算库使用 FP16 减少精度以优化矩阵乘法
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # 定义张量的选项，指定在 CUDA 设备上使用 FP16 数据类型
        tensor_options = {"device": "cuda", "dtype": torch.float16}

        # 创建非连续存储的张量 a 和 b，使用给定的选项
        a = torch.empty_strided((50257, 32768), (1, 50304), **tensor_options)
        b = torch.empty_strided((32768, 768), (768, 1), **tensor_options)

        # 断言检查 config 对象中是否包含 "rocm" 属性
        assert "rocm" in dir(config)

        # 在 config 的上下文中，使用指定的配置参数运行以下代码块
        with config.patch(
            {
                "max_autotune": True,                            # 开启自动调优功能
                "autotune_in_subproc": True,                     # 在子进程中执行自动调优
                "max_autotune_gemm_backends": max_autotune_gemm_backends,  # 设置自动调优的 GEMM 后端
                "compile_threads": 2,                            # 编译线程数为 2
                "rocm.ck_dir": self.ck_dir,                      # 设置 ROCm 的编译核心目录
                "rocm.n_max_profiling_configs": 2,               # 最大的性能配置数为 2
            }
        ):
            # 定义一个不使用动态计算的矩阵乘法函数 mm
            @torch.compile(dynamic=False)
            def mm(a, b):
                return a @ b

            # 使用 mm 函数对张量 a 和 b 进行编译计算
            Y_compiled = mm(a, b)
            # 直接计算张量 a 和 b 的乘积
            Y_eager = a @ b
            # 使用测试函数检查编译结果 Y_compiled 是否与直接计算结果 Y_eager 接近
            torch.testing.assert_close(Y_compiled, Y_eager)
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._inductor.utils 模块中导入 is_big_gpu 函数
    from torch._inductor.utils import is_big_gpu
    
    # 设置环境以确保在持续集成环境中正常工作
    # 条件检查：HAS_CUDA 和 HAS_CPU 均为真，并且第一个 GPU 是大型 GPU
    if HAS_CUDA and HAS_CPU and is_big_gpu(0):
        # 如果条件满足，运行测试函数 run_tests()
        run_tests()
```