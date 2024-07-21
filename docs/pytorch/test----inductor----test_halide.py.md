# `.\pytorch\test\inductor\test_halide.py`

```
# Owner(s): ["oncall: pt2"]
# 导入所需的标准库和第三方库
import os
import sys
import textwrap
import unittest

# 导入PyTorch相关模块
import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor import config
from torch._inductor.codecache import HalideCodeCache
from torch._inductor.runtime.hints import HalideInputSpec, HalideMeta
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import parallel_num_threads

# 导入测试相关的工具函数和常量
from torch.testing._internal.common_utils import IS_CI, IS_MACOS, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CPU

# 如果运行在Windows且是CI环境下，则输出相应的错误信息并退出
if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# 尝试导入Halide模块，设置HAS_HALIDE标志位
try:
    import halide

    HAS_HALIDE = halide is not None
except ImportError:
    HAS_HALIDE = False

# 尝试导入test_torchinductor模块，如果失败则尝试从当前目录导入
try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor

# 创建名为make_halide的配置字典，设置相关配置项
make_halide = config.patch(
    {
        "cpu_backend": "halide",
        "cuda_backend": "halide",
        "fallback_random": True,  # TODO(jansel): support random
    }
)

# 定义一个unittest测试类，如果没有Halide则跳过测试
@unittest.skipUnless(HAS_HALIDE, "requires halide")
class HalideTests(TestCase):
    # 定义单元测试方法，用于测试代码缓存功能
    def test_codecache(self):
        # 调用 HalideCodeCache 类的 generate_halide 方法，生成 Halide 代码并返回函数
        fn = HalideCodeCache.generate_halide(
            # 构造 HalideMeta 对象，描述 Halide 代码的输入输出及目标平台信息
            HalideMeta(
                argtypes=[
                    # 第一个输入参数的描述
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    # 第二个输入参数的描述
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    # 输出参数的描述
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                # Halide 代码的目标平台
                target="host-no_runtime",
                # 使用的调度器
                scheduler="Mullapudi2016",
                # 调度器的标志参数，例如并行度
                scheduler_flags={
                    "parallelism": parallel_num_threads(),
                },
            ),
            # Halide 代码的实际内容，使用了 textwrap.dedent 去除缩进
            textwrap.dedent(
                """
                import halide as hl

                @hl.generator(name="kernel")
                class Kernel:
                    # 输入缓冲区定义
                    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                    in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                    # 输出缓冲区定义
                    out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                    def generate(g):
                        # 将输入缓冲区和输出缓冲区绑定到本地变量
                        in_ptr0 = g.in_ptr0
                        in_ptr1 = g.in_ptr1
                        out_ptr0 = g.out_ptr0
                        # 定义变量 xindex
                        xindex = hl.Var('xindex')
                        x0 = xindex
                        # 定义临时函数 tmp0，并为其赋值
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[x0]
                        # 定义临时函数 tmp1，并为其赋值
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[x0]
                        # 定义临时函数 tmp2，并为其赋值
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        # 将 tmp2 的值写入输出缓冲区 out_ptr0
                        out_ptr0[x0] = tmp2[xindex]

                        # 断言使用了自动调度器
                        assert g.using_autoscheduler()
                        # 设置输入缓冲区的估计值
                        in_ptr0.set_estimates([hl.Range(1024, 1024)])
                        in_ptr1.set_estimates([hl.Range(1024, 1024)])
                        # 设置输出缓冲区的估计值
                        out_ptr0.set_estimates([hl.Range(1024, 1024)])

                # 如果脚本作为主程序执行，调用 hl.main()
                __name__ == '__main__' and hl.main()
                """
            ),
        )
        # 生成随机的输入数据张量
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        # 调用生成的 Halide 函数 fn，计算结果并存入 c
        fn(a, b, c)
        # 断言 c 的值等于 a 和 b 相加的结果
        self.assertEqual(c, a + b)
    # 定义一个测试方法，用于测试手动调度生成的 Halide 代码
    def test_manual_schedule(self):
        # 生成 Halide 代码并返回函数
        fn = HalideCodeCache.generate_halide(
            # HalideMeta 对象，描述 Halide 函数的元数据
            HalideMeta(
                argtypes=[
                    # 输入参数规格：指向 float 类型数据的指针，长度为 1024
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    HalideInputSpec(
                        ctype="float*",
                        name="in_ptr1",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                    # 输出参数规格：指向 float 类型数据的指针，长度为 1024
                    HalideInputSpec(
                        ctype="float*",
                        name="out_ptr0",
                        shape=["1024L"],
                        stride=["1L"],
                        offset="0",
                    ),
                ],
                # 指定目标为 host-no_runtime，表示在没有运行时的主机上执行
                target="host-no_runtime",
                # 调度器为空，表示使用手动调度
                scheduler=None,
            ),
            # Halide 代码文本，使用 textwrap.dedent 去除缩进
            textwrap.dedent(
                """
                import halide as hl

                @hl.generator(name="kernel")
                class Kernel:
                    in_ptr0 = hl.InputBuffer(hl.Float(32), 1)
                    in_ptr1 = hl.InputBuffer(hl.Float(32), 1)
                    out_ptr0 = hl.OutputBuffer(hl.Float(32), 1)

                    def generate(g):
                        # 从 generator 对象 g 中获取输入和输出指针
                        in_ptr0 = g.in_ptr0
                        in_ptr1 = g.in_ptr1
                        out_ptr0 = g.out_ptr0
                        # 定义一个名为 xindex 的变量
                        xindex = hl.Var('xindex')
                        # 将 in_ptr0 中的数据复制到 tmp0 中
                        tmp0 = hl.Func()
                        tmp0[xindex] = in_ptr0[xindex]
                        # 将 in_ptr1 中的数据复制到 tmp1 中
                        tmp1 = hl.Func()
                        tmp1[xindex] = in_ptr1[xindex]
                        # 计算 tmp0 和 tmp1 的和，结果存储到 tmp2 中
                        tmp2 = hl.Func()
                        tmp2[xindex] = tmp0[xindex] + tmp1[xindex]
                        # 将 tmp2 的值存储到 out_ptr0 中
                        out_ptr0[xindex] = tmp2[xindex]

                        # 断言不使用自动调度器
                        assert not g.using_autoscheduler()
                        # 定义两个变量 i 和 j
                        i = hl.Var()
                        j = hl.Var()
                        # 将 out_ptr0 计算放到根节点
                        out_ptr0.compute_root()
                        # 在 xindex 上进行分割，分为 i 和 j，每次处理 32 个元素
                        out_ptr0.split(xindex, i, j, 32)
                        # 并行化处理 i 轴
                        out_ptr0.parallel(i)
                        # 对 j 轴进行向量化处理
                        out_ptr0.vectorize(j)
                        # 将 tmp2 在 i 处计算
                        tmp2.compute_at(out_ptr0, i)
                        # 将 tmp2 在 i 处存储
                        tmp2.store_at(out_ptr0, i)
                        # 将 tmp1 内联处理
                        tmp1.compute_inline()

                # 如果是主模块则运行 hl.main()
                __name__ == '__main__' and hl.main()
                """
            ),
        )
        # 生成随机数据 a, b, c，每个长度为 1024
        a = torch.randn(1024)
        b = torch.randn(1024)
        c = torch.randn(1024)
        # 调用生成的 Halide 函数 fn，将 a 和 b 存储到 c 中
        fn(a, b, c)
        # 断言 c 的值等于 a 和 b 的和
        self.assertEqual(c, a + b)
# 如果 test_torchinductor 模块有定义 HAS_CPU 并且当前环境有 Halide 支持，则执行以下操作
if test_torchinductor.HAS_CPU and HAS_HALIDE:
    # 使用 make_halide 函数生成 Halide 版本的测试用例 SweepInputsCpuHalideTest
    SweepInputsCpuHalideTest = make_halide(test_torchinductor.SweepInputsCpuTest)
    # 使用 make_halide 函数生成 Halide 版本的测试用例 CpuHalideTests
    CpuHalideTests = make_halide(test_torchinductor.CpuTests)

# 如果 test_torchinductor 模块有定义 HAS_GPU、当前环境有 Halide 支持，且环境变量 TEST_HALIDE_GPU 设置为 "1"
if (
    test_torchinductor.HAS_GPU
    and HAS_HALIDE
    and os.environ.get("TEST_HALIDE_GPU") == "1"
):
    # 使用 make_halide 函数生成 Halide 版本的测试用例 SweepInputsGPUHalideTest
    SweepInputsGPUHalideTest = make_halide(test_torchinductor.SweepInputsGPUTest)
    # 使用 make_halide 函数生成 Halide 版本的测试用例 GPUHalideTests
    GPUHalideTests = make_halide(test_torchinductor.GPUTests)

# 如果当前脚本作为主程序运行，并且具备 CPU 支持、非 macOS 系统，并且当前环境有 Halide 支持
if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS and HAS_HALIDE:
        # 运行测试，其中指定需要的依赖为 "filelock"
        run_tests(needs="filelock")
```