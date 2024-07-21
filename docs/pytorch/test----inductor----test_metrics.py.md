# `.\pytorch\test\inductor\test_metrics.py`

```py
# Owner(s): ["module: inductor"]

# 导入 PyTorch 库
import torch
# 导入需要的模块和函数
from torch._inductor import config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import collect_defined_kernels
from torch._inductor.wrapper_benchmark import get_kernel_category_by_source_code
from torch.testing._internal.common_device_type import largeTensorTest
# 导入内部工具模块
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

# 定义示例的内核代码字符串
example_kernel = """
@triton_heuristics.reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'},
        'device': 0,
        'device_type': 'GPU_TYPE',
        'constants': {},
        'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={
        'autotune_hints': set(),
        'kernel_name': 'triton_red_fused_add_sum_2',
        'mutated_arg_names': ['in_out_ptr0'],
        'no_x_dim': False,
        'kernel_num_gb': 0.0083968
    }
)
@triton.jit
# 定义使用 Triton 框架加速的内核函数
def triton_red_fused_add_sum_2(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    # 初始化变量和参数
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    # 循环迭代计算
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp4 + tmp2
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
""".replace(
    "GPU_TYPE", GPU_TYPE
)

# 测试类，用于测试 metrics 模块中的函数
class TestMetrics(TestCase):
    # 测试解析内核函数代码是否以 "def " 开头
    def test_parse_proper_kernel_fn_code(self):
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(example_kernel)
        assert proper_kernel_fn_code.startswith("def ")

    # 测试统计内核函数代码中参数的个数是否正确
    def test_count_args(self):
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(example_kernel)
        self.assertEqual(6, metrics._count_args(proper_kernel_fn_code))

    # 测试内核函数代码中特定模式出现的次数是否正确
    def test_count_pattern(self):
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(example_kernel)
        self.assertEqual(2, metrics._count_pattern(proper_kernel_fn_code, "tl.load"))
        self.assertEqual(1, metrics._count_pattern(proper_kernel_fn_code, "tl.store"))
        self.assertEqual(1, metrics._count_pattern(proper_kernel_fn_code, "for "))
    # 定义测试方法，用于测试解析缩减提示函数
    def test_parse_reduction_hint(self):
        # 获取示例内核的内核类别
        kernel_category = get_kernel_category_by_source_code(example_kernel)
        # 断言内核类别为 "reduction"
        self.assertEqual("reduction", kernel_category)
        # 解析内核缩减提示，预期为 "INNER"
        self.assertEqual(
            "INNER", metrics._parse_reduction_hint(kernel_category, example_kernel)
        )

    # 定义测试原子加函数
    def test_atomic_add(self):
        # 声明编译装饰器函数
        @torch.compile
        def f(lhs, index, rhs):
            # 执行原子加操作
            return lhs.index_put_([index], rhs, accumulate=True)

        # 初始化左操作数，随机张量
        lhs = torch.randn(1024, device=GPU_TYPE)
        # 初始化索引，随机整数张量
        index = torch.randint(0, 1024, [32], device=GPU_TYPE, dtype=torch.int32)
        # 初始化右操作数，随机张量
        rhs = torch.randn(32, device=GPU_TYPE)

        # 初始化内核列表
        kernel_list = []
        # 收集定义的内核
        with collect_defined_kernels(kernel_list):
            # 调用原子加函数
            f(lhs, index, rhs)

        # 断言内核列表长度为1
        self.assertEqual(len(kernel_list), 1)
        # 获取内核代码
        kernel_code = kernel_list[0]
        # 断言内核代码中 "tl.atomic_add" 模式出现次数为1
        self.assertEqual(metrics._count_pattern(kernel_code, "tl.atomic_add"), 1)

    # 定义测试内核参数数量函数，使用大张量测试装饰器和配置补丁
    @largeTensorTest(25e7 * 2 * 4, device=GPU_TYPE)
    @config.patch("benchmark_kernel", True)
    def test_kernel_args_num_gb(self):
        # 声明编译装饰器函数
        @torch.compile
        def f(x):
            # 执行点对点操作
            return x + 1

        # 初始化输入张量 x，随机张量
        x = torch.randn(int(25e7), device=GPU_TYPE)
        # 初始化内核列表
        kernel_list = []
        # 收集定义的内核
        with collect_defined_kernels(kernel_list):
            # 调用函数 f
            f(x)

        # 断言内核列表长度为1
        self.assertEqual(len(kernel_list), 1)
        # 获取内核代码
        kernel_code = kernel_list[0]
        # 解析内核参数数量（GB），预期为 2.0
        self.assertEqual(
            metrics._parse_kernel_args_num_gb(kernel_code, "pointwise"), 2.0
        )
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 如果系统配置表明有GPU可用
    if HAS_GPU:
        # 运行测试函数
        run_tests()
```