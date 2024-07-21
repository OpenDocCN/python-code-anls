# `.\pytorch\test\inductor\test_minifier.py`

```py
# 导入所需的模块和类
import unittest
from unittest.mock import patch

# 导入需要测试的模块和配置
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch._inductor import config
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu

# 定义测试类，继承自MinifierTestBase
class MinifierTests(MinifierTestBase):
    
    # 测试在AOT之后是否可以重现编译和准确性错误（CPU和CUDA）
    def _test_after_aot(self, device, expected_error):
        # 程序故意设计得很简单，只触发一个缩小步骤，不再多余（专门的缩小器测试应该只测试缩小器）
        run_code = f"""\
@torch.compile()
def inner(x):
    x = torch.relu(x)
    x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("{device}"))
"""
        self._run_full_test(run_code, "aot", expected_error, isolate=False)

    # 跳过Jetson平台上的测试
    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_cpu_compile_error(self):
        self._test_after_aot("cpu", "CppCompileError")

    # 跳过Jetson平台上的测试
    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_cpu_accuracy_error(self):
        self._test_after_aot("cpu", "AccuracyError")

    # 需要GPU的测试，修补编译错误
    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_gpu_compile_error(self):
        self._test_after_aot(GPU_TYPE, "SyntaxError")

    # 需要GPU的测试，修补准确性错误
    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_gpu_accuracy_error(self):
        self._test_after_aot(GPU_TYPE, "AccuracyError")

    # 修补准确性错误的测试
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_constant_in_graph(self):
        run_code = """\
@torch.compile()
def inner(x):
    return torch.tensor(2) + torch.relu(x)

inner(torch.randn(2))
"""
        self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)

    # 测试RMSE是否在ATOL上有所改善
    @requires_gpu
    @patch.object(config, "joint_graph_constant_folding", False)
    def test_rmse_improves_over_atol(self):
        # 来自https://twitter.com/itsclivetime/status/1651135821045719041?s=20
        run_code = """
@torch.compile()
def inner(x):
    return x - torch.tensor(655, dtype=torch.half, device='GPU_TYPE') * 100

inner(torch.tensor(655 * 100, dtype=torch.half, device='GPU_TYPE'))
# 将字符串中的 "GPU_TYPE" 替换为变量 GPU_TYPE 的值，生成新的字符串
""".replace(
            "GPU_TYPE", GPU_TYPE
        )

# 使用配置禁用 fp64 会触发精度错误，因为 torch.compile 的增加精度改变了结果
# 655 * 100 的计算结果
with dynamo_config.patch("same_two_models_use_fp64", False):
    # 运行完整测试，使用 "aot" 模式，预期捕获 "AccuracyError" 异常，
    # 需要关闭隔离模式
    self._run_full_test(
        run_code,
        "aot",
        "AccuracyError",
        isolate=False,
        # 注意：需要这样设置以避免在 fp64 不起作用时拒绝缩小化
        # （由于上面的配置补丁导致）
        minifier_args=["--strict-accuracy"],
    )

# 使用 fp64 时，我们看到预期的语义是增加了精度，因此报告没有问题
self._run_full_test(run_code, "aot", None, isolate=False)

# 使用配置注入 relu 和 log1p 的错误进行测试
@inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
@inductor_config.patch("cpp.inject_log1p_bug_TESTING_ONLY", "accuracy")
def test_accuracy_vs_strict_accuracy(self):
    # 定义一个运行代码块，使用 torch.compile 编译装饰器
    run_code = """
@torch.compile()
def inner(x):
    # 计算 x 的对数加一
    y = torch.log1p(x)
    # 计算 y 是否大于零的布尔值
    b = y > 0
    # 确保后缀移除命中布尔输出
    b = torch.logical_not(b)
    b = torch.logical_not(b)
    # 计算 x 的 relu
    x = torch.relu(x)
    return torch.where(b, x, x)

inner(torch.randn(20))
"""

# 严格精度模式将由于布尔掩码的差异而遇到问题，这将把错误局限于 sigmoid，
# 即使对最终结果并不重要
res = self._run_full_test(
    run_code,
    "aot",
    "AccuracyError",
    isolate=False,
    minifier_args=["--strict-accuracy"],
)
# 断言预期的内联输出
self.assertExpectedInline(
    res.repro_module(),
    """\
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1):
        log1p = torch.ops.aten.log1p.default(arg0_1);  arg0_1 = None
        return (log1p,)""",
)

# FP 精度将拒绝在输出上提升 logical_not，因此你将进入 relu 部分
res = self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)
# 断言预期的内联输出
self.assertExpectedInline(
    res.repro_module(),
    """\
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1):
        relu = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        return (relu,)""",
)

# 使用配置注入 relu 的错误进行测试
@inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
def test_offload_to_disk(self):
    # 仅仅是一个烟雾测试，实际上并没有测试内存使用是否下降。
    # 测试案例被精心构造以命中 delta 调试。
    run_code = """\
@torch.compile()
def inner(x):
    x = torch.sin(x)
    # 计算张量 x 中每个元素的正弦值，并更新 x
    x = torch.sin(x)
    
    # 计算张量 x 中每个元素的余弦值，并更新 x
    x = torch.cos(x)
    
    # 计算张量 x 中每个元素的 ReLU（整流线性单元）激活函数值，并更新 x
    x = torch.relu(x)
    
    # 返回经过正弦、余弦和 ReLU 操作后的张量 x
    return x
inner(torch.randn(20, 20))
"""
调用名为 inner 的函数，并传入一个大小为 (20, 20) 的随机张量作为参数。
"""

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # 如果当前脚本不在 macOS 环境下，并且不是在 ASAN 测试模式下
    # 则执行 run_tests() 函数来运行测试用例
    if not IS_MACOS and not TEST_WITH_ASAN:
        run_tests()
```