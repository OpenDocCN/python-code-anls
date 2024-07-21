# `.\pytorch\test\jit\test_unsupported_ops.py`

```py
# Owner(s): ["oncall: jit"]

# 导入标准库和第三方库
import os
import sys
import unittest

# 导入 PyTorch 库
import torch

# 让 test/ 中的辅助文件可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行此文件，抛出运行时错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# NOTE: FIXING FAILING TESTS
# 如果你在此文件中看到测试失败，请注意，这意味着 JIT 和 Python API 之间的差异
# 在修复测试之前，你必须更新文档中指出的不受支持的行为，参见 `jit_unsupported.rst`


# 定义一个测试类，继承自 JitTestCase
class TestUnsupportedOps(JitTestCase):

    # 测试工厂操作需要梯度失败的情况
    def test_factory_ops_requires_grad_fail(self):
        # 关键字参数 `{name} unknown` 是 JIT 特有的错误消息，
        # 因此这些函数在 eager 模式下成功，在 JIT 模式下失败

        # 定义一个返回全为 1 的张量，并要求梯度
        def ones():
            return torch.ones([2], requires_grad=True)

        # 断言在 JIT 模式下运行 torch.jit.script(ones) 会抛出异常
        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.ones"
        ):
            torch.jit.script(ones)

        # 定义一个返回服从标准正态分布的张量，并要求梯度
        def randn():
            return torch.randn([2], requires_grad=True)

        # 断言在 JIT 模式下运行 torch.jit.script(randn) 会抛出异常
        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.randn"
        ):
            torch.jit.script(randn)

        # 定义一个返回全为 0 的张量，并要求梯度
        def zeros():
            return torch.zeros([2], requires_grad=True)

        # 断言在 JIT 模式下运行 torch.jit.script(zeros) 会抛出异常
        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.zeros"
        ):
            torch.jit.script(zeros)

    # 如果 PyTorch 编译时没有包含 Lapack 库，则跳过此测试
    @unittest.skipIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")
    # 测试初始化操作
    def test_init_ops(self):
        # 返回 LeakyReLU 激活函数的增益值
        def calculate_gain():
            return torch.nn.init.calculate_gain("leaky_relu", 0.2)

        # 初始化一个单位矩阵
        def eye_():
            return torch.nn.init.eye_(torch.zeros([2, 2]))

        # 初始化一个 Dirac delta 函数
        def dirac_():
            return torch.nn.init.dirac_(torch.empty(3, 16, 5, 5))

        # 使用 Kaiming 均匀分布初始化
        def kaiming_uniform_():
            return torch.nn.init.kaiming_normal_(torch.empty(3, 5))

        # 使用正交初始化
        def orthogonal_():
            return torch.nn.init.orthogonal_(torch.empty(3, 5))

        # 使用稀疏初始化
        def sparse():
            return torch.nn.init.sparse_(torch.empty(3, 5), sparsity=0.1)

        # 遍历所有初始化函数，确保在 JIT 模式下能抛出异常
        for func in [
            calculate_gain,
            eye_,
            dirac_,
            kaiming_uniform_,
            orthogonal_,
            sparse,
        ]:
            # 在 eager 模式下调用函数不会报错
            func()
            # 在 JIT 模式下调用函数应该抛出异常
            with self.assertRaisesRegex(Exception, ""):
                torch.jit.script(func)
```