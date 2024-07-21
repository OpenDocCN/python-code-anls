# `.\pytorch\test\export\test_tools.py`

```
# Owner(s): ["oncall: export"]

import torch  # 导入 PyTorch 库
from torch._dynamo.test_case import TestCase  # 导入 TestCase 类
from torch._export.tools import report_exportability  # 导入 report_exportability 函数

from torch.testing._internal.common_utils import run_tests  # 导入运行测试的辅助函数

# 定义一个 Torch 库，描述一个操作，标记为 pt2_compliant_tag
torch.library.define(
    "testlib::op_missing_meta",
    "(Tensor(a!) x, Tensor(b!) z) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)


# 使用 Torch 库的实现修饰符，在 CPU 上实现 op_missing_meta 操作
@torch.library.impl("testlib::op_missing_meta", "cpu")
@torch._dynamo.disable  # 禁用动态 Torch 功能
def op_missing_meta(x, z):
    x.add_(5)  # 张量 x 增加 5
    z.add_(5)  # 张量 z 增加 5
    return x + z  # 返回 x 和 z 的和


# 定义一个 TestCase 子类 TestExportTools
class TestExportTools(TestCase):
    # 测试 report_exportability 函数的基本用法
    def test_report_exportability_basic(self):
        # 定义一个简单的 Module 类
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()  # 实例化 Module 类
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))  # 输入数据

        report = report_exportability(f, inp)  # 调用 report_exportability 函数
        self.assertTrue(len(report) == 1)  # 断言报告长度为 1
        self.assertTrue(report[""] is None)  # 断言空键值对应为 None

    # 测试 report_exportability 函数处理有问题情况的用法
    def test_report_exportability_with_issues(self):
        # 定义一个不支持的 Module 类
        class Unsupported(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib.op_missing_meta(x, x.cos())  # 调用 op_missing_meta 操作

        # 定义一个支持的 Module 类
        class Supported(torch.nn.Module):
            def forward(self, x):
                return x.sin()  # 返回 x 的正弦值

        # 定义主 Module 类，包含 Unsupported 和 Supported 实例
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unsupported = Unsupported()
                self.supported = Supported()

            def forward(self, x):
                y = torch.nonzero(x)  # 对输入 x 进行非零索引查找
                return self.unsupported(y) + self.supported(y)  # 返回 Unsupported 和 Supported 的结果之和

        f = Module()  # 实例化 Module 类
        inp = (torch.ones(4, 4),)  # 输入数据

        # 调用 report_exportability 函数，使用严格模式，预先分派
        report = report_exportability(f, inp, strict=False, pre_dispatch=True)

        self.assertTrue(report[""] is not None)  # 断言空键不为空
        self.assertTrue(report["unsupported"] is not None)  # 断言不支持键不为空
        self.assertTrue(report["supported"] is None)  # 断言支持键为空


if __name__ == "__main__":
    run_tests()  # 运行所有测试
```