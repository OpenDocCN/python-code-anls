# `.\pytorch\test\jit\test_script_profile.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os
import sys

import torch
from torch import nn

# 将测试目录中的辅助文件变成可导入的
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果作为主程序执行，则抛出运行时错误，提示正确的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个继承自 nn.Module 的神经网络模型 Sequence
class Sequence(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化模型中的 LSTM 层和线性层
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    # 定义模型的前向传播方法
    def forward(self, input):
        outputs = []
        # 初始化 LSTM 单元的隐藏状态和细胞状态
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)
        h_t2 = torch.zeros(input.size(0), 51)
        c_t2 = torch.zeros(input.size(0), 51)

        # 遍历输入的每一个时间步
        for input_t in input.split(1, dim=1):
            # LSTM 层的前向计算
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # 线性层的前向计算
            output = self.linear(h_t2)
            outputs += [output]
        # 将所有时间步的输出拼接在一起
        outputs = torch.cat(outputs, dim=1)
        return outputs

# 定义一个测试类，继承自 JitTestCase
class TestScriptProfile(JitTestCase):
    # 测试基本功能
    def test_basic(self):
        seq = torch.jit.script(Sequence())
        p = torch.jit._ScriptProfile()
        p.enable()
        seq(torch.rand((10, 100)))
        p.disable()
        # 断言输出的统计信息不为空
        self.assertNotEqual(p.dump_string(), "")

    # 测试脚本化函数
    def test_script(self):
        seq = Sequence()

        p = torch.jit._ScriptProfile()
        p.enable()

        @torch.jit.script
        def fn():
            _ = seq(torch.rand((10, 100)))

        fn()
        p.disable()

        # 断言输出的统计信息不为空
        self.assertNotEqual(p.dump_string(), "")

    # 测试多次运行
    def test_multi(self):
        seq = torch.jit.script(Sequence())
        profiles = [torch.jit._ScriptProfile() for _ in range(5)]
        for p in profiles:
            p.enable()

        last = None
        while len(profiles) > 0:
            seq(torch.rand((10, 10)))
            p = profiles.pop()
            p.disable()
            stats = p.dump_string()
            # 断言每次的统计信息不为空，并且与上次的不同
            self.assertNotEqual(stats, "")
            if last:
                self.assertNotEqual(stats, last)
            last = stats

    # 测试不同参数的情况
    def test_section(self):
        seq = Sequence()

        @torch.jit.script
        def fn(max: int):
            _ = seq(torch.rand((10, max)))

        p = torch.jit._ScriptProfile()
        p.enable()
        fn(100)
        p.disable()
        s0 = p.dump_string()

        fn(10)
        p.disable()
        s1 = p.dump_string()

        p.enable()
        fn(10)
        p.disable()
        s2 = p.dump_string()

        # 断言相同输入下的统计信息相同，不同输入下的统计信息不同
        self.assertEqual(s0, s1)
        self.assertNotEqual(s1, s2)

    # 测试空函数
    def test_empty(self):
        p = torch.jit._ScriptProfile()
        p.enable()
        p.disable()
        # 断言空函数的统计信息为空字符串
        self.assertEqual(p.dump_string(), "")
```