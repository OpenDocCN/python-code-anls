# `.\pytorch\test\distributed\_tools\test_mod_tracker.py`

```
# Owner(s): ["module: unknown"]

# 导入必要的库和模块
from copy import copy
import torch
from torch.distributed._tools.mod_tracker import ModTracker
from torch.testing._internal.common_utils import run_tests, TestCase, xfailIfTorchDynamo
from torch.utils.checkpoint import checkpoint

# 定义测试类 TestModTracker，继承自 TestCase
class TestModTracker(TestCase):

    # 标记该测试在 Torch Dynamo 模式下会失败，参见 GitHub 问题链接
    @xfailIfTorchDynamo
    def test_module_hierarchy(self):
        # 初始化记录正向传播和反向传播信息的列表
        seen_fw = []
        seen_bw = []

        # 定义一个名为 Foo 的模块，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 定义 forward 方法，执行模块的前向传播操作
            def forward(self, x):
                # 修改输入张量 x 中键为 "a" 的数据，并应用 ReLU 激活函数
                x = x["a"].relu_()
                # 将当前模块及其父模块信息与反向传播标志记录到 seen_fw 列表中
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                # 注册钩子函数，记录当前模块及其父模块信息与反向传播标志到 seen_bw 列表中
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                # 返回处理后的张量，以字典形式包含键 "a" 的矩阵乘法结果
                return {"a": torch.mm(x, x)}

        # 定义一个名为 Mod 的模块，继承自 torch.nn.Module
        class Mod(torch.nn.Module):
            # 构造函数，初始化模块 a, b, c
            def __init__(self):
                super().__init__()
                self.a = Foo()  # 创建 Foo 类的实例作为模块 a
                self.b = torch.nn.ModuleDict({"nest": Foo()})  # 创建嵌套模块字典
                self.c = torch.nn.ModuleList([Foo()])  # 创建模块列表

            # 定义 forward 方法，执行模块的前向传播操作
            def forward(self, x):
                # 使用模块 c 中的第一个模块处理输入 x，并返回结果
                x = self.c[0](x)
                # 依次调用模块 b 中名为 "nest" 的模块和模块 a 处理输入 x，并返回结果
                return self.b["nest"](self.a(x))

        # 创建 Mod 类的实例 mod
        mod = Mod()

        # 使用 ModTracker 上下文管理器来跟踪模块中的运行状态
        with ModTracker() as tracker:
            # 调用 mod 的 forward 方法处理输入张量，对结果求和并执行反向传播
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})["a"].sum().backward()
            # 再次调用 mod 的 forward 方法处理输入张量，对结果求和并执行反向传播
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})["a"].sum().backward()

        # 断言 seen_fw 列表是否符合预期的正向传播结果
        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
            ],
        )

        # 断言 seen_bw 列表是否符合预期的反向传播结果
        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
            ],
        )

    # 定义测试反向传播检测的方法
    def test_bw_detection(self):
        # 创建一个线性模块，输入维度为 2，输出维度为 2
        mod = torch.nn.Linear(2, 2)

        # 使用 ModTracker 上下文管理器来跟踪模块中的运行状态
        with ModTracker() as tracker:
            # 调用 mod 处理随机生成的张量，对结果求和并执行反向传播
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            # 断言 tracker.is_bw 属性为 False
            self.assertFalse(tracker.is_bw)
            # 断言 tracker.parents 包含全局作用域
            self.assertEqual(tracker.parents, {"Global"})

    # 标记该测试在 Torch Dynamo 模式下会失败
    @xfailIfTorchDynamo
    def test_user_hooks(self):
        # 定义一个名为 test_user_hooks 的测试方法
        class Bar(torch.nn.Module):
            # 定义一个名为 Bar 的自定义神经网络模块
            def __init__(self):
                # Bar 类的初始化函数
                super().__init__()
                # 调用父类的初始化方法
                self.foo = torch.nn.Linear(10, 10)
                # 创建一个名为 foo 的线性层，输入和输出都是 10 维

            def forward(self, x):
                # 定义前向传播函数 forward，接受输入 x
                return self.foo(x).relu_()
                # 将输入 x 经过 self.foo 线性层，然后应用 ReLU 激活函数

        mt = ModTracker()
        # 创建一个名为 mt 的 ModTracker 对象
        test_op = []
        # 创建一个空列表 test_op，用于存储测试操作的结果

        def hook(mod, hook_name):
            # 定义一个名为 hook 的函数，用于记录钩子函数的调用信息
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            # 获取 mod 的完全限定名，如果 mod 不为 None，则调用 ModTracker 的 get_known_fqn 方法，否则为 None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))
            # 将 hook_name、mfqn、mfqn 是否在 mt.parents 中以及 mt.is_bw 的值作为元组加入 test_op 列表

        mod = Bar()
        # 创建一个 Bar 类的实例 mod

        mt.register_user_hooks(
            lambda m, inp: hook(m, "pre_fw"),
            lambda m, inp, op: hook(m, "post_fw"),
            lambda m, gop: hook(m, "pre_bw"),
            lambda m, ginp: hook(m, "post_bw"),
        )
        # 使用 ModTracker 对象 mt 的 register_user_hooks 方法注册用户定义的钩子函数

        with mt:
            # 使用 mt 的上下文管理器
            mod(torch.rand(10, 10, requires_grad=True)).sum().backward()
            # 对模型 mod 输入随机的 10x10 张量，计算其输出的和，并进行反向传播
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", "Bar", True, True),
            ("pre_bw", "Bar.foo", True, True),
            ("post_bw", "Bar", True, True),
            ("post_bw", "Bar.foo", True, True),
        ]
        # 预期的测试操作结果，包含了前向和后向传播阶段的钩子调用信息
        self.assertEqual(test_op, expected_op)
        # 使用 unittest 框架的 assertEqual 方法断言 test_op 和 expected_op 相等

        with self.assertRaises(AssertionError):
            # 使用 unittest 框架的 assertRaises 方法检查是否抛出 AssertionError 异常
            mt.register_user_hooks(lambda x, y: x, None, None, None)

        test_op.clear()
        # 清空 test_op 列表
        with mt:
            # 使用 mt 的上下文管理器
            loss = mod(torch.rand(10, 10, requires_grad=True)).sum()
            # 对模型 mod 输入随机的 10x10 张量，计算其输出的和
            del mod
            # 删除模型 mod
            loss.backward()
            # 对损失 loss 进行反向传播
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", None, False, True),
            ("pre_bw", None, False, True),
            ("post_bw", None, False, True),
            ("post_bw", None, False, True),
        ]
        # 预期的测试操作结果，包含了删除模型后的反向传播阶段的钩子调用信息
        self.assertEqual(test_op, expected_op)
        # 使用 unittest 框架的 assertEqual 方法断言 test_op 和 expected_op 相等

    @xfailIfTorchDynamo
    # 定义一个测试方法，用于测试神经网络模型的自动检查点功能
    def test_ac(self):
        # 定义一个继承自 torch.nn.Module 的神经网络模型类 Foo
        class Foo(torch.nn.Module):
            # 初始化方法，接收层数 n_layers、维度 dim 和是否使用自动检查点 use_ac 作为参数
            def __init__(self, n_layers: int, dim: int, use_ac: bool = False):
                super().__init__()
                # 初始化一个空的神经网络层列表
                self.linears = torch.nn.ModuleList()
                # 记录是否使用自动检查点
                self.use_ac = use_ac
                # 根据 n_layers 循环添加线性层到 self.linears 列表中
                for _ in range(n_layers):
                    self.linears.append(torch.nn.Linear(dim, dim))

            # 前向传播方法，接收输入张量 x 作为参数
            def forward(self, x):
                # 遍历 self.linears 列表中的每个块
                for i, block in enumerate(self.linears):
                    # 如果层数大于等于1且使用自动检查点，则使用检查点函数
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        # 否则直接调用块进行前向传播
                        x = block(x)
                    # 断言 x 不为空
                    assert x is not None
                    # 对输出应用 ReLU 激活函数
                    x = torch.nn.functional.relu(x)
                # 返回最终的输出张量 x
                return x

        # 定义测试用例中的批大小、维度和层数
        bsz = 2
        dim = 8
        n_layers = 2
        # 初始化测试操作列表
        test_op = []

        # 定义一个用于记录模块操作的钩子函数
        def hook(mod, mt, hook_name):
            # 获取模块的完全限定名（mfqn），如果模块为 None 则返回 None
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            # 将钩子名、模块完全限定名、模块是否在父模块中、是否为反向传播的钩子结果追加到 test_op 列表中
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        # 初始化模块跟踪器对象 mt
        mt = ModTracker()
        # 注册用户定义的前向传播和反向传播钩子函数
        mt.register_user_hooks(
            lambda m, i: hook(m, mt, "pre_fw"),
            lambda m, i, o: hook(m, mt, "post_fw"),
            lambda m, go: hook(m, mt, "pre_bw"),
            lambda m, gi: hook(m, mt, "post_bw"),
        )
        # 创建一个 Foo 类型的模型实例 model
        model = Foo(n_layers, dim, True)
        # 生成一个随机输入张量 x
        x = torch.randn(bsz, dim)
        # 使用模块跟踪器 mt 来执行以下代码块
        with mt:
            # 对模型进行前向传播，并对其结果求和后进行反向传播
            model(x).sum().backward()

        # 预期的操作序列，用于与实际测试操作 test_op 进行比较
        expected_op = [
            ("pre_fw", "Foo", True, False),
            ("pre_fw", "Foo.linears.0", True, False),
            ("post_fw", "Foo.linears.0", True, False),
            ("pre_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo", True, False),
            ("pre_bw", "Foo", True, True),
            ("pre_bw", "Foo.linears.1", True, True),
            ("pre_fw", "Foo.linears.1", True, True),
            ("post_fw", "Foo.linears.1", True, True),
            ("post_bw", "Foo.linears.1", True, True),
            ("pre_bw", "Foo.linears.0", True, True),
        ]
        # 断言实际的测试操作与预期的操作序列 expected_op 相等
        self.assertEqual(test_op, expected_op)
# 如果这个脚本被直接运行而非被导入作为模块时，执行下面的代码块
if __name__ == "__main__":
    # 调用一个函数来执行测试用例
    run_tests()
```