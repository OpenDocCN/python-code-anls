# `.\pytorch\test\export\test_pass_infra.py`

```
# 导入必要的模块和库
import copy  # 导入copy模块，用于对象的深拷贝操作
import unittest  # 导入unittest模块，用于编写和运行单元测试

import torch  # 导入PyTorch库

from functorch.experimental import control_flow  # 导入functorch库中的control_flow模块，用于控制流操作
from torch._dynamo.eval_frame import is_dynamo_supported  # 导入torch._dynamo.eval_frame模块中的is_dynamo_supported函数，用于检查Dynamo支持情况
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse  # 导入torch._export.pass_base模块中的_ExportPassBaseDeprecatedDoNotUse类，已不推荐使用
from torch.export import export  # 导入torch.export模块中的export函数，用于模型导出
from torch.fx.passes.infra.pass_base import PassResult  # 导入torch.fx.passes.infra.pass_base模块中的PassResult类，表示转换过程的结果
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase  # 导入torch.testing._internal.common_utils模块中的IS_WINDOWS、run_tests和TestCase，用于测试相关的公用函数和类

# 如果Dynamo不支持，则跳过此测试类
@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPassInfra(TestCase):
    # 测试_ExportPassBaseDeprecatedDoNotUse类的导出功能
    def test_export_pass_base(self) -> None:
        # 定义一个简单的模型Foo
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        # 定义一个空的导出Pass，继承自_ExportPassBaseDeprecatedDoNotUse类
        class NullPass(_ExportPassBaseDeprecatedDoNotUse):
            pass

        # 对模型f进行导出，输入参数为torch.ones(3, 2)
        ep = export(f, (torch.ones(3, 2),))
        old_nodes = ep.graph.nodes  # 获取导出结果的图节点

        # 应用_NullPass导出Pass，获取新的图节点
        ep = ep._transform_do_not_use(NullPass())
        new_nodes = ep.graph.nodes

        # 遍历新的图节点
        for node in new_nodes:
            if node.op != "call_function":  # 如果节点操作不是"call_function"，则跳过
                continue
            self.assertTrue(hasattr(node, "stack_trace"))  # 断言节点具有"stack_trace"属性
            self.assertIsNotNone(node.stack_trace)  # 断言节点的"stack_trace"属性不为None

        # 断言新旧节点的数量相等
        self.assertEqual(len(new_nodes), len(old_nodes))
        # 逐个比较新旧节点的操作和目标
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)  # 断言新旧节点的操作相同
            self.assertEqual(new_node.target, old_node.target)  # 断言新旧节点的目标相同

    # 如果在Windows系统上运行测试，则跳过此测试方法
    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_cond(self) -> None:
        # 定义一个带条件分支的模型M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, x, y):
                # 定义一个true分支函数
                def true_fn(x, y):
                    b = x.item()
                    torch._check(b >= 2)
                    torch._check(b <= 5)
                    return x - y

                # 定义一个false分支函数
                def false_fn(x, y):
                    c = y.item()
                    torch._check(c >= 2)
                    torch._check(c <= 5)
                    return x + y

                # 使用control_flow.cond根据pred选择执行true_fn或false_fn，并返回结果
                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])  # 创建张量x
        y = torch.tensor([5])  # 创建张量y
        mod = M()  # 实例化模型M
        _ = export(mod, (torch.tensor(True), x, y))._transform_do_not_use(
            _ExportPassBaseDeprecatedDoNotUse()
        )  # 对模型进行导出，并应用_ExportPassBaseDeprecatedDoNotUse导出Pass
    def test_node_name_stability(self) -> None:
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation

        # 定义一个自定义的 PyTorch 模块
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # 定义一个参数
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # 定义两个缓冲区
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # 在 forward 方法中使用参数、缓冲区和两个输入
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # 修改一个缓冲区（例如，增加其值 1）
                self.my_buffer2.add_(1.0)

                return output

        # 创建输入张量
        inps = (torch.rand(1), torch.rand(1))

        # 实例化自定义模块
        m = CustomModule()

        # 导出模型的计算图
        ep_before = export(m, inps)

        # 执行一个没有意义的转换，不会对节点进行任何有意义的更改
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        # 检查转换前后每个节点的名称是否保持不变
        for before_node, after_node in zip(ep_before.graph.nodes, ep_after.graph.nodes):
            self.assertEqual(before_node.name, after_node.name)
    def test_graph_signature_updated_after_transformation(self) -> None:
        # 检查在转换后，通过基础设施正确更新图签名

        # 定义一个自定义模块，继承自 torch.nn.Module
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # 添加一个模块参数
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # 注册两个缓冲区
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # 在 forward 方法中使用参数、缓冲区和两个输入张量
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        # 创建一个 CustomModule 的实例
        my_module = CustomModule()

        # 创建两个输入张量
        input_tensor1 = torch.tensor(5.0)
        input_tensor2 = torch.tensor(6.0)

        # 使用 torch.export.export 方法导出模块
        ep_before = torch.export.export(my_module, (input_tensor1, input_tensor2))

        # 导入 PassResult 类
        from torch.fx.passes.infra.pass_base import PassResult

        # 定义一个修改输入输出的转换函数
        def modify_input_output_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        # 在 ep_before 上应用转换函数，得到 ep_after
        ep_after = ep_before._transform_do_not_use(modify_input_output_pass)

        # 获取转换后的新图签名
        new_signature = ep_after.graph_signature

        # 断言所有用户输出节点名都包含 "_modified"
        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        # 获取转换前的旧图签名
        old_signature = ep_before.graph_signature

        # 断言新旧图签名的用户输出不相等
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)
    def test_replace_hook_basic(self) -> None:
        # 定义一个自定义的神经网络模块
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # 创建一个可训练的参数
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # 注册两个缓冲区
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            # 前向传播方法，使用参数、缓冲区和两个输入
            def forward(self, x1, x2):
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        # 创建一个自定义模块的实例
        my_module = CustomModule()
        
        # 定义输入
        inputs = (torch.tensor(6.0), torch.tensor(7.0))
        
        # 导出模块并获取导出结果
        ep_before = export(my_module, inputs)

        # 定义替换函数
        def replace_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        # 深度复制导出的图模块和签名
        gm = copy.deepcopy(ep_before.graph_module)
        sig = copy.deepcopy(ep_before.graph_signature)

        # 使用替换钩子替换图中的函数调用节点名称
        with gm._set_replace_hook(sig.get_replace_hook()):
            replace_pass(gm)

        # 断言修改后的节点名称包含 "_modified"
        for node_name in sig.user_outputs:
            self.assertTrue("_modified" in node_name)

        # 断言新旧签名不相等
        old_signature = ep_before.graph_signature
        self.assertNotEqual(sig.user_outputs, old_signature.user_outputs)
# 如果这个模块是直接运行的入口
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```