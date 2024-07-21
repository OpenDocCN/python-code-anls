# `.\pytorch\test\fx\test_fx_const_fold.py`

```
# Owner(s): ["module: fx"]

# 导入必要的库和模块
import operator
import torch
import torch.fx
from torch.fx.experimental import const_fold
from torch.fx.passes.shape_prop import _extract_tensor_metadata, ShapeProp
from torch.testing._internal.common_utils import TestCase

# 定义测试类 TestConstFold，继承自 TestCase
class TestConstFold(TestCase):

    # 获取节点的属性值
    def _get_attr(self, node):
        mod = node.graph.owning_module  # 获取节点所在的模块
        target = str(node.target)  # 获取节点的目标属性名
        target_atoms = target.split(".")  # 将目标属性名按点号分隔成列表
        curr_obj = mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(curr_obj, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target '{'.'.join(target_atoms[:i])}'; "
                    f" original whole target: '{target}'"
                )
            curr_obj = getattr(curr_obj, atom)  # 获取当前对象的属性
        return curr_obj

    # 验证常量折叠后的模块
    def _verify_const_fold_mod(self, mod_folded: const_fold.FoldedGraphModule):
        self.assertTrue(mod_folded.const_subgraph_module is not None)  # 断言常量子图模块不为空

        # 检查是否在图中找到已折叠的属性，并排除了常量和非常量折叠的图形，确保有常量折叠的 get_attr
        found_folded_attrs = False
        for n in mod_folded.graph.nodes:
            if n.op == "get_attr" and n.target.startswith("_FX_CONST_FOLDED_ATTRS"):
                found_folded_attrs = True
            elif n.op == "call_module":
                self.assertTrue(n.target not in {"submod_0", "submod_1"})  # 断言不包含特定的调用模块名称
        self.assertTrue(found_folded_attrs)  # 断言至少找到一个已折叠的属性
        def test_const_fold_basic_one_attr_no_name_collision(self):
            r"""
            Perform constant folding conversion, from original mod to split constant folding
            module with two split subgraphs, where there's a single attr to fold and
            a single output attr result to replace.

               attr1                 attr1
                | |                   | |
            x   add                   add
             \ /                       |
             sub   y                 output     (becomes attr add_1)
                \ /         ==> -------+------- (const/base subgraph split)
                mul  attr2       x   /          (input from previous subgraph
                  \ /             \ /            is attr)
                  add             sub   y
                   |                 \ /
                 output              mul  attr2
                                       \ /
                                       add
                                        |
                                      output
            """

            class ConstFoldTestModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # Define attribute attr_1 initialized with a tensor [-0.9]
                    self.attr_1 = torch.nn.Parameter(torch.tensor([[-0.9]]))
                    # Define attribute attr_2 initialized with a tensor [17.1]
                    self.attr_2 = torch.nn.Parameter(torch.tensor([[17.1]]))

                def forward(self, x, y):
                    # Perform addition of attr_1 with itself
                    a = self.attr_1 + self.attr_1
                    # Subtract a from input x
                    x = x - a
                    # Compute x multiplied by y and add attr_2
                    return x * y + self.attr_2

            # Instantiate ConstFoldTestModule
            mod = ConstFoldTestModule()
            # Apply constant folding by splitting subgraphs
            mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
            # Verify the folded module against expected behavior
            self._verify_const_fold_mod(mod_folded)

            # Now run both folded and non-folded to check results equal.
            in_x, in_y = torch.tensor([[-0.45]]), torch.tensor([0.9])
            # Compute result using original module
            base_result = mod(in_x, in_y)
            # Compute result using folded module
            fold_result = mod_folded(in_x, in_y)
            # Assert that both results are equal
            self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_basic_one_attr_name_collision(self):
        r"""
        Perform constant folding conversion, from original mod to split constant folding
        module with two split subgraphs, where there's a single attr to fold and
        a single output attr result to replace. Name the attrs such that they will
        collide by name with folded attrs.

           add_1                 add_1
            | |                   | |
        x   add                   add
         \ /                       |
         sub   y                 output     (becomes attr add_1)
            \ /         ==> -------+------- (const/base subgraph split)
            mul  add_2       x   /          (input from previous subgraph
              \ /             \ /            is attr)
              add             sub   y
               |                 \ /
             output              mul  add_2
                                   \ /
                                   add
                                    |
                                  output
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Note: Named as such to result in name collision.
                self.add_1__CF = torch.nn.Parameter(torch.tensor([[1.0]]))
                self.add_2__CF = torch.nn.Parameter(torch.tensor([[17.1]]))

            def forward(self, x, y):
                # Compute 'a' by adding the folded attribute twice
                a = self.add_1__CF + self.add_1__CF
                # Subtract 'a' from 'x'
                x = x - a
                # Return the result of multiplying 'x' and 'y', then adding the folded attribute
                return x * y + self.add_2__CF

        # Create an instance of ConstFoldTestModule
        mod = ConstFoldTestModule()
        # Perform split constant subgraph folding on the module
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        # Verify the constant folding modifications
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        # Define input tensors 'in_x' and 'in_y'
        in_x, in_y = torch.tensor([[5.0]]), torch.tensor([4.0])
        # Compute result using the original module
        base_result = mod(in_x, in_y)
        # Compute result using the folded module
        fold_result = mod_folded(in_x, in_y)
        # Assert that the results from both computations are equal
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_basic_placeholder_reordered(self):
        """
        测试占位符在普通操作节点之后的代码路径
        """

        class ConstFoldTestModule(torch.nn.Module):
            def forward(self, x, y):
                # 返回 x * 2 + y 的结果
                return x * 2 + y

        mod = ConstFoldTestModule()
        # 对模块进行符号化跟踪
        mod = torch.fx.symbolic_trace(mod)
        yy = None
        for n in mod.graph.nodes:
            # 查找占位符节点，并且目标是 "y"
            if n.op == "placeholder" and n.target == "y":
                yy = n
            # 当找到目标占位符节点之后，将其前置到当前节点之前
            elif yy is not None and n.op == "call_function":
                yy.prepend(n)
                break

        # 对模块进行常量折叠
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)

        # 断言折叠后的子图模块为 None，因为没有进行常量折叠
        self.assertTrue(mod_folded.const_subgraph_module is None)
        
        # 现在运行折叠和非折叠的模块，检查结果是否相等
        in_x = torch.tensor([[-0.45]])
        in_y = torch.tensor([[0.45]])
        base_result = mod(in_x, in_y)
        fold_result = mod_folded(in_x, in_y)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_noop(self):
        r"""
        检查没有常量折叠的图表现正常处理的情况。

        x  attr1
         \ /
         sub
          |
        output
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr1 = torch.nn.Parameter(torch.tensor([[-0.9]]))

            def forward(self, x):
                # 返回 x - self.attr1 的结果
                return x - self.attr1

        mod = ConstFoldTestModule()
        # 对模块进行常量折叠
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)

        # 断言折叠后的子图模块为 None，因为没有进行常量折叠
        self.assertTrue(mod_folded.const_subgraph_module is None)

        # 现在运行折叠和非折叠的模块，检查结果是否相等
        in_x = torch.tensor([[-0.45]])
        base_result = mod(in_x)
        fold_result = mod_folded(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_basic_two_attr_three_input(self):
        r"""
        执行常量折叠转换，从原始的模块到拆分常量折叠模块，其中有两个拆分子图，
        其中有两个属性被折叠到单个输出中，并且有三个占位符输入。

        attr1   attr2         attr1   attr2
            \   /                 \   /
         x   add                   add
          \ /                       |
          sub     y               output     (becomes attr add_1)
             \   /     ==>   -------+------- (const/base subgraph split)
              mul  z           x   /         (input from previous subgraph
                \ /             \ /           is attr)
                div              sub  y
                 |                 \ /
               output              mul  z
                                     \ /
                                     div
                                      |
                                    output
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模块的参数 attr1 和 attr2，这里的 attr2 实际上应该是 attr2，
                # 修正为 self.attr2
                self.attr1 = torch.nn.Parameter(torch.tensor([[-0.9]]))
                self.attr2 = torch.nn.Parameter(torch.tensor([[1.32]]))

            def forward(self, x, y, z):
                # 使用模块内的 attr1 和 attr2 进行常量折叠操作
                a = self.attr1 + self.attr2  # attr1 和 attr2 相加
                sub = x - a  # x 减去 a
                mul = sub * y  # sub 乘以 y
                return mul / z  # 返回结果除以 z

        # 创建 ConstFoldTestModule 的实例 mod
        mod = ConstFoldTestModule()
        # 对 mod 进行常量折叠分割
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        # 验证常量折叠后的模块 mod_folded 的正确性
        self._verify_const_fold_mod(mod_folded)

        # 现在运行常量折叠后和未折叠前的模块，以检查结果是否相等。
        # 定义输入张量 in_x, in_y, in_z
        in_x, in_y, in_z = (
            torch.tensor([[-0.45]]),
            torch.tensor([0.9]),
            torch.tensor([1.1]),
        )
        # 使用原始模块 mod 计算 base_result
        base_result = mod(in_x, in_y, in_z)
        # 使用折叠后的模块 mod_folded 计算 fold_result
        fold_result = mod_folded(in_x, in_y, in_z)
        # 断言折叠后的结果与原始结果是否相等
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_basic_two_attr(self):
        r"""
        Perform constant folding conversion, from original mod to split constant
        folding module with two split subgraphs, where there are two attrs to
        fold into a single output.

        attr1  attr2                attr1  attr2
            \ /                         \ /
        x   add                         add       (becomes attr add_1)
         \ /            ==>       -------+------- (const/base subgraph split)
         sub                         x   |        (input from previous subgraph is attr)
          |                           \ /
        output                        sub
                                       |
                                     output
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模块的两个参数，attr1 和 attr2
                self.attr1 = torch.nn.Parameter(torch.randn(2, 3))
                self.attr2 = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # 计算两个参数的加和
                y = self.attr1 + self.attr2
                # 返回输入 x 加上计算结果 y
                return x + y

        # 创建 ConstFoldTestModule 的实例
        mod = ConstFoldTestModule()
        # 将模块进行常量折叠分割
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        # 验证常量折叠后模块的正确性
        self._verify_const_fold_mod(mod_folded)

        # 现在运行常量折叠后的模块和原始模块，以检查结果是否相等。
        in_x = torch.randn(2, 3)
        fold_result = mod_folded(in_x)
        base_result = mod(in_x)
        # 断言两个结果是否相等
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_multi_const_folded_attrs(self):
        r"""
        Perform constant folding conversion, from original mod to split constant
        folding module with two split subgraphs, where there are two attrs to
        fold into two new attrs.

           attr1        attr2          attr1     attr2
           /    \         |           /     \      |
        permute  |       sum       permute   |    sum
            \   /        /                \ /      |
         x   add    y   /                 add      |
          \ /        \ /                   |       |
          sub        add                 output  output     (become attrs add_1 and mul_1)
             \       /        ==>   --------+-------+------ (const/base subgraph split)
              \     /                   x   |   y   |       (inputs from previous subgraph
                add                      \ /     \ /         are attrs)
                 |                       sub     add
               linear                       \   /
                 |                           add
               sigmoid                        |
                 |                          linear
               output                         |
                                            sigmoid
                                              |
                                            output
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Define two trainable attributes initialized with random values
                self.attr1 = torch.nn.Parameter(torch.randn(4, 4))
                self.attr2 = torch.nn.Parameter(torch.randn(4, 4))
                # Define a linear layer
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x, y):
                # Compute a new attribute 'a' as the sum of 'attr1' and its permuted form
                a = self.attr1 + self.attr1.permute(1, 0)
                # Modify input 'x' by subtracting 'a'
                x = x - a
                # Compute the sum along dimension 1 of 'attr2'
                amax = torch.sum(self.attr2, dim=1)
                # Modify input 'y' by adding 'amax'
                y = y + amax
                # Forward pass through a linear layer and sigmoid activation
                return torch.sigmoid(self.lin(x + y))

        # Create an instance of the module
        mod = ConstFoldTestModule()
        # Perform constant folding by splitting subgraphs
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        # Verify the correctness of the folded module
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equality.
        # Generate random input tensors 'in_x' and 'in_y'
        in_x, in_y = torch.randn(4, 4), torch.randn(4)
        # Forward pass through the folded module
        fold_result = mod_folded(in_x, in_y)
        # Forward pass through the original module
        base_result = mod(in_x, in_y)
        # Assert that the outputs of both passes are equal
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_submod_hierarchy(self):
        r"""
        Perform constant folding conversion, from original mod to split constant folding
        module where one of the folded attrs comes from a submod deeper in the hierarchy
        of the base module.
        """
        # 定义一个测试方法，用于测试常量折叠的子模块层次结构
        # 将原始模块转换为拆分的常量折叠模块，在其中一个折叠的属性来自基础模块层次结构中更深处的子模块

        class TracedThroughModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个内部属性，作为一个带参数的神经网络模块
                self.internal_attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self):
                # 返回这个内部属性
                return self.internal_attr

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个TracedThroughModule的实例，作为自身的子模块
                self.my_mod = TracedThroughModule()
                # 定义一个属性，作为一个带参数的神经网络模块
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # 返回两个属性和输入张量x的总和
                return self.attr + self.my_mod() + x

        # 创建一个ConstFoldTestModule的实例
        mod = ConstFoldTestModule()
        # 对模块进行常量折叠，返回一个FoldedGraphModule的实例
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        # 验证常量折叠后的模块
        self._verify_const_fold_mod(mod_folded)

        # 现在运行折叠和非折叠的模块，以检查结果是否相等
        in_x = torch.randn(2, 3)
        # 使用折叠后的模块处理输入数据in_x
        fold_result = mod_folded(in_x)
        # 使用原始模块处理输入数据in_x
        base_result = mod(in_x)
        # 断言折叠后的结果与原始结果是否相等
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_retain_node_meta(self):
        r"""
        Perform constant folding conversion, and validate that node meta is retained.
        """

        # 定义一个测试用的神经网络模块 ConstFoldTestModule
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个模型参数 attr，形状为 (2, 3)
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # 在 forward 方法中执行常量折叠操作
                a = self.attr + self.attr
                return x - a

        # 创建 ConstFoldTestModule 实例 mod
        mod = ConstFoldTestModule()
        # 对模型进行符号化追踪，得到图模型 gm
        gm = torch.fx.symbolic_trace(mod)

        # 为每个节点添加元数据以便在常量折叠后进行验证
        for idx, node in enumerate(gm.graph.nodes):
            if node.op != "output":
                node.meta["meta_idx"] = idx

        # 对 gm 进行常量折叠，得到常量折叠后的图模型 gm_folded
        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(gm)
        # 调用验证函数，验证常量折叠后的模型
        self._verify_const_fold_mod(gm_folded)

        # 检查常量折叠后的节点索引情况
        for node in gm_folded.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(node.meta["meta_idx"], 0)
            elif node.op == "get_attr":
                self.assertEqual(node.meta["meta_idx"], 2)
            elif node.op == "call_function" and node.target == operator.sub:
                self.assertEqual(node.meta["meta_idx"], 3)
            else:
                self.assertEqual(node.op, "output")

        # 使用折叠和非折叠的模型运行，检查结果是否一致
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_has_inlined_call_module_node(self):
        # 定义一个测试用的神经网络模块 ConstFoldTestModule
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个模型参数 attr，形状为 (2, 3)
                self.attr = torch.nn.Parameter(torch.randn(2, 3))
                # 添加一个子模块 Identity 和其方法 relu
                self.mod = torch.nn.Identity()
                self.mod.relu = torch.nn.ReLU()

            def forward(self, x):
                # 在 forward 方法中执行常量折叠操作
                a = self.attr + self.attr
                return self.mod.relu(x - a)

        # 创建 ConstFoldTestModule 实例 mod
        mod = ConstFoldTestModule()
        # 对模型进行常量折叠，得到折叠后的模型 gm_folded
        gm_folded = const_fold.split_const_subgraphs(mod)

        # 使用折叠和非折叠的模型运行，检查结果是否一致
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_const_fold_module_attr(self):
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))  # 创建一个形状为 (2, 3) 的参数张量 const
                self.mod = torch.nn.Identity()  # 创建一个 Identity 模块实例 mod
                self.mod.attr = torch.nn.Parameter(torch.randn(2, 3))  # 将形状为 (2, 3) 的参数张量赋给 mod 的 attr 属性

            def forward(self, x):
                a = self.const + self.mod.attr  # 计算 const 和 mod.attr 的和，赋值给变量 a
                x = x + a  # 计算输入 x 和 a 的和，赋值给 x
                return x + self.mod.attr  # 返回 x 和 mod.attr 的和作为输出

        mod = ConstFoldTestModule()  # 创建 ConstFoldTestModule 的实例 mod
        gm_folded = const_fold.split_const_subgraphs(mod)  # 对 mod 进行常量折叠分割

        # 现在运行折叠和非折叠版本以检查结果是否相等。
        in_x = torch.randn(2, 3)  # 创建形状为 (2, 3) 的随机张量 in_x
        fold_result = gm_folded(in_x)  # 使用折叠后的模块处理输入 in_x
        base_result = mod(in_x)  # 使用原始模块处理输入 in_x
        self.assertTrue(torch.equal(fold_result, base_result))  # 断言折叠后的结果与原始结果相等

    def test_const_fold_unused_placeholder(self):
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))  # 创建一个形状为 (2, 3) 的参数张量 const

            def forward(self, x, y, z):
                a = self.const + self.const  # 计算 const 和 const 的和，赋值给变量 a
                return y + a  # 返回 y 和 a 的和作为输出

        mod = ConstFoldTestModule()  # 创建 ConstFoldTestModule 的实例 mod
        gm_folded = const_fold.split_const_subgraphs(mod)  # 对 mod 进行常量折叠分割

        # 现在运行折叠和非折叠版本以检查结果是否相等。
        in_x = torch.randn(2, 3)  # 创建形状为 (2, 3) 的随机张量 in_x
        fold_result = gm_folded(in_x, in_x, in_x)  # 使用折叠后的模块处理输入 in_x
        base_result = mod(in_x, in_x, in_x)  # 使用原始模块处理输入 in_x
        self.assertTrue(torch.equal(fold_result, base_result))  # 断言折叠后的结果与原始结果相等

    def test_dict_output(self):
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))  # 创建一个形状为 (2, 3) 的参数张量 const

            def forward(self, x):
                a = self.const + self.const  # 计算 const 和 const 的和，赋值给变量 a
                return {"result": x + a}  # 返回一个字典，包含键 "result" 对应的值为 x 和 a 的和

        mod = ConstFoldTestModule()  # 创建 ConstFoldTestModule 的实例 mod
        gm_folded = const_fold.split_const_subgraphs(mod)  # 对 mod 进行常量折叠分割

        # 现在运行折叠和非折叠版本以检查结果是否相等。
        in_x = torch.randn(2, 3)  # 创建形状为 (2, 3) 的随机张量 in_x
        fold_result = gm_folded(in_x)  # 使用折叠后的模块处理输入 in_x
        base_result = mod(in_x)  # 使用原始模块处理输入 in_x
        self.assertTrue(torch.equal(fold_result["result"], base_result["result"]))  # 断言折叠后的结果字典中 "result" 键对应的值与原始结果中相等

    def test_two_outputs(self):
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))  # 创建一个形状为 (2, 3) 的参数张量 const

            def forward(self, x):
                a = self.const + self.const  # 计算 const 和 const 的和，赋值给变量 a
                return x, x + a  # 返回两个张量，分别是输入 x 和 x+a

        mod = ConstFoldTestModule()  # 创建 ConstFoldTestModule 的实例 mod
        gm_folded = const_fold.split_const_subgraphs(mod)  # 对 mod 进行常量折叠分割

        # 现在运行折叠和非折叠版本以检查结果是否相等。
        in_x = torch.randn(2, 3)  # 创建形状为 (2, 3) 的随机张量 in_x
        fold_result = gm_folded(in_x)  # 使用折叠后的模块处理输入 in_x
        base_result = mod(in_x)  # 使用原始模块处理输入 in_x
        self.assertTrue(torch.equal(fold_result[0], base_result[0]))  # 断言折叠后的第一个输出与原始结果的第一个输出相等
        self.assertTrue(torch.equal(fold_result[1], base_result[1]))  # 断言折叠后的第二个输出与原始结果的第二个输出相等
    def test_three_outputs(self):
        # 定义一个测试函数，测试模块包含常量折叠的情况
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个随机的参数张量
                self.const = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # 在前向传播中使用常量张量进行加法操作
                a = self.const + self.const
                # 返回三个结果：原始输入，原始输入与加法结果相加，再次与加法结果相加
                return x, x + a, x + a

        # 创建一个 ConstFoldTestModule 的实例
        mod = ConstFoldTestModule()
        # 对模块进行常量折叠，得到折叠后的图模块
        gm_folded = const_fold.split_const_subgraphs(mod)

        # 现在运行折叠后和未折叠前的模块，检查结果是否相等
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        # 断言三个输出张量的内容是否完全一致
        self.assertTrue(torch.equal(fold_result[0], base_result[0]))
        self.assertTrue(torch.equal(fold_result[1], base_result[1]))
        self.assertTrue(torch.equal(fold_result[2], base_result[2]))

    def test_check_inline_non_const(self):
        r"""
        执行常量折叠转换，并检查非常量模块是否正确内联。
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个随机的参数张量
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # 在前向传播中使用参数张量进行加法操作
                a = self.attr + self.attr
                # 返回经过复杂计算的结果
                return (x - a * x) / 2

        # 创建一个 ConstFoldTestModule 的实例
        mod = ConstFoldTestModule()
        # 使用 symbolic_trace 将模块转换为图模块
        gm = torch.fx.symbolic_trace(mod)

        # 对模块进行常量折叠，得到折叠后的图模块
        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(gm)
        # 调用方法验证常量折叠后的模块是否正确
        self._verify_const_fold_mod(gm_folded)

        # 检查图中是否没有调用模块，因为它们已经被内联或提取用于常量折叠
        for node in gm_folded.graph.nodes:
            self.assertNotEqual(node.op, "call_module")

        # 现在运行折叠后和未折叠前的模块，检查结果是否相等
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        # 断言折叠后的结果与原始模块的结果是否完全一致
        self.assertTrue(torch.equal(fold_result, base_result))
    def test_check_inline_non_const_mult_return(self):
        r"""
        Perform constant folding conversion and check that the non-const module is inlined
        correctly.
        """

        # 定义一个测试用的神经网络模块 ConstFoldTestModule
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个形状为 (2, 3) 的参数，内容为随机数的张量，并将其设为模块的属性
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # 在模型的前向传播过程中，使用模型的属性进行加法操作
                a = self.attr + self.attr
                # 返回两个张量：一个是输入 x 减去 a 的结果，另一个是输入 x 除以 2 的结果
                return x - a, x / 2

        # 创建 ConstFoldTestModule 的实例 mod
        mod = ConstFoldTestModule()
        # 对模型进行符号跟踪，得到图形模块 gm
        gm = torch.fx.symbolic_trace(mod)

        # 对图形模块 gm 进行常量折叠操作，得到折叠后的图形模块 gm_folded
        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(gm)
        # 调用 _verify_const_fold_mod 方法验证常量折叠后的模块是否正确
        self._verify_const_fold_mod(gm_folded)

        # 检查图形模块 gm_folded 中是否存在调用模块，因为它们可能已被内联或提取用于常量折叠
        for node in gm_folded.graph.nodes:
            self.assertNotEqual(node.op, "call_module")

        # 现在运行折叠和非折叠的模块，检查结果是否相等
        in_x = torch.randn(2, 3)
        # 对折叠后的模块 gm_folded 进行前向传播得到结果 fold_result
        fold_result = gm_folded(in_x)
        # 对原始模块 mod 进行前向传播得到基准结果 base_result
        base_result = mod(in_x)
        # 断言折叠后的结果和基准结果的第一个张量是否相等
        self.assertTrue(torch.equal(fold_result[0], base_result[0]))
        # 断言折叠后的结果和基准结果的第二个张量是否相等
        self.assertTrue(torch.equal(fold_result[1], base_result[1]))
    def test_check_skip_folding_quant_dequant_pattern(self):
        r"""
        Set up skip_folding_quant_dequant function to skip quant/dequant pattern.
        This example shows how to use skip_folding_node_fn.
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                quant_weight = torch.quantize_per_tensor(
                    self.weight, 0.5, 3, torch.quint8
                )
                dequant_weight = torch.dequantize(quant_weight)
                output = torch.nn.functional.linear(x, dequant_weight, self.bias)
                return self.relu(output)

        mod = ConstFoldTestModule()
        in_x = torch.randn(2, 4)
        gm = torch.fx.symbolic_trace(mod)

        def skip_folding_quant_dequant(node: torch.fx.Node):
            # 检查节点是否为 torch.quantize_per_tensor 函数调用
            if node.target != torch.quantize_per_tensor:
                return False
            # 检查该节点的用户中是否包含 torch.dequantize 函数调用
            for user in node.users:
                if user.target == torch.dequantize:
                    return True
            return False

        # 将模型中的常量子图分割，并跳过指定的节点模式
        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            gm, skip_folding_node_fn=skip_folding_quant_dequant
        )

        # 检查常量折叠后的图模块是否为 None，因为没有进行折叠操作
        self.assertTrue(gm_folded.const_subgraph_module is None)

        # 现在运行折叠和非折叠的模型来检查结果是否相等
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_fold_module(self):
        r"""
        Perform constant folding with a call_module node.
        """

        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_input = torch.nn.Parameter(torch.randn(4, 4))
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(self.lin_input) + x

        mod = ConstFoldTestModule()
        
        # 对模型中的常量子图进行折叠操作
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        
        # 验证常量折叠后的模型
        self._verify_const_fold_mod(mod_folded)

        # 现在运行折叠和非折叠的模型来检查结果是否相等
        inp = torch.randn(4, 4)
        self.assertTrue(torch.equal(mod_folded(inp), mod(inp)))

    def test_const_fold_tensor_meta(self):
        # 测试对张量元数据进行常量折叠
        self._test_const_fold_tensor_meta(True)
        self._test_const_fold_tensor_meta(False)
    def _test_const_fold_tensor_meta(self, requires_grad):
        """
        Verify tensor_meta is handled correctly.
        """

        # 定义一个测试用的模块，继承自torch.nn.Module
        class ConstFoldTestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义两个可训练参数，并指定是否需要梯度
                self.attr_1 = torch.nn.Parameter(torch.tensor([[-0.9]]), requires_grad)
                self.attr_2 = torch.nn.Parameter(torch.tensor([[17.1]]), requires_grad)

            def forward(self, x, y):
                # 在前向传播中进行张量操作
                a = self.attr_1 + self.attr_1  # 计算两次self.attr_1的和
                x = x - a  # 将x减去a
                return x * y + self.attr_2  # 返回计算结果

        # 创建ConstFoldTestModule的实例
        mod = ConstFoldTestModule()
        # 对模块进行符号跟踪，生成图形化表示
        gm = torch.fx.symbolic_trace(mod)
        # 定义输入张量
        in_x, in_y = torch.tensor([[-0.45]]), torch.tensor([0.9])
        # 使用ShapeProp对象传播输入形状信息
        ShapeProp(gm).propagate(in_x, in_y)
        # 将图形化模块转化为常量折叠图形模块
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            gm, device_for_folded_attrs="cpu"
        )
        # 验证常量折叠后的模块
        self._verify_const_fold_mod(mod_folded)

        # 运行常量折叠操作
        mod_folded.run_folding()

        # 遍历折叠后模块的所有节点
        for n in mod_folded.graph.nodes:
            if n.op == "get_attr":
                # 获取节点对应的属性
                attr = self._get_attr(n)
                # 将属性中提取的张量元数据与节点的meta信息进行比较验证
                self.assertEqual(_extract_tensor_metadata(attr), n.meta["tensor_meta"])

        # 分别运行折叠前后的模块，验证结果是否相等
        base_result = mod(in_x, in_y)
        fold_result = mod_folded(in_x, in_y)
        # 断言折叠后的结果与折叠前的结果相等
        self.assertTrue(torch.equal(fold_result, base_result))
```