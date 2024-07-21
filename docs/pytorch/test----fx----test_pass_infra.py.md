# `.\pytorch\test\fx\test_pass_infra.py`

```
# Owner(s): ["module: fx"]

# 导入必要的库和模块
import torch
import torch.fx as fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.infra.pass_manager import (
    _topological_sort_passes,
    pass_result_wrapper,
    PassManager,
    this_before_that_pass_constraint,
)

# 导入测试相关的工具类
from torch.testing._internal.common_utils import TestCase


# PassBase 类型的优化通行证，将所有 torch.add 替换为 torch.mul
class ReplaceAddWithMulPass(PassBase):
    def call(self, gm) -> PassResult:
        modified = False
        # 遍历计算图的所有节点
        for node in gm.graph.nodes:
            # 如果节点是调用函数且目标是 torch.add，则替换为 torch.mul
            if node.op == "call_function" and node.target == torch.add:
                node.target = torch.mul
                modified = True
        # 返回优化后的 PassResult 对象
        return PassResult(gm, modified)


# 可调用的优化通行证函数，将所有 torch.mul 替换为 torch.div
def replace_mul_with_div_pass(gm) -> PassResult:
    modified = False
    # 遍历计算图的所有节点
    for node in gm.graph.nodes:
        # 如果节点是调用函数且目标是 torch.mul，则替换为 torch.div
        if node.op == "call_function" and node.target == torch.mul:
            node.target = torch.div
            modified = True
    # 返回优化后的 PassResult 对象
    return PassResult(gm, modified)


# PassBase 类型的优化通行证，将所有 torch.div 替换为 torch.sub
class ReplaceDivWithSubPass(PassBase):
    def call(self, gm) -> PassResult:
        # 遍历计算图的所有节点
        for node in gm.graph.nodes:
            # 如果节点是调用函数且目标是 torch.div，则替换为 torch.sub
            if node.op == "call_function" and node.target == torch.div:
                node.target = torch.sub


# 可调用的优化通行证函数，将所有 torch.sub 替换为 torch.add
def replace_sub_with_add_pass(gm) -> PassResult:
    # 遍历计算图的所有节点
    for node in gm.graph.nodes:
        # 如果节点是调用函数且目标是 torch.sub，则替换为 torch.add
        if node.op == "call_function" and node.target == torch.sub:
            node.target = torch.add


# 示例模块，包含一个简单的神经网络模型
class AddModule(torch.nn.Module):
    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


# PassManager 的测试用例类，继承自 TestCase
class TestPassManager(TestCase):
    def test_pass_manager(self):
        """
        Tests that the pass manager runs the passes correctly.
        """

        # 创建示例模型
        m = AddModule()
        # 对模型进行符号化追踪
        traced_m = torch.fx.symbolic_trace(m)
        # 创建 PassManager 对象，传入四个优化通行证和迭代次数
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                pass_result_wrapper(ReplaceDivWithSubPass()),
                pass_result_wrapper(replace_sub_with_add_pass),
            ],
            steps=5,
        )

        # 验证优化通行证的约束条件
        pm.validate_constraints()
        # 断言 PassManager 中包含四个优化通行证
        self.assertEqual(len(pm.passes), 4)

        # 执行 PassManager 对追踪后的模型进行优化
        res = pm(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, fx.GraphModule)

        # 检查修改后的计算图中所有的调用函数节点是否为 torch.add
        for node in modified_m.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, torch.add)
    def test_this_before_that_pass_constraint(self):
        """
        Tests the construction of constraints
        """
        # 创建一个包含 10 个 lambda 函数的列表，每个函数都是一个计算 2 * x 的函数
        passes = [lambda x: 2 * x for _ in range(10)]
        # 创建 PassManager 对象，将 passes 列表作为参数传入
        pm = PassManager(passes)

        # 添加一个无法实现的约束条件
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        # 断言运行时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            pm.validate_constraints()

    def test_pass_manager_checks(self):
        """
        Tests that users can add in check functions correctly
        """
        # 创建 AddModule 对象
        m = AddModule()
        # 对 AddModule 进行符号跟踪
        traced_m = fx.symbolic_trace(m)
        # 创建 PassManager 对象，指定两个转换函数作为转换 passes
        pm = PassManager(passes=[ReplaceAddWithMulPass(), replace_mul_with_div_pass])

        # 定义一个检查函数 check_div_target
        def check_div_target(graph_module):
            # 遍历图中的每个节点
            for node in graph_module.graph.nodes:
                # 如果节点是 "call_function" 类型且目标不是 torch.div 函数，则抛出 ValueError 异常
                if node.op == "call_function" and node.target != torch.div:
                    raise ValueError("Target should be div!")

        # 将 check_div_target 函数添加到 PassManager 的检查函数列表中
        pm.add_checks(check_div_target)

        # 断言运行时会抛出 ValueError 异常
        with self.assertRaises(ValueError):
            pm(traced_m)

    def test_pass_manager_bad_checks(self):
        """
        Checks that we error if we pass in a check function with the wrong parameters
        """

        # 定义一个错误的检查函数 check_bad_args，它有两个参数而不是一个
        def check_bad_args(graph_module, i):
            pass

        # 创建 PassManager 对象
        pm = PassManager()
        # 断言调用 pm.add_checks(check_bad_args) 时会抛出 TypeError 异常
        self.assertRaises(TypeError, pm.add_checks, check_bad_args)
    def test_topological_sort(self):
        """
        Tests that passes are correctly ordered based on constraints.
        测试通过约束条件正确排序的函数。
        """

        def pass0(x):
            return x
        # 定义一个函数 pass0，返回输入的参数 x

        def pass1(x):
            return x + 1
        # 定义一个函数 pass1，返回输入的参数 x 加 1

        def pass2(x):
            return x + 2
        # 定义一个函数 pass2，返回输入的参数 x 加 2

        def pass3(x):
            return x + 3
        # 定义一个函数 pass3，返回输入的参数 x 加 3

        def pass4(x):
            return x + 4
        # 定义一个函数 pass4，返回输入的参数 x 加 4

        def pass5(x):
            return x + 5
        # 定义一个函数 pass5，返回输入的参数 x 加 5

        # Not passing any constraints should keep the original order
        # 如果没有任何约束条件，则保持原始顺序
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        # 创建一个函数列表 passes 包含 pass0 到 pass5
        sorted = _topological_sort_passes(passes, [])
        # 调用函数 _topological_sort_passes 对 passes 进行拓扑排序，传入空的约束列表
        self.assertEqual(sorted, passes)
        # 断言排序后的结果与 passes 相同

        # Graph that we are constructing:
        #     5 ---->  0  <---- 4
        #     |                 |
        #     +-> 2 -> 3 -> 1 <-+
        # Which has a possible topological order of: [4, 5, 0, 2, 3, 1]
        # 我们正在构建的图：
        #     5 ---->  0  <---- 4
        #     |                 |
        #     +-> 2 -> 3 -> 1 <-+
        # 其可能的拓扑顺序为：[4, 5, 0, 2, 3, 1]
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        # 创建一个函数列表 passes 包含 pass0 到 pass5
        constraints = [
            this_before_that_pass_constraint(pass5, pass0),
            this_before_that_pass_constraint(pass5, pass2),
            this_before_that_pass_constraint(pass4, pass0),
            this_before_that_pass_constraint(pass4, pass1),
            this_before_that_pass_constraint(pass2, pass3),
            this_before_that_pass_constraint(pass3, pass1),
        ]
        # 创建一个约束条件列表 constraints，规定函数执行顺序
        sorted = _topological_sort_passes(passes, constraints)
        # 调用函数 _topological_sort_passes 对 passes 进行拓扑排序，传入约束条件 constraints
        self.assertEqual(sorted, [pass4, pass5, pass0, pass2, pass3, pass1])
        # 断言排序后的结果与预期的顺序相同

        # Circular dependency should result in the circular_dep flag being set
        # 循环依赖应导致 circular_dep 标志设置
        passes = [pass0, pass1, pass2]
        # 创建一个函数列表 passes 包含 pass0 到 pass2
        constraints = [
            this_before_that_pass_constraint(passes[0], passes[1]),
            this_before_that_pass_constraint(passes[1], passes[2]),
            this_before_that_pass_constraint(passes[2], passes[0]),
        ]
        # 创建一个约束条件列表 constraints，规定函数执行顺序，其中存在循环依赖
        with self.assertRaises(RuntimeError) as e:
            _topological_sort_passes(passes, constraints)
        # 使用断言捕获 RuntimeError 异常
        expected_error_msg = (
            f"Circular dependency detected within the following passes: {passes}"
        )
        # 期望的错误消息，指明发现了循环依赖的函数列表
        self.assertEqual(e.exception.args[0], expected_error_msg)
        # 断言捕获的异常消息与预期的错误消息相同
    def test_pass_manager_error(self):
        """
        Tests error catching + debug
        """

        # 定义一个内部函数，用于抛出运行时错误
        def pass_fail(graph_module):
            raise RuntimeError("bad")

        # 创建一个 AddModule 实例
        m = AddModule()
        # 对 AddModule 进行符号化跟踪
        traced_m = torch.fx.symbolic_trace(m)
        # 创建一个 PassManager 实例，包含多个转换 passes
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),  # 替换加法为乘法的转换 pass
                replace_mul_with_div_pass,  # 替换乘法为除法的转换 pass
                ReplaceDivWithSubPass(),  # 替换除法为减法的转换 pass
                pass_result_wrapper(replace_sub_with_add_pass),  # 调用结果包装器转换 pass
            ],
        )

        # 设置错误消息的正则表达式，用于断言捕获的异常信息
        error_msg = (
            "ReplaceDivWithSubPass.*ReplaceAddWithMulPass.*replace_mul_with_div_pass"
        )
        # 使用断言来检查 PassManager 的运行是否引发了预期的异常
        with self.assertRaisesRegex(Exception, error_msg):
            pm(traced_m)

        # 创建另一个 PassManager 实例，包含一个会引发错误的转换 pass
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),  # 替换加法为乘法的转换 pass
                replace_mul_with_div_pass,  # 替换乘法为除法的转换 pass
                pass_result_wrapper(ReplaceDivWithSubPass()),  # 调用结果包装器转换 pass
                pass_result_wrapper(replace_sub_with_add_pass),  # 调用结果包装器转换 pass
                pass_fail,  # 会引发运行时错误的转换 pass
            ],
        )

        # 设置错误消息的正则表达式，用于断言捕获的异常信息
        error_msg = "pass_fail.*ReplaceAddWithMulPass.*replace_mul_with_div_pass.*ReplaceDivWithSubPass.*replace_sub_with_add_pass"
        # 使用断言来检查 PassManager 的运行是否引发了预期的异常
        with self.assertRaisesRegex(Exception, error_msg):
            pm(traced_m)
```