# `.\pytorch\test\jit\test_profiler.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os
import sys

import torch
from torch.testing._internal.common_utils import skipIfTorchDynamo

# Make the helper files in test/ importable
# 获取当前脚本所在目录，并将其加入系统路径，以便导入测试文件
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import FileCheck, JitTestCase, warmup_backward

# 如果作为主程序运行，则抛出运行时错误，建议通过指定的方式运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 装饰器，跳过 Torch Dynamo 模式下的测试
@skipIfTorchDynamo()
class TestProfiler(JitTestCase):
    def setUp(self):
        # 设置 JIT 分析执行器为开启状态，并保存之前的状态
        self.prev_exec = torch._C._jit_set_profiling_executor(True)
        # 设置 JIT 图优化为开启状态，并保存之前的状态
        self.prev_profiling = torch._C._get_graph_executor_optimize(True)
        # 设置自动微分子图内联为关闭状态，并保存之前的状态
        self.inline_autodiff = torch._C._debug_set_autodiff_subgraph_inlining(False)
        # 获取当前是否启用了 TExpr 融合，保存之前的状态
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        # 获取当前 CPU 是否支持融合，保存之前的状态
        self.can_fuse_on_cpu = torch._C._jit_can_fuse_on_cpu()
        # 强制开启 TExpr 融合
        torch._C._jit_set_texpr_fuser_enabled(True)
        # 覆盖允许 CPU 上的融合
        torch._C._jit_override_can_fuse_on_cpu(True)
        # 保存当前默认的数据类型，并设置为双精度浮点型
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)
        # 获取当前是否启用了 TExpr 减少操作优化，保存之前的状态，并强制开启
        self.old_reduction_enabled = torch._C._jit_set_texpr_reductions_enabled(True)
        # 获取当前融合组内联策略，保存之前的状态，并设置为关闭
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        # 获取当前是否必须使用 LLVM CPU，保存之前的状态，并设置为不强制
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

    def tearDown(self):
        # 恢复 JIT 分析执行器的状态
        torch._C._jit_set_profiling_executor(self.prev_exec)
        # 恢复 JIT 图优化的状态
        torch._C._get_graph_executor_optimize(self.prev_profiling)
        # 恢复自动微分子图内联的状态
        torch._C._debug_set_autodiff_subgraph_inlining(self.inline_autodiff)
        # 恢复 TExpr 融合的状态
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        # 恢复 CPU 融合的状态
        torch._C._jit_override_can_fuse_on_cpu(self.can_fuse_on_cpu)
        # 恢复默认的数据类型
        torch.set_default_dtype(self.default_dtype)
        # 恢复 TExpr 减少操作优化的状态
        torch._C._jit_set_texpr_reductions_enabled(self.old_reduction_enabled)
        # 恢复融合组内联策略的状态
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        # 恢复是否必须使用 LLVM CPU 的状态
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)
    # 定义一个测试函数，用于验证张量类型是否由输入决定
    def test_tensor_type_not_determined_by_inputs(self):
        # 使用 Torch Script 注解将函数标记为脚本化
        @torch.jit.script
        def scalar_type_input(x, y, z):
            # 执行张量计算：x + y + 4 + z.item()
            return x + y + 4 + z.item()

        # 创建一个形状为 [2, 2] 的整型张量 x
        x = torch.tensor([2, 2])
        # 调用脚本化函数 scalar_type_input，传入 x, x, 以及整型张量 [1]
        scalar_type_input(x, x, torch.tensor(1))
        # 再次调用 scalar_type_input，传入相同的参数
        scalar_type_input(x, x, torch.tensor(1))
        # 第三次调用 scalar_type_input，但是参数 z 是浮点型张量 [1.0]
        scalar_type_input(x, x, torch.tensor(1.0))
        # 获取最近执行的优化图
        g = torch.jit.last_executed_optimized_graph()

        # 验证在优化图中，item 和 add 操作没有被合并到融合组中
        # 期望在 IR dump 中看到 Fusion Group (item / add) Fusion Group
        FileCheck().check("TensorExpr").check("Scalar = aten::item").check_next(
            "Tensor = aten::add"
        ).check("TensorExpr").run(g)

        # 定义一个新的 Torch Script 函数，用于验证非常数数据类型的情况
        @torch.jit.script
        def non_const_dtype(x, y, cond: bool):
            # 根据条件选择数据类型为 torch.int16 或 torch.int32
            dtype = torch.int16 if cond else torch.int32
            # 计算 (x + y + 3) 的和，并指定数据类型为 dtype
            return (x + y + 3).sum(dtype=dtype)

        # 调用 non_const_dtype 函数两次，传入相同的参数 x, x 和 True
        non_const_dtype(x, x, True)
        non_const_dtype(x, x, True)
        # 获取最近执行的优化图
        g = torch.jit.last_executed_optimized_graph()

        # 因为数据类型不是常量，所以 sum 操作不应该被合并到融合组中
        FileCheck().check("TensorExpr").check("TensorExpr").check_not("aten::sum").run(
            g
        )

    # 定义一个测试函数，用于测试特化反向传播
    def test_specialize_backward(self):
        # 定义一个函数 test_fuse，计算 a * b * b 的结果
        def test_fuse(a, b):
            c = a * b
            d = c * b
            return d

        # 禁用 Torch JIT 函数缓存
        test_fuse.__disable_jit_function_caching__ = True

        # 将 test_fuse 函数脚本化
        scripted_f = torch.jit.script(test_fuse)
        # 创建一个 requires_grad=True 的大小为 [1] 的张量 x
        x = torch.ones(1, requires_grad=True)
        # 创建一个 requires_grad=True 的大小为 [1] 的张量 y
        y = torch.ones(1, requires_grad=True)
        # 调用脚本化函数 scripted_f，传入 x, y
        scripted_f(x, y)
        # 再次调用 scripted_f，传入相同的参数 x, y
        b = scripted_f(x, y)
        # 执行一次反向传播的预热
        warmup_backward(b)
        # 获取最近执行的优化图
        g = torch.jit.last_executed_optimized_graph()

        # 验证反向传播中是否存在 if 节点来保护特化版本，
        # 在 if 节点的 true 分支中是否只有一个保护 tensorexpr 组的 if 节点
        optimized_block = next(g.findNode("prim::If").blocks())
        if_nodes = list(optimized_block.findAllNodes("prim::If"))
        self.assertEqual(len(if_nodes), 1)
        FileCheck().check("Group[Subgraph").run(str(if_nodes[0]))
        # 验证没有广播发生，sum_to_size 被特化掉了
        self.assertIsNone(optimized_block.findNode("aten::_grad_sum_to_size"))

        # 再次脚本化 test_fuse 函数
        broadcast_f = torch.jit.script(test_fuse)
        # 创建一个大小为 [2, 2] 的 requires_grad=True 的张量 x
        x = torch.ones([2, 2], requires_grad=True)
        # 创建一个大小为 [1] 的 requires_grad=True 的张量 y
        y = torch.ones([1], requires_grad=True)
        # 调用脚本化函数 broadcast_f，传入 x, y
        broadcast_f(x, y)
        # 再次调用 broadcast_f，传入相同的参数 x, y
        b = broadcast_f(x, y)
        # 对 b 执行反向传播，使用大小为 [2, 2] 的浮点张量，保留计算图
        b.backward(torch.ones([2, 2], dtype=torch.float), retain_graph=True)
        # 对 b 再次执行反向传播，使用大小为 [2, 2] 的浮点张量
        b.backward(torch.ones([2, 2], dtype=torch.float))
        # 获取最近执行的优化图
        g = torch.jit.last_executed_optimized_graph()
        
        # 获取优化图中的优化块
        optimized_block = next(g.findNode("prim::If").blocks())
        # 验证是否存在广播，目前期望看到 aten::_grad_sum_to_size
        self.assertIsNotNone(optimized_block.findNode("aten::_grad_sum_to_size"))
    # 定义测试方法，用于测试特定类型的操作
    def test_specialized_types(self):
        # 使用 Torch Script 注解定义一个函数 test_fuse，对输入的张量 a 和 b 执行一系列数学运算
        @torch.jit.script
        def test_fuse(a, b):
            # 计算 a 和 b 的乘积
            c = a * b
            # 计算 c 和 b 的乘积
            d = c * b
            # 返回结果 d
            return d

        # 创建一个包含单个浮点数值的张量 x
        x = torch.tensor([0.5])
        # 多次调用 test_fuse 函数，对 x 进行操作
        for _ in range(3):
            test_fuse(x, x)

        # 获取最近执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 进行类型检查和融合操作的类型检查输出应保持特化
        FileCheck().check("Double(").check_same("prim::TypeCheck").check_same(
            "\n"
        ).check("Double").check_same("TensorExpr").run(g)

        # 其他输出不应特化
        FileCheck().check("Tensor = prim::If").run(g)

    # 定义测试方法，用于测试别名合并的情况
    def test_aliasing_merge(self):
        # 使用 Torch Script 注解定义一个函数 foo，对输入的张量 a 和 b 执行一系列数学操作
        @torch.jit.script
        def foo(a, b):
            # 计算 a 和 b 的乘积
            c = a * b
            # 计算 c 和 b 的乘积
            d = c * b
            # 将 b 添加到 d 上（就地操作）
            d.add_(b)
            # 计算 d 和 b 的乘积
            e = d * b
            # 返回 d 和 e 的和
            return d + e

        # 创建包含单个元素值为 1 的张量 x 和 y
        x = torch.ones(1)
        y = torch.ones(1)
        # 调用 foo 函数，对 x 和 y 执行操作
        foo(x, y)
        # 再次调用 foo 函数，对 x 和 y 执行操作
        b = foo(x, y)
        # 获取最近执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 断言存在两个 prim::TypeCheck 节点
        self.assertEqual(len(list(g.findAllNodes("prim::TypeCheck"))), 2)
        # 检查图形中是否包含特定的操作序列
        FileCheck().check("TensorExpr").check("aten::add_").check("TensorExpr").run(g)

    # 定义测试方法，用于测试未被分析的情况
    def test_use_not_profiled(self):
        # 定义函数 foo，对输入的四个张量执行求和操作，根据条件选择性返回结果
        def foo(t1, t2, t3, t4, t: float):
            # 对 t1、t2、t3、t4 四个张量进行求和
            h = t1 + t2 + t3 + t4
            # 如果 t 大于 0.5，则执行条件分支（实际上不会执行）
            if t > 0.5:
                # 在永远不执行的条件分支中使用 t1，防止对其进行优化
                return t1 + 1
            # 返回求和结果 h
            return h

        # 创建一个包含 8 个随机浮点数值的张量 t
        t = torch.rand(8, dtype=torch.float)

        # 将函数 foo 转换为 Torch Script 形式
        foo_script = torch.jit.script(foo)
        # 多次调用 foo_script 函数，对 t 进行操作
        for _ in range(torch._C._jit_get_num_profiled_runs() + 1):
            foo_script(t, t, t, t, 0.1)

        # 断言未分析的 foo 函数与分析后的 foo_script 函数返回相同的结果
        self.assertEqual(foo(t, t, t, t, 0.1), foo_script(t, t, t, t, 0.1))
        # 获取最近执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 断言所有的加法操作都已融合
        FileCheck().check("graph").check_not("aten::add").check("prim::If").run(g)

    # 定义测试方法，用于测试不融合标量操作的情况
    def test_not_fusing_scalar_ops(self):
        # 使用 Torch Script 注解定义一个函数 foo，对输入的整数 x 和 y 执行一系列加法操作
        @torch.jit.script
        def foo(x: int, y: int):
            return x + y + 2 + 4 + 5 + 6

        # 调用 foo 函数，对输入参数执行一系列加法操作
        foo(1, 2)
        foo(2, 3)
        # 获取最近执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 检查图形中是否不包含 TensorExpr
        FileCheck().check_not("TensorExpr").run(g)

    # 定义测试方法，用于测试不优化属性操作的情况
    def test_not_optimizing_property(self):
        # 使用 Torch Script 注解定义一个函数 foo，对输入的张量 x 和 y 执行一系列加法和属性操作
        @torch.jit.script
        def foo(x, y):
            return x + y + 1 + 2 + 3, x.size()

        # 创建一个包含单个元素值为 1 的张量 x
        x = torch.ones(1)
        # 调用 foo 函数，对 x 和 x 执行操作
        foo(x, x)
        foo(x, x)
        # 获取最近执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 检查图形中是否包含特定的操作序列
        FileCheck().check("aten::size").run(g)
        # 创建一个形状为 [2, 3, 5] 的张量 x
        x = torch.ones([2, 3, 5])
        # 断言优化后的 foo 函数与未优化的 foo 函数返回相同的结果
        self.assertEqual(foo(x, x), (x + x + 1 + 2 + 3, x.size()))

    # 定义测试方法，用于测试回退图形未特化的情况
    def test_fallback_graph_not_specialized(self):
        # 使用 Torch Script 注解定义一个函数 foo，对输入的张量 a 和 b 执行一系列数学操作
        @torch.jit.script
        def foo(a, b):
            # 计算 a 和 b 的乘积
            c = a * b
            # 计算 c 和 b 的乘积
            d = c * b
            # 计算 d 和 b 的乘积
            e = d * b
            # 返回 d 和 e 的和
            return d + e

        # 创建包含单个元素值为 1 的张量 x 和 y
        x = torch.ones(1)
        y = torch.ones(1)
        # 调用 foo 函数，对 x 和 y 执行操作
        foo(x, y)
        foo(x, y)
        # 获取最近执行的优化图形
        g = torch.jit.last_executed_optimized_graph()
        # 检查图形中是否包含特定的操作序列
        FileCheck().check("CallFunction").check_next("Tensor = prim::TupleUnpack").run(
            g
        )
    # 定义一个测试函数，用于测试自动微分回退图形
    def test_autograd_fallback_graph(self):
        # 定义一个 TorchScript 函数 foo，接受两个参数 a 和 b
        @torch.jit.script
        def foo(a, b):
            # 计算 c = a * b
            c = a * b
            # 计算 d = c * b
            d = c * b
            # 计算 e = d * b
            e = d * b
            # 返回 d + e
            return d + e

        # 创建两个值为 1 的张量 x 和 y，并标记需要梯度计算
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # 调用 foo 函数，计算结果但不保存
        foo(x, y)
        # 再次调用 foo 函数，计算结果并保存在变量 b 中
        b = foo(x, y)
        # 对 b 进行反向传播，梯度为 [1.0]，保留计算图
        b.backward(torch.ones([1], dtype=torch.float), retain_graph=True)
        # 对 b 进行反向传播，梯度为 [1.0]，不保留计算图
        b.backward(torch.ones([1], dtype=torch.float))

        # 获取最后执行的优化图形 g
        g = torch.jit.last_executed_optimized_graph()
        # 使用 FileCheck 验证优化图形中包含 "fallback_function" 和 "CallFunction" 的内容
        FileCheck().check("fallback_function").check_next("CallFunction").run(g)

    # 定义一个测试函数，用于测试张量常量
    def test_tensor_constant(self):
        # 定义一个普通的 Python 函数 foo，接受两个参数 a 和 b
        def foo(a, b):
            # 返回 a + b + [2]
            return a + b + torch.tensor([2])

        # 创建一个值为 1 的张量 x，并标记不需要梯度计算
        x = torch.ones(1, requires_grad=False)
        # 将 foo 函数转换为 TorchScript 函数
        foo_script = torch.jit.script(foo)
        # 调用 foo_script 函数两次，传入相同的参数 x
        foo_script(x, x)
        foo_script(x, x)

        # 使用断言检查 foo_script(x, x) 的结果与 foo(x, x) 的结果是否相等
        self.assertEqual(foo_script(x, x), foo(x, x))
        # 获取最后执行的优化图形 g
        g = torch.jit.last_executed_optimized_graph()
        # 使用 FileCheck 验证优化图形中 "aten::add" 出现的次数为 2
        FileCheck().check_count("aten::add", 2, exactly=True).run(g)

    # 定义一个测试函数，用于测试本地融合策略
    def test_local_fusion_strategy(self):
        # 定义一个 TorchScript 函数 foo，接受一个参数 x
        @torch.jit.script
        def foo(x):
            # 返回 x + x + x
            return x + x + x

        # 设置静态融合策略 [("STATIC", 1)]
        torch.jit.set_fusion_strategy([("STATIC", 1)])
        # 多次循环调用 foo 函数，传入形状为 [10] 的随机张量
        for _ in range(3):
            foo(torch.rand([10]))

        # 设置静态融合策略 [("STATIC", 10)]
        torch.jit.set_fusion_strategy([("STATIC", 10)])

        # 循环调用 foo 函数，传入形状为 [i] 的随机张量，其中 i 从 0 到 9
        for i in range(10):
            foo(torch.rand([i]))
            foo(torch.rand([i]))

        # 获取最后执行的优化图形 g
        g = torch.jit.last_executed_optimized_graph()
        # 使用 FileCheck 验证优化图形中 ":TensorExprGroup" 出现的次数为 2
        FileCheck().check_count(":TensorExprGroup", 2, exactly=True).run(g)

    # 定义一个测试函数，用于测试迭代融合
    def test_iterative_fusion(self):
        # 定义一个 TorchScript 函数 foo，接受四个参数 a、b、c、d
        @torch.jit.script
        def foo(a, b, c, d):
            # 计算 a = a + b
            a = a + b
            # 在 b 上执行原地加法 b.add_(3)
            b.add_(3)
            # 计算 c = c + b + d
            c = c + b + d
            # 计算 a = a + 1
            a = a + 1
            # 返回 a 和 c
            return a, c

        # 创建一个值为 1 的张量 x，并标记不需要梯度计算
        x = torch.ones(1, requires_grad=False)
        # 多次循环调用 foo 函数，传入相同的参数 x
        foo(x, x, x, x)
        foo(x, x, x, x)

        # 获取最后执行的优化图形 g
        g = torch.jit.last_executed_optimized_graph()
        # 使用断言检查在 g 中找到的 "prim::TensorExprGroup" 节点的数量是否为 2
        self.assertEqual(len(list(g.findAllNodes("prim::TensorExprGroup"))), 2)
```