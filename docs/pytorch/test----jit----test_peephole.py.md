# `.\pytorch\test\jit\test_peephole.py`

```py
# Owner(s): ["oncall: jit"]

# 引入单元测试模块
import unittest
# 引入类型提示相关模块
from typing import Callable, List

# 引入 PyTorch 核心模块
import torch
# 引入 PyTorch 中的神经网络模块
from torch import nn
# 引入 PyTorch 中的文件检查工具
from torch.testing import FileCheck
# 引入 PyTorch 内部的 JIT 测试工具
from torch.testing._internal.jit_utils import _inline_everything, JitTestCase, RUN_CUDA

# 如果该模块作为主程序执行，则抛出运行时错误，提示正确的执行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自 JitTestCase
class TestPeephole(JitTestCase):
    
    # 测试函数：测试带有写操作的情况
    def test_peephole_with_writes(self):
        # 定义内部函数 test_write，接受参数 x
        def test_write(x):
            # 初始化变量 s 为 0
            s = 0
            # 将 x 加到 s 上
            s += x
            # 再次将 x 加到 s 上
            s += x
            # 返回 s
            return s

        # 对 test_write 进行脚本化检查，传入参数为 torch.ones(4, 4)
        self.checkScript(test_write, (torch.ones(4, 4),))

    # 测试函数：测试带有非输出写操作的情况
    def test_peephole_with_non_output_writes(self):
        # 使用 @torch.jit.ignore 装饰器定义内部函数 nomnom，接受参数 x
        @torch.jit.ignore
        def nomnom(x):
            pass

        # 定义内部函数 test_write，接受参数 x
        def test_write(x):
            # 创建 t，其形状与 x 相同且所有元素为 1
            t = torch.ones_like(x)
            # 克隆 x 到 z
            z = x.clone()
            # 将 z 加 0，并赋值给 y
            y = z + 0
            # 将 t 加到 z 上（原地加法）
            z.add_(t)
            # 确保 z 不会因为没有返回或者没有用于副作用的方式而消失
            nomnom(z)
            # 返回 y 加 y 的结果
            return y + y

        # 创建一个 4x4 全为 1 的张量 a
        a = torch.ones(4, 4)
        # 对 test_write 进行脚本化检查，传入参数为 a
        j = self.checkScript(test_write, (a,))

    # 测试函数：测试没有输出别名的情况
    def test_peephole_no_output_aliasing(self):
        # 定义内部函数 test_peephole，接受参数 x
        def test_peephole(x):
            # 将 x 加 0，并赋值给 y
            y = x + 0
            # 返回 x 和 y
            return x, y

        # 创建一个 4x4 全为 1 的张量 a
        a = torch.ones(4, 4)
        # 对 test_peephole 进行脚本化检查，传入参数为 a
        j = self.checkScript(test_peephole, (a,))
        # 执行 j(a) 并将结果分别赋给 r1 和 r2
        r1, r2 = j(a)
        # 断言 r1 和 r2 的数据指针不同
        self.assertNotEqual(r1.data_ptr(), r2.data_ptr())

    # 测试函数：测试 peephole 优化
    def test_peephole(self):
        # 创建张量 a，b 和 c，分别为 [0.4], [0.7] 和 [0]（类型为 torch.int32）
        a = torch.tensor([0.4])
        b = torch.tensor([0.7])
        c = torch.tensor([0], dtype=torch.int32)

        # 定义函数 f，接受 x 和 y 作为参数，返回 x 的类型转换为 y 的结果
        def f(x, y):
            return x.type_as(y)

        # 对函数 f 进行 JIT 跟踪，跟踪参数为 (a, b)，返回跟踪后的结果 tf
        tf = torch.jit.trace(f, (a, b))
        # 使用 FileCheck 检查 tf.graph 中是否包含 "type_as"
        FileCheck().check("type_as").run(str(tf.graph))
        # 运行名为 "peephole" 的优化 pass 到 tf.graph
        self.run_pass("peephole", tf.graph)
        # 使用 FileCheck 检查 tf.graph 中是否不包含 "type_as"
        FileCheck().check_not("type_as").run(str(tf.graph))

        # 再次对函数 f 进行 JIT 跟踪，跟踪参数为 (a, c)，返回跟踪后的结果 tf2
        tf2 = torch.jit.trace(f, (a, c))
        # 将 tf2.graph 转换为字符串并赋值给 s
        s = str(tf2.graph)
        # 运行名为 "peephole" 的优化 pass 到 tf2.graph
        self.run_pass("peephole", tf2.graph)
        # 断言 s 与 str(s) 相等
        self.assertEqual(s, str(s))

    # 测试函数：测试动态 peephole 优化
    def test_peephole_dynamic(self):
        # 定义函数 f，接受 x 和 y 作为参数，返回 x 的类型转换为 y 的结果
        def f(x, y):
            return x.type_as(y)

        # 对函数 f 进行 JIT 脚本化，返回脚本化后的结果 fn
        fn = torch.jit.script(f)
        # 将 fn.graph 转换为字符串并赋值给 s
        s = str(fn.graph)
        # 执行 "peephole" 优化 pass 到 fn.graph
        torch._C._jit_pass_peephole(fn.graph)
        # 断言 s 与 str(fn.graph) 相等
        self.assertEqual(s, str(fn.graph))
    def test_peephole_list_ops(self):
        # 定义一个脚本化函数 foo，返回列表 [x, y, z] 的长度
        @torch.jit.script
        def foo(x, y, z):
            return len([x, y, z])

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 检查计算图中是否包含 "value=3" 和 "return"，并运行验证
        FileCheck().check("value=3").check_next("return").run(foo.graph)

        # 定义一个脚本化函数 foo，创建列表 [x, y, z]，然后循环向其添加 x 的长度个元素
        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            for i in range(len(x)):
                li.append(x)
            return len([x, y, z])

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 检查计算图中是否不再包含 "aten::len"，并运行验证
        FileCheck().check_not("aten::len").run(foo.graph)

        # 定义一个脚本化函数 foo，创建列表 [x, y, z]，返回索引为 1 和 -2 的元素
        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            return li[1], li[-2]

        # 检查计算图中是否包含 "aten::__getitem__"，并运行验证
        FileCheck().check("aten::__getitem__").run(foo.graph)
        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 检查计算图中是否不再包含 "aten::__getitem__"，并运行验证
        FileCheck().check_not("aten::__getitem__").run(foo.graph)

        # 定义一个脚本化函数 foo，创建列表 [x, y, z]，返回索引为 -7 的元素
        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            return li[-7]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 检查计算图中是否包含 "aten::__getitem__"，并运行验证
        FileCheck().check("aten::__getitem__").run(foo.graph)

        # 定义一个脚本化函数 foo，创建列表 [x, y, z]，然后循环向其添加 x 的长度个元素，返回索引为 -2 的元素
        @torch.jit.script
        def foo(x, y, z):
            li = [x, y, z]
            for i in range(len(x)):
                li.append(x)
            return li[-2]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 检查计算图中是否包含 "aten::__getitem__"，并运行验证
        FileCheck().check("aten::__getitem__").run(foo.graph)

    @unittest.skipIf(not RUN_CUDA, "cpp tests require CUDA")
    def test_peephole_cuda(self):
        # 创建在 CPU 和 CUDA 设备上的张量
        a = torch.tensor([0.4], device="cpu")
        b = torch.tensor([0.7], device="cuda")
        c = torch.tensor([0.7], device="cuda")

        # 定义一个函数 f，返回 x 转换为与 y 类型相同的张量
        def f(x, y):
            return x.type_as(y)

        # 跟踪函数 f 在输入 (a, c) 上的计算图
        trace = torch.jit.trace(f, (a, c))
        # 将计算图转换为字符串表示
        s = str(trace.graph)
        # 在 trace 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", trace.graph)
        # 验证计算图的字符串表示没有改变
        self.assertEqual(s, str(trace.graph))
        # 再次跟踪函数 f 在输入 (b, c) 上的计算图
        trace = torch.jit.trace(f, (b, c))
        # 在 trace 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", trace.graph)
        # 在 trace 的计算图上运行名为 "dce" 的优化传递
        self.run_pass("dce", trace.graph)
        # 检查计算图中是否不再包含 "type_as"，并运行验证
        FileCheck().check_not("type_as").run(str(trace.graph))

    @_inline_everything
    def test_peephole_type_refinements(self):
        # 定义一个函数 refine，接受一个可选的张量 x 作为参数，返回 x 或 torch.tensor(3)
        def refine(x):
            # type: (Optional[Tensor]) -> Tensor
            return x if x is not None else torch.tensor(3)

        # 脚本化函数 test，调用 refine 函数并传入 torch.tensor(4)
        @torch.jit.script
        def test():
            return refine(torch.tensor(4))

        # 检查计算图中是否包含 "prim::unchecked_cast"，并运行验证
        FileCheck().check("prim::unchecked_cast").run(test.graph)
        # 在 test 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", test.graph)
        # 检查计算图中是否不再包含 "prim::unchecked_cast"，并运行验证
        FileCheck().check_not("prim::unchecked_cast").run(test.graph)

        # 定义一个函数 is_int_tensor，接受一个张量 x 作为参数，返回根据 x.item() 的类型的不同返回值
        def is_int_tensor(x):
            scalar = x.item()
            if isinstance(scalar, int):
                return scalar + 3
            else:
                return 8

        # 检查脚本化版本的 is_int_tensor 在输入为 torch.tensor(2) 时的行为
        self.checkScript(is_int_tensor, (torch.tensor(2),))
        # 检查脚本化版本的 is_int_tensor 在输入为 torch.tensor(2.5) 时的行为
        self.checkScript(is_int_tensor, (torch.tensor(2.5),))
        # 获取 is_int_tensor 的计算图
        graph = torch.jit.script(is_int_tensor).graph
        # 在 graph 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", graph)
        # 检查计算图中是否包含 "prim::unchecked_cast"，并运行验证
        FileCheck().check("prim::unchecked_cast").run(graph)
    def test_short_circuit_optimization(self):
        @torch.jit.script
        def const_expressions(x):
            # 定义一个 TorchScript 函数，接受一个整数参数 x，返回两个布尔值的元组
            # 第一个元素是 x 是否等于 1 并且为 False
            # 第二个元素是 x 是否等于 1 或者为 True
            return x == 1 and False, x == 1 or True

        # 运行名为 "constant_propagation" 的优化传递，传递 TorchScript 函数的图形
        self.run_pass("constant_propagation", const_expressions.graph)
        # 检查图形中不应包含 "prim::If" 和 "aten::eq" 操作
        FileCheck().check_not("prim::If").check_not("aten::eq").run(
            const_expressions.graph
        )
        # 断言调用 const_expressions(1) 应返回 (False, True)
        self.assertEqual(const_expressions(1), (False, True))

        @torch.jit.script
        def redundant_expressions(x):
            # 定义一个 TorchScript 函数，接受一个整数参数 x，返回两个布尔值的元组
            # 第一个元素是 x 是否等于 1 并且为 True
            # 第二个元素是 x 是否等于 1 或者为 False
            return x == 1 and True, x == 1 or False

        # 运行名为 "peephole" 的优化传递，传递 TorchScript 函数的图形
        self.run_pass("peephole", redundant_expressions.graph)
        # 断言调用 redundant_expressions(1) 应返回 (True, True)
        # 断言调用 redundant_expressions(0) 应返回 (False, False)
        self.assertEqual(redundant_expressions(1), (True, True))
        self.assertEqual(redundant_expressions(0), (False, False))
        # 检查图形中应移除 "and True" 和 "or False" 操作
        FileCheck().check("aten::eq").check_not("prim::If").run(
            redundant_expressions.graph
        )

    def test_conv_dim_folding(self):
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        for mod in modules:

            class ConvDim(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个 ConvDim 类的实例，其中包含一个指定的卷积模块
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    # 在给定输入 x 上应用卷积操作
                    x = self.conv(x)
                    # 返回卷积操作结果的维度
                    return x.dim()

            # 将 ConvDim 类的 TorchScript 版本创建为 conv_dim
            conv_dim = torch.jit.script(ConvDim())
            # 运行名为 "inline" 的优化传递，传递 conv_dim 的图形
            self.run_pass("inline", conv_dim.graph)
            # 运行名为 "peephole" 的优化传递，传递 conv_dim 的图形
            self.run_pass("peephole", conv_dim.graph)
            # 检查图形中不应包含 "conv" 和 "dim" 操作
            FileCheck().check_not("conv").check_not("dim").run(conv_dim.graph)

            class ConvDimMutate(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # 创建一个 ConvDimMutate 类的实例，其中包含一个指定的卷积模块
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    # 在给定输入 x 上应用卷积操作
                    x = self.conv(x)
                    # 修改 x 的大小并返回其维度
                    x.resize_([4, 4])
                    return x.dim()

            # 将 ConvDimMutate 类的 TorchScript 版本创建为 conv_dim
            conv_dim = torch.jit.script(ConvDimMutate())
            # 运行名为 "inline" 的优化传递，传递 conv_dim 的图形
            self.run_pass("inline", conv_dim.graph)
            # 运行名为 "peephole" 的优化传递，传递 conv_dim 的图形
            self.run_pass("peephole", conv_dim.graph)
            # 检查图形中应包含 "conv" 和 "dim" 操作
            FileCheck().check("conv").check("dim").run(conv_dim.graph)

    def test_normalized_rsub(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        def convertible_rsub(x, y):
            # 定义一个函数 convertible_rsub，接受两个张量参数 x 和 y
            # 返回两个张量之间的减法运算和 y 和 x 之间的反向减法运算
            return (x - y), torch.rsub(y, x)

        # 使用 checkScript 方法检查 convertible_rsub 函数的 TorchScript 版本
        self.checkScript(convertible_rsub, (a, b))
        # 获取 convertible_rsub 函数的图形
        op_graph = torch.jit.script(convertible_rsub).graph
        # 检查图形中 "aten::sub" 操作的确切数量为 2 次
        FileCheck().check_count("aten::sub", 2, exactly=True).run(op_graph)
        # 检查图形中 "aten::rsub" 操作的确切数量为 0 次
        FileCheck().check_count("aten::rsub", 0, exactly=True).run(op_graph)
    # 定义测试方法，验证 `is` 操作的情况
    def test_normalized_is_op(self):
        # 定义内部函数 `convertible_is_op`，接受两个布尔值参数，并返回三个比较结果
        def convertible_is_op(x: bool, y: bool):
            return x is True, False is x, x is y

        # 使用测试框架的辅助方法验证 `convertible_is_op` 函数的脚本化输出
        self.checkScript(convertible_is_op, (True, False))

        # 从脚本化的 `convertible_is_op` 函数中获取其计算图
        op_graph = torch.jit.script(convertible_is_op).graph
        # 使用文件检查工具，确保计算图中 `aten::eq` 操作的数量为 3
        FileCheck().check_count("aten::eq", 3, exactly=True).run(op_graph)
        # 使用文件检查工具，确保计算图中 `aten::__is__` 操作的数量为 0
        FileCheck().check_count("aten::__is__", 0, exactly=True).run(op_graph)

    # 定义测试方法，验证 `is not` 操作的情况
    def test_normalized_isnot_op(self):
        # 定义内部函数 `convertible_isnot_op`，接受两个布尔值参数，并返回三个比较结果
        def convertible_isnot_op(x: bool, y: bool):
            return x is not True, False is not x, x is not y

        # 使用测试框架的辅助方法验证 `convertible_isnot_op` 函数的脚本化输出
        self.checkScript(convertible_isnot_op, (True, False))

        # 从脚本化的 `convertible_isnot_op` 函数中获取其计算图
        op_graph = torch.jit.script(convertible_isnot_op).graph
        # 使用文件检查工具，确保计算图中 `aten::ne` 操作的数量为 3
        FileCheck().check_count("aten::ne", 3, exactly=True).run(op_graph)
        # 使用文件检查工具，确保计算图中 `aten::__isnot__` 操作的数量为 0
        FileCheck().check_count("aten::__isnot__", 0, exactly=True).run(op_graph)
    def test_integer_refinement(self):
        # 定义内部函数，运行常量值优化、常量传播和死代码消除的 passes
        def run_peephole_and_check_const_value(graph, const_string):
            self.run_pass("refine_integer_values", graph)
            self.run_pass("constant_propagation", graph)
            self.run_pass("dce", graph)
            # 使用 FileCheck 检查是否存在指定的常量字符串和下一个 "return"
            FileCheck().check(const_string).check_next("return").run(graph)

        # 声明一个 TorchScript 函数 foo，接受两个整数参数 x 和 y
        @torch.jit.script
        def foo(x: int, y: int):
            # 如果 x 不等于 4 或者 y 不等于 5，抛出异常
            if x != 4 or y != 5:
                raise Exception("")  # noqa: TRY002

            return x + y

        # 获取 foo 函数的图形对象
        graph = foo.graph
        # 运行常量值优化 pass
        self.run_pass("refine_integer_values", graph)
        # 运行常量传播 pass
        self.run_pass("constant_propagation", graph)
        # 运行死代码消除 pass
        self.run_pass("dce", graph)

        # 调用 run_peephole_and_check_const_value 函数检查图形对象的常量值
        run_peephole_and_check_const_value(foo.graph, "value=9")
        # 断言调用 foo(4, 5) 的结果为 9
        self.assertEqual(foo(4, 5), 9)
        # 断言调用 foo(2, 4) 会抛出异常
        with self.assertRaises(Exception):
            foo(2, 4)

        # 声明另一个 TorchScript 函数 foo，接受两个整数参数 x 和 y
        @torch.jit.script
        def foo(x: int, y: int):
            # 如果 x 等于 4 并且 y 等于 5，则通过，否则抛出异常
            if x == 4 and y == 5:
                pass
            else:
                raise Exception("hi")  # noqa: TRY002

            return x + y

        # 调用 run_peephole_and_check_const_value 函数检查图形对象的常量值
        run_peephole_and_check_const_value(foo.graph, "value=9")
        # 断言调用 foo(4, 5) 的结果为 9
        self.assertEqual(foo(4, 5), 9)
        # 断言调用 foo(2, 4) 会抛出异常
        with self.assertRaises(Exception):
            foo(2, 4)

        # 声明另一个 TorchScript 函数 foo，接受三个整数参数 x、y 和 z
        @torch.jit.script
        def foo(x: int, y: int, z: int):
            # 如果 x 不等于 4，则抛出异常
            if x != 4:
                raise Exception("..")  # noqa: TRY002
            else:
                # 如果 y 不等于 8，则抛出异常
                if y != 8:
                    raise Exception("...")  # noqa: TRY002
                else:
                    # 如果 z 等于 3，则通过，否则抛出异常
                    if z == 3:
                        pass
                    else:
                        raise Exception("...")  # noqa: TRY002

            return x + y * z

        # 调用 run_peephole_and_check_const_value 函数检查图形对象的常量值
        run_peephole_and_check_const_value(foo.graph, "value=28")
        # 断言调用 foo(4, 8, 3) 的结果为 28
        self.assertEqual(foo(4, 8, 3), 28)
        # 断言调用 foo(1, 2, 3) 会抛出异常
        with self.assertRaises(Exception):
            foo(1, 2, 3)

        # 声明一个 TorchScript 函数 foo，接受一个整数参数 x 和一个布尔型参数 cond
        @torch.jit.script
        def foo(x: int, cond: bool):
            # 如果 x 等于 4，则根据 cond 的值返回 x 或者 4
            if x == 4:
                if cond:
                    return x
                return 4

            return 4

        # 调用 run_peephole_and_check_const_value 函数检查图形对象的常量值
        run_peephole_and_check_const_value(foo.graph, "value=4")

        # 声明一个 TorchScript 函数 foo，接受两个整数参数 x 和 y
        @torch.jit.script
        def foo(x: int, y: int):
            # 断言 x 等于 4 或者 y 等于 5
            assert x == 4 or y == 5
            return x + y

        # 对 foo 的图形对象运行 peephole 优化，设置 refine_list_len 为 True
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        # 运行常量传播 pass
        self.run_pass("constant_propagation", foo.graph)
        # 使用 FileCheck 检查是否存在 "aten::add"
        FileCheck().check("aten::add").run(foo.graph)

    # 定义测试函数 test_optimize_out_comparison_same_value
    def test_optimize_out_comparison_same_value(self):
        # 定义函数 foo，接受一个整数参数 x，返回 x == x 和 x != x 的结果
        def foo(x: int):
            return x == x, x != x

        # 定义函数 foo2，接受一个整数列表参数 x，返回 x == x 和 x != x 的结果
        def foo2(x: List[int]):
            return x == x, x != x

        # 遍历 foo 和 foo2 函数及其输入参数的组合
        for func, inp in zip([foo, foo2], [1, [2, 3]]):
            # 将 func 转换为 TorchScript 函数对象
            func_s = torch.jit.script(func)
            # 对 func_s 的图形对象运行 peephole 优化
            self.run_pass("peephole", func_s.graph)
            # 使用 FileCheck 检查 func_s 的图形对象中不包含 "aten::eq" 和 "aten::neq"
            FileCheck().check_not("aten::eq").check_not("aten::neq").run(func_s.graph)
            # 断言调用 func(inp) 和 func_s(inp) 的结果相等
            self.assertEqual(func(inp), func_s(inp))
    def test_peephole_add_zero(self):
        @torch.jit.script
        def foo(x: int):
            # 返回值为 x 和 x 的和，但这里并未直接使用 aten::add 运算
            return x + 0, 0 + x

        # 在 foo 的计算图上运行 peephole 优化
        self.run_pass("peephole", foo.graph)
        # 检查计算图中不应包含 "aten::add" 操作
        FileCheck().check_not("aten::add")
        # 断言调用 foo(3) 返回 (3, 3)
        self.assertEqual(foo(3), (3, 3))

    def test_noop_peephole(self):
        # 测试不成功
        def foo1(x):
            # 返回值为 x，实际上这是一个无操作的 peephole 优化
            return x + 0

        def foo2():
            x = torch.zeros([2, 2])
            x.sub_(3)
            # 返回 x，此处为无操作的 peephole 优化
            return x + 0

        def foo3():
            x = torch.zeros([2, 2])
            # 返回 x 和 x 的和，实际上这是一个无操作的 peephole 优化
            return x, x + 0

        def foo4():
            x = torch.zeros([2, 2])
            # 返回值为浮点数的 x + 0.0，也是一个无操作的 peephole 优化
            return x + 0.0

        funcs = foo1, foo2, foo3, foo4
        inps = (torch.ones([2]),), (), (), ()
        for func, inp in zip(funcs, inps):
            # 对每个函数 func 进行 TorchScript 编译
            foo_s = torch.jit.script(func)
            # 在 foo_s 的计算图上运行 peephole 优化
            self.run_pass("peephole", foo_s.graph)
            # 检查计算图中确切包含一次 "aten::add" 操作
            FileCheck().check_count("aten::add", 1, exactly=True).run(foo_s.graph)
            # 断言 func(*inp) 与 foo_s(*inp) 的结果相等
            self.assertEqual(func(*inp), foo_s(*inp))

        # 测试成功
        def func(x):
            # 返回 (x + 0) * 1 - 5，初始运行应当 bail on modified value first
            return (x + 0) * 1 - 5

        func_s = torch.jit.script(func)
        # 在 func_s 的计算图上运行 peephole 优化
        self.run_pass("peephole", func_s.graph)
        # 检查计算图中不应包含 "aten::add"，但应包含 "aten::mul"
        FileCheck().check_not("aten::add").check("aten::mul").run(func_s.graph)
        # 第二次运行 peephole 优化应成功
        self.run_pass("peephole", func_s.graph)
        # 检查计算图中不应包含 "aten::add" 和 "aten::mul"
        FileCheck().check_not("aten::add").check_not("aten::mul").run(func_s.graph)
        # 断言 func(torch.ones([2, 2])) 与 func_s(torch.ones([2, 2])) 的结果相等
        self.assertEqual(func(torch.ones([2, 2])), func_s(torch.ones([2, 2])))

        def func(x):
            # 返回 (x + 0.0) - 5，针对浮点数运行 peephole 优化
            return (x + 0.0) - 5

        func_s = torch.jit.script(func)
        inp = next(func_s.graph.inputs())
        # 设置输入张量类型为随机生成的张量类型
        inp.setType(torch._C.TensorType.create_from_tensor(torch.rand([2, 2])))
        # 执行 peephole 优化，禁用形状 peephole 优化
        torch._C._jit_pass_peephole(func_s.graph, disable_shape_peepholes=True)
        # 检查计算图中应包含 "aten::add" 操作
        FileCheck().check("aten::add").run(func_s.graph)
        # 再次运行 peephole 优化，启用形状 peephole 优化
        torch._C._jit_pass_peephole(func_s.graph, disable_shape_peepholes=False)
        # 检查计算图中不应包含 "aten::add" 操作
        FileCheck().check_not("aten::add").run(func_s.graph)

    def test_refine_integer_values(self):
        @torch.jit.script
        def foo(x: int):
            y = 1
            if x == 1:
                # 如果 x 等于 1，则返回 y
                return y
            else:
                # 否则返回 x
                return x

        # 在 foo 的计算图上运行 refine_integer_values 优化
        self.run_pass("refine_integer_values", foo.graph)
        # 在 foo 的计算图上运行 constant_propagation 优化
        self.run_pass("constant_propagation", foo.graph)
        # 在 foo 的计算图上运行 dce 优化
        self.run_pass("dce", foo.graph)
        # 检查计算图中应有 "graph" 和 "return" 出现
        FileCheck().check("graph").check_next("return").run(foo.graph)
        # 断言 foo(2) 返回 2
        self.assertEqual(foo(2), 2)
        # 断言 foo(1) 返回 1
        self.assertEqual(foo(1), 1)
    def test_peephole_len_list(self):
        # 定义一个 TorchScript 函数 foo，返回输入张量 x 的 size 的长度
        @torch.jit.script
        def foo(x):
            return len(x.size())

        # 对 foo 函数的计算图应用 "peephole" 优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证图中是否包含 "aten::len" 操作符
        FileCheck().check("aten::len").run(foo.graph)
        # 获取输入节点列表并更新类型信息，将大小信息设置为 [None, None]
        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([None, None]))
        # 再次对 foo 函数的计算图应用 "peephole" 优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证图中是否不再包含 "aten::len" 操作符
        FileCheck().check_not("aten::len").run(foo.graph)
        # 断言 foo 函数对输入张量 torch.rand([3, 1]) 的返回值为 2
        self.assertEqual(2, foo(torch.rand([3, 1])))

        # 定义另一个 TorchScript 函数 foo，返回输入张量 x 的 size 的长度
        @torch.jit.script
        def foo(x):
            # 获取输入张量 x 的 size
            li = x.size()
            # 将值 4 添加到列表 li 中
            li.append(4)
            # 返回列表 li 的长度
            return len(li)

        # 获取输入节点列表并更新类型信息，将大小信息设置为 [None, None]
        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([None, None]))
        # 对 foo 函数的计算图应用 "peephole" 优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证图中是否包含 "aten::len" 操作符
        FileCheck().check("aten::len").run(foo.graph)
        # 断言 foo 函数对输入张量 torch.rand([3, 1]) 的返回值为 3
        self.assertEqual(3, foo(torch.rand([3, 1])))

    def test_peephole_optional_refine(self):
        # 定义一个 TorchScript 函数 foo，根据条件 cond 返回 z 或 z2
        @torch.jit.script
        def foo(z: int, z2: int, cond: bool):
            if cond:
                return z
            else:
                return z2

        # 获取条件节点的输出并设置为 OptionalType(IntType)
        out = next(foo.graph.findNode("prim::If").outputs())
        out.setType(torch._C.OptionalType(torch._C.IntType.get()))
        # 对 foo 函数的计算图应用 "peephole" 优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证图中是否不包含 "int?" 类型
        FileCheck().check_not("int?").run(foo.graph)

    def test_peephole_int(self):
        # 定义一个 TorchScript 函数 foo，将输入 x 转换为整数类型
        @torch.jit.script
        def foo(x):
            # type: (number)
            return int(x)

        # 使用 FileCheck 验证图中是否包含 "aten::Int" 操作符
        FileCheck().check("aten::Int").run(foo.graph)
        # 获取下一个输入节点并将其类型设置为整数类型
        next(foo.graph.inputs()).setType(torch._C.IntType.get())
        # 对 foo 函数的计算图应用 "peephole" 优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证图中是否不再包含 "aten::Int" 操作符
        FileCheck().check_not("aten::Int").run(foo.graph)

    def test_peephole_arith(self):
        # 定义一个 TorchScript 函数 foo，进行多种算术运算后返回列表
        @torch.jit.script
        def foo(input0: int, input1: int, input2: int, input3: int):
            _1 = torch.add(input1, 2)
            _3 = torch.add(input3, 2)
            _5 = torch.add(1, torch.sub(_1, 3) // 1)
            _6 = torch.add(1 * torch.sub(_3, 3) // 1, 1) / 1
            return [_5, int(_6)]

        # 使用 FileCheck 验证图中是否包含多种算术运算操作符
        FileCheck().check("aten::add").check("aten::sub").check("aten::mul").check(
            "aten::floordiv"
        ).check("aten::div").run(foo.graph)
        # 对 foo 函数的计算图应用 "peephole" 优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证图的结构，检查是否包含 ListConstruct 和 return
        FileCheck().check("graph").check("):").check_next("ListConstruct").check_next(
            "return"
        ).run(foo.graph)
        # 断言 foo 函数对输入参数 0, 1, 2, 3 的返回值为 [1, 3]
        self.assertEqual(foo(0, 1, 2, 3), [1, 3])
    def test_peephole_dict_getitem_simple(self):
        # 定义一个 Torch 脚本函数 foo，接受两个整数参数 a 和 b
        @torch.jit.script
        def foo(a: int, b: int):
            # 创建一个字典 d，键为整数 0 和 1，对应的值分别为参数 a 和 b
            d = {0: a, 1: b}
            # 获取字典 d 中键为 1 的值，赋给变量 x
            x = d[1]
            # 获取字典 d 中键为 0 的值，赋给变量 y
            y = d[0]
            # 返回 x 和 y
            return x, y

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证优化后的计算图中不包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check_not("DictConstruct").check_not("__getitem__").run(foo.graph)
        # 验证 foo(0, 1) 返回 (1, 0)
        self.assertEqual(foo(0, 1), (1, 0))

        # 定义另一个 Torch 脚本函数 foo，接受两个整数参数 a 和 b
        @torch.jit.script
        def foo(a: int, b: int):
            # 创建一个字典 d，键为字符串 "0" 和 "1"，对应的值分别为参数 a 和 b
            d = {"0": a, "1": b}
            # 获取字典 d 中键为 "1" 的值，赋给变量 x
            x = d["1"]
            # 获取字典 d 中键为 "0" 的值，赋给变量 y
            y = d["0"]
            # 返回 x 和 y
            return x, y

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证优化后的计算图中不包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check_not("DictConstruct").check_not("__getitem__").run(foo.graph)
        # 验证 foo(0, 1) 返回 (1, 0)

        self.assertEqual(foo(0, 1), (1, 0))

        # 定义另一个 Torch 脚本函数 foo，接受两个整数参数 a 和 b
        @torch.jit.script
        def foo(a: int, b: int):
            # 创建一个字典 d，键为浮点数 0.0 和 1.0，对应的值分别为参数 a 和 b
            d = {0.0: a, 1.0: b}
            # 获取字典 d 中键为 1.0 的值，赋给变量 x
            x = d[1.0]
            # 获取字典 d 中键为 0.0 的值，赋给变量 y
            y = d[0.0]
            # 返回 x 和 y
            return x, y

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证优化后的计算图中不包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check_not("DictConstruct").check_not("__getitem__").run(foo.graph)
        # 验证 foo(0, 1) 返回 (1, 0)

        self.assertEqual(foo(0, 1), (1, 0))

    def test_peephole_dict_getitem_no_optimization_missing_key(self):
        # 定义一个 Torch 脚本函数 foo，无参数
        @torch.jit.script
        def foo():
            # 创建一个字典 d，键为整数 0，值为整数 1
            d = {0: 1}
            # 尝试返回字典 d 中键为 2 的值，但该键不存在
            return d[2]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证计算图中包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_get_input_arg(self):
        # 定义一个 Torch 脚本函数 foo，接受一个整数参数 a
        # 这里无法确定输入参数是否在字典中，因此无法进行优化
        @torch.jit.script
        def foo(a: int):
            # 创建一个字典 d，键为整数 0，值为整数 1
            d = {0: 1}
            # 返回字典 d 中键为 a 的值
            return d[a]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证计算图中包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
        # 验证 foo(0) 返回 1
        self.assertEqual(foo(0), 1)

    def test_peephole_dict_getitem_no_optimization_dict_modified(self):
        # 定义一个 Torch 脚本函数 foo，无参数
        @torch.jit.script
        def foo():
            # 创建一个字典 d，键为整数 0，值为整数 1
            d = {0: 1}
            # 修改字典 d 中键为 0 的值为 2
            d[0] = 2
            # 返回字典 d 中键为 0 的值
            return d[0]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证计算图中包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
        # 验证 foo() 返回 2
        self.assertEqual(foo(), 2)

    def test_peephole_dict_getitem_no_optimization_overlapping_keys(self):
        # 定义一个 Torch 脚本函数 foo，无参数
        @torch.jit.script
        def foo():
            # 创建一个字典 d，键为整数 0，值为整数 2，这里键 0 会被后面的键值对覆盖
            d = {0: 1, 0: 2}  # noqa: F601
            # 返回字典 d 中键为 0 的值
            return d[0]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证计算图中包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_keys_might_overlap(self):
        # 定义一个 Torch 脚本函数 foo，接受一个整数参数 x
        @torch.jit.script
        def foo(x: int):
            # 创建一个字典 d，键为整数 0 和参数 x，值分别为 1 和 2
            d = {0: 1, x: 2}
            # 返回字典 d 中键为 x 的值
            return d[x]

        # 在 foo 的计算图上运行名为 "peephole" 的优化传递
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证计算图中包含 "DictConstruct" 和 "__getitem__" 字符串
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
    # 定义一个测试函数，用于测试不支持类型优化时的字典项获取操作
    def test_peephole_dict_getitem_no_optimization_unsupported_type(self):
        # 使用 Torch 脚本装饰器定义一个函数 foo
        @torch.jit.script
        def foo():
            # 创建一个随机的 2x2 的 Tensor
            a = torch.rand((2, 2))
            # 创建一个字典 d，将 Tensor a 映射到整数 1
            d = {a: 1}
            # 返回字典 d 中键为 a 的值
            return d[a]

        # 在 foo 函数的计算图上运行名为 "peephole" 的优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 检查计算图，验证是否包含 "DictConstruct" 和 "__getitem__" 操作
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
        # 断言 foo 函数的执行结果为 1
        self.assertEqual(foo(), 1)

    # 定义一个测试函数，用于测试字典长度计算的优化
    def test_peephole_dict_len(self):
        # 使用 Torch 脚本装饰器定义一个函数 foo
        @torch.jit.script
        def foo():
            # 创建一个包含两个键值对的字典 d
            d = {0: 1, 1: 2}
            # 返回字典 d 的长度
            return len(d)

        # 在 foo 函数的计算图上运行名为 "peephole" 的优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 检查计算图，验证不包含 "DictConstruct" 和 "len" 操作
        FileCheck().check_not("DictConstruct").check_not("len").run(foo.graph)
        # 断言 foo 函数的执行结果为 2
        self.assertEqual(foo(), 2)

    # 定义一个测试函数，用于测试包含重复键的字典长度计算的优化
    def test_peephole_dict_len_no_optimization_overlapping_keys(self):
        # 使用 Torch 脚本装饰器定义一个函数 foo
        @torch.jit.script
        def foo():
            # 创建一个包含重复键的字典 d
            d = {0: 1, 0: 2}  # noqa: F601
            # 返回字典 d 的长度
            return len(d)

        # 在 foo 函数的计算图上运行名为 "peephole" 的优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 检查计算图，验证包含 "DictConstruct" 和 "len" 操作
        FileCheck().check("DictConstruct").check("len").run(foo.graph)
        # 断言 foo 函数的执行结果为 1
        self.assertEqual(foo(), 1)

    # 定义一个测试函数，用于测试包含可能重叠键的字典长度计算的优化
    def test_peephole_dict_len_no_optimization_keys_might_overlap(self):
        # 使用 Torch 脚本装饰器定义一个函数 foo，参数为整数 x
        @torch.jit.script
        def foo(x: int):
            # 创建一个字典 d，包含键值对 (0, 1) 和 (x, 2)
            d = {0: 1, x: 2}
            # 返回字典 d 的长度
            return len(d)

        # 在 foo 函数的计算图上运行名为 "peephole" 的优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 检查计算图，验证包含 "DictConstruct" 和 "len" 操作
        FileCheck().check("DictConstruct").check("len").run(foo.graph)

    # 定义一个测试函数，用于测试不支持类型优化时的字典长度计算操作
    def test_peephole_dict_len_no_optimization_unsupported_type(self):
        # 使用 Torch 脚本装饰器定义一个函数 foo
        @torch.jit.script
        def foo():
            # 创建一个随机的 2x2 的 Tensor
            a = torch.rand((2, 2))
            # 创建一个字典 d，将 Tensor a 映射到整数 1
            d = {a: 1}
            # 返回字典 d 的长度
            return len(d)

        # 在 foo 函数的计算图上运行名为 "peephole" 的优化 pass
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 检查计算图，验证包含 "DictConstruct" 和 "len" 操作
        FileCheck().check("DictConstruct").check("len").run(foo.graph)
        # 断言 foo 函数的执行结果为 1
        self.assertEqual(foo(), 1)

    # 定义一个测试函数，用于测试包含所有三个参数的切片操作的优化
    def test_peephole_slice_all_three_args(self):
        # 定义一个函数 foo，参数为整数 x
        def foo(x: int):
            # 返回列表的切片操作结果，从索引 -5 到 6，步长为 2
            return [1, 2, x, 4, 5, 6, 7][-5:6:2]

        # 获取函数 foo 的计算图
        graph = torch.jit.script(foo).graph
        # 在计算图上运行名为 "peephole" 的优化 pass
        self.run_pass("peephole", graph)
        # 使用 FileCheck 检查计算图，验证不包含 "aten::slice" 操作
        FileCheck().check_not("aten::slice").run(graph)
        # 调用 self.checkScript 函数验证 foo 函数的执行结果与预期相符
        self.checkScript(foo, (3,))

    # 定义一个测试函数，用于测试包含一个空参数的切片操作的优化
    def test_peephole_slice_one_empty_arg(self):
        # 定义一个辅助函数 check_helper，接受一个函数 fn 作为参数，并返回空值
        def check_helper(fn: Callable[[int], None]) -> None:
            # 获取函数 fn 的计算图
            graph = torch.jit.script(fn).graph
            # 在计算图上运行名为 "peephole" 的优化 pass
            self.run_pass("peephole", graph)
            # 使用 FileCheck 检查计算图，验证不包含 "aten::slice" 操作
            FileCheck().check_not("aten::slice").run(graph)
            # 调用 self.checkScript 函数验证 fn 函数的执行结果与预期相符
            self.checkScript(fn, (3,))

        # 定义一个函数 foo，参数为整数 x
        def foo(x: int):
            # 返回列表的切片操作结果，从索引 1 开始到末尾，步长为 2
            return [1, 2, x, 4, 5, 6, 7][1::2]

        # 调用 check_helper 函数验证 foo 函数的切片操作优化情况
        check_helper(foo)

        # 定义一个函数 foo，参数为整数 x
        def foo(x: int):
            # 返回列表的切片操作结果，从索引 0 开始到索引 4，步长为 3
            return [1, 2, x, 4, 5, 6, 7][:5:3]

        # 调用 check_helper 函数验证 foo 函数的切片操作优化情况
        check_helper(foo)

        # 定义一个函数 foo，参数为整数 x
        def foo(x: int):
            # 返回列表的切片操作结果，从索引 0 开始到索引 4
            return [1, 2, x, 4, 5, 6, 7][0:4]

        # 调用 check_helper 函数验证 foo 函数的切片操作优化情况
        check_helper(foo)
    # 定义测试函数，用于测试 peephole 优化在切片操作中的应用情况（当切片参数为空的情况）
    def test_peephole_slice_two_empty_args(self):
        # 定义辅助函数，用于检查和运行 JIT 脚本化函数的图形
        def check_helper(fn: Callable[[int], None]) -> None:
            # 获取 fn 的 JIT 脚本化图形
            graph = torch.jit.script(fn).graph
            # 在 JIT 脚本化图形上运行 peephole 优化
            self.run_pass("peephole", graph)
            # 使用 FileCheck 验证在图形中是否不存在 "aten::slice" 操作
            FileCheck().check_not("aten::slice").run(graph)
            # 检查 JIT 脚本化函数的运行结果
            self.checkScript(fn, (3,))

        # 定义一个函数 foo，根据输入 x 返回列表的每隔两个元素的切片
        def foo(x: int):
            return [1, 2, x, 4, 5, 6, 7][::2]

        # 调用辅助函数，测试函数 foo 的 peephole 优化情况
        check_helper(foo)

        # 定义一个函数 foo，根据输入 x 返回列表的前五个元素的切片
        def foo(x: int):
            return [1, 2, x, 4, 5, 6, 7][:5]

        # 调用辅助函数，测试函数 foo 的 peephole 优化情况
        check_helper(foo)

        # 定义一个函数 foo，根据输入 x 返回列表从索引 1 开始到末尾的切片
        def foo(x: int):
            return [1, 2, x, 4, 5, 6, 7][1:]

        # 调用辅助函数，测试函数 foo 的 peephole 优化情况
        check_helper(foo)

    # 定义测试函数，用于测试 peephole 优化在切片操作中的应用情况（当列表被修改后）
    def test_peephole_slice_optimization_not_applied_list_modified(self):
        # 使用 JIT 脚本化装饰器定义函数 foo
        @torch.jit.script
        def foo():
            # 创建列表 li，并修改其第一个元素
            li = [1, 2, 3, 4, 5, 6, 7]
            li[0] = 0
            # 返回列表 li 的切片结果（从索引 2 到索引 5 的元素）
            return li[2:5]

        # 在 foo 的图形上运行 peephole 优化
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证在图形中是否存在 "aten::slice" 操作
        FileCheck().check("aten::slice").run(foo.graph)

    # 定义测试函数，用于测试 peephole 优化在切片操作中的应用情况（当切片参数不是常量的情况）
    def test_peephole_slice_optimization_not_applied_non_const_args(self):
        # 使用 JIT 脚本化装饰器定义函数 foo，接受输入 x 和 y
        @torch.jit.script
        def foo(x: int, y: int):
            # 创建列表 li，并根据输入 x 和 y 返回切片结果
            li = [1, 2, 3, 4, 5, 6, 7]
            return li[x:y]

        # 在 foo 的图形上运行 peephole 优化
        self.run_pass("peephole", foo.graph)
        # 使用 FileCheck 验证在图形中是否存在 "aten::slice" 操作
        FileCheck().check("aten::slice").run(foo.graph)
```