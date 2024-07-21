# `.\pytorch\test\jit\test_remove_mutation.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os
import sys
from typing import List

import torch
from torch.testing import FileCheck

# 将 test/ 目录下的辅助文件设为可导入状态
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import freeze_rng_state, JitTestCase

# 如果当前脚本作为主程序运行，则抛出运行时错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestRemoveMutation，继承自 JitTestCase
class TestRemoveMutation(JitTestCase):
    
    # 定义测试方法 test_aten_inplace
    def test_aten_inplace(self):
        
        # 定义内部函数 test_not_new_alias，参数为 x
        def test_not_new_alias(x):
            # 创建 y，为 x 的第一个元素
            y = x[0]
            # 在 y 上执行原地加法操作
            y.add_(2)
            return y
        
        # 将 test_not_new_alias 函数转换为 Torch 脚本
        fn = torch.jit.script(test_not_new_alias)
        # 获取生成的图形对象
        graph = fn.graph
        # 运行名为 "remove_mutation" 的优化 pass
        self.run_pass("remove_mutation", graph)
        # 使用 FileCheck 验证图形中是否包含 "aten::add_" 操作
        FileCheck().check("aten::add_").run(graph)
        # 断言 Torch 脚本的执行结果与普通 Python 函数的结果相同
        self.assertEqual(fn(torch.ones([2, 2])), test_not_new_alias(torch.ones([2, 2])))
        
        # 定义内部函数 test_no_lowering
        def test_no_lowering():
            # 创建张量 x
            x = torch.tensor([2, 2])
            # 修改 x 的第一个元素为 3
            x[0] = 3
            return x
        
        # 将 test_no_lowering 函数转换为 Torch 脚本
        fn = torch.jit.script(test_no_lowering)
        # 获取生成的图形对象
        graph = fn.graph
        # 运行名为 "remove_mutation" 的优化 pass
        self.run_pass("remove_mutation", graph)
        # 使用 FileCheck 验证图形中是否包含 "aten::copy_" 操作
        FileCheck().check("aten::copy_").run(graph)
        # 断言 Torch 脚本的执行结果与普通 Python 函数的结果相同
        self.assertEqual(fn(), test_no_lowering())
        
        # 定义内部函数 test_move_before_not_valid
        def test_move_before_not_valid():
            # 创建张量 y
            y = torch.tensor([2, 2])
            # 创建张量 z，作为 y 加 2 的结果
            z = y + 2
            # 在 y 上执行原地加法操作
            y.add_(2)
            return y, z
        
        # 将 test_move_before_not_valid 函数转换为 Torch 脚本
        fn = torch.jit.script(test_move_before_not_valid)
        # 获取生成的图形对象
        graph = fn.graph
        # 运行名为 "remove_mutation" 的优化 pass
        self.run_pass("remove_mutation", graph)
        # 使用 FileCheck 验证图形中是否包含 "aten::add_" 操作
        FileCheck().check("aten::add_").run(graph)
        # 断言 Torch 脚本的执行结果与普通 Python 函数的结果相同
        self.assertEqual(fn(), test_move_before_not_valid())
        
        # 定义内部函数 test_successful
        def test_successful():
            # 创建张量 x
            x = torch.tensor([2, 2])
            # 在 x 上执行原地加法操作
            x.add_(1)
            x.add_(3)
            # 创建张量 y，作为 x 加 4 的结果
            y = x + 4
            return x, y
        
        # 将 test_successful 函数转换为 Torch 脚本
        fn = torch.jit.script(test_successful)
        # 获取生成的图形对象
        graph = fn.graph
        # 运行名为 "remove_mutation" 的优化 pass
        self.run_pass("remove_mutation", graph)
        # 使用 FileCheck 验证图形中不包含 "aten::add_" 操作
        FileCheck().check_not("aten::add_").run(graph)
        # 断言 Torch 脚本的执行结果与普通 Python 函数的结果相同
        self.assertEqual(test_successful(), fn())
        
        # 定义内部函数 test_intermediary_use
        def test_intermediary_use():
            # 创建张量 x
            x = torch.tensor([2, 2])
            # 在 x 上执行原地加法操作
            x.add_(1)
            # 创建张量 y，作为 x 加 4 的结果
            y = x + 4
            # 再次在 x 上执行原地加法操作
            x.add_(3)
            return x, y
        
        # 将 test_intermediary_use 函数转换为 Torch 脚本
        fn = torch.jit.script(test_intermediary_use)
        # 获取生成的图形对象
        graph = fn.graph
        # 使用 FileCheck 验证图形中 "aten::add_" 操作出现的次数为 2 次
        FileCheck().check_count("aten::add_", 2).run(graph)
        # 运行名为 "remove_mutation" 的优化 pass
        self.run_pass("remove_mutation", graph)
        # 由于 y = x + 4 的存在，无法删除第二个原地加法操作
        # 将来可能会通过复制 x 的临时值并替换其中间使用，来解决这个问题（只要别名安全）
        # 使用 FileCheck 验证图形中 "aten::add_" 操作出现的次数为 1 次
        FileCheck().check_count("aten::add_", 1).run(graph)
        # 断言 Torch 脚本的执行结果与普通 Python 函数的结果相同
        self.assertEqual(test_intermediary_use(), fn())
    def test_if_output(self):
        # 定义内部函数 foo，根据条件 cond 返回不同的结果
        def foo(x, cond: bool):
            # 根据条件 cond 选择不同的操作
            if cond:
                y = x + 5  # 如果条件为真，将 x 加 5
            else:
                y = x + 2  # 如果条件为假，将 x 加 2
            y.add_(4)  # 给 y 加 4，注意此处会修改 y 的值
            return y  # 返回处理后的 y

        # 使用 foo 函数进行即时执行
        out_eager = foo(torch.tensor(5), True)
        # 对 foo 函数进行脚本化，生成一个脚本化的 TorchScript 版本
        foo_script = torch.jit.script(foo)
        # 检查 TorchScript 中是否包含 "aten::add_" 操作，并在结果中运行 FileCheck
        FileCheck().check("aten::add_").run(foo_script.graph)
        # 在 foo_script 的计算图中运行 "remove_mutation" 优化 pass
        self.run_pass("remove_mutation", foo_script.graph)
        # 再次检查 TorchScript 中是否不包含 "aten::add_" 操作，并在结果中运行 FileCheck
        FileCheck().check_not("aten::add_").run(foo_script.graph)

        # 断言即时执行和 TorchScript 版本的结果是否一致
        self.assertEqual(out_eager, foo_script(torch.tensor(5), True))

    def test_if_output_fail(self):
        # 使用 TorchScript 的方式定义 foo 函数
        @torch.jit.script
        def foo(cond: bool):
            li = []
            # 根据条件 cond 执行不同的操作
            if cond:
                x = torch.tensor(1)
                li.append(x)  # 将 x 加入列表 li
            else:
                x = torch.tensor(2)
            y = x.add_(2)  # 将 x 加 2，并将结果赋给 y，注意此处会修改 x 的值
            return y, li  # 返回处理后的 y 和列表 li

        # 在 foo 函数的计算图中运行 "inline" 优化 pass
        self.run_pass("inline", foo.graph)
        # 在 foo 函数的计算图中运行 "remove_mutation" 优化 pass
        self.run_pass("remove_mutation", foo.graph)
        # 检查 foo 函数的计算图中是否包含 "aten::add_" 操作，并在结果中运行 FileCheck
        FileCheck().check("aten::add_").run(foo.graph)

        # 使用 TorchScript 的方式定义 foo 函数
        @torch.jit.script
        def foo(cond: bool, y):
            # 根据条件 cond 执行不同的操作
            if cond:
                x = y
            else:
                x = torch.tensor(2)
            z = x.add_(2)  # 将 x 加 2，并将结果赋给 z，注意此处会修改 x 的值
            return z  # 返回处理后的 z

        # 在 foo 函数的计算图中运行 "inline" 优化 pass
        self.run_pass("inline", foo.graph)
        # 在 foo 函数的计算图中运行 "remove_mutation" 优化 pass
        self.run_pass("remove_mutation", foo.graph)
        # 检查 foo 函数的计算图中是否包含 "aten::add_" 操作，并在结果中运行 FileCheck
        FileCheck().check("aten::add_").run(foo.graph)

    def test_special_mapped_op(self):
        # 定义内部函数 test_successful，测试成功的情况
        def test_successful():
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.zero_()  # 将 x 张量清零
            y.fill_(3)  # 将 y 张量填充为 3
            return x, y  # 返回处理后的 x 和 y

        # 对 test_successful 函数进行 TorchScript 脚本化
        fn = torch.jit.script(test_successful)
        graph = fn.graph
        # 在 fn 的计算图中运行 "remove_mutation" 优化 pass
        self.run_pass("remove_mutation", graph)
        # 检查计算图中是否不包含 "aten::zero_" 和 "aten::fill_" 操作，并在结果中运行 FileCheck
        FileCheck().check_not("aten::zero_").check_not("aten::fill_").run(graph)
        # 断言 test_successful 函数的直接调用结果和 TorchScript 版本的调用结果是否一致
        self.assertEqual(test_successful(), fn())

        # 定义内部函数 test_successful，测试成功的情况
        def test_successful():
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.fill_(y)  # 使用 y 张量填充 x 张量
            return x + x  # 返回处理后的 x + x 结果

        # 对 test_successful 函数进行 TorchScript 脚本化
        fn = torch.jit.script(test_successful)
        graph = fn.graph
        # 在 fn 的计算图中运行 "remove_mutation" 优化 pass
        self.run_pass("remove_mutation", graph)
        # 检查计算图中是否不包含 "aten::fill_" 操作，并在结果中运行 FileCheck
        FileCheck().check_not("aten::fill_").run(graph)

        # 定义普通函数 normal，用于测试特殊情况
        def normal():
            # 注意：由于 `self.run_pass` 中的 `torch._C._jit_pass_remove_mutation` 调用，
            # 将 `torch.randn(..., dtype=None).normal_()` 替换为 `aten::normal` 调用，即使默认 dtype 是 float，
            # 所以我们必须在此显式设置 dtype 为 float
            return torch.rand(2, 1, 3, 4, dtype=torch.float).normal_()

        # 对 normal 函数进行 TorchScript 脚本化
        fn = torch.jit.script(normal)
        graph = fn.graph
        # 在 fn 的计算图中运行 "remove_mutation" 优化 pass
        self.run_pass("remove_mutation", graph)
        # 检查计算图中是否不包含 "normal_" 操作，并在结果中运行 FileCheck
        FileCheck().check_not("normal_").run(graph)
        # 使用 `freeze_rng_state()` 确保比较两次执行的结果一致
        with freeze_rng_state():
            out_eager = normal()
        with freeze_rng_state():
            out_script = fn()
        # 断言即时执行和 TorchScript 版本的结果是否一致
        self.assertEqual(out_eager, out_script)
    # 定义测试方法 test_lists_append，用于测试列表的 append 操作
    def test_lists_append(self):
        # 定义成功移除元素的函数 successful_remove
        def successful_remove():
            # 使用列表推导式创建包含0到4的列表
            return [i for i in range(5)]  # noqa: C416

        # 使用 Torch 的 JIT 编译 successful_remove 函数
        fn = torch.jit.script(successful_remove)
        # 获取函数的计算图
        graph = fn.graph
        # 运行名为 "loop_unrolling" 的优化 Pass
        self.run_pass("loop_unrolling", graph)
        # 运行名为 "remove_mutation" 的优化 Pass
        self.run_pass("remove_mutation", graph)
        # 运行名为 "constant_propagation" 的优化 Pass
        self.run_pass("constant_propagation", graph)
        # 使用 FileCheck 检查计算图，验证是否包含 "graph"、"Constant"、"return"
        FileCheck().check("graph").check_next("Constant").check_next("return").run(
            graph
        )
        # 断言 successful_remove() 的执行结果与 successful_remove() 函数的 Torch JIT 版本执行结果相同
        self.assertEqual(successful_remove(), successful_remove())

        # 定义使用中间变量的函数 intermediary_use
        def intermediary_use():
            a = [1, 2]
            b = len(a)
            # 向列表 a 中追加元素 3
            a.append(3)
            return a

        # 使用 Torch 的 JIT 编译 intermediary_use 函数
        fn = torch.jit.script(intermediary_use)
        # 获取函数的计算图
        graph = fn.graph
        # 使用 FileCheck 检查计算图，验证是否包含 "append"
        FileCheck().check("append").run(graph)
        # 运行名为 "remove_mutation" 的优化 Pass
        self.run_pass("remove_mutation", graph)
        # 在这里可能可以移除 append 操作，但目前没有相关逻辑处理
        FileCheck().check_not("append").run(graph)
        # 断言 intermediary_use() 的执行结果与 fn() 函数执行结果相同
        self.assertEqual(intermediary_use(), fn())

    # 定义测试方法 test_lists_insert，用于测试列表的 insert 操作
    def test_lists_insert(self):
        # 定义成功移除元素的函数 successful_remove
        def successful_remove():
            # 创建空列表 a
            a: List[int] = []
            # 在索引 0 处插入元素 1
            a.insert(0, 1)
            # 在索引 0 处插入元素 2
            a.insert(0, 2)
            # 在索引 -10 处插入元素 3
            a.insert(-10, 3)
            # 在索引 -9 处插入元素 4
            a.insert(-9, 4)
            # 在索引 10 处插入元素 5
            a.insert(10, 5)
            return a

        # 使用 Torch 的 JIT 编译 successful_remove 函数
        fn = torch.jit.script(successful_remove)
        # 获取函数的计算图
        graph = fn.graph
        # 调用 Torch 提供的函数来移除计算图中的变异操作
        torch._C._jit_pass_remove_mutation(graph)
        # 调用 Torch 提供的函数来进行常量传播优化
        torch._C._jit_pass_constant_propagation(graph)
        # 使用 FileCheck 检查计算图，验证是否包含 "graph"、"Constant"、"return"
        FileCheck().check("graph").check_next("Constant").check_next("return").run(
            graph
        )
        # 断言 successful_remove() 的执行结果与 fn() 函数执行结果相同
        self.assertEqual(successful_remove(), fn())
    def test_list_indexing_removal(self):
        # 定义一个 TorchScript 函数，用于测试越界索引
        @torch.jit.script
        def out_of_bounds():
            # 创建列表 x 包含元素 [1, 2]
            x = [1, 2]
            # 尝试对列表 x 的索引 4 进行赋值操作，这是一个越界访问
            x[4] = 3
            return x

        # 移除 TorchScript 函数中的突变操作
        torch._C._jit_pass_remove_mutation(out_of_bounds.graph)
        # 使用 FileCheck 验证是否有 "set_item" 操作
        FileCheck().check("set_item").run(out_of_bounds.graph)

        # 定义一个 TorchScript 函数，测试未知索引的赋值
        @torch.jit.script
        def unknown(y: int):
            # 创建列表 x 包含元素 [1, 2]
            x = [1, 2]
            # 使用参数 y 作为索引，尝试赋值操作
            x[y] = 3
            return x

        # 移除 TorchScript 函数中的突变操作
        torch._C._jit_pass_remove_mutation(out_of_bounds.graph)
        # 使用 FileCheck 验证是否有 "set_item" 操作
        FileCheck().check("set_item").run(out_of_bounds.graph)

        # 定义一个普通 Python 函数，测试正常的索引赋值
        def successful():
            # 创建列表 x 包含元素 [1, 2, 3]
            x = [1, 2, 3]
            # 修改列表 x 的第一个元素为 4
            x[0] = 4
            # 修改列表 x 的最后一个元素为 0
            x[-1] = 0
            return x

        # 将成功函数转换为 TorchScript 函数
        scripted_fn = torch.jit.script(successful)
        # 移除 TorchScript 函数中的突变操作
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        # 使用 FileCheck 验证 TorchScript 中没有 "set_item" 操作
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        # 调用自定义的检查函数检查 TorchScript 函数的正确性
        self.checkScript(successful, ())

        # 以下是重复的成功函数定义和相同的 TorchScript 检查过程
        # 定义一个普通 Python 函数，测试正常的索引赋值
        def successful():
            # 创建列表 x 包含元素 [1, 2, 3]
            x = [1, 2, 3]
            # 修改列表 x 的第一个元素为 4
            x[0] = 4
            # 修改列表 x 的最后一个元素为 0
            x[-1] = 0
            return x

        # 将成功函数转换为 TorchScript 函数
        scripted_fn = torch.jit.script(successful)
        # 移除 TorchScript 函数中的突变操作
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        # 使用 FileCheck 验证 TorchScript 中没有 "set_item" 操作
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        # 调用自定义的检查函数检查 TorchScript 函数的正确性
        self.checkScript(successful, ())

        # 以下是重复的成功函数定义和相同的 TorchScript 检查过程
        # 定义一个普通 Python 函数，测试只包含一个元素的列表索引赋值
        def successful():
            # 创建列表 x 包含元素 [1]
            x = [1]
            # 修改列表 x 的倒数第一个元素为 3
            x[-1] = 3
            return x

        # 将成功函数转换为 TorchScript 函数
        scripted_fn = torch.jit.script(successful)
        # 移除 TorchScript 函数中的突变操作
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        # 使用 FileCheck 验证 TorchScript 中没有 "set_item" 操作
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        # 调用自定义的检查函数检查 TorchScript 函数的正确性
        self.checkScript(successful, ())
    def test_common_pytorch_list_ops(self):
        # 针对一些常见的 PyTorch 列表操作进行测试，如 cat、stack、vstack、hstack、dstack
        for op in ["cat", "stack", "vstack", "hstack", "dstack"]:

            class OpMod(torch.nn.Module):
                def __init__(self, op):
                    super().__init__()
                    self.op = torch_op

                def forward(self):
                    # 创建一个包含元素 [1, 2, 3, 4] 的张量 x
                    x = torch.tensor([1, 2, 3, 4])
                    # 将张量 x 中的每个元素都加上 3
                    x.add_(3)
                    # 创建一个列表 y，包含两个相同的张量 x
                    y = [x, x]
                    # 执行指定的操作 op，并对结果再加上 3
                    return self.op(y) + 3

            # 获取 torch 模块中对应操作 op 的函数
            torch_op = getattr(torch, op)
            # 使用 OpMod 类创建一个模块实例 mod
            mod = OpMod(torch_op)
            # 对模块 mod 进行脚本化
            mod_script = torch.jit.script(mod)
            # 运行名为 "remove_mutation" 的处理器，作用于 mod_script 的前向计算图
            self.run_pass("remove_mutation", mod_script.forward.graph)
            # 检查在 mod_script 的前向计算图中不出现 "aten::add_" 操作
            FileCheck().check_not("aten::add_").run(mod_script.forward.graph)
            # 断言模块 mod 和脚本化后的 mod_script 的输出相等
            self.assertEqual(mod(), mod_script())

            # 测试输出不对输入进行别名化处理
            for inputs in [torch.rand(2, 2)], [torch.rand(2, 2) for _ in range(2)]:
                # 使用 torch_op 对输入执行操作，将结果存储在 result 中
                result = torch_op(inputs)
                # 对结果中的每个张量计算 sum
                sums = [ten.sum() for ten in result]

                # 将输入中的每个张量的所有元素都设置为 10
                for inp in inputs:
                    inp.fill_(10)

                # 断言修改后的输入不会影响到 sums 中的元素之和
                self.assertEqual(sums, [ten.sum() for ten in result])

        @torch.jit.script
        def test_multiple_uses():
            # 创建一个包含元素 [1, 2, 3, 4] 的张量 x
            x = torch.tensor([1, 2, 3, 4])
            # 将张量 x 中的每个元素都加上 3
            x.add_(3)
            # 创建一个列表 y，包含两个相同的张量 x
            y = [x, x]
            # 返回对 y 执行 torch.cat 操作的结果和列表 y 自身
            return torch.cat(y), y

        # 运行名为 "remove_mutation" 的处理器，作用于 test_multiple_uses 函数的计算图
        self.run_pass("remove_mutation", mod_script.forward.graph)
        # 检查在 test_multiple_uses 的计算图中有 "aten::add_" 操作
        FileCheck().check("aten::add_").run(test_multiple_uses.graph)
```