# `.\pytorch\test\jit\test_alias_analysis.py`

```
# Owner(s): ["oncall: jit"]

# 导入 torch 库
import torch
# 导入解析 IR 的函数
from torch._C import parse_ir
# 导入临时文件名生成工具
from torch.testing._internal.common_utils import TemporaryFileName
# 导入 JIT 测试用例基类
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行此文件，抛出 RuntimeError
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestAliasAnalysis，继承自 JitTestCase
class TestAliasAnalysis(JitTestCase):
    # 测试方法：test_becomes_wildcard_annotations
    def test_becomes_wildcard_annotations(self):
        # 定义包含图形 IR 的字符串
        graph_str = """
        graph(%a.1 : Tensor, %b.1 : Tensor):
            %11 : NoneType = prim::Constant()
            %8 : int = prim::Constant[value=0]()
            %7 : int = prim::Constant[value=1]()
            %x.1 : Tensor = aten::add(%a.1, %b.1, %7)
            %y.1 : Tensor[] = aten::split(%x.1, %7, %8)
            return ()
        """
        # 解析图形 IR 字符串为图形对象
        graph = parse_ir(graph_str)
        # 获取图形的别名数据库
        alias_db = graph.alias_db()
        # 查找图中名称为 "aten::split" 的节点
        split_node = graph.findNode("aten::split")
        # 断言：split 输入进入通配符集合，列表初始化为包含通配符集合
        self.assertTrue(
            alias_db.may_contain_alias(next(split_node.inputs()), split_node.output())
        )
        # 因为 %x.1 进入通配符集合，现在它别名其他通配符集合成员（图的输入）
        self.assertTrue(
            alias_db.may_contain_alias(next(split_node.inputs()), next(graph.inputs()))
        )

    # 测试方法：test_nested_list_construct_not_wildcard
    def test_nested_list_construct_not_wildcard(self):
        # 使用 torch.jit.script 装饰器定义函数 foo
        @torch.jit.script
        def foo(x):
            # 创建随机张量 y
            y = torch.rand([2, 2])
            # 返回列表包含 y
            return [y]

        # 获取 foo 函数的图形对象
        graph = foo.graph
        # 获取图形的别名数据库
        graph.alias_db()
        alias_db = graph.alias_db()
        # 查找图中名称为 "aten::rand" 的节点的输出
        ten_construct = graph.findNode("aten::rand").output()
        # 获取图的输出
        output = next(graph.outputs())
        # 断言：输出可能包含与 ten_construct 的别名
        self.assertTrue(alias_db.may_contain_alias(ten_construct, output))
        # 断言：图的输入不与 ten_construct 别名
        self.assertFalse(
            alias_db.may_contain_alias(next(graph.inputs()), ten_construct)
        )
    def test_recursive_calls(self):
        # 定义一个使用 Torch JIT 脚本装饰器的函数 foo，实现对输入张量 x 进行原地加一，然后返回 x + y
        @torch.jit.script
        def foo(x, y):
            x.add_(1)
            return x + y

        # 定义一个使用 Torch JIT 脚本装饰器的函数 caller，调用 foo 函数两次并返回结果
        @torch.jit.script
        def caller():
            # 创建一个形状为 [2, 2] 的随机张量 a 和全为 1 的张量 b
            a = torch.rand([2, 2])
            b = torch.ones([2, 2])
            # 调用 foo 函数，将结果存储在 out1 中
            out1 = foo(a, b)
            # 创建一个形状为 [1] 的随机张量 c 和形状为 [2] 的全为 1 的张量 d
            c = torch.rand([1])
            d = torch.ones([2])
            # 再次调用 foo 函数，将结果存储在 out2 中
            out2 = foo(d, c)
            return out1, out2

        # 初始化变量，设置是否冻结以及是否向下递归调用函数
        isFrozen = False
        descend_function_calls = True
        # 获取 caller 函数的图形别名数据库
        alias_db = caller.graph.alias_db(isFrozen, descend_function_calls)
        # 查找所有名为 "prim::CallFunction" 的节点
        func_calls = caller.graph.findAllNodes("prim::CallFunction")
        # 断言找到的函数调用节点数量为 2
        self.assertEqual(len(func_calls), 2)
        # 遍历每个函数调用节点
        for node in func_calls:
            # 获取节点的输入列表
            inps = list(node.inputs())
            # 断言第二个输入是否有写入操作
            self.assertTrue(alias_db.has_writers(inps[1]))
            # 断言第三个输入是否没有写入操作
            self.assertFalse(alias_db.has_writers(inps[2]))

        # 定义一个继承自 torch.nn.Module 的模块 Mod
        class Mod(torch.nn.Module):
            # 实现模块的前向传播函数
            def forward(self):
                # 创建一个形状为 [2, 2] 的随机张量 a 和全为 1 的张量 b
                a = torch.rand([2, 2])
                b = torch.ones([2, 2])
                # 调用模块自身的 foo2 方法，将结果存储在 out1 中
                out1 = self.foo2(a, b)
                # 创建一个形状为 [1] 的随机张量 c 和形状为 [2] 的全为 1 的张量 d
                c = torch.rand([1])
                d = torch.ones([2])
                # 再次调用模块自身的 foo2 方法，将结果存储在 out2 中
                out2 = self.foo2(d, c)
                return out1, out2

            # 定义模块自身的方法 foo2，实现对输入张量 x 进行原地加一，然后返回 x + y
            def foo2(self, x, y):
                x.add_(1)
                return x + y

        # 使用 Torch JIT 脚本化 Mod 类实例化模块 mod
        mod = torch.jit.script(Mod())
        # 获取 mod 模块的图形别名数据库
        alias_db = mod.graph.alias_db(isFrozen, descend_function_calls)
        # 查找所有名为 "prim::CallMethod" 的节点
        func_calls = mod.graph.findAllNodes("prim::CallMethod")
        # 断言找到的方法调用节点数量为 2
        self.assertEqual(len(func_calls), 2)
        # 遍历每个方法调用节点
        for node in func_calls:
            # 获取节点的输入列表
            inps = list(node.inputs())
            # 断言第二个输入是否有写入操作
            self.assertTrue(alias_db.has_writers(inps[1]))
            # 断言第三个输入是否没有写入操作
            self.assertFalse(alias_db.has_writers(inps[2]))
    def test_multiple_compilation_units(self):
        # 这是我们遇到的一个内部问题的复现。
        # 在这里，我们有大量（40个）模块，每个模块都有相同的名称（MyModuleCUTest）。
        # AliasDB 使用一些哈希表来对类型进行哈希；这 40 个模块并非完全相同，因为它们有不同的编译单元，但它们的名称相同。
        # 因此，如果我们仅基于模块名称进行哈希（我们之前的做法），那么所有这些模块类型都会发生哈希冲突。
        #
        # flat_hash_map 对于这种哈希冲突行为性能非常差（指数增长）。
        # 在修复之前会导致内存溢出。
        N = 40

        class MultiTmpFile:
            def __init__(self, N):
                self.N = N
                # 创建 N 个临时文件名对象列表，用于写入模块脚本
                self.ctxs = [
                    TemporaryFileName(mode="w", suffix=".py") for _ in range(N)
                ]

            def __enter__(self):
                # 进入上下文管理器，返回所有临时文件名对象的上下文
                return [x.__enter__() for x in self.ctxs]

            def __exit__(self, exc_type, exc_value, traceback):
                # 退出上下文管理器，调用所有临时文件名对象的退出方法
                return [x.__exit__(exc_type, exc_value, traceback) for x in self.ctxs]

        class ModuleWrapper(torch.nn.Module):
            def __init__(self, module_list):
                super().__init__()
                self.module_list = module_list

            def forward(self, x):
                # 遍历模块列表，对每个模块调用 forward 方法
                for mod in self.module_list:
                    x = mod(x)
                return x

        # 使用 MultiTmpFile 上下文管理器创建临时文件名列表 fnames
        with MultiTmpFile(N) as fnames:
            module_list = torch.nn.ModuleList()
            global MyModuleCUTest

            # 定义一个名为 MyModuleCUTest 的模块类，继承自 torch.nn.Module
            class MyModuleCUTest(torch.nn.Module):
                def forward(self, x):
                    return x + 2

            # 遍历临时文件名列表 fnames，为每个文件名创建脚本化的 MyModuleCUTest 模块并保存
            for _, fname in enumerate(fnames):
                mod = torch.jit.script(MyModuleCUTest())
                torch.jit.save(mod, fname)
                loaded_mod = torch.jit.load(fname)
                module_list.append(loaded_mod)

            # 创建 ModuleWrapper 对象，将 module_list 作为参数传入
            mod = ModuleWrapper(module_list)
            # 对 ModuleWrapper 对象进行脚本化
            mod = torch.jit.script(mod)
            # 执行模型的 forward 方法，传入一个 2x2 的零张量作为输入
            mod(torch.zeros((2, 2)))
```