# `.\pytorch\test\jit\test_slice.py`

```
# Owner(s): ["oncall: jit"]

import os  # 导入操作系统接口模块
import sys  # 导入系统相关功能模块
from typing import List  # 导入类型提示模块中的List类型

import torch  # 导入PyTorch深度学习库

# Make the helper files in test/ importable
# 获取当前脚本所在目录的上一级目录路径，并将其添加到系统路径中，使得test/目录下的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase  # 导入PyTorch的测试工具类

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# Tests that Python slice class is supported in TorchScript
# 定义一个测试类，用于测试Python切片类在TorchScript中的支持
class TestSlice(JitTestCase):
    
    # 测试带关键字参数的切片函数
    def test_slice_kwarg(self):
        def slice_kwarg(x: List[int]):
            return x[slice(1, stop=2)]  # 返回从索引1到索引2的切片
        
        # 检查是否引发了期望的异常信息
        with self.assertRaisesRegex(
            RuntimeError, "Slice does not accept any keyword arguments"
        ):
            torch.jit.script(slice_kwarg)

    # 测试三个None参数的切片函数
    def test_slice_three_nones(self):
        def three_nones(x: List[int]):
            return x[slice(None, None, None)]  # 返回完整切片
        
        # 使用JitTestCase中的方法检查函数是否可以成功转换为TorchScript
        self.checkScript(three_nones, (range(10),))

    # 测试两个None参数的切片函数
    def test_slice_two_nones(self):
        def two_nones(x: List[int]):
            return x[slice(None, None)]  # 返回从头到尾的切片
        
        self.checkScript(two_nones, (range(10),))

    # 测试一个None参数的切片函数
    def test_slice_one_none(self):
        def one_none(x: List[int]):
            return x[slice(None)]  # 返回从头到尾的切片
        
        self.checkScript(one_none, (range(10),))

    # 测试只有stop参数的切片函数
    def test_slice_stop_only(self):
        def fn(x: List[int]):
            return x[slice(5)]  # 返回前5个元素的切片
        
        self.checkScript(fn, (range(10),))

    # 测试带有None和stop参数的切片函数
    def test_slice_stop_only_with_nones(self):
        def fn(x: List[int]):
            return x[slice(None, 5, None)]  # 返回从头到第5个元素的切片
        
        self.checkScript(fn, (range(10),))

    # 测试带有start和stop参数的切片函数
    def test_slice_start_stop(self):
        def fn(x: List[int]):
            return x[slice(1, 5)]  # 返回从索引1到索引5的切片
        
        self.checkScript(fn, (range(10),))

    # 测试带有start、stop和step参数的切片函数
    def test_slice_start_stop_with_none(self):
        def fn(x: List[int]):
            return x[slice(1, 5, None)]  # 返回从索引1到索引5的切片
        
        self.checkScript(fn, (range(10),))

    # 测试带有start、stop和step参数，且针对多维数据进行切片的函数
    def test_slice_start_stop_step(self):
        def fn(x: List[int]):
            return x[slice(0, 6, 2)]  # 返回从索引0到索引6，步长为2的切片
        
        self.checkScript(fn, (range(10),))

    # 测试对字符串进行切片的函数
    def test_slice_string(self):
        def fn(x: str):
            return x[slice(None, 3, 1)]  # 返回字符串前3个字符的切片
        
        self.checkScript(fn, ("foo_bar",))

    # 测试对Tensor进行切片的函数
    def test_slice_tensor(self):
        def fn(x: torch.Tensor):
            return x[slice(None, 3, 1)]  # 返回Tensor的前3个元素的切片
        
        self.checkScript(fn, (torch.ones(10),))

    # 测试对多维Tensor进行切片的函数
    def test_slice_tensor_multidim(self):
        def fn(x: torch.Tensor):
            return x[slice(None, 3, 1), 0]  # 返回Tensor的前3个元素的第0维切片
        
        self.checkScript(fn, (torch.ones((10, 10)),))

    # 测试对多维Tensor进行切片，并使用省略号的函数
    def test_slice_tensor_multidim_with_dots(self):
        def fn(x: torch.Tensor):
            return x[slice(None, 3, 1), ...]  # 返回Tensor的前3个元素的全部维度切片
        
        self.checkScript(fn, (torch.ones((10, 10)),))

    # 测试将切片对象作为变量使用的函数
    def test_slice_as_variable(self):
        def fn(x: List[int]):
            a = slice(1)  # 创建一个切片对象
            return x[a]  # 返回根据切片对象a对列表x进行的切片操作
        
        self.checkScript(fn, (range(10),))
    # 定义测试方法，测试对列表切片操作的处理情况
    def test_slice_stop_clipped(self):
        # 定义一个函数 fn，接受一个整数列表 x，并返回 x 的前1000个元素
        def fn(x: List[int]):
            return x[slice(1000)]

        # 调用自定义的检查方法 checkScript，验证函数 fn 对 range(10) 的处理结果
        self.checkScript(fn, (range(10),))

    # 定义测试方法，测试动态索引切片操作的处理情况
    def test_slice_dynamic_index(self):
        # 定义函数 t，接受参数 x
        def t(x):
            # 切片操作 slice1 获取 x 的第一个元素
            slice1 = x[0:1]
            # 定义变量 zero 和 one，分别为 0 和 1
            zero = 0
            one = zero + 1
            # 切片操作 slice2 获取 x 的第一个元素（通过 zero 和 one 变量）
            slice2 = x[zero:one]
            # 返回两个切片操作的结果相加
            return slice1 + slice2

        # 调用自定义的检查方法 checkScript，验证函数 t 对 torch.zeros(3, 2, 3) 的处理结果
        self.checkScript(t, (torch.zeros(3, 2, 3),))

    # 定义测试方法，测试元组切片操作的处理情况
    def test_tuple_slicing(self):
        # 定义函数 tuple_slice，接受参数 a
        def tuple_slice(a):
            # 如果 a 为真值，赋值元组 b 为 (1, 2, 3, 4)，否则赋值为 (4, 3, 2, 1)
            if bool(a):
                b = (1, 2, 3, 4)
            else:
                b = (4, 3, 2, 1)
            # 切片操作 c 获取 b 的所有元素
            c = b[-4:4]
            # 切片操作 e 获取 c 的第二个到倒数第二个元素
            e = c[1:-1]
            # 返回切片操作 e 的结果
            return e

        # 调用自定义的检查方法 checkScript，验证函数 tuple_slice 对 torch.tensor([1]) 的处理结果，并启用优化
        self.checkScript(tuple_slice, (torch.tensor([1]),), optimize=True)
        # 对 tuple_slice 函数进行脚本化处理
        scripted_fn = torch.jit.script(tuple_slice)
        # 断言脚本化函数对 torch.tensor(1) 的执行结果
        self.assertEqual(scripted_fn(torch.tensor(1)), (2, 3))
        # 获取脚本化函数的图形表示
        tuple_graph = scripted_fn.graph
        # 查找所有的元组构造节点
        slices = tuple_graph.findAllNodes("prim::TupleConstruct")
        # 计算每个元组节点的输出元素数量
        num_outputs = {len(x.output().type().elements()) for x in slices}
        # 断言只有一个输出元素数量为 2 的元组构造节点
        self.assertTrue(num_outputs == {2})
        # 在 tuple_graph 上运行 "lower_all_tuples" 优化传递
        self.run_pass("lower_all_tuples", tuple_graph)
        # 断言 tuple_graph 中不再包含 "Tuple" 字符串
        self.assertTrue("Tuple" not in str(tuple_graph))

    # 定义测试方法，测试模块列表切片操作的处理情况
    def test_module_list_slicing(self):
        # 定义类 Bar，继承自 torch.nn.Module
        class Bar(torch.nn.Module):
            # 初始化方法，接受标识符 identifier
            def __init__(self, identifier: str):
                super().__init__()
                self.identifier = identifier

            # 前向传播方法，返回 0
            def forward(self):
                return 0

        # 定义类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建模块列表 module_list，包含五个 Bar 实例
                module_list = [Bar("A"), Bar("B"), Bar("C"), Bar("D"), Bar("E")]
                # 创建 ModuleList 实例 self.test，并赋值为 module_list
                self.test = torch.nn.ModuleList(module_list)

            # 前向传播方法
            def forward(self):
                # 返回 self.test 列表的倒数第二个到第一个元素的逆序列表和第二个到第四个元素的子列表
                return self.test[::-2], self.test[1:4:]

        # 对 Foo 类进行脚本化处理
        scripted_foo = torch.jit.script(Foo())
        # 执行脚本化的 Foo 实例，并获取结果 result1 和 result2
        result1, result2 = scripted_foo()

        # 断言 result1 列表的长度为 3，且其元素的 identifier 属性分别为 "E", "C", "A"
        self.assertEqual(len(result1), 3)
        self.assertEqual(result1[0].identifier, "E")
        self.assertEqual(result1[1].identifier, "C")
        self.assertEqual(result1[2].identifier, "A")

        # 断言 result2 列表的长度为 3，且其元素的 identifier 属性分别为 "B", "C", "D"
        self.assertEqual(len(result2), 3)
        self.assertEqual(result2[0].identifier, "B")
        self.assertEqual(result2[1].identifier, "C")
        self.assertEqual(result2[2].identifier, "D")
```