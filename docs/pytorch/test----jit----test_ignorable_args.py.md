# `.\pytorch\test\jit\test_ignorable_args.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os
import sys

# 导入 PyTorch 相关模块
import torch
from torch._C import parse_ir  # 导入 Torch 的内部 IR 解析功能
from torch.testing import FileCheck  # 导入用于文件检查的测试工具

# Make the helper files in test/ importable
# 设置测试目录并添加到系统路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase  # 导入测试用例的基类


if __name__ == "__main__":
    # 如果被直接运行，抛出运行时错误，建议使用特定方式运行测试文件
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


# Tests that Python slice class is supported in TorchScript
# 定义测试类，验证 Python 的切片类在 TorchScript 中的支持
class TestIgnorableArgs(JitTestCase):

    # 测试切片在 TorchScript 中处理忽略参数的情况
    def test_slice_ignorable_args_for_slice(self):
        # 定义包含 TorchScript IR 的字符串表示形式
        graph_str = """graph():
            %13 : int = prim::Constant[value=0]()
            %10 : bool = prim::Constant[value=0]()
            %8 : NoneType = prim::Constant()
            %0 : int = prim::Constant[value=1]()
            %1 : int = prim::Constant[value=2]()
            %2 : int = prim::Constant[value=3]()
            %3 : int = prim::Constant[value=4]()
            %4 : int = prim::Constant[value=9]()
            %5 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %6 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %7 : int[][] = prim::ListConstruct(%5, %6)
            %val.1 : Tensor = aten::tensor(%7, %8, %8, %10)
            %16 : Tensor = aten::slice(%val.1, %13, %1, %8, %0)
            %20 : Tensor = aten::slice(%16, %0, %8, %0, %0)
            return (%20)"""
        
        # 解析 IR 字符串为 TorchScript 图对象
        graph = parse_ir(graph_str)
        # 创建 TorchScript 函数对象
        function = self.createFunctionFromGraph(graph)
        # 创建函数的导出和导入副本
        function_copy = self.getExportImportCopy(function)
        # 获取 TorchScript 函数的源代码字符串表示
        src = str(function.code)
        
        # 使用 FileCheck 检查源代码中的特定字符串，验证 TorchScript 转换正确性
        FileCheck().check(
            "torch.slice(torch.slice(torch.tensor(_0), 0, 2), 1, None, 1)"
        ).run(src)
        
        # 调用函数和其导入导出的副本，并比较结果
        self.assertEqual(function(), function_copy())

    # 测试在 TorchScript 中添加忽略参数的情况
    def test_add_out_ignorable_args(self):
        # 使用 Torch 脚本装饰器定义函数
        @torch.jit.script
        def fn(x: torch.Tensor, y: torch.Tensor):
            # 调用 Torch 函数 torch.add()，指定输出参数 out=y
            torch.add(x, y, out=y)

        # 使用 FileCheck 检查 Torch 脚本中的特定字符串，验证 TorchScript 转换正确性
        FileCheck().check("torch.add(x, y, out=y)").run(fn.code)
```