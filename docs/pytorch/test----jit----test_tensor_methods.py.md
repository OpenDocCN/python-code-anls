# `.\pytorch\test\jit\test_tensor_methods.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os
import sys

import torch  # 导入 PyTorch 库

# Make the helper files in test/ importable
# 获取当前脚本的父目录并添加到系统路径中，以便导入 test/ 目录下的文件
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing import FileCheck  # 导入文件检查工具
from torch.testing._internal.jit_utils import JitTestCase  # 导入 JIT 测试用例类

if __name__ == "__main__":
    # 如果直接运行该文件，则抛出运行时错误，提示正确的使用方式
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自 JitTestCase
class TestTensorMethods(JitTestCase):
    
    # 测试 torch.Tensor 的 __getitem__ 方法
    def test_getitem(self):
        def tensor_getitem(inp: torch.Tensor):
            indices = torch.tensor([0, 2], dtype=torch.long)  # 创建索引张量
            return inp.__getitem__(indices)  # 使用索引张量获取对应的子张量

        inp = torch.rand(3, 4)  # 创建一个随机张量
        self.checkScript(tensor_getitem, (inp,))  # 检查 tensor_getitem 函数的脚本化版本

        scripted = torch.jit.script(tensor_getitem)  # 对 tensor_getitem 函数进行脚本化
        FileCheck().check("aten::index").run(scripted.graph)  # 使用 FileCheck 检查生成的图中是否包含 "aten::index" 操作

    # 测试 torch.Tensor 的无效 __getitem__ 调用
    def test_getitem_invalid(self):
        def tensor_getitem_invalid(inp: torch.Tensor):
            return inp.__getitem__()  # 无参数调用 __getitem__，应该抛出错误

        # 使用 assertRaisesRegexWithHighlight 断言捕获特定错误信息并高亮显示 "inp.__getitem__"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "expected exactly 1 argument", "inp.__getitem__"
        ):
            torch.jit.script(tensor_getitem_invalid)  # 尝试对 tensor_getitem_invalid 进行脚本化
```