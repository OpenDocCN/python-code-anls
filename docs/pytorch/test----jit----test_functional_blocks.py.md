# `.\pytorch\test\jit\test_functional_blocks.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os
import sys

import torch
from torch.testing import FileCheck

# 将 test/ 目录下的 helper 文件添加到模块搜索路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果这个文件被直接运行，则抛出 RuntimeError 提示用户不直接运行该文件
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类 TestFunctionalBlocks，继承自 JitTestCase
class TestFunctionalBlocks(JitTestCase):
    
    # 定义测试函数 test_subgraph_creation
    def test_subgraph_creation(self):
        
        # 定义一个函数 fn，接受三个参数 x, y, z
        def fn(x, y, z):
            # 对 x, y, z 进行一系列数学操作
            x = x + 1
            y = y + 1
            z = z + 1
            z.add_(2)  # 原地加法操作
            z = z * z   # z 的平方
            y = y * z   # y 乘以 z 的平方
            if y < 2:   # 条件判断
                y = y + 5
            return x + y + z  # 返回 x + y + z 的结果
        
        # 使用 torch.jit.script 将 fn 函数转换为 Torch Script，并获取其图形表示
        graph = torch.jit.script(fn).graph
        
        # 运行名为 "create_functional_graphs" 的优化 pass
        self.run_pass("create_functional_graphs", graph)
        
        # 对图形进行检查，确保所有对 x 和 y 的使用被下沉（sunk）
        FileCheck().check(r"%x").check_not(r"%x").check("FunctionalGraph").check(
            r"%x"
        ).run(graph)
        FileCheck().check(r"%y").check_not(r"%y").check("FunctionalGraph").check(
            r"%y"
        ).run(graph)
        
        # 检查图形，确保没有任何逃逸出作用域的输出，因此图中会有一次最终的加法操作
        FileCheck().check("Tensor = prim::Functional").check_next("aten::add").run(
            graph
        )
        
        # 检查图形，确保 z + 1, z.add_(2) 被认为是非函数式的，而 z = z * z 应被认为是函数式的
        FileCheck().check("add").check("add_").check_not("mul").check(
            "FunctionalGraph"
        ).run(graph)
```