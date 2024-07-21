# `.\pytorch\test\jit\test_modules.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os
import sys

# 导入 PyTorch 库及其测试工具
import torch
from torch.testing._internal.jit_utils import JitTestCase

# 将测试文件所在目录添加到系统路径，使得其中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果此脚本被直接运行，则抛出运行时错误，建议使用特定命令运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestModules，继承自 JitTestCase
class TestModules(JitTestCase):
    
    # 定义测试方法 test_script_module_with_constants_list
    def test_script_module_with_constants_list(self):
        """
        Test that a module that has __constants__ set to something
        that is not a set can be scripted.
        """

        # 定义一个名为 Net 的子类，继承自 torch.nn.Linear
        # 设置 x 作为 torch.jit.Final[int] 类型的属性
        class Net(torch.nn.Linear):
            x: torch.jit.Final[int]

            # 初始化方法
            def __init__(self):
                # 调用父类的初始化方法，创建一个线性层，输入维度为 5，输出维度为 10
                super().__init__(5, 10)
                # 初始化 self.x 为 0
                self.x = 0

        # 使用 JitTestCase 类中的方法 checkModule 对定义的 Net 模块进行测试
        # 参数为 Net() 实例和一个包含 torch.randn(5) 的元组
        self.checkModule(Net(), (torch.randn(5),))
```