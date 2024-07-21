# `.\pytorch\test\jit\test_dce.py`

```
# Owner(s): ["oncall: jit"]

# 导入 PyTorch 库
import torch
# 导入用于测试的文件检查工具
from torch.testing import FileCheck
# 导入 JIT 测试相关的实用工具和基类
from torch.testing._internal.jit_utils import JitTestCase, make_global

# 定义一个继承自 JitTestCase 的测试类 TestDCE
class TestDCE(JitTestCase):
    
    # 测试函数：测试在没有别名数据库情况下的属性设置
    def test_setattr_no_aliasdb(self):
        # 定义一个简单的神经网络模型类 Net
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个空的张量 self.x
                self.x = torch.empty([2, 2])

            def forward(self):
                # 在 forward 方法中生成一个随机张量 x
                x = torch.rand([3, 3])
                # 将生成的随机张量赋值给 self.x
                self.x = x

        # 使用 torch.jit.script 方法将 Net 类实例化为脚本化模型 net
        net = torch.jit.script(Net())

        # 使用 FileCheck 检查 net 的计算图中是否包含 "prim::SetAttr" 操作

    # 测试函数：测试属性设置后是否被删除
    def test_setattr_removed(self):
        # 使用 torch.jit.script 装饰器将类 Thing1 脚本化
        @torch.jit.script
        class Thing1:
            def __init__(self):
                # 初始化一个全零张量 self.x
                self.x = torch.zeros([2, 2])

        # 将 Thing1 类注册为全局对象
        make_global(Thing1)

        # 定义一个继承自 torch.nn.Module 的类 Thing2
        class Thing2(torch.nn.Module):
            def forward(self):
                # 生成两个随机张量 x 和 y
                x = torch.rand([2, 2])
                y = torch.rand([2, 2])
                # 实例化一个 Thing1 对象 t1，并将随机张量 x 赋值给 t1.x
                t1 = Thing1()
                t1.x = x
                return y

        # 创建一个未脚本化的 Thing2 类实例 unscripted
        unscripted = Thing2()

        # 使用 torch.jit.script 方法将 unscripted 脚本化为 t2
        t2 = torch.jit.script(unscripted)
        # 设定 t2 为评估模式
        t2.eval()

        # 冻结 t2，即将其内联化 t1.__init__() 后，执行 DCE（死代码消除）
        t2 = torch.jit.freeze(t2)
        # 使用 FileCheck 检查 t2 的计算图中是否不包含 "prim::SetAttr" 操作
```