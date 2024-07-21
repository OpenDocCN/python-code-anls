# `.\pytorch\test\jit\test_attr.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
from typing import NamedTuple, Tuple

import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行此文件，抛出运行时错误，建议通过指定的方式来运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自 JitTestCase
class TestGetDefaultAttr(JitTestCase):
    
    # 测试获取带默认值的属性
    def test_getattr_with_default(self):
        
        # 定义一个继承自 torch.nn.Module 的类 A
        class A(torch.nn.Module):
            
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.init_attr_val = 1.0  # 初始化一个属性 init_attr_val 为 1.0

            # 前向传播方法
            def forward(self, x):
                y = getattr(self, "init_attr_val")  # 获取属性 init_attr_val 的值
                w: list[float] = [1.0]  # 定义一个类型为 float 的列表 w
                z = getattr(self, "missing", w)  # 获取属性 "missing" 的值，如果不存在则使用默认值 w
                z.append(y)  # 将 y 添加到 z 中
                return z

        # 创建类 A 的实例，并调用 forward 方法
        result = A().forward(0.0)
        self.assertEqual(2, len(result))  # 断言结果列表 result 的长度为 2
        graph = torch.jit.script(A()).graph  # 对类 A 进行脚本化，并获取其图表示

        # 断言图中存在获取属性 "init_attr_val" 的操作
        FileCheck().check('prim::GetAttr[name="init_attr_val"]').run(graph)
        # 断言图中不存在获取属性 "missing" 的操作，因此不应该有对应的 GetAttr
        FileCheck().check_not("missing").run(graph)
        # 相反，getattr 调用将会生成默认值的操作，这里是一个包含一个 float 元素的列表
        FileCheck().check("float[] = prim::ListConstruct").run(graph)

    # 测试获取命名元组的属性
    def test_getattr_named_tuple(self):
        global MyTuple
        
        # 定义一个命名元组 MyTuple
        class MyTuple(NamedTuple):
            x: str
            y: torch.Tensor
        
        # 定义一个函数 fn，接受一个 MyTuple 类型的参数 x，返回一个 Tuple[str, torch.Tensor, int]
        def fn(x: MyTuple) -> Tuple[str, torch.Tensor, int]:
            return (
                getattr(x, "x", "fdsa"),  # 获取 x 中的属性 "x"，如果不存在则返回 "fdsa"
                getattr(x, "y", torch.ones((3, 3))),  # 获取 x 中的属性 "y"，如果不存在则返回一个 3x3 全为 1 的 Tensor
                getattr(x, "z", 7),  # 获取 x 中的属性 "z"，如果不存在则返回 7
            )

        inp = MyTuple(x="test", y=torch.ones(3, 3) * 2)  # 创建一个 MyTuple 的实例 inp
        ref = fn(inp)  # 调用函数 fn，并将结果保存在 ref 中
        fn_s = torch.jit.script(fn)  # 对函数 fn 进行脚本化
        res = fn_s(inp)  # 对 inp 执行脚本化后的函数 fn_s
        self.assertEqual(res, ref)  # 断言脚本化后的结果与预期的 ref 相同

    # 测试获取元组的属性
    def test_getattr_tuple(self):
        
        # 定义一个函数 fn，接受一个 Tuple[str, int] 类型的参数 x，返回一个 int
        def fn(x: Tuple[str, int]) -> int:
            return getattr(x, "x", 2)  # 获取 x 中的属性 "x"，如果不存在则返回 2

        # 使用 self.assertRaisesRegex 断言在对 fn 进行脚本化时会抛出 RuntimeError，且错误信息中包含 "but got a normal Tuple"
        with self.assertRaisesRegex(RuntimeError, "but got a normal Tuple"):
            torch.jit.script(fn)
```