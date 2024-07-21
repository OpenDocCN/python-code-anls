# `.\pytorch\test\export\test_functionalized_assertions.py`

```py
# Owner(s): ["oncall: export"]
# 导入 PyTorch 库
import torch
# 导入测试相关的工具类和函数
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义测试类 TestFuntionalAssertions，继承自 TestCase
class TestFuntionalAssertions(TestCase):
    
    # 测试异步消息功能的断言
    def test_functional_assert_async_msg(self) -> None:
        # 创建依赖令牌
        dep_token = torch.ops.aten._make_dep_token()
        
        # 调用 _functional_assert_async.msg 方法，验证返回的依赖令牌
        self.assertEqual(
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(1), "test msg", dep_token
            ),
            dep_token,
        )
        
        # 使用断言检查运行时异常是否包含指定消息
        with self.assertRaisesRegex(RuntimeError, "test msg"):
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(0), "test msg", dep_token
            ),

    # 测试符号约束范围功能
    def test_functional_sym_constrain_range(self) -> None:
        # 创建依赖令牌
        dep_token = torch.ops.aten._make_dep_token()
        
        # 调用 _functional_sym_constrain_range 方法，验证返回的依赖令牌
        self.assertEqual(
            torch.ops.aten._functional_sym_constrain_range(
                3, min=2, max=5, dep_token=dep_token
            ),
            dep_token,
        )

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```