# `.\pytorch\test\jit\test_parametrization.py`

```
# Owner(s): ["oncall: jit"]

# 导入PyTorch相关模块
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

# 导入测试相关的内部工具
from torch.testing._internal.jit_utils import JitTestCase

# 如果当前脚本作为主程序运行，抛出运行时错误，建议使用特定命令运行测试文件
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自JitTestCase
class TestParametrization(JitTestCase):
    
    # 定义一个模型类，继承自nn.Module，用于对权重进行对称化处理
    class Symmetric(nn.Module):
        def forward(self, X):
            # 返回上三角部分与其转置的和
            return X.triu() + X.triu(1).mT

    # 测试模型的可追踪性
    def test_traceable(self):
        r"""Test the jit scripting and tracing of a parametrized model."""
        # 创建一个线性模型
        model = nn.Linear(5, 5)
        # 将权重参数注册为对称化处理
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)  # 创建输入张量
        y = model(x)  # 对输入进行模型预测

        # 检查追踪功能是否正常工作，因为追踪后的函数无法直接调用，比较激活值
        traced_model = torch.jit.trace_module(model, {"forward": x})
        y_hat = traced_model(x)
        self.assertEqual(y, y_hat)  # 断言相等性

        # 检查追踪模型在缓存中是否正常工作
        with parametrize.cached():
            y_hat = traced_model(x)
            self.assertEqual(y, y_hat)  # 断言相等性

        # 检查追踪过程在缓存时是否会抛出错误
        with self.assertRaisesRegex(RuntimeError, "Cannot trace a model while caching"):
            with parametrize.cached():
                traced_model = torch.jit.trace_module(model, {"forward": x})

    def test_scriptable(self):
        # TODO: Need to fix the scripting in parametrizations
        #       Currently, all the tests below will throw torch.jit.Error
        
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)
        y = model(x)

        with self.assertRaises(torch.jit.Error):
            # 检查脚本化是否正常工作
            scripted_model = torch.jit.script(model)
            y_hat = scripted_model(x)
            self.assertEqual(y, y_hat)  # 断言相等性

            with parametrize.cached():
                # 检查脚本化模型在缓存中是否正常工作
                y_hat = scripted_model(x)
                self.assertEqual(y, y_hat)  # 断言相等性

                # 检查脚本化过程在缓存时是否会抛出错误
                with self.assertRaisesRegex(RuntimeError, "Caching is not implemented"):
                    scripted_model = torch.jit.trace_module(model)
```