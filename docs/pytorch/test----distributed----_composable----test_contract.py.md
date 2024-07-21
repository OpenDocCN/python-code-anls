# `.\pytorch\test\distributed\_composable\test_contract.py`

```py
# Owner(s): ["oncall: distributed"]

# 从标准库中导入 deepcopy 函数和 Tuple 类型
from copy import deepcopy
from typing import Tuple

# 导入 PyTorch 库
import torch
import torch.nn as nn
# 导入分布式相关的模块
from torch.distributed._composable import _get_registry, contract
# 导入测试相关的工具函数和类
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# 定义一个简单的神经网络模型
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 nn.Sequential 定义两层线性变换的序列 seq1 和 seq2
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        # 创建一个可学习的参数 self.p，并初始化为随机张量
        self.p = nn.Parameter(torch.randn(10, 10), requires_grad=True)
        # 创建一个张量 self.b，并初始化为全零，作为缓冲区（buffer）
        self.b = torch.zeros(1)  # buffer

    # 定义模型的前向传播方法
    def forward(self, x, y):
        # 使用 torch.no_grad() 上下文管理器，使得以下操作不会被记录到计算图中
        with torch.no_grad():
            # 更新缓冲区 self.b 的值为 x 和 y 的元素和
            self.b += x.sum() + y.sum()

        # 返回模型输出，是参数 self.p 与 seq1(x) 和 seq2(y) 的线性变换结果之和
        return self.p + self.seq1(x) + self.seq2(y)


# 测试用例类 TestContract，继承自 TestCase 类
class TestContract(TestCase):
    # 装饰器函数，如果在 Torch Dynamo 环境下，则跳过该测试用例
    @skipIfTorchDynamo("Dynamo does not support the state key")
    # 测试添加钩子函数的功能
    def test_add_hooks(self):
        # 定义前向预处理钩子函数
        def forward_pre_hook(
            module: nn.Module, inp: Tuple[torch.Tensor]
        ) -> Tuple[torch.Tensor]:
            return inp

        # 定义前向钩子函数
        def forward_hook(
            module: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor
        ) -> torch.Tensor:
            return out

        # 定义反向预处理钩子函数
        def backward_pre_hook(
            module: nn.Module, grad_output: torch.Tensor
        ) -> torch.Tensor:
            return grad_output

        # 定义反向钩子函数
        def backward_hook(
            module: nn.Module,
            grad_input: Tuple[torch.Tensor],
            grad_output: torch.Tensor,
        ) -> Tuple[torch.Tensor]:
            return grad_input

        # 装饰器函数，定义一个空操作的 API 函数
        @contract()
        def noop_api(module: nn.Module) -> nn.Module:
            # 分别注册前向预处理钩子、前向钩子、完整反向预处理钩子和完整反向钩子到模块 module 上
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_hook)
            module.register_full_backward_pre_hook(backward_pre_hook)
            module.register_full_backward_hook(backward_hook)
            return module

        # 创建 ToyModel 的实例 model
        model = ToyModel()
        # 深度复制 model，得到 model_with_hooks
        model_with_hooks = deepcopy(model)
        # 对 model 的 seq1 和 seq2 应用 noop_api 函数，添加钩子函数
        noop_api(model.seq1)
        noop_api(model.seq2)

        # 创建随机输入 x 和 y
        x, y = torch.randn(10, 10), torch.randn(10, 10)
        # 分别对 model 和 model_with_hooks 进行前向传播和反向传播
        model(x, y).sum().backward()
        model_with_hooks(x, y).sum().backward()

        # 验证两个模型的参数是否相等
        for p1, p2 in zip(model.parameters(), model_with_hooks.parameters()):
            self.assertEqual(p1, p2)

    # 装饰器函数，如果在 Torch Dynamo 环境下，则跳过该测试用例
    @skipIfTorchDynamo("Dynamo does not support the state key")
    # 测试修改 FQN（Fully Qualified Name）的功能
    def test_modify_fqn(self):
        # 定义一个模块包装器 ModelWrapper，封装输入模块 module
        class ModelWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

        # 装饰器函数，定义一个包装模块的 API 函数
        @contract()
        def wrap_module(module: nn.Module) -> nn.Module:
            return ModelWrapper(module)

        # 创建 ToyModel 的实例 model
        model = ToyModel()

        # 使用 self.assertRaisesRegex 断言捕获 RuntimeError 异常，并检查异常消息是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            "Check parameters, Composable distributed API implementations cannot modify FQNs",
        ):
            # 尝试对 model 的 seq1 应用 wrap_module 函数，捕获预期的异常
            wrap_module(model.seq1)
    @skipIfTorchDynamo("Dynamo does not support the state key")
    # 定义测试函数，检查并更新模型状态钩子
    def test_state(self):
        def check_and_update_state_hook(
            module: nn.Module, inp: Tuple[torch.Tensor]
        ) -> Tuple[torch.Tensor]:
            # 断言模型的dummy_state属性为7
            self.assertEqual(api.state(module).dummy_state, 7)
            # 更新模型的dummy_state属性为8
            api.state(module).dummy_state = 8
            return inp

        # FIXME: 循环引用看起来有点奇怪。我们应该将.state作为顶级API而不是附加到contract API吗？
        # 定义API函数，设置模型的dummy_state属性为7，并注册前向钩子
        @contract()
        def api(module: nn.Module) -> nn.Module:
            api.state(module).dummy_state = 7
            module.register_forward_pre_hook(check_and_update_state_hook)
            return module

        # 创建ToyModel实例
        model = ToyModel()
        # 调用api函数，作用于model.seq1
        api(model.seq1)

        # 断言模型seq1的dummy_state属性为7
        self.assertEqual(api.state(model.seq1).dummy_state, 7)
        # 对模型进行前向传播
        model(torch.zeros(10, 10), torch.zeros(10, 10))
        # 再次断言模型seq1的dummy_state属性为8
        self.assertEqual(api.state(model.seq1).dummy_state, 8)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    # 定义测试函数，测试API注册功能
    def test_registry(self):
        # 定义API函数api1，返回输入模型module
        @contract()
        def api1(module: nn.Module) -> nn.Module:
            return module

        # 定义API函数api2，返回输入模型module
        @contract()
        def api2(module: nn.Module) -> nn.Module:
            return module

        # 创建ToyModel实例
        model = ToyModel()
        # 使用api1函数作用于model
        model = api1(model)
        # 断言_registry中的注册数为1
        self.assertEqual(1, len(_get_registry(model)))
        # 断言_registry中包含"api1"键
        self.assertTrue("api1" in _get_registry(model))
        # 使用api2函数作用于model
        model = api2(model)
        # 断言_registry中的注册数为2
        self.assertEqual(2, len(_get_registry(model)))
        # 断言_registry的键为["api1", "api2"]
        self.assertTrue([_get_registry(model).keys()], ["api1", "api2"])
        # 断言model.seq1在_registry中为None
        self.assertEqual(None, _get_registry(model.seq1))
        # 断言model.seq2在_registry中为None
        self.assertEqual(None, _get_registry(model.seq2))

        # 使用assertRaisesRegex断言抛出异常AssertionError，异常信息包含"api1 has already been applied"
        with self.assertRaisesRegex(AssertionError, "api1 has already been applied"):
            model = api1(model)
# 如果当前脚本作为主程序运行（而不是被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 调用函数 run_tests()，用于执行测试或主程序的功能
    run_tests()
```