# `.\pytorch\test\test_functional_optim.py`

```
# Owner(s): ["oncall: distributed"]

import unittest  # 导入单元测试框架
from typing import List, Optional, Tuple  # 导入类型提示相关模块

import torch  # 导入PyTorch库
import torch.distributed  # 导入PyTorch分布式模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from torch import Tensor  # 导入张量类型
from torch.optim import Adam, AdamW, SGD  # 导入优化器
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关工具类

# 定义一个简单的神经网络模块
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(3, 3, bias=False)  # 定义一个线性层，输入3维，输出3维，无偏置
        self.lin2 = nn.Linear(3, 3, bias=False)  # 定义另一个线性层，输入3维，输出3维，无偏置

    def forward(self, t1):
        return self.lin2(F.relu(self.lin1(t1)))  # 前向传播函数，先经过lin1，再经过ReLU激活，最后经过lin2


# dummy class to showcase custom optimizer registration with functional wrapper
# 自定义类，演示如何使用函数包装器注册自定义优化器
class MyDummyFnOptimizer:
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        _allow_empty_param_list: bool = False,
    ):
        # 参数合法性检查
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 初始化默认参数字典
        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }

        # 检查参数列表是否为空
        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

    # 不支持的优化器参数更新函数
    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        with torch.no_grad():
            raise RuntimeError(
                "MyDummyFnOptimizer does not support step_param() as of now"
            )

    # 不支持的优化器整体参数更新函数
    def step(self, gradients: List[Optional[Tensor]]):
        with torch.no_grad():
            raise RuntimeError("MyDummyFnOptimizer does not support step() as of now")


# 检查是否支持分布式环境，如果支持，则导入相关工具
if torch.distributed.is_available():
    from torch.distributed.optim.utils import (
        functional_optim_map,
        register_functional_optim,
    )

# 单元测试类，测试函数式优化器的兼容性
@unittest.skipIf(
    not torch.distributed.is_available(), "These are testing distributed functions"
)
class TestFunctionalOptimParity(TestCase):
    # 验证参数函数，比较两个参数列表的元素是否相等
    def _validate_parameters(self, params_1, params_2):
        for p1, p2 in zip(params_1, params_2):
            self.assertEqual(p1, p2)

    # 在Python 3.8/3.11下，Dynamo编译此部分会失败
    # 因为在实际测试代码中编译通过
    # 我们在这里禁用Dynamo
    @torch._disable_dynamo(recursive=False)
    # 测试函数：验证优化器的功能和性能
    def _test_functional_optim_parity(self, optim_cls, *args, **kwargs):
        # 创建一个模块实例用于优化
        module_optim = MyModule()
        # 创建另一个模块实例用于功能性对比
        module_functional = MyModule()
        # 获取优化模块的参数
        optim_params = module_optim.parameters()
        # 获取功能性模块的参数
        functional_params = module_functional.parameters()
        # 创建指定类型的优化器实例
        optim = optim_cls(optim_params, *args, **kwargs)
        # 根据优化器类型获取功能性优化器类
        functional_optim_cls = functional_optim_map.get(optim_cls, None)
        # 如果没有找到对应的功能性优化器类，则抛出错误
        if not functional_optim_cls:
            raise ValueError(f"Functional optimizer not implemented for {optim_cls}")
        # 使用功能性优化器类创建功能性优化器实例，允许空参数列表
        optim_functional = functional_optim_cls(
            [], *args, **kwargs, _allow_empty_param_list=True
        )
        # 如果功能性优化器类没有实现 `step_param` 方法，则抛出错误
        if not hasattr(optim_functional, "step_param"):
            raise ValueError(
                f"Functional optimizer class {optim_functional} must implement step_param method."
            )

        # 验证初始权重应该匹配
        self._validate_parameters(
            module_optim.parameters(), module_functional.parameters()
        )
        # 保存旧参数以验证优化器是否修改它们
        old_module_optim_params = [
            param.clone().detach() for param in module_optim.parameters()
        ]
        old_module_functional_params = [
            param.clone().detach() for param in module_functional.parameters()
        ]

        # 创建一个形状为 (3, 3) 的随机张量
        t1 = torch.randn(3, 3)
        # 进行 10 次迭代
        for _ in range(10):
            # 清零优化器的梯度
            module_optim.zero_grad()
            module_functional.zero_grad()
            # 前向传播 + 反向传播
            optim_out = module_optim(t1).sum()
            functional_out = module_functional(t1).sum()
            optim_out.backward()
            functional_out.backward()
            # 执行优化器的步骤
            optim.step()
            # 执行功能性优化器的 step_param 方法
            for param in module_functional.parameters():
                grad = param.grad
                optim_functional.step_param(param, grad)

            # 验证参数是否相等
            for optim_param, functional_param in zip(
                module_optim.parameters(), module_functional.parameters()
            ):
                self.assertEqual(optim_param, functional_param)
            # 验证参数是否被修改
            for i, (optim_param, functional_param) in enumerate(
                zip(module_optim.parameters(), module_functional.parameters())
            ):
                self.assertNotEqual(old_module_optim_params[i], optim_param)
                self.assertNotEqual(old_module_functional_params[i], functional_param)

    # 测试函数：验证功能性优化器的注册过程
    def _test_functional_optim_registration(self):
        # 函数映射的关键字
        fn_map_key = "MyDummyFnOptimizer"
        # 获取指定的功能性优化器类
        fn_optim = MyDummyFnOptimizer
        # 将函数映射关键字和功能性优化器类注册到全局映射表中
        register_functional_optim(fn_map_key, fn_optim)
        # 获取注册后的功能性优化器类
        functional_optim_cls = functional_optim_map.get(fn_map_key, None)
        # 如果没有找到注册的功能性优化器类，则抛出错误
        if not functional_optim_cls:
            raise ValueError(f"Functional optimizer not registered for {fn_map_key}")
    # 测试函数，调用 _test_functional_optim_registration 方法进行功能测试
    def test_functional_optim_registration(self):
        self._test_functional_optim_registration()
    
    # 测试函数，调用 _test_functional_optim_parity 方法进行功能测试，使用 SGD 优化器
    def test_functional_optim_parity_sgd(self):
        self._test_functional_optim_parity(SGD, 1e-2, momentum=0.9, weight_decay=0.01)
    
    # 测试函数，调用 _test_functional_optim_parity 方法进行功能测试，使用 Adam 优化器
    def test_functional_optim_parity_adam(self):
        self._test_functional_optim_parity(Adam, 1e-2, betas=(0.9, 0.999), eps=1e-6)
    
    # 测试函数，调用 _test_functional_optim_parity 方法进行功能测试，使用 AdamW 优化器
    def test_functional_optim_parity_adam_w(self):
        self._test_functional_optim_parity(AdamW, 1e-2, betas=(0.9, 0.999), eps=1e-6)
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```