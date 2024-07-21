# `.\pytorch\test\distributed\optim\test_apply_optimizer_in_backward.py`

```
# 导入单元测试模块
import unittest
# 从标准库中导入深拷贝函数
from copy import deepcopy

# 导入PyTorch库
import torch
import torch.nn as nn

# 从PyTorch分布式优化模块中导入相关函数和类
from torch.distributed.optim import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)

# TODO (rohan-varma): Add FSDP & DDP tests once supported

# 定义一个函数用于验证参数列表
def _validate_params(params_list, fn):
    # 取第一个模型的参数作为参考参数
    ref_params = params_list[0]
    # 遍历比较所有参数列表的对应参数
    for param_list in params_list[1:]:
        for p1, p2 in zip(ref_params, param_list):
            fn(p1, p2)

# 定义一个单元测试类
class ApplyOverlappedOptimizerTest(unittest.TestCase):

    # 定义一个方法来运行训练循环并验证结果
    def _run_training_loop_and_validate(self, inp, models, optimizers):
        for i in range(6):
            for model in models:
                # 对模型进行前向计算和反向传播
                model(inp).sum().backward()
            for opt in optimizers:
                # 执行优化器的参数更新
                opt.step()

            # 使用子测试功能来验证每一轮迭代的参数是否符合预期
            with self.subTest(i):
                _validate_params(
                    [model.parameters() for model in models],
                    torch.testing.assert_allclose,
                )

            for opt in optimizers:
                # 清空优化器的梯度信息
                opt.zero_grad(set_to_none=True)

    # 定义一个测试函数，测试在反向传播中应用优化器的功能
    def _test_apply_optimizer_in_backward(self, share_params) -> None:
        # 设置权重和偏置优化器的参数
        weight_optimizer_kwargs = {"lr": 1.0}
        bias_optimizer_kwargs = {"lr": 0.5}
        
        # 创建一个包含两个线性层的模型
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        
        # 如果需要共享参数，则将第一个线性层的权重与第二个线性层的权重共享
        if share_params:
            model[0].weight = model[1].weight

        # 分别获取权重和偏置的优化器
        weights = [m.weight for m in model]
        biases = [m.bias for m in model]
        optim_weight = torch.optim.SGD(weights, **weight_optimizer_kwargs)
        optim_bias = torch.optim.SGD(biases, **bias_optimizer_kwargs)
        
        # 深拷贝模型以备份当前状态
        model_with_opt_in_bwd = deepcopy(model)

        # 在反向传播中分别应用权重和偏置的优化器
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.weight for m in model_with_opt_in_bwd],
            optimizer_kwargs=weight_optimizer_kwargs,
        )

        _apply_optimizer_in_backward(
            torch.optim.SGD,
            [m.bias for m in model_with_opt_in_bwd],
            optimizer_kwargs=bias_optimizer_kwargs,
        )

        # 验证原始模型和应用优化器后的模型参数是否一致
        _validate_params(
            [
                model.parameters(),
                model_with_opt_in_bwd.parameters(),
            ],
            torch.testing.assert_allclose,
        )

        # 运行训练循环并验证结果
        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [optim_weight, optim_bias],
        )

    # 测试在反向传播中应用优化器的功能（不共享参数）
    def test_apply_optimizer_in_backward(self) -> None:
        self._test_apply_optimizer_in_backward(share_params=False)

    # 测试在反向传播中应用优化器的功能（共享参数）
    def test_apply_optimizer_in_backward_shared_params(self) -> None:
        self._test_apply_optimizer_in_backward(share_params=True)
    def test_no_register_hook(self):
        # 创建一个带有注册hook的模型，包含两个线性层
        model_with_hook = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        # 深拷贝带有hook的模型作为初始模型
        initial_model = deepcopy(model_with_hook)
        # 深拷贝另一个带有hook的模型
        model_no_hook = deepcopy(model_with_hook)
        # 对带有hook的模型应用优化器（SGD），学习率为0.03
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_hook.parameters(),
            optimizer_kwargs={"lr": 0.03},
        )
        # 对不带hook的模型应用优化器（SGD），学习率为0.03，但不注册hook
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_no_hook.parameters(),
            optimizer_kwargs={"lr": 0.03},
            register_hook=False,
        )
        # 创建输入数据
        inp = torch.randn(4, 10)
        # 对带hook的模型进行前向传播、求和、反向传播
        model_with_hook(inp).sum().backward()
        # 对不带hook的模型进行前向传播、求和、反向传播
        model_no_hook(inp).sum().backward()

        # 验证每个参数的值是否相同，预期会抛出AssertionError
        for p1, p2 in zip(model_with_hook.parameters(), initial_model.parameters()):
            with self.assertRaises(AssertionError):
                torch.testing.assert_allclose(p1, p2)

        # 验证每个参数的值是否相同，预期每个参数的值会很接近
        for p1, p2 in zip(model_no_hook.parameters(), initial_model.parameters()):
            torch.testing.assert_allclose(p1, p2)

    def test_multiple_optim_for_params(self) -> None:
        # 创建一个简单的模型，包含两个线性层
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        opt_0_kwargs = {"lr": 0.03}
        opt_1_kwargs = {"lr": 0.01}
        # 创建两个不同的SGD优化器，分别针对模型的参数
        opt_0 = torch.optim.SGD(model.parameters(), **opt_0_kwargs)
        opt_1 = torch.optim.SGD(model.parameters(), **opt_1_kwargs)
        # 深拷贝带有优化器的模型
        model_with_opt_in_bwd = deepcopy(model)
        # 对带有优化器的模型应用优化器（SGD），使用第一个优化器的参数
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_0_kwargs,
        )
        # 对带有优化器的模型应用优化器（SGD），使用第二个优化器的参数
        _apply_optimizer_in_backward(
            torch.optim.SGD,
            model_with_opt_in_bwd.parameters(),
            optimizer_kwargs=opt_1_kwargs,
        )
        # 运行训练循环并验证结果
        self._run_training_loop_and_validate(
            torch.randn(4, 10),
            [model, model_with_opt_in_bwd],
            [opt_0, opt_1],
        )

    def test_get_optimizers_in_backward(self):
        # 创建一个简单的测试模型类
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 2)

        model = TestModel()

        # 在反向传播中应用优化器（SGD），学习率为0.01
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {"lr": 0.01})
        # 获取所有在反向传播中使用的优化器
        in_backward_optims = _get_in_backward_optimizers(model)
        # 断言模型的参数数量与反向优化器数量相同
        self.assertEqual(len(list(model.parameters())), len(in_backward_optims))
        # 检查结果集合与预期集合是否相同
        result = set(in_backward_optims)
        expected = {
            optim for p in model.parameters() for optim in p._in_backward_optimizers
        }
        self.assertEqual(result, expected)
```