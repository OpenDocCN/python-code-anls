# `.\pytorch\test\distributed\_tensor\test_optimizers.py`

```py
# Owner(s): ["oncall: distributed"]

from copy import deepcopy  # 导入深拷贝函数

import torch  # 导入PyTorch库

import torch.nn as nn  # 导入PyTorch神经网络模块

from torch.distributed._tensor import (  # 导入分布式张量相关模块
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests  # 导入测试相关的通用函数

from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量测试相关模块
    DTensorTestBase,
    MLPModule,
    with_comms,
)


# shard function to do full sharding on all parameters of a module
def shard_fn(name, module, device_mesh):
    if isinstance(module, nn.Linear):  # 如果模块是线性层
        for name, param in module.named_parameters():  # 遍历模块的所有参数
            dist_param = torch.nn.Parameter(  # 创建分布式参数
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            # make sure partial sum get cleared after backward()
            dist_param.register_hook(
                lambda grad: grad.redistribute(placements=[Shard(0)])
            )
            module.register_parameter(name, dist_param)  # 注册分布式参数到模块中


# prepare input
def input_fn(mod, inputs, device_mesh):
    # split the input tensor to be sharded input
    dist_inp = distribute_tensor(inputs[0], device_mesh, [Shard(0)])  # 将输入张量进行分布式切分
    return dist_inp


# prepare output to be local torch.Tensor
def output_fn(mod, outputs, device_mesh):
    assert isinstance(outputs, DTensor)  # 断言输出是分布式张量
    return outputs.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()  # 将输出重分布为本地张量


class TestDTensorOptimizer(DTensorTestBase):
    def _assert_optimizer(
        self,
        mesh,
        model,
        optim,
        dist_model,
        dist_optim,
        inputs,
        *,
        rtol: float = 1.3e-6,
        atol: float = 1e-5,
    ):
        for iter_idx in range(2):
            # run forward/backward/optim for original model
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))  # 清空原模型优化器的梯度
            out = model(inputs)  # 原模型前向传播
            loss = out.sum()  # 计算原模型输出的总和作为损失
            loss.backward()  # 原模型反向传播
            optim.step()  # 原模型优化器执行优化步骤

            # run forward/backward/optim for distributed model
            dist_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))  # 清空分布式模型优化器的梯度
            dist_out = dist_model(inputs)  # 分布式模型前向传播
            dist_loss = dist_out.sum()  # 计算分布式模型输出的总和作为损失
            dist_loss.backward()  # 分布式模型反向传播
            dist_optim.step()  # 分布式模型优化器执行优化步骤

            # check that the optimizer update parameters with same numerics
            for p1, p2 in zip(model.parameters(), dist_model.parameters()):  # 检查模型参数是否相同
                p2 = p2.full_tensor()  # 获取分布式模型参数的完整张量
                # Default 'rtol' and 'atol' for attr:`~torch.float32` are ``1.3e-6`` and ``1e-5``
                self.assertEqual(p1, p2, atol=atol, rtol=rtol)  # 断言两个参数张量相等，使用指定的误差容限

    def test_optimizer_foreach_supported_types_include_DTensor(self):
        from torch.optim.optimizer import _foreach_supported_types

        self.assertTrue(DTensor in _foreach_supported_types)  # 断言分布式张量类型在支持的类型中

    @with_comms
    # 定义一个测试方法，用于测试1维Adam优化器在分片环境中的行为
    def test_adam_1d_sharding(self):
        # 创建设备网格对象，指定设备类型和全局设备编号列表
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 对于 capturable=False 和 foreach=True 的情况，不支持以 Tensor 形式传入的学习率 lr
        adam_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]
        # 包含融合优化的 Adam 配置列表，设置了 "fused": True
        fused_adam_float_lr_configs = [
            {"lr": 0.1, "fused": True},
            {"lr": 0.1, "weight_decay": 0.05, "amsgrad": True, "fused": True},
            {
                "lr": 0.1,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]
        # 当 fused=True 时，Adam 优化器的学习率 lr 可以是 Tensor 或 float
        fused_adam_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adam_float_lr_configs
        ]
        fused_adam_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adam_float_lr_configs
            ]
        )
        # 汇总所有 Adam 优化器的配置
        adam_configs = [
            *adam_float_lr_configs,
            *fused_adam_float_lr_configs,
            *fused_adam_tensor_lr_configs,
        ]

        # 遍历所有 Adam 配置
        for config in adam_configs:
            # 创建一个 MLP 模型
            mod = MLPModule(self.device_type)
            # 初始化 Adam 优化器，应用于模型的参数，使用给定的配置
            opt = torch.optim.Adam(mod.parameters(), **config)

            # 在分布式环境中，将模型分发到设备网格上，并应用分布式优化
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            dist_opt = torch.optim.Adam(dist_mod.parameters(), **config)

            # 创建输入张量，确保在不同的设备上输入一致
            inp = torch.ones(8, 10, device=self.device_type)
            # 调用辅助方法验证优化器的行为和输出
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    # 带有通信装饰器的方法将在分布式环境中运行
    @with_comms
    # 定义测试方法，用于测试 AdamW 优化器在 1D 分片情况下的表现
    def test_adamw_1d_sharding(self):
        # 创建设备网格，使用指定设备类型和全局大小的范围列表
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 当 capturable=False 且 foreach=True 时，Tensor 类型的 lr 不被支持
        adamw_float_lr_configs = [
            {"lr": 0.1, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},
            {"lr": 0.1, "weight_decay": 0.05},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
            },
        ]

        fused_adamw_float_lr_configs = [
            {"lr": 0.1, "weight_decay": 0.05, "fused": True},
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "amsgrad": True,
                "fused": True,
            },
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
                "amsgrad": True,
                "fused": True,
            },
        ]

        # 当 fused=True 时，lr 可能是 Tensor 类型或 float 类型，用于 adamW 优化器
        fused_adamw_tensor_lr_configs = [
            {**config, "lr": torch.tensor(0.1)}
            for config in fused_adamw_float_lr_configs
        ]
        fused_adamw_tensor_lr_configs.extend(
            [
                {**config, "lr": torch.tensor([0.1])}
                for config in fused_adamw_float_lr_configs
            ]
        )

        # 将所有的 adamW 配置组合在一起
        adamw_configs = [
            *adamw_float_lr_configs,
            *fused_adamw_float_lr_configs,
            *fused_adamw_tensor_lr_configs,
        ]

        # 遍历所有 adamW 配置
        for config in adamw_configs:
            # 创建 MLP 模块实例
            mod = MLPModule(self.device_type)
            # 使用当前配置创建 AdamW 优化器
            opt = torch.optim.AdamW(mod.parameters(), **config)

            # 分布式模块化，复制模块，网格，分片函数，输入输出函数进行分发
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 使用当前配置创建分布式 AdamW 优化器
            dist_opt = torch.optim.AdamW(dist_mod.parameters(), **config)

            # 使用 ones 确保单机模型在不同 rank 上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 断言优化器的行为，检查分布式和非分布式模块的一致性
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    # 使用通信装饰器，将当前测试方法标记为带有通信行为的测试用例
    @with_comms
    # 定义一个测试方法，用于测试单机模型在分片环境中的表现
    def test_sgd_1d_sharding(self):
        # 创建设备网格对象，包含指定数量的设备
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义多个 SGD 配置字典，每个字典包含不同的优化器参数
        sgd_configs = [
            {"lr": 0.1, "foreach": False},  # 第一个配置，设置学习率和 foreach 参数
            {"lr": 0.1, "momentum": 0.05, "foreach": False},  # 第二个配置，加入动量参数
            {"lr": 0.1, "momentum": 0.05},  # 第三个配置，仅学习率和动量参数
            {"lr": 0.1, "momentum": 0.06, "dampening": 0.07},  # 第四个配置，包括阻尼参数
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
                "foreach": False,
            },  # 第五个配置，包含权重衰减、Nesterov 和 maximize 参数
            {
                "lr": 0.1,
                "momentum": 0.08,
                "weight_decay": 0.05,
                "nesterov": True,
                "maximize": True,
            },  # 第六个配置，同样的参数但 foreach 参数为默认值
        ]

        # 遍历每个 SGD 配置
        for config in sgd_configs:
            # 创建一个 MLP 模型对象
            mod = MLPModule(self.device_type)
            # 使用当前配置创建一个 SGD 优化器
            opt = torch.optim.SGD(mod.parameters(), **config)

            # 在分布式环境中复制模型并分发到设备网格上，使用自定义的分片和输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 在分布式环境中使用当前配置创建一个 SGD 优化器
            dist_opt = torch.optim.SGD(dist_mod.parameters(), **config)

            # 创建输入数据，使用全 1 的张量确保单机模型在不同设备上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 调用辅助方法，断言单机和分布式优化器的行为和输出一致
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    # 定义测试函数 test_adagrad_1d_sharding，用于测试 Adagrad 优化器在分布式环境下的行为
    def test_adagrad_1d_sharding(self):
        # 创建设备网格对象，指定设备类型和世界大小范围
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义多组 Adagrad 配置参数列表
        adagrad_configs = [
            {"lr": 0.1, "foreach": False},  # 基本配置，学习率为 0.1，不针对每个参数分别更新
            {"lr": 0.1, "lr_decay": 0.05, "foreach": False},  # 添加学习率衰减率配置
            {"lr": 0.1, "lr_decay": 0.02, "weight_decay": 0.05, "foreach": False},  # 添加权重衰减配置
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "foreach": False,
            },  # 添加初始累加器值配置
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "foreach": False,
            },  # 添加数值稳定性常数配置
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
                "foreach": False,
            },  # 添加优化最大化配置
            {
                "lr": 0.1,
                "lr_decay": 0.02,
                "weight_decay": 0.05,
                "initial_accumulator_value": 0.03,
                "eps": 1e-6,
                "maximize": True,
            },  # 添加优化最大化配置，但未指定 foreach 参数，默认为 False
        ]

        # 遍历每个配置项
        for config in adagrad_configs:
            # 创建 MLP 模型对象
            mod = MLPModule(self.device_type)
            # 使用当前配置创建 Adagrad 优化器
            opt = torch.optim.Adagrad(mod.parameters(), **config)

            # 深度复制模型，分发到设备网格上，并使用相应的分片函数和输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 在分布式模型上使用相同的配置创建 Adagrad 优化器
            dist_opt = torch.optim.Adagrad(dist_mod.parameters(), **config)

            # 创建输入张量，全为 1，确保单机模型在不同排名上具有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 断言优化器行为的一致性
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    # 定义测试函数，用于测试分布式环境下的 RMSprop 优化器配置
    def test_RMSprop_1d_sharding(self):
        # 创建设备网格，指定设备类型和世界大小
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 不同的 RMSprop 配置列表
        RMSprop_configs = [
            {"lr": 0.1, "foreach": False},  # RMSprop 参数配置示例1
            {"lr": 0.1, "alpha": 0.85, "foreach": False},  # RMSprop 参数配置示例2
            {"lr": 0.1, "alpha": 0.88, "eps": 1e-6, "foreach": False},  # RMSprop 参数配置示例3
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },  # RMSprop 参数配置示例4
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "foreach": False,
            },  # RMSprop 参数配置示例5
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "foreach": False,
            },  # RMSprop 参数配置示例6
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
                "foreach": False,
            },  # RMSprop 参数配置示例7
            {
                "lr": 0.1,
                "alpha": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "centered": True,
                "maximize": True,
            },  # RMSprop 参数配置示例8
        ]

        # 遍历 RMSprop 配置列表
        for config in RMSprop_configs:
            # 创建 MLPModule 实例
            mod = MLPModule(self.device_type)
            # 使用当前配置创建 RMSprop 优化器
            opt = torch.optim.RMSprop(mod.parameters(), **config)

            # 在分布式环境下，复制模型并分发到网格中，使用指定的函数进行分片和输入输出处理
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 使用当前配置创建分布式 RMSprop 优化器
            dist_opt = torch.optim.RMSprop(dist_mod.parameters(), **config)

            # 创建输入数据，确保单机模型在不同 rank 上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 调用辅助函数来断言优化器的行为
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_adadelta_1d_sharding(self):
        # 创建设备网格对象，用于设备分布
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义多个 Adadelta 优化器的配置
        adadelta_configs = [
            {"lr": 0.1, "foreach": False},  # 基本配置
            {"lr": 0.1, "rho": 0.85, "foreach": False},  # 包含 rho 参数的配置
            {"lr": 0.1, "rho": 0.88, "eps": 1e-5, "foreach": False},  # 包含 rho 和 eps 参数的配置
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },  # 包含 rho, eps, weight_decay 参数的配置
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
            },  # 包含 rho, eps, weight_decay 参数的配置，没有 foreach 参数
            {
                "lr": 0.1,
                "rho": 0.88,
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
            },  # 包含所有参数的配置，且设置 maximize 为 True
        ]

        # 遍历不同的 Adadelta 配置
        for config in adadelta_configs:
            # 创建 MLP 模块
            mod = MLPModule(self.device_type)
            # 根据配置创建 Adadelta 优化器
            opt = torch.optim.Adadelta(mod.parameters(), **config)

            # 分布模块，使用深拷贝的模块，设备网格，分片函数，输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 根据配置创建分布式 Adadelta 优化器
            dist_opt = torch.optim.Adadelta(dist_mod.parameters(), **config)

            # 使用全为 1 的张量作为输入，确保单机模型在不同的排名上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 执行优化器测试断言
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    @with_comms
    def test_nadam_1d_sharding(self):
        # 创建设备网格对象，用于设备分布
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义多个 Nadam 优化器的配置
        nadam_configs = [
            {"lr": 0.1, "foreach": False},  # 基本配置
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},  # 包含 weight_decay 参数的配置
            {"lr": 0.1, "weight_decay": 0.05},  # 只包含 weight_decay 参数的配置
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },  # 包含 betas, eps, weight_decay 参数的配置
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
            },  # 包含所有参数的配置，且设置 decoupled_weight_decay 为 True
        ]

        # 遍历不同的 Nadam 配置
        for config in nadam_configs:
            # 创建 MLP 模块
            mod = MLPModule(self.device_type)
            # 根据配置创建 Nadam 优化器
            opt = torch.optim.NAdam(mod.parameters(), **config)

            # 分布模块，使用深拷贝的模块，设备网格，分片函数，输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 根据配置创建分布式 Nadam 优化器
            dist_opt = torch.optim.NAdam(dist_mod.parameters(), **config)

            # 使用全为 1 的张量作为输入，确保单机模型在不同的排名上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 执行优化器测试断言
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)
    # 定义测试函数 test_radam_1d_sharding，用于测试 1D 分片下的 RAdam 优化器行为
    def test_radam_1d_sharding(self):
        # 创建设备网格对象，使用给定设备类型和全局大小列表
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义多个 RAdam 优化器的配置参数列表
        radam_configs = [
            {"lr": 0.1, "foreach": False},  # 基本配置，学习率为 0.1，不使用 foreach 参数
            {"lr": 0.1, "weight_decay": 0.05, "foreach": False},  # 包含权重衰减参数
            {
                "lr": 0.1,
                "weight_decay": 0.05,
            },  # 只设置学习率和权重衰减参数，没有 foreach 参数
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },  # 包含动量参数 betas 和 epsilon 参数
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "decoupled_weight_decay": True,
            },  # 同上，并启用解耦权重衰减
        ]

        # 遍历每个配置，进行测试
        for config in radam_configs:
            # 创建 MLP 模型对象
            mod = MLPModule(self.device_type)
            # 使用当前配置创建 RAdam 优化器
            opt = torch.optim.RAdam(mod.parameters(), **config)

            # 分布式模型分片，使用深拷贝的模型，设备网格，分片函数和输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 使用当前配置创建分布式 RAdam 优化器
            dist_opt = torch.optim.RAdam(dist_mod.parameters(), **config)

            # 准备输入数据，全为 1 的张量，确保单机模型在不同 rank 上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 调用辅助函数，断言优化器的行为符合预期
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)

    # 使用装饰器声明的测试函数，用于测试 1D 分片下的 Adamax 优化器行为
    @with_comms
    def test_adamax_1d_sharding(self):
        # 创建设备网格对象，使用给定设备类型和全局大小列表
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义多个 Adamax 优化器的配置参数列表
        adamax_configs = [
            {"lr": 0.1, "foreach": False},  # 基本配置，学习率为 0.1，不使用 foreach 参数
            {"lr": 0.1, "betas": (0.6, 0.66), "foreach": False},  # 包含动量参数 betas
            {"lr": 0.1, "betas": (0.6, 0.66), "eps": 1e-6, "foreach": False},  # 包含 betas 和 epsilon 参数
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "foreach": False,
            },  # 包含 betas、epsilon 和权重衰减参数
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
            },  # 同上，但不使用 foreach 参数
            {
                "lr": 0.1,
                "betas": (0.6, 0.66),
                "eps": 1e-6,
                "weight_decay": 0.05,
                "maximize": True,
            },  # 包含最大化参数 maximize
        ]

        # 遍历每个配置，进行测试
        for config in adamax_configs:
            # 创建 MLP 模型对象
            mod = MLPModule(self.device_type)
            # 使用当前配置创建 Adamax 优化器
            opt = torch.optim.Adamax(mod.parameters(), **config)

            # 分布式模型分片，使用深拷贝的模型，设备网格，分片函数和输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 使用当前配置创建分布式 Adamax 优化器
            dist_opt = torch.optim.Adamax(dist_mod.parameters(), **config)

            # 准备输入数据，全为 1 的张量，确保单机模型在不同 rank 上有相同的输入
            inp = torch.ones(8, 10, device=self.device_type)
            # 调用辅助函数，断言优化器的行为符合预期
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)
    # 定义一个名为 test_asgd_1d_sharding 的测试方法
    def test_asgd_1d_sharding(self):
        # 创建一个设备网格对象，使用给定的设备类型和世界大小范围
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # 定义一组 ASGD 优化器的配置参数列表
        asgd_configs = [
            {"lr": 0.1, "foreach": False},  # ASGD 参数配置示例 1
            {"lr": 0.1, "lambd": 0.001, "foreach": False},  # ASGD 参数配置示例 2
            {"lr": 0.1, "lambd": 0.001, "alpha": 0.85, "foreach": False},  # ASGD 参数配置示例 3
            {"lr": 0.1, "lambd": 0.001, "alpha": 0.85, "t0": 1e5, "foreach": False},  # ASGD 参数配置示例 4
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": False,
            },  # ASGD 参数配置示例 5
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": True,
            },  # ASGD 参数配置示例 6
            {
                "lr": 0.1,
                "lambd": 0.001,
                "alpha": 0.85,
                "t0": 1e5,
                "weight_decay": 0.05,
                "foreach": True,
                "maximize": True,
            },  # ASGD 参数配置示例 7
        ]

        # 遍历 ASGD 参数配置列表
        for config in asgd_configs:
            # 创建一个 MLPModule 模型对象
            mod = MLPModule(self.device_type)
            # 使用当前配置创建一个 ASGD 优化器对象
            opt = torch.optim.ASGD(mod.parameters(), **config)

            # 将模型对象深度复制并分发到设备网格上，使用给定的分片函数和输入输出函数
            dist_mod = distribute_module(
                deepcopy(mod), mesh, shard_fn, input_fn, output_fn
            )
            # 使用当前配置创建一个分布式 ASGD 优化器对象
            dist_opt = torch.optim.ASGD(dist_mod.parameters(), **config)

            # 创建一个输入张量，全部为 1，使用给定的设备类型
            inp = torch.ones(8, 10, device=self.device_type)

            # 断言优化器的行为，确保在不同的机器模型上具有相同的输入
            # TODO: 暂时保留 ASGD 优化器的单元测试，但需要进一步研究为什么在比较模型参数时需要更高的 atol 和 rtol
            # 默认的 'rtol' 和 'atol' 对于 torch.float32 分别是 1.3e-6 和 1e-5
            # 参考：https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L65
            self._assert_optimizer(
                mesh, mod, opt, dist_mod, dist_opt, inp, atol=1.3e-5, rtol=1e-4
            )
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```