# `.\pytorch\test\distributed\_tensor\debug\test_comm_mode_features.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from typing import Any, Dict

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.api import distribute_tensor, DTensor
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    MLPStacked,
    ModelArgs,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    Transformer,
    with_comms,
)

# Importing C10D functional operations
c10d_functional = torch.ops.c10d_functional

class TestCommModeFeatures(DTensorTestBase):
    # checks if parameter / sharding info is the same as ground truth
    def check_same_set_of_keys(self, dict1, dict2):
        """
        Used to ensure the comm_mode parameter/sharding dictionaries contain the same information produced by the
        ground truth
        """
        dict1_keys = []
        dict2_keys = []

        # Collect all keys from dict1
        for key in dict1:
            for nested_key in dict1[key]:
                dict1_keys.append((key, nested_key))

        # Collect all keys from dict2
        for key in dict2:
            for nested_key in dict2[key]:
                dict2_keys.append((key, nested_key))

        # Ensure both dictionaries have the same number of keys
        self.assertEqual(len(dict1_keys), len(dict2_keys))

        # Check each key-value pair to ensure they are identical
        for i in range(len(dict1_keys)):
            self.assertEqual(dict1_keys[i], dict2_keys[i])

    # generates the ground truth parameter and sharding info
    def ground_truth(self, model):
        """
        Used to generate the ground-truth parameter and sharding info for a given distributed model to
        verify comm_mode correctness
        """
        module_parameters_dict: Dict[str, Any] = {}
        module_sharding_dict: Dict[str, Any] = {}

        # Iterate through all named parameters of the model
        for name, parameters in model.named_parameters():
            # Extract module name and parameter name
            module_name = model.__class__.__name__ + "." + name.rsplit(".", 1)[0]
            parameter_name = name.rsplit(".", 1)[1]

            # Initialize module parameters dictionary if it doesn't exist
            if module_name not in module_parameters_dict:
                module_parameters_dict[module_name] = {}

            # Store parameter data in the module parameters dictionary
            module_parameters_dict[module_name][parameter_name] = parameters.data

            # If the parameter is an instance of DTensor, store its placement information
            if isinstance(parameters.data, DTensor):
                key_name = module_name + "." + parameter_name
                module_sharding_dict[key_name] = parameters.data.placements

        return module_parameters_dict, module_sharding_dict

    @with_comms
    def test_MLP_distributed_sharding_display(self):
        """
        tests parameters and sharding on a module level
        """
        # 创建设备网格对象，使用给定的设备类型和设备索引范围
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        # 定义输入数据大小
        inp_size = [8, 10]
        # 设置随机种子
        torch.manual_seed(0)
        # 创建随机输入张量，指定设备类型
        inp = torch.rand(*inp_size, device=self.device_type)
        # 创建 MLP 模型对象
        model = MLPModule(self.device_type)

        # 定义并行化计划，将不同部分的网络分别并行化处理
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        # 并行化模型
        model = parallelize_module(model, device_mesh, parallelize_plan)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 进入通信调试模式
        with comm_mode:
            # 对模型进行前向传播
            output_tp = model(inp)
            # 对输出结果进行求和并进行反向传播
            output_tp.sum().backward()

        # 获取模型的参数字典和分片信息字典，作为参考对比
        module_parameters_dict, module_sharding_dict = self.ground_truth(model)

        # 检查参数信息是否与通信调试模式中的一致
        self.check_same_set_of_keys(
            module_parameters_dict, comm_mode.get_parameter_info()
        )
        # 检查分片信息是否与通信调试模式中的一致
        self.check_same_set_of_keys(module_sharding_dict, comm_mode.get_sharding_info())

    @with_comms
    def test_MLPStacked_distributed_sharding_display(self):
        """
        tests model with nested modules and makes sure comm_mode correctly resets parameter and sharding information
        """
        # 创建设备网格对象，使用给定的设备类型和设备索引范围
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )

        # 定义输入数据大小
        inp_size = [8, 10]
        # 设置随机种子
        torch.manual_seed(0)
        # 创建随机输入张量，指定设备类型
        inp = torch.rand(*inp_size, device=self.device_type)
        # 创建 MLP 模型对象
        model = MLPModule(self.device_type)

        # 定义并行化计划，将不同部分的网络分别并行化处理
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        # 并行化模型
        model = parallelize_module(model, device_mesh, parallelize_plan)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 进入通信调试模式
        with comm_mode:
            # 对模型进行前向传播
            output_tp = model(inp)
            # 对输出结果进行求和并进行反向传播
            output_tp.sum().backward()

        # 创建 MLPStacked 模型对象
        model2 = MLPStacked(self.device_type)

        # 定义并行化计划，将不同部分的网络分别并行化处理
        parallelize_plan = {
            "MLPStacked.layers.0.net1": ColwiseParallel(),
            "MLPStacked.layers.0.net2": RowwiseParallel(),
            "MLPStacked.layers.1.net1": ColwiseParallel(),
            "MLPStacked.layers.1.net2": RowwiseParallel(),
        }

        # 并行化模型
        model2 = parallelize_module(model2, device_mesh, parallelize_plan)

        # 进入通信调试模式
        with comm_mode:
            # 确保通信调试模式正确重置参数和分片信息
            self.assertEqual(comm_mode.get_parameter_info(), {})
            self.assertEqual(comm_mode.get_sharding_info(), {})

            # 对模型进行前向传播
            output_tp = model2(inp)

        # 获取模型的参数字典和分片信息字典，作为参考对比
        module_parameters_dict, module_sharding_dict = self.ground_truth(model2)

        # 检查参数信息是否与通信调试模式中的一致
        self.check_same_set_of_keys(
            module_parameters_dict, comm_mode.get_parameter_info()
        )
        # 检查分片信息是否与通信调试模式中的一致，并且确认分片信息数量为 8
        self.check_same_set_of_keys(module_sharding_dict, comm_mode.get_sharding_info())
        self.assertEqual(len(comm_mode.get_sharding_info()), 8)
    def test_MLP_module_tracing(self):
        """
        tests module-level tracing for MLP module
        """

        # 创建一个设备网格对象，指定设备类型和设备索引范围
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        
        # 定义输入数据的大小
        inp_size = [8, 10]
        # 设置随机种子为0，生成在指定设备上的随机输入数据
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        
        # 创建一个 MLPModule 模型对象，使用指定的设备类型
        model = MLPModule(self.device_type)

        # 定义并行化计划，将模型的不同部分分别并行化
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        # 并行化模型的各个部分，根据设备网格和并行化计划
        model = parallelize_module(model, device_mesh, parallelize_plan)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()

        # 进入通信调试模式上下文
        with comm_mode:
            # 在模型上执行输入数据的前向传播计算
            output_tp = model(inp)
            # 对输出结果进行求和并执行反向传播
            output_tp.sum().backward()

        # 检查所有子模块是否成功添加到模块深度字典中
        self.assertEqual(len(comm_mode.advanced_module_tracker.module_depth_dict), 5)

        # 检查所有集合操作是否在模块级别正确跟踪
        self.assertEqual(
            comm_mode.comm_module_counts["Global"][c10d_functional.all_reduce], 1
        )
        self.assertEqual(
            comm_mode.comm_module_counts["MLPModule"][c10d_functional.all_reduce], 1
        )
        self.assertEqual(
            comm_mode.comm_module_counts["MLPModule.net2"][c10d_functional.all_reduce],
            1,
        )
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```