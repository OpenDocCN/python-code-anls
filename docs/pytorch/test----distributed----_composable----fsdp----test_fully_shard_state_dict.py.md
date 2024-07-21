# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_state_dict.py`

```py
# Owner(s): ["oncall: distributed"]  # 代码所有者，负责分布式部分的人员

import copy  # 导入复制模块，用于对象的深复制操作
import functools  # 导入函数工具模块，支持高阶函数的操作
import unittest  # 导入单元测试模块，用于编写和运行测试

from typing import Dict  # 导入类型提示模块，用于类型注解

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard  # 导入FSDP相关模块
from torch.distributed._tensor import distribute_tensor, DTensor  # 导入张量分布和DTensor相关模块
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh  # 导入设备网格和初始化函数
from torch.distributed.tensor.parallel import (  # 导入张量并行化相关模块
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入CUDA相关测试工具
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试的GPU数量检查
from torch.testing._internal.common_fsdp import FSDPTest, FSDPTestMultiThread, MLP  # 导入FSDP测试和MLP模型
from torch.testing._internal.common_utils import run_tests  # 导入测试运行工具
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量通用DTensor相关模块
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFullyShardStateDictMultiProcess(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())  # 返回最小的4和当前CUDA设备数量的值

    @skip_if_lt_x_gpu(2)  # 如果GPU数量小于2，则跳过此测试
    def test_1d_state_dict_save_load(self):
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5]},  # 运行子测试，使用不同的MLP维度参数
            self._test_1d_state_dict_save_load,
        )

    def _test_1d_state_dict_save_load(self, mlp_dim: int):
        torch.manual_seed(42)  # 设置随机种子为42
        base_model = nn.Sequential(
            MLP(mlp_dim),  # 构建包含MLP层的序列模型
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),  # 添加MLP和线性层的嵌套序列
            MLP(mlp_dim),  # 添加另一个MLP层
        )
        # 检查基本的`reshard_after_forward=True`
        model1 = copy.deepcopy(base_model)  # 深复制基础模型
        for module in model1:
            fully_shard(module)  # 对每个模块进行完全分片
        fully_shard(model1)  # 对整个模型进行完全分片
        self._test_state_dict_save_load(model1)  # 执行状态字典保存和加载测试

        # 检查在前向传播之前和之后`reshard_after_forward=False`
        model2 = copy.deepcopy(base_model)  # 深复制基础模型
        for module in model2:
            fully_shard(module, reshard_after_forward=False)  # 对每个模块进行分片，但前向传播后不进行重分片
        fully_shard(model2, reshard_after_forward=False)  # 对整个模型进行分片，但前向传播后不进行重分片
        self._test_state_dict_save_load(model2)  # 执行状态字典保存和加载测试
        ref_sharded_sd = model2.state_dict()  # 获取参考分片后的状态字典
        inp = torch.randn((2, mlp_dim), device="cuda")  # 在CUDA设备上生成随机输入数据
        model2(inp)  # 执行前向传播，此后参数不会被重新分片
        # 检查状态字典钩子是否重新分片
        sharded_sd = model2.state_dict()  # 获取当前模型的分片状态字典
        self.assertEqual(set(ref_sharded_sd.keys()), set(sharded_sd.keys()))  # 检查键集合是否相等
        for key, value in ref_sharded_sd.items():
            self.assertEqual(value, sharded_sd[key])  # 检查每个键值对是否相等

    @skip_if_lt_x_gpu(2)  # 如果GPU数量小于2，则跳过此测试
    # 定义测试函数，用于测试一维状态字典在 CPU 离载策略下的行为
    def test_1d_state_dict_cpu_offload(self):
        # 设置 MLP 维度为 4
        mlp_dim = 4
        # 创建 CPU 离载策略对象，设置 pin_memory=True
        offload_policy = CPUOffloadPolicy(pin_memory=True)
        # 设定随机种子为 42
        torch.manual_seed(42)
        # 在 "meta" 设备上创建神经网络模型
        with torch.device("meta"):
            # 构建包含两个线性层的序列模型，无偏置
            model = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim, bias=False),
                nn.Linear(mlp_dim, mlp_dim, bias=False),
            )
        # 对模型中的每个模块应用完全分片函数，使用给定的离载策略
        for module in model:
            fully_shard(module, offload_policy=offload_policy)
        # 对整个模型应用完全分片函数，使用给定的离载策略
        fully_shard(model, offload_policy=offload_policy)

        # 将完整的状态字典拆分为多个部分
        # 用于测试在 `strict=False` 的情况下加载
        state_dicts = []
        for name, dtensor in model.named_parameters():
            # 创建随机张量，与参数张量相同的大小
            full_tensor = torch.randn(dtensor.size())
            # 将完整张量分发到指定的设备网格和位置
            sharded_tensor = distribute_tensor(
                full_tensor, dtensor.device_mesh, dtensor.placements
            )
            state_dicts.append({name: sharded_tensor})

        # 检查是否可以在仍有部分参数在 "meta" 设备上的情况下加载模型
        for sd in state_dicts:
            model.load_state_dict(sd, assign=True, strict=False)

        # 在没有错误的情况下进行懒初始化
        inp = torch.rand((mlp_dim, mlp_dim), device="cuda")
        model(inp)

        # 获取当前模型的状态字典
        state_dict = model.state_dict()
        # 验证所有参数张量均在 "cpu" 设备上
        for name, dtensor in state_dict.items():
            self.assertEqual(dtensor.device.type, "cpu")

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_2d_state_dict_save_load(self):
        # 设定分布式并行尺寸为 2
        dp_size = 2
        # 初始化全局设备网格，使用 "cuda" 设备和给定的网格维度名称
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        # 运行子测试，参数包括 MLP 维度和特定的测试函数
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5]},
            functools.partial(self._test_2d_state_dict_save_load, global_mesh),
        )

    # 实际执行二维状态字典保存与加载的测试函数
    def _test_2d_state_dict_save_load(self, global_mesh: DeviceMesh, mlp_dim: int):
        # 获取全局设备网格中的 dp 和 tp 网格
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        # 设定随机种子为 42
        torch.manual_seed(42)
        # 创建包含三个 MLP 模块的序列模型
        model = nn.Sequential(*[MLP(mlp_dim) for _ in range(3)])
        # 将模型并行化，使用给定的设备网格和并行化计划
        model = parallelize_module(
            model,
            device_mesh=tp_mesh,
            parallelize_plan={
                "0.in_proj": ColwiseParallel(),
                "0.out_proj": RowwiseParallel(),
                "1.in_proj": ColwiseParallel(),
                "1.out_proj": RowwiseParallel(),
                "2.in_proj": ColwiseParallel(),
                "2.out_proj": RowwiseParallel(),
            },
        )
        # 对模型中的每个 MLP 模块应用完全分片函数，使用 dp 网格
        for mlp in model:
            fully_shard(mlp, mesh=dp_mesh)
        # 对整个模型应用完全分片函数，使用 dp 网格
        fully_shard(model, mesh=dp_mesh)
        # 执行状态字典保存与加载的测试函数
        self._test_state_dict_save_load(model)
# 定义一个测试类，继承自FSDPTestMultiThread，用于测试分布式状态字典的多线程功能
class TestFullyShardStateDictMultiThread(FSDPTestMultiThread):

    # 定义一个属性，返回当前测试的分布式环境中的进程数
    @property
    def world_size(self):
        return 2

    # 如果没有 CUDA 支持，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_rank0_offload_full_state_dict(self):
        # 在所有进程上构建一个未分片的参考模型
        model_args = ModelArgs(dropout_p=0.0)
        torch.manual_seed(42)
        # 使用 CUDA 构建 Transformer 模型作为参考模型
        ref_model = Transformer(model_args).cuda()
        # 将参考模型的参数广播到所有进程
        for param in ref_model.parameters():
            torch.distributed.broadcast(param.detach(), src=0)

        # 在所有进程上构建一个分片模型及其分片状态字典
        model = copy.deepcopy(ref_model)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)
        sharded_sd = model.state_dict()

        # 在进程不是0的情况下，删除参考模型
        if self.rank != 0:
            del ref_model
        else:
            # 在进程0上，保存参考模型的 GPU 全状态字典，并删除参考模型的 GPU 状态
            ref_gpu_full_sd = ref_model.state_dict()
            ref_full_sd = {k: v.cpu() for k, v in ref_gpu_full_sd.items()}
            del ref_gpu_full_sd

        # 将 GPU 分片状态字典转换为 CPU 全状态字典，仅在进程0上进行操作
        full_sd = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if self.rank == 0:
                full_sd[param_name] = full_param.cpu()
            else:
                del full_param

        # 检查是否只在进程0上存在 CPU 全状态字典
        if self.rank == 0:
            # 断言 CPU 全状态字典的长度与参考全状态字典长度一致
            self.assertEqual(len(full_sd), len(ref_full_sd))
            # 断言 CPU 全状态字典的键列表与参考全状态字典的键列表一致
            self.assertEqual(list(full_sd.keys()), list(ref_full_sd.keys()))
            # 逐一比较 CPU 全状态字典的每个参数与参考全状态字典的对应参数
            for (param_name, param), ref_param in zip(
                full_sd.items(), ref_full_sd.values()
            ):
                # 断言当前参数在 CPU 上
                self.assertEqual(param.device, torch.device("cpu"))
                # 断言当前参数与参考参数在相同设备上
                self.assertEqual(param.device, ref_param.device)
                # 断言当前参数与参考参数的数值相等
                self.assertEqual(param, ref_param)
        else:
            # 对于非进程0，断言 CPU 全状态字典为空
            self.assertEqual(len(full_sd), 0)


if __name__ == "__main__":
    run_tests()
```