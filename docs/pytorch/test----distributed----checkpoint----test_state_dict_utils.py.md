# `.\pytorch\test\distributed\checkpoint\test_state_dict_utils.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入必要的库和模块
import copy  # 导入深拷贝模块
import io  # 导入用于处理输入输出的模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
import torch.distributed._functional_collectives as funcol  # 导入PyTorch分布式函数集合

from torch.distributed._state_dict_utils import (  # 导入PyTorch分布式状态字典相关工具
    _check_state_dict_similarity,  # 检查状态字典相似性的函数
    _copy_state_dict,  # 复制状态字典的函数
    _create_cpu_state_dict,  # 创建CPU状态字典的函数
    _gather_state_dict,  # 收集状态字典的函数
    _offload_state_dict_to_cpu,  # 将状态字典转移到CPU的函数
)
from torch.distributed._tensor import DTensor  # 导入PyTorch分布式张量
from torch.distributed._tensor.placement_types import Shard  # 导入Shard分片类型
from torch.testing._internal.common_utils import run_tests  # 导入用于运行测试的工具函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入常用的分布式张量测试基类
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


class TestStateDictUtils(DTensorTestBase):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_gather_state_dict_dtensor(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]  # 设定分片规格
        torch.random.manual_seed(dist.get_rank())  # 使用当前进程的分布式进程等级设定随机数种子
        local_tensor = torch.randn(3, 3, 3)  # 生成本地随机张量
        # 从本地张量创建分布式张量
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        state_dict = {"dtensor": dist_tensor}  # 创建状态字典，包含分布式张量

        # 收集分布式状态字典
        gathered_state_dict = _gather_state_dict(state_dict)
        # 预期的收集到的分布式张量
        expected_gathered_dtensor = funcol.all_gather_tensor(
            dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )
        # 断言收集到的分布式张量与预期的一致
        self.assertEqual(expected_gathered_dtensor, gathered_state_dict["dtensor"])
        self.assertTrue(gathered_state_dict["dtensor"].is_cuda)  # 断言收集到的分布式张量在GPU上

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_with_cpu_and_ranks_only(self):
        # 构建设备网格
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]  # 设定分片规格
        torch.random.manual_seed(dist.get_rank())  # 使用当前进程的分布式进程等级设定随机数种子
        local_tensor = torch.randn(3, 3, 3)  # 生成本地随机张量
        # 从本地张量创建分布式张量
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        state_dict = {"dtensor": dist_tensor}  # 创建状态字典，包含分布式张量

        # 使用CPU卸载并仅收集指定进程的状态字典
        gathered_state_dict = _gather_state_dict(
            state_dict, cpu_offload=True, ranks_only=(0, 2)
        )
        # 预期的收集到的分布式张量
        expected_gathered_dtensor = funcol.all_gather_tensor(
            dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )
        # 如果当前进程在(0, 2)范围内
        if dist.get_rank() in (0, 2):
            # 断言收集到的分布式张量与预期的一致
            self.assertEqual(expected_gathered_dtensor, gathered_state_dict["dtensor"])
            self.assertFalse(gathered_state_dict["dtensor"].is_cuda)  # 断言收集到的分布式张量不在GPU上
        else:
            self.assertEqual(gathered_state_dict, {})  # 如果不在(0, 2)范围内，收集到的状态字典为空字典

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_cpu_and_ranks_only(self):
        # 使用 CUDA 设备
        device = torch.device("cuda")
        # 创建包含两个张量的状态字典，存储在 CUDA 设备上
        state_dict = {
            "tensor1": torch.arange(10, device=device),
            "tensor2": torch.ones(10, device=device),
        }

        # 将状态字典中的部分张量转移到 CPU，仅包括排名为 0 和 2 的设备
        cpu_state_dict = _offload_state_dict_to_cpu(state_dict, ranks_only=(0, 2))
        # 如果当前进程的分布式排名在 (0, 2) 中
        if dist.get_rank() in (0, 2):
            # 检查所有 CPU 张量
            for v in cpu_state_dict.values():
                self.assertFalse(v.is_cuda)
            # 检查特定张量是否与预期的 CPU 张量相等
            self.assertEqual(cpu_state_dict["tensor1"], torch.arange(10))
            self.assertEqual(cpu_state_dict["tensor2"], torch.ones(10))
        else:
            # 如果不在 (0, 2) 中，期望得到空的状态字典
            self.assertEqual(cpu_state_dict, {})

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_complicated_dict(self):
        # 创建分布式张量的函数
        def create_dtensor():
            # 构建设备网格
            device_mesh = self.build_device_mesh()
            # 使用单个分片规范
            shard_spec = [Shard(0)]
            # 设置随机种子
            torch.random.manual_seed(dist.get_rank())
            # 本地张量（CUDA 设备上）
            local_tensor = torch.randn(3, 3, 3)
            # 转换为分布式张量
            dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
            # 使用全局组进行所有收集操作
            tensor = funcol.all_gather_tensor(
                dist_tensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )
            return tensor, dist_tensor

        # 初始化空列表
        ltensor, ldtensor = [], []
        # 循环创建张量并添加到列表中
        for i in range(10):
            tensor, dtensor = create_dtensor()
            ltensor.append(tensor)
            ltensor.append(torch.ones(10, device=torch.device("cuda")))
            ldtensor.append(dtensor)
            ldtensor.append(torch.ones(10, device=torch.device("cuda")))

        # 最终创建张量并添加到列表中
        tensor, dtensor = create_dtensor()
        # 分布式状态字典，包含本地张量、列表和 arange 张量
        dist_state_dict = {
            "local": dtensor,
            "list": ldtensor,
            "arange": torch.arange(10, device=torch.device("cuda")),
        }
        # 本地状态字典，与分布式状态字典汇总结果进行比较
        state_dict = {
            "local": tensor,
            "list": ltensor,
            "arange": torch.arange(10, device=torch.device("cuda")),
        }
        # 断言本地状态字典与汇总状态字典是否相等
        self.assertEqual(state_dict, _gather_state_dict(dist_state_dict))

    @skip_if_lt_x_gpu(2)
    def test_create_cpu_state_dict(self):
        device = torch.device("cuda")
        buffer = io.BytesIO()
        torch.save(torch.ones(10), buffer)
        buffer.seek(0)
        state_dict = {
            "tensor1": torch.arange(10, device=device),
            "tensor2": torch.ones(10, device=device),
            "non_tensor_bytes_io": copy.deepcopy(buffer),
            "non_tensor_bytes": buffer.read(),
            "step": torch.tensor(7, dtype=torch.float),
            "lr": 1.5,
            "nested": {"list": [1, 2, 3, 4]},
        }

        def _verify(cpu_state_dict):
            # 验证 _check_state_dict_similarity() 的正确性
            self.assertTrue(_check_state_dict_similarity(state_dict, cpu_state_dict))
            # 备份 tensor1，然后修改它并验证是否仍然相似
            tensor1 = cpu_state_dict["tensor1"]
            cpu_state_dict["tensor1"] = torch.arange(11)
            self.assertFalse(_check_state_dict_similarity(state_dict, cpu_state_dict))
            cpu_state_dict["tensor1"] = tensor1

            # 使用 _copy_state_dict 复制 state_dict 到 cpu_state_dict
            _copy_state_dict(state_dict, cpu_state_dict)

            # 验证 _copy_state_dict 的工作情况
            for v in cpu_state_dict.values():
                if isinstance(v, torch.Tensor):
                    self.assertFalse(v.is_cuda)
            self.assertEqual(cpu_state_dict["tensor1"], torch.arange(10))
            self.assertEqual(cpu_state_dict["tensor2"], torch.ones(10))
            buffer.seek(0)
            cpu_state_dict["non_tensor_bytes_io"].seek(0)
            # 验证非张量数据（BytesIO 对象）是否正确复制
            self.assertEqual(
                cpu_state_dict["non_tensor_bytes_io"].read(), buffer.read()
            )
            buffer.seek(0)
            # 验证非张量数据（BytesIO 对象中的字节流）是否正确复制
            self.assertEqual(cpu_state_dict["non_tensor_bytes"], buffer.read())
            self.assertEqual(cpu_state_dict["lr"], 1.5)
            self.assertEqual(cpu_state_dict["step"], 7)
            self.assertEqual(cpu_state_dict["nested"], {"list": [1, 2, 3, 4]})

        # 创建 CPU 上的状态字典，并进行验证
        cpu_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True)
        _verify(cpu_state_dict)
        cpu_state_dict = _create_cpu_state_dict(state_dict, share_memory=True)
        _verify(cpu_state_dict)
        cpu_state_dict = _create_cpu_state_dict(
            state_dict, share_memory=True, pin_memory=True
        )
        _verify(cpu_state_dict)
# 如果当前模块是主程序，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```