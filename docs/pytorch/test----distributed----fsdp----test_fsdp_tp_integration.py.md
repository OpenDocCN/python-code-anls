# `.\pytorch\test\distributed\fsdp\test_fsdp_tp_integration.py`

```py
# 导入必要的模块和库
import copy  # 导入深拷贝模块
import sys  # 导入系统模块
from collections import OrderedDict  # 导入有序字典模块
from typing import Dict, List, Optional, Tuple  # 导入类型提示相关模块

import torch  # 导入PyTorch库
from torch import distributed as dist  # 导入分布式模块
from torch.distributed._tensor import (  # 导入分布式张量相关模块
    DeviceMesh,
    distribute_module,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug import CommDebugMode  # 导入通信调试模块
from torch.distributed.fsdp.fully_sharded_data_parallel import (  # 导入全分片数据并行模块
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.tensor.parallel import (  # 导入张量并行模块
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试相关模块
from torch.testing._internal.common_fsdp import FSDPTest  # 导入FSDP测试相关模块
from torch.testing._internal.common_utils import (  # 导入通用测试工具模块
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量通用模块
    MLPModule,
    RMSNormPython,
)

# 检查分布式功能是否可用，若不可用则输出信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 检查是否在开发调试ASAN模式下，若是则输出信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(5, 8)  # 定义第一个线性层
        self.relu = torch.nn.ReLU()  # 定义ReLU激活函数
        self.net2 = torch.nn.Linear(8, 4)  # 定义第二个线性层
        self.net3 = torch.nn.Linear(4, 12)  # 定义第三个线性层

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))  # 前向传播函数

    @staticmethod
    def get_sharded_param_names() -> List[str]:
        return ["net1.weight", "net1.bias", "net2.weight"]  # 返回被分片的参数名称列表

    @staticmethod
    def get_non_sharded_param_names() -> List[str]:
        return ["net3.weight", "net3.bias"]  # 返回未被分片的参数名称列表


def distribute_rmsnorm(module, device_mesh):
    def prepare_input_fn(mod, inputs, device_mesh):
        shard_tensor = DTensor.from_local(inputs[0], device_mesh, [Shard(0)])  # 准备输入函数，将输入张量划分为分片张量
        return shard_tensor

    def prepare_output_fn(mod, outputs, device_mesh):
        return outputs.to_local()  # 准备输出函数，将输出张量转为本地张量

    return distribute_module(
        module, device_mesh, input_fn=prepare_input_fn, output_fn=prepare_output_fn
    )  # 使用分布式模块函数，分布式部署模块与设备网格


class TestTPFSDPIntegration(FSDPTest):
    def _get_params_and_sharding_info(
        self,
        model: SimpleModel,
        sharded_param_names: List[str],
        tensor_parallel_size: int,
    ) -> Tuple[Dict[str, int], Dict[str, Tuple[torch.Size, int]]]:
        """
        返回两个字典：参数名称到其元素数量的映射和参数名称到分片信息的映射。
        """
        assert (
            type(model) is SimpleModel
        ), "Expects a `SimpleModel` since the sharding cases on the model definition"
        # 断言模型类型为 SimpleModel，因为模型定义上有分片的情况
        param_name_to_numel = OrderedDict()
        param_name_to_sharding_info = OrderedDict()
        for param_name, param in model.named_parameters():
            if param_name not in sharded_param_names:
                # 如果参数不在被分片的参数名列表中，则直接取其元素数量
                param_name_to_numel[param_name] = param.numel()
            else:
                # 如果参数在被分片的参数名列表中，则按照分片大小计算元素数量，并记录分片信息
                param_name_to_numel[param_name] = param.numel() // tensor_parallel_size
                param_name_to_sharding_info[param_name] = (
                    param.size(),
                    0 if "net1" in param_name else 1,
                )
        return param_name_to_numel, param_name_to_sharding_info

    def _get_sub_pgs(self, tensor_parallel_size: int):
        """
        生成 TP 和 FSDP 子进程组。`tensor_parallel_size` 指定了 TP 进程组的大小。

        例如，如果全局世界大小是 8，而 tensor parallel 大小是 2，则会创建：
        - 4 个 TP 子进程组：[0, 1], [2, 3], [4, 5], [6, 7]
        - 2 个 FSDP 子进程组：[0, 2, 4, 6], [1, 3, 5, 7]
        """
        # 二维网格是 [dp, tp]
        twod_mesh = DeviceMesh(
            device_type="cuda",
            mesh=torch.arange(0, self.world_size).view(-1, tensor_parallel_size),
        )

        fsdp_pg = twod_mesh.get_group(mesh_dim=0)
        tp_pg = twod_mesh.get_group(mesh_dim=1)
        return twod_mesh, fsdp_pg, tp_pg

    def _sync_tp_grads(
        self,
        tp_fsdp_model: FSDP,
        tp_pg: dist.ProcessGroup,
        param_name_to_numel: Dict[str, int],
        non_sharded_param_names: List[str],
        ):
        """
        同步 TP 梯度。

        Args:
            tp_fsdp_model (FSDP): 带有 FSDP 的 TP 模型。
            tp_pg (dist.ProcessGroup): TP 进程组。
            param_name_to_numel (Dict[str, int]): 参数名称到元素数量的映射。
            non_sharded_param_names (List[str]): 非分片参数的名称列表。
        """
    ) -> None:
        """
        Syncs the tensor parallel parameters' gradients following the data
        parallel paradigm where gradients are averaged over ranks (in this
        case, the ones in the tensor parallel process group).
        """
        tp_world_size = tp_pg.size()  # 获取张量并行过程组的大小
        fsdp_world_size = self.world_size // tp_world_size  # 计算每个张量并行组中的数据并行世界大小
        assert (
            type(tp_fsdp_model) is FSDP
            and len([m for m in tp_fsdp_model.modules() if type(m) is FSDP]) == 1
        ), (
            "The following logic assumes a single top-level-only FSDP wrapping "
            "the model with TP already applied"
        )  # 断言确保 tp_fsdp_model 是 FSDP 类型的单个顶级封装模型
        for flat_param in tp_fsdp_model.params:
            splits = tuple(param_name_to_numel.values())  # 获取参数名到元素数的映射
            # 创建一个用于手动减少梯度元素的掩码
            unsharded_size = torch.Size([flat_param.numel() * fsdp_world_size])
            unsharded_zeros = torch.zeros(unsharded_size, device=flat_param.device)
            per_param_masks = unsharded_zeros.split(splits)  # 拆分为每个参数的掩码
            for param_idx, param_name in enumerate(
                param_name_to_numel.keys()
            ):  # 假设固定顺序
                if param_name not in non_sharded_param_names:
                    per_param_masks[param_idx][:] = 1  # 根据非分片参数名设置掩码为1
            unsharded_mask = (
                torch.cat(per_param_masks).contiguous().type(torch.BoolTensor)
            )  # 将所有参数的掩码连接并转换为布尔张量
            sharded_mask = unsharded_mask.chunk(fsdp_world_size)[
                self.rank // tp_world_size
            ]  # 根据排名和张量并行组的大小分割掩码
            grad_device = flat_param.grad.device  # 获取梯度所在设备
            grad = flat_param.grad.detach().clone().cuda(self.rank)  # 克隆并移动梯度到指定的 GPU 设备
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_pg)  # 对梯度进行全局归约操作
            grad = grad.to(grad_device)  # 将梯度移动回原始设备
            flat_param.grad[~sharded_mask] = grad[~sharded_mask]  # 通过掩码更新平坦参数的梯度
            # 平均所有梯度元素以匹配仅使用 FSDP 的语义
            flat_param.grad /= tp_world_size

    def _get_grads_as_flattened(
        self,
        model: FSDP,
        uses_tp: bool,
        param_name_to_numel: Dict[str, int],
        param_name_to_sharding_info: Dict[str, Tuple[torch.Size, int]],
        tp_pg: Optional[dist.ProcessGroup],
        fsdp_pg: Optional[dist.ProcessGroup],
        sharded_param_names: Optional[List[str]],
    ) -> torch.Tensor:
        """
        Returns all unsharded gradients as a single flattened tensor. This
        returns the same value on all ranks.
        """
        # Flatten and concatenate gradients of all model parameters into a single tensor
        local_grads_as_flattened = (
            torch.cat(
                [
                    torch.flatten(param.grad)
                    if param.grad is not None  # Check if gradient exists for the parameter
                    else torch.zeros_like(torch.flatten(param))  # Use zeros if gradient is None
                    for param in model.parameters()
                ]
            )
            .contiguous()  # Ensure contiguous memory layout
            .cuda(self.rank)  # Move tensor to GPU specified by self.rank
        )
        
        # Create an empty tensor to gather all gradients from different ranks
        all_grads_as_flattened = torch.cat(
            [torch.empty_like(local_grads_as_flattened) for _ in range(fsdp_pg.size())]
        ).contiguous()
        
        # All gather operation to collect local gradients from all ranks into all_grads_as_flattened
        dist.all_gather_into_tensor(
            all_grads_as_flattened, local_grads_as_flattened, group=fsdp_pg
        )
        
        # If not using Tensor Parallelism (TP), return the gathered gradients
        if not uses_tp:
            return all_grads_as_flattened
        
        # Split gradients according to sizes defined in param_name_to_numel
        splits = tuple(param_name_to_numel.values())
        all_grads_per_param = list(all_grads_as_flattened.split(splits))
        
        # Iterate over parameters and handle sharded gradients
        for param_idx, param_name in enumerate(param_name_to_numel.keys()):
            if param_name in sharded_param_names:
                # Adjust local tensor size based on sharding information
                local_tensor_size = list(param_name_to_sharding_info[param_name][0])
                sharding_dim = param_name_to_sharding_info[param_name][1]
                local_tensor_size[sharding_dim] //= tp_pg.size()
                
                # View and gather local tensors across Tensor Parallelism groups
                local_tensor = all_grads_per_param[param_idx].view(*local_tensor_size)
                local_tensors = [
                    torch.empty_like(local_tensor) for _ in range(tp_pg.size())
                ]
                dist.all_gather(local_tensors, local_tensor, group=tp_pg)
                
                # Concatenate tensors along sharding dimension and reshape
                all_grads_per_param[param_idx] = torch.cat(
                    local_tensors, dim=sharding_dim
                ).reshape(-1)
        
        # Concatenate all gradients across parameters and return
        return torch.cat(all_grads_per_param).contiguous()

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_integration(self):
        self.run_subtests(
            {
                "cpu_offload": [
                    CPUOffload(offload_params=False),
                    CPUOffload(offload_params=True),
                ],
                "sharding_strategy": [None, ShardingStrategy.SHARD_GRAD_OP],
                "use_orig_params": [False, True],
            },
            self._test_fsdp_tp_integration,
        )

    def _test_fsdp_tp_integration(
        self, cpu_offload, sharding_strategy, use_orig_params
    ):
        # This function is the actual test function for fsdp-tp integration
        # It takes parameters related to CPU offload, sharding strategy, and original parameters usage
        # and runs subtests using self.run_subtests
        pass
    def test_fsdp_tp_extension_grad(self):
        """
        Tests TP + FSDP extension with correct gradient (i.e. no ACT)
        """
        # 初始化一个二维设备网格，使用 CUDA，分配给当前进程的 GPU 设备数目的一半，网格维度名称为 dp 和 tp
        mesh_2d = init_device_mesh(
            "cuda", (self.world_size // 2, 2), mesh_dim_names=["dp", "tp"]
        )

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 MLP 模块，使用 CUDA
                self.mlp = MLPModule("cuda")
                # 初始化一个 RMSNormPython 实例，设定参数为 10
                self.mlp_norm = RMSNormPython(10)

            def forward(self, x):
                # 将输入 x 先经过 self.mlp_norm 正规化，再通过 self.mlp 前馈计算
                return self.mlp(self.mlp_norm(x))

        # 创建一个 TestModel 的实例，并分配到当前进程的 GPU 设备
        model = TestModel().cuda(self.rank)

        # 在 TP 网格上并行化模型
        tp_mesh = mesh_2d["tp"]
        tp_model = parallelize_module(
            model,
            tp_mesh,
            {
                "mlp.net1": ColwiseParallel(input_layouts=Shard(0)),
                "mlp.net2": RowwiseParallel(output_layouts=Shard(0)),
            },
        )
        # 在 TP 网格上分发 RMSNorm 操作
        distribute_rmsnorm(tp_model.mlp_norm, tp_mesh)

        # 创建一个 FSDP 模型，基于 tp_model，并分配到当前进程的 dp 网格设备
        fsdp_2d_model = FSDP(tp_model, device_mesh=mesh_2d["dp"])
        comm_mode = CommDebugMode()

        # 进入通信调试模式
        with comm_mode:
            # 在模型上进行前向传播和反向传播，输入随机数据，分配到当前进程的 GPU 设备
            fsdp_2d_model(torch.rand(2, 10).cuda(self.rank)).sum().backward()

        # 引用 Torch 的 c10d_functional 操作模块
        funcol = torch.ops.c10d_functional
        # 引用 Torch 的 c10d 操作模块
        c10d_ops = torch.ops.c10d
        # 获取通信统计信息
        comm_counts = comm_mode.get_comm_counts()
        # 断言总通信数为 7
        self.assertEqual(comm_mode.get_total_counts(), 7)
        # 断言 TP 网格的通信数
        self.assertEqual(comm_counts[funcol.reduce_scatter_tensor], 2)
        self.assertEqual(comm_counts[funcol.all_gather_into_tensor], 2)
        self.assertEqual(comm_counts[funcol.all_reduce], 1)
        # 断言 FSDP 模型的通信数
        self.assertEqual(comm_counts[c10d_ops._allgather_base_], 1)
        self.assertEqual(comm_counts[c10d_ops._reduce_scatter_base_], 1)

        # 收集所有有梯度的参数的梯度值
        grads = [p.grad for p in fsdp_2d_model.parameters() if p.grad is not None]

        # 断言所有梯度值不含 NaN
        for grad in grads:
            self.assertFalse(grad.isnan().any().item())
    def test_fsdp_tp_sync_module_state(self):
        # 初始化包含 "dp" 和 "tp" 两个维度的设备网格，使用 CUDA，每个维度大小为 self.world_size // 2
        mesh_2d = init_device_mesh(
            "cuda", (self.world_size // 2, 2), mesh_dim_names=["dp", "tp"]
        )
        # 获取 "tp" 维度的网格
        tp_mesh = mesh_2d["tp"]
        # 获取 "dp" 维度的网格
        dp_mesh = mesh_2d["dp"]

        # 为每个进程设置随机种子
        torch.manual_seed(mesh_2d.get_rank())

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 从本地创建分布式张量 replicated_dt，使用 "tp" 网格进行复制
                replicated_dt = DTensor.from_local(
                    torch.randn(8, 8), tp_mesh, [Replicate()], run_check=False
                )
                # 从本地创建分布式缓冲张量 replicated_buffer_dt，使用 "tp" 网格进行复制
                replicated_buffer_dt = DTensor.from_local(
                    torch.randn(8, 8), tp_mesh, [Replicate()], run_check=False
                )
                # 将 replicated_dt 设为模型的参数
                self.param = torch.nn.Parameter(replicated_dt)
                # 注册 replicated_buffer_dt 为模型的缓冲区
                self.register_buffer("buf", replicated_buffer_dt)

            def forward(self, x):
                # 模型的前向传播，返回 param 加 buffer 加 1 的结果
                return self.param + self.buffer + 1

        # 创建测试模型实例
        model = TestModel()

        def assert_local_shard_across_ranks(local_tensor, group, check_equal=True):
            # 创建空张量列表 gathered_tensors，用于存储收集的张量
            gathered_tensors = [
                torch.empty_like(local_tensor) for _ in range(group.size())
            ]
            # 使用分布式通信收集 local_tensor 到 gathered_tensors 中，使用指定的 group
            dist.all_gather(gathered_tensors, local_tensor, group=group)
            # 选取第一个 gathered_tensors 中的张量作为比较对象
            tensor_to_compare = gathered_tensors[0]
            # 检查 gathered_tensors 中的张量是否相等
            for tensor in gathered_tensors[1:]:
                if check_equal:
                    self.assertTrue(torch.equal(tensor, tensor_to_compare))
                else:
                    self.assertFalse(torch.equal(tensor, tensor_to_compare))

        # 获取 "dp" 维度的分组
        dp_group = dp_mesh.get_group()

        # 检查在 "dp" 维度上，模型的参数 local tensor 不相等
        local_param = model.param.to_local()
        assert_local_shard_across_ranks(local_param, dp_group, check_equal=False)
        # 检查在 "dp" 维度上，模型的缓冲区 local tensor 不相等
        local_buf = model.buf.to_local()
        assert_local_shard_across_ranks(local_buf, dp_group, check_equal=False)

        # 使用 fsdp 同步参数应该同步 "dp" 维度的模块状态
        fsdp_mod = FSDP(model, device_mesh=dp_mesh, sync_module_states=True)
        with fsdp_mod.summon_full_params(fsdp_mod):
            # 在 fsdp 同步后，检查 "dp" 维度上，模型的参数 local tensor 相等
            local_param = fsdp_mod.param.to_local()
            assert_local_shard_across_ranks(local_param, dp_group, check_equal=True)

            # 在 fsdp 同步后，检查 "dp" 维度上，模型的缓冲区 local tensor 相等
            local_buf = fsdp_mod.buf.to_local()
            assert_local_shard_across_ranks(local_buf, dp_group, check_equal=True)
# 实例化一个带参数的测试，使用 TestTPFSDPIntegration 类来设置测试的参数化
instantiate_parametrized_tests(TestTPFSDPIntegration)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```