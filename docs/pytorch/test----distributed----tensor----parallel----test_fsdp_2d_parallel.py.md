# `.\pytorch\test\distributed\tensor\parallel\test_fsdp_2d_parallel.py`

```
# Owner(s): ["oncall: distributed"]
# 导入所需的库和模块
import io
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

import torch.nn.functional as F
from torch.distributed._tensor import DTensor as DT, init_device_mesh, Replicate, Shard
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    clean_tensor_name,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Tensor-Parallel degree
# 定义并行度为2
TP_DEGREE = 2
# 学习率设定为3e-5
LR = 3e-5


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义简单的神经网络模型，包括三个线性层和ReLU激活函数
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        # 前向传播函数，依次经过三个线性层和ReLU激活函数
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        # 生成一个随机的输入张量，设备为CUDA
        return torch.rand(4, 5, device="cuda")


class SimpleModelUneven(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # 定义不均匀的神经网络模型，包括四个线性层和ReLU激活函数
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        # 前向传播函数，依次经过四个线性层和ReLU激活函数
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        # 生成一个随机的输入张量，设备为CUDA
        return torch.rand(4, 5, device="cuda")


# TODO: add additional tests for multi_param_group, optim_in_backward,
# and fsdp_nested.
# 定义用于测试的类，继承自DTensorTestBase基类
class TestNew2dParallelTraining(DTensorTestBase):
    # 比较两个模型的参数是否相同
    def _compare_params(self, m1, m2):
        # 使用 m1 的完整参数配置上下文
        with FSDP.summon_full_params(m1):
            # 使用 m2 的完整参数配置上下文
            with FSDP.summon_full_params(m2):
                # 遍历 m1 和 m2 的命名参数对
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    # 获取参数 p1 和 p2
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    # 断言参数名是否相同或者是包含关系
                    if n_p1[0] != n_p2[0]:
                        self.assertTrue(n_p1[0] in n_p2[0])
                    # 获取参数名
                    name = n_p1[0]
                    # 如果参数名为 "net2.bias" 并且当前进程不是第 0 等级，跳过
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    # 如果 p2 的类型是 DT（假设为数据类型），则重新分发到本地设备
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()
                    # 断言两个参数在数值上是否接近
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    # 测试函数：检查是否会引发无效的 TP 组合异常
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_raise_invalid_tp_composition(self):
        # 断言捕获特定的 RuntimeError 异常，包含指定的错误消息
        with self.assertRaisesRegex(
            RuntimeError, r"Found TP device_mesh on the \d dimension of its parent mesh"
        ):
            # 初始化一个二维设备网格
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("tp", "dp")
            )
            # 并行化计划：net1 使用 ColwiseParallel，net2 使用 RowwiseParallel
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            # 在二维设备网格上并行化 SimpleModel 模型
            model_2d = parallelize_module(
                SimpleModel().cuda(), mesh_2d["tp"], parallelize_plan
            )

    # 测试函数：验证二维 FSDP 状态下的扩展是否启用
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_state_enable_extension(self):
        # 初始化一个二维设备网格
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        # 在二维设备网格上启用 FSDP 模型
        model = FSDP(
            SimpleModel().cuda(),
            device_mesh=mesh_2d["dp"],
        )
        # 获取 FSDP 模型的状态
        fsdp_state = _get_module_fsdp_state(model)
        # 断言 FSDP 扩展是否为 DTensorExtensions 类型
        self.assertTrue(isinstance(fsdp_state._fsdp_extension, DTensorExtensions))

    # 辅助函数：测试二维端到端训练
    def _test_2d_e2e_training(
        self,
        use_orig_params=False,
        recompute_activation=False,
    ) -> None:
        # 设置随机种子为0，确保结果可复现
        torch.manual_seed(0)
        # 创建一个简单模型并移动到指定 GPU
        model = SimpleModel().cuda(self.rank)
        # 使用 FSDP 对模型进行分布式数据并行
        model = FSDP(model, use_orig_params=use_orig_params)
        # 使用 Adam 优化器优化模型参数，学习率为0.01
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        # 再次设置随机种子为0，确保结果可复现
        torch.manual_seed(0)
        # 初始化设备网格，设备类型为 self.device_type，尺寸为 (2, self.world_size // 2)，网格维度名称为 ("dp", "tp")
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        # 获取 TP 维度的网格
        tp_mesh = mesh_2d["tp"]
        # 获取 DP 维度的网格
        dp_mesh = mesh_2d["dp"]
        # 定义并行化计划，net1 使用列并行化，net2 使用行并行化
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        # 将简单模型在 TP 维度上进行并行化
        model_2d = parallelize_module(SimpleModel().cuda(), tp_mesh, parallelize_plan)
        # 使用 FSDP 对 TP 维度并行化后的模型进行进一步分布式数据并行
        model_2d = FSDP(
            model_2d,
            device_mesh=dp_mesh,
            use_orig_params=use_orig_params,
        )
        # 使用 Adam 优化器优化 TP 维度并行化后的模型参数，学习率为0.01
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        # 如果需要重新计算激活值，对 TP 维度并行化后的模型进行输入重分片
        if recompute_activation:
            model_2d = input_reshard(model_2d, mesh_2d["tp"], 0)

        # 检查命名参数是否一致
        param_names_2d = [
            clean_tensor_name(name) for name, _ in model_2d.named_parameters()
        ]
        for name, _ in model.named_parameters():
            name = clean_tensor_name(name)
            if name not in param_names_2d:
                print(name, param_names_2d)
            # 断言命名参数在 TP 维度并行化后的模型中
            self.assertTrue(name in param_names_2d)
        # 比较原始模型和 TP 维度并行化后的模型参数是否一致
        self._compare_params(model, model_2d)

        # TODO: 添加多参数组和优化器在反向传播中的额外测试。

        # 执行5次循环
        for i in range(5):
            # 确保所有 TP 维度上的输入数据相同
            # TODO: 添加一个 get_group_rank() 函数到 DeviceMesh 中。
            torch.manual_seed(i + dist.get_rank(dp_mesh.get_group(mesh_dim=0)))
            # 创建随机输入数据，大小为 (4, 5)，并移动到指定 GPU
            input = torch.rand(4, 5).cuda(self.rank)
            # 在原始模型上进行前向传播
            output = model(input)
            # 在 TP 维度并行化后的模型上进行前向传播
            output_2d = model_2d(input)
            # 断言前向传播输出结果相等
            self.assertEqual(output, output_2d)
            # 对原始模型进行反向传播求梯度并更新优化器
            output.sum().backward()
            optim.step()
            # 对 TP 维度并行化后的模型进行反向传播求梯度并更新优化器
            output_2d.sum().backward()
            optim_2d.step()
            # 再次断言前向传播输出结果相等
            self.assertEqual(model(input), model_2d(input))

        # 确保优化器更新后模型参数仍然一致
        self._compare_params(model, model_2d)

    @with_comms
    @skip_if_lt_x_gpu(4)
    # 测试默认情况下的二维端到端训练
    def test_2d_e2e_training_default(self):
        self._test_2d_e2e_training()

    @with_comms
    @skip_if_lt_x_gpu(4)
    # 测试使用原始参数进行的二维端到端训练
    def test_2d_e2e_training_use_orig_params(self):
        self._test_2d_e2e_training(use_orig_params=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    # 测试不使用原始参数进行的二维端到端训练
    def test_2d_e2e_training_not_use_orig_params(self):
        # TODO: 需要重新审视 input_reshard API 为什么会导致多 GPU 测试失败。
        # self._test_2d_e2e_training(recompute_activation=True)
        # 执行二维端到端训练，不重新计算激活值
        self._test_2d_e2e_training(recompute_activation=False)
# TODO: 将所有状态字典单元测试更新为使用 distributed.checkpoint.state_dict，
# 并将所有状态字典测试整合到 test.distributed.checkpoint 中。
class TestNew2dParallelStateDict(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_fsdp_2d_extension(self):
        """
        测试 FSDPstate 的 _fsdp_extension 是否设置正确。
        """
        # 初始化一个二维设备网格
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        # 并行化计划，将不同网络分别指定为列并行和行并行
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
            "net3": ColwiseParallel(),
        }
        # 在 GPU 上并行化 SimpleModel，并应用设备网格
        model_2d = parallelize_module(
            SimpleModel().cuda(),
            mesh_2d["tp"],
            parallelize_plan=parallelize_plan,
        )
        # 将模型应用 FSDP 包装器，指定设备网格和使用原始参数
        model_2d = FSDP(model_2d, device_mesh=mesh_2d["dp"], use_orig_params=True)
        # 获取模型的 FSDP 状态
        model_2d_fsdp_state = _get_module_fsdp_state(model_2d)
        # 断言模型的 _fsdp_extension 是 DTensorExtensions 类型的实例
        self.assertTrue(
            isinstance(model_2d_fsdp_state._fsdp_extension, DTensorExtensions)
        )

        # 初始化一个一维设备网格
        mesh_1d = init_device_mesh("cuda", (self.world_size,))
        # 在 GPU 上应用 FSDP 包装器到一维 SimpleModel，指定设备网格和使用原始参数
        model_1d = FSDP(SimpleModel().cuda(), device_mesh=mesh_1d, use_orig_params=True)
        # 获取模型的 FSDP 状态
        model_1d_fsdp_state = _get_module_fsdp_state(model_1d)
        # 断言模型的 _fsdp_extension 为 None
        self.assertEqual(model_1d_fsdp_state._fsdp_extension, None)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    # 定义一个测试方法，用于测试2D状态字典的功能
    def test_2d_state_dict(self, is_even_sharded_model):
        # 根据条件选择SimpleModel或SimpleModelUneven作为模型类
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # 设置随机种子，并创建一个未使用包装器的模型，并将其放置在指定的GPU设备上
        torch.manual_seed(0)
        no_wrap_model = simple_model().cuda(self.rank)
        no_wrap_state_dict = no_wrap_model.state_dict()

        # 设置随机种子，并创建一个使用2D FSDP + TP进行分片的模型
        torch.manual_seed(0)
        # 初始化设备网格，指定了dp和tp的维度名称
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        # 定义并并行化计划
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        # 并行化模型，并应用FSDP进行参数分片
        model_2d = parallelize_module(simple_model().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(model_2d, device_mesh=dp_mesh, use_orig_params=True)

        # 设置模型的状态字典类型为SHARDED_STATE_DICT
        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict_2d = model_2d.state_dict()

        # 比较未使用包装器的模型和使用2D状态字典的模型的状态字典内容
        for no_wrap_items, two_d_items in zip(
            no_wrap_state_dict.items(), state_dict_2d.items()
        ):
            no_wrap_k, no_wrap_v = no_wrap_items
            two_d_k, two_d_v = two_d_items

            # 断言键相同
            self.assertEqual(no_wrap_k, two_d_k)

            # 检查2D状态字典中所有值是否为DTensor类型
            self.assertTrue(isinstance(two_d_v, DT))
            self.assertEqual(len(two_d_v.placements), 2)
            # 外部维度是FSDP维度，放置始终是Shard(0)
            self.assertEqual(two_d_v.placements[0], Shard(0))
            self.assertEqual(two_d_v.device_mesh, mesh_2d)

            # 检查参数值在2D模型和未使用包装器的模型之间是否相同
            all_gather_two_d_v = two_d_v.redistribute(
                mesh_2d, (Replicate(), Replicate())
            )
            self.assertEqual(
                torch.allclose(no_wrap_v, all_gather_two_d_v.to_local()), True
            )
    # 定义一个测试函数，用于加载和比较二维模型的状态字典
    def test_2d_load_state_dict(self, is_even_sharded_model):
        # 根据条件选择简单模型类
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # 设置随机种子为0
        torch.manual_seed(0)
        # 初始化设备网格为二维网格，根据设备类型和世界大小分配网格维度命名为("dp", "tp")
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        # 获取tp维度的网格
        tp_mesh = mesh_2d["tp"]
        # 获取dp维度的网格
        dp_mesh = mesh_2d["dp"]
        
        # 并行化计划，包含两个网络"net1"和"net2"，分别使用不同的并行化策略
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        # 并行化模型，将简单模型实例移到CUDA设备上，并按tp_mesh的计划并行化
        model_2d = parallelize_module(simple_model().cuda(), tp_mesh, parallelize_plan)
        # 使用FSDP（Fully Sharded Data Parallel）模块，指定dp_mesh设备网格和使用原始参数
        model_2d = FSDP(model_2d, device_mesh=dp_mesh, use_orig_params=True)
        # 使用Adam优化器，针对model_2d的参数，设置学习率为0.01
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        # 设置FSDP模型的状态字典类型为SHARDED_STATE_DICT
        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 创建一个内存中的字节流对象checkpoint，用于保存模型状态字典
        checkpoint = io.BytesIO()
        # 将model_2d的状态字典保存到checkpoint中
        torch.save(model_2d.state_dict(), checkpoint)
        # 深拷贝当前状态字典，以便后续比较加载回来的状态字典
        ref_state_dict = deepcopy(model_2d.state_dict())

        # 更新模型参数，使model_2d的状态字典与ref_state_dict不同
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()

        # 加载回ref_state_dict
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model_2d.load_state_dict(load_ref_state_dict)
        new_state_dict = model_2d.state_dict()

        # 检查new_state_dict是否与ref_state_dict相同
        for (k1, v1), (k2, v2) in zip(ref_state_dict.items(), new_state_dict.items()):
            # 检查键名k1和k2是否相同
            self.assertEqual(k1, k2)

            # 检查v1和v2的类型是否为DT（可能是自定义类型）
            self.assertEqual(type(v1), DT)
            self.assertEqual(type(v2), DT)

            # 检查v1和v2的本地张量是否相同
            # TODO: 目前不支持二维DTensor的比较，因此暂时比较规格和本地张量。
            # TODO: 更新为一旦支持二维DTensor比较，就比较两个DTensors。
            self.assertEqual(v1.to_local(), v2.to_local())
            # 检查v1和v2的设备网格是否相同
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            # 检查v1和v2的放置是否相同
            self.assertEqual(v1.placements, v2.placements)
# 实例化参数化测试用例，使用 TestNew2dParallelStateDict 类作为参数
instantiate_parametrized_tests(TestNew2dParallelStateDict)

# 如果当前脚本作为主程序运行，则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```