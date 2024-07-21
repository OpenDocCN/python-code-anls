# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_init.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入所需的库和模块
import copy  # 导入 copy 模块，用于对象复制
import itertools  # 导入 itertools 模块，用于迭代操作
import unittest  # 导入 unittest 模块，用于编写和运行测试
from typing import List  # 导入 List 类型提示

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入分布式训练相关模块
import torch.nn as nn  # 导入神经网络模块
from torch.distributed._composable import replicate  # 导入复制相关功能
from torch.distributed._composable.fsdp import fully_shard  # 导入 FSDP 相关功能
from torch.distributed._composable.fsdp._fsdp_init import (
    _get_managed_modules,  # 导入获取受管理模块的函数
    _get_managed_states,  # 导入获取受管理状态的函数
)
from torch.distributed._composable.fsdp._fsdp_param import ParamModuleInfo  # 导入参数模块信息
from torch.distributed._composable.fsdp._fsdp_param_group import _get_param_module_infos  # 导入获取参数模块信息的函数
from torch.distributed._tensor import (  # 导入分布式张量相关功能
    DeviceMesh,  # 设备网格
    distribute_tensor,  # 分布式张量分发
    DTensor,  # 分布式张量
    Replicate,  # 复制
    Shard,  # 分片
)
from torch.distributed.device_mesh import init_device_mesh  # 初始化设备网格
from torch.distributed.fsdp._init_utils import (  # 导入初始化工具函数
    _init_inter_node_process_group,  # 初始化跨节点进程组
    _init_intra_node_process_group,  # 初始化内节点进程组
)
from torch.distributed.tensor.parallel import (  # 导入张量并行操作相关模块
    ColwiseParallel,  # 列并行
    parallelize_module,  # 并行化模块
    RowwiseParallel,  # 行并行
)
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入测试 CUDA 相关功能
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP  # 导入 FSDP 测试相关类和 MLP 模型
from torch.testing._internal.common_utils import run_tests  # 导入运行测试函数
from torch.testing._internal.distributed._tensor.common_dtensor import (  # 导入分布式张量的公共测试相关模块
    ModelArgs,  # 模型参数
    Transformer,  # 变换器
    TransformerBlock,  # 变换器块
)


class TestFullyShardDeviceTensor(FSDPTestMultiThread):
    """Tests that tensor parameters are moved to the expected device."""

    @property
    def world_size(self) -> int:
        return 1  # 设置测试所需的全局世界大小为 1

    @unittest.skipIf(not TEST_CUDA, "no cuda")  # 如果没有 CUDA 支持，则跳过测试
    def test_move_states_to_device_tensor(self):
        model = MLP(8, torch.device("cpu"), with_buffer=True)  # 创建一个 MLP 模型对象，使用 CPU 设备
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, torch.device("cpu"))  # 断言模型参数和缓冲区在 CPU 设备上
        fully_shard(model)  # 对模型进行完全分片
        cuda_device = torch.device("cuda", torch.cuda.current_device())  # 获取当前 CUDA 设备
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, cuda_device)  # 断言模型参数和缓冲区已经移动到 CUDA 设备


class TestFullyShardDeviceDTensor(FSDPTestMultiThread):
    """Tests that DTensor parameters are moved to the expected device."""

    @property
    def world_size(self) -> int:
        return 4  # 设置测试所需的全局世界大小为 4

    @unittest.skipIf(not TEST_CUDA, "no cuda")  # 如果没有 CUDA 支持，则跳过测试
    # 定义一个测试方法，用于验证将状态移动到设备上时的行为是否正确
    def test_move_states_to_device_dtensor_valid(self):
        # 断言当前的世界大小是否大于等于4，否则抛出异常并显示当前世界大小
        assert self.world_size >= 4, f"{self.world_size}"
        # 设定每个数据并行组的大小为2
        dp_size = 2
        # 初始化全局设备网格为 CUDA 设备，划分为 (dp_size, self.world_size // dp_size) 的网格尺寸，指定网格维度名称为 ("dp", "tp")
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        # 获取 dp_mesh 和 tp_mesh 两个子网格
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        # 创建一个包含缓冲区的 MLP 模型，输入维度为 8，设备为 CPU
        model = MLP(8, torch.device("cpu"), with_buffer=True)
        # 将模型并行化，使用 tp_mesh 网格，指定输入投影和输出投影的并行方式
        parallelize_module(
            model,
            tp_mesh,
            {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
        )
        # 获取当前 CUDA 设备的设备对象
        cuda_device = torch.device("cuda", torch.cuda.current_device())
        # 遍历模型的参数和缓冲区
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            if isinstance(tensor, DTensor):
                # 如果 tensor 是 DTensor 类型，则断言其设备为 CUDA 设备
                # DTensor 构造函数将其移动到所属网格的设备上
                self.assertEqual(tensor.device, cuda_device)
                self.assertEqual(tensor._local_tensor.device, cuda_device)
            else:
                # 否则断言其设备为 CPU
                self.assertEqual(tensor.device, torch.device("cpu"))
        # 将模型完全分片，使用 dp_mesh 网格
        fully_shard(model, mesh=dp_mesh)
        # 再次验证模型的参数和缓冲区是否都在 CUDA 设备上
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, cuda_device)
            if isinstance(tensor, DTensor):
                self.assertEqual(tensor._local_tensor.device, cuda_device)

    # 根据测试是否支持 CUDA，决定是否跳过该测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_move_states_to_device_dtensor_invalid(self):
        # 断言当前的世界大小是否大于等于4，否则抛出异常并显示当前世界大小
        assert self.world_size >= 4, f"{self.world_size}"
        # 设定每个数据并行组的大小为2
        dp_size = 2
        # 初始化全局 CUDA 设备网格，划分为 (dp_size, self.world_size // dp_size) 的网格尺寸，指定网格维度名称为 ("dp", "tp")
        global_cuda_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        # 初始化全局 CPU 设备网格，划分为 (dp_size, self.world_size // dp_size) 的网格尺寸，指定网格维度名称为 ("dp", "tp")
        global_cpu_mesh = init_device_mesh(
            "cpu", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        # 获取 dp_mesh 和 tp_mesh 两个子网格，其中 tp_mesh 使用 CPU 设备（不匹配的网格！）
        dp_mesh = global_cuda_mesh["dp"]
        tp_mesh = global_cpu_mesh["tp"]
        # 创建一个包含缓冲区的 MLP 模型，输入维度为 8，设备为 CPU
        model = MLP(8, torch.device("cpu"), with_buffer=True)
        # 将模型并行化，使用 tp_mesh 网格，指定输入投影和输出投影的并行方式
        parallelize_module(
            model,
            tp_mesh,
            {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
        )
        # 遍历模型的参数和缓冲区
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            # 断言所有 tensor 的设备为 CPU
            self.assertEqual(tensor.device, torch.device("cpu"))
            if isinstance(tensor, DTensor):
                # 如果 tensor 是 DTensor 类型，断言其本地张量的设备也为 CPU
                self.assertEqual(tensor._local_tensor.device, torch.device("cpu"))
        # 准备匹配的错误信息正则表达式
        regex = r"Requires DTensor to have mesh of the same type as the FSDP mesh but got cpu for DTensor and cuda for FSDP"
        # 断言调用 fully_shard 函数时抛出 ValueError 异常，并且异常消息符合预期的正则表达式格式
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model, mesh=dp_mesh)
class TestFullyShardMeshArg(FSDPTestMultiThread):
    """Tests the ``mesh`` argument."""

    @property
    def world_size(self) -> int:
        # 返回固定的世界大小为 2
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_invalid_mesh_ndim(self):
        # 初始化一个 CUDA 设备上的 mesh，形状为 (2, 1, 1)
        mesh = init_device_mesh("cuda", (self.world_size, 1, 1))
        model = MLP(8)
        # 定义一个用于匹配异常消息的正则表达式
        regex = r"fully\_shard expects a 1D or 2D DeviceMesh but got DeviceMesh\(\[\[\[0\]\], \[\[1\]\]\]\)"
        # 断言调用 fully_shard 函数时会抛出 ValueError 异常，并匹配指定的正则表达式
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model, mesh=mesh)


class TestFullyShardManagedModulesAndStates(FSDPTestMultiThread):
    """Tests getting the managed modules/states for a ``fully_shard`` module."""

    @property
    def world_size(self) -> int:
        # 返回固定的世界大小为 1
        return 1

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_single(self):
        model = MLP(8)
        # 假设在模型上调用 `fully_shard`
        managed_modules = _get_managed_modules(model)
        expected_managed_modules = list(model.modules())
        # 检查获取的 managed_modules 是否与预期的 expected_managed_modules 相同
        self._check_managed_modules(managed_modules, expected_managed_modules)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_nested(self):
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        fully_shard(model[0])
        # 假设在模型上调用 `fully_shard`
        managed_modules = _get_managed_modules(model)
        expected_managed_modules = list(model[1].modules()) + [model]
        # 检查获取的 managed_modules 是否与预期的 expected_managed_modules 相同
        self._check_managed_modules(managed_modules, expected_managed_modules)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_nested_fully_shard_and_replicate(self):
        model = nn.Sequential(*[MLP(8) for _ in range(3)])
        replicate(model[0])
        fully_shard(model[2])
        # 假设在模型上调用 `fully_shard`
        managed_modules = _get_managed_modules(model)
        expected_managed_modules = list(model[1].modules()) + [model]
        # 检查获取的 managed_modules 是否与预期的 expected_managed_modules 相同
        self._check_managed_modules(managed_modules, expected_managed_modules)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_duplicate(self):
        mlp = MLP(8)
        model = nn.Sequential(mlp, mlp)  # 复制 MLP 模块
        # 假设在模型上调用 `fully_shard`
        managed_modules = _get_managed_modules(model)
        # 检查重复模块只计数一次
        expected_managed_modules = list(mlp.modules()) + [model]
        self._check_managed_modules(managed_modules, expected_managed_modules)

    def _check_managed_modules(
        self,
        managed_modules: List[nn.Module],
        expected_managed_modules: List[nn.Module],
    ):
        # 断言 managed_modules 和 expected_managed_modules 的长度相同
        self.assertEqual(len(managed_modules), len(expected_managed_modules))
        # 检查集合是否相同，因为不要求顺序一致
        self.assertEqual(set(managed_modules), set(expected_managed_modules))

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义一个测试方法，测试管理状态、共享参数和缓冲区
    def test_managed_states_shared_params_and_buffers(self):
        # 创建一个包含三个 MLP 模型的序列模型
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(3)])
        # 将第一个模型的输入投影权重共享给第二个模型
        model[0].in_proj.weight = model[1].in_proj.weight
        # 将第二个模型的输入投影权重共享给第三个模型
        model[2].in_proj.weight = model[1].in_proj.weight
        # 将第三个模型的缓冲区共享给第二个模型
        model[1].buffer = model[2].buffer
        # 假设在 `model` 上调用 `fully_shard` 方法
        managed_modules = _get_managed_modules(model)
        # 获取管理模块的状态：参数和缓冲区
        params, buffers = _get_managed_states(managed_modules)
        # 获取预期的参数列表，去重共享的参数
        expected_params = list(model.parameters())
        # 获取预期的缓冲区列表，去重共享的缓冲区
        expected_buffers = list(model.buffers())
        # 调用私有方法 `_check_managed_states` 检查管理状态
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    # 如果未启用 CUDA，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_states_nested_fully_shard(self):
        # 创建一个包含两个 MLP 模型的序列模型
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(2)])
        # 假设在第一个模型上调用 `fully_shard` 方法
        fully_shard(model[0])
        # 假设在 `model` 上调用 `fully_shard` 方法
        managed_modules = _get_managed_modules(model)
        # 获取管理模块的状态：参数和缓冲区
        params, buffers = _get_managed_states(managed_modules)
        # 获取预期的参数列表，第二个模型的参数列表（因为第一个模型被 shard 了）
        expected_params = list(model[1].parameters())
        # 获取预期的缓冲区列表，第二个模型的缓冲区列表（因为第一个模型被 shard 了）
        expected_buffers = list(model[1].buffers())
        # 调用私有方法 `_check_managed_states` 检查管理状态
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    # 检查管理状态的私有方法
    def _check_managed_states(
        self,
        managed_params: List[nn.Parameter],
        managed_buffers: List[torch.Tensor],
        expected_managed_params: List[nn.Parameter],
        expected_managed_buffers: List[torch.Tensor],
    ):
        # 断言管理参数的长度与预期管理参数的长度相等
        self.assertEqual(len(managed_params), len(expected_managed_params))
        # 断言管理缓冲区的长度与预期管理缓冲区的长度相等
        self.assertEqual(len(managed_buffers), len(expected_managed_buffers))
        # 断言管理参数的集合与预期管理参数的集合相等（去重共享的参数）
        self.assertEqual(set(managed_params), set(expected_managed_params))
        # 断言管理缓冲区的集合与预期管理缓冲区的集合相等（去重共享的缓冲区）
        self.assertEqual(set(managed_buffers), set(expected_managed_buffers))
# 定义一个测试类 `TestFullyShardParamModuleInfos`，继承自 `FSDPTestMultiThread`
class TestFullyShardParamModuleInfos(FSDPTestMultiThread):

    # 定义一个属性方法 `world_size`，返回整数值 2
    @property
    def world_size(self) -> int:
        return 2

    # 如果未开启 CUDA 测试，则跳过该测试方法
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义一个测试方法 `test_get_param_module_infos_shared_params`
    def test_get_param_module_infos_shared_params(self):
        # 创建一个包含两个 MLP 模块的序列模型 `model`
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        # 共享第一个 MLP 模块的输入投影层权重给第二个 MLP 模块的输入投影层权重
        model[0].in_proj.weight = model[1].in_proj.weight
        # 获取管理的模块列表 `managed_modules`
        managed_modules = _get_managed_modules(model)
        # 获取管理的状态信息 `params` 和 `_`
        params, _ = _get_managed_states(managed_modules)
        # 获取参数模块信息 `param_module_infos`
        param_module_infos = _get_param_module_infos(params, model)
        # 断言 `param_module_infos` 的长度与 `params` 的长度相等
        self.assertEqual(len(param_module_infos), len(params))
        # 期望 `params` 中已经包含了去重后的共享参数
        expected_param_module_infos = [
            ParamModuleInfo(model[0].in_proj, "weight", [model[1].in_proj], ["weight"]),
            ParamModuleInfo(model[0].in_proj, "bias", [], []),
            ParamModuleInfo(model[0].out_proj, "weight", [], []),
            ParamModuleInfo(model[0].out_proj, "bias", [], []),
            ParamModuleInfo(model[1].in_proj, "bias", [], []),
            ParamModuleInfo(model[1].out_proj, "weight", [], []),
            ParamModuleInfo(model[1].out_proj, "bias", [], []),
        ]
        # 断言 `param_module_infos` 的长度与期望的参数模块信息列表长度相等
        self.assertEqual(len(param_module_infos), len(expected_param_module_infos))
        # 断言 `param_module_infos` 等于期望的参数模块信息列表
        self.assertEqual(param_module_infos, expected_param_module_infos)

    # 如果未开启 CUDA 测试，则跳过该测试方法
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义一个测试方法 `test_get_param_module_infos_duplicates`
    def test_get_param_module_infos_duplicates(self):
        # 创建一个 MLP 模块 `mlp`
        mlp = MLP(8)
        # 创建一个包含两个共享 MLP 模块的序列模型 `model`
        model = nn.Sequential(mlp, mlp)  # 共享的 MLP
        # 获取模型 `model` 的所有参数列表 `params`
        params = list(model.parameters())
        # 获取参数模块信息 `param_module_infos`
        param_module_infos = _get_param_module_infos(params, model)
        # 断言 `param_module_infos` 的长度与 `params` 的长度相等
        self.assertEqual(len(param_module_infos), len(params))
        # 期望的参数模块信息列表，包含共享参数信息
        expected_param_module_infos = [
            ParamModuleInfo(mlp.in_proj, "weight", [mlp.in_proj], ["weight"]),
            ParamModuleInfo(mlp.in_proj, "bias", [mlp.in_proj], ["bias"]),
            ParamModuleInfo(mlp.out_proj, "weight", [mlp.out_proj], ["weight"]),
            ParamModuleInfo(mlp.out_proj, "bias", [mlp.out_proj], ["bias"]),
        ]
        # 断言 `param_module_infos` 的长度与期望的参数模块信息列表长度相等
        self.assertEqual(len(param_module_infos), len(expected_param_module_infos))
        # 断言 `param_module_infos` 等于期望的参数模块信息列表
        self.assertEqual(param_module_infos, expected_param_module_infos)

        # 创建一个包含两个 MLP 模块的序列模型 `model`
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        # 共享第一个 MLP 模块的输入投影层给第二个 MLP 模块的输入投影层
        model[0].in_proj = model[1].in_proj  # 共享的输入投影层
        # 获取模型 `model` 的所有参数列表 `params`
        params = list(model.parameters())
        # 获取参数模块信息 `param_module_infos`
        param_module_infos = _get_param_module_infos(params, model)
        # 断言 `param_module_infos` 的长度与 `params` 的长度相等
        self.assertEqual(len(param_module_infos), len(params))
        # 期望的参数模块信息列表，包含共享参数信息
        expected_param_module_infos = [
            ParamModuleInfo(model[0].in_proj, "weight", [model[1].in_proj], ["weight"]),
            ParamModuleInfo(mlp.in_proj, "bias", [], []),
            ParamModuleInfo(mlp.out_proj, "weight", [], []),
            ParamModuleInfo(mlp.out_proj, "bias", [], []),
        ]
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 如果 TEST_CUDA 不为真，则跳过测试，显示 "no cuda"
    def test_shard_tensor_parameters(self):
        # 使用奇数维度大小来测试不均匀分片
        model = nn.Sequential(*[MLP(3, dim_multiplier=3) for _ in range(3)])
        # 复制模型参数，以备后续比较
        orig_params = [param.detach().clone() for param in model.parameters()]
        # 对模型进行完全分片处理
        fully_shard(model)
        # 获取分片后的参数列表
        sharded_params = list(model.parameters())
        # 调用函数检查一维分片后的参数是否正确
        self._check_1d_sharded_parameters(orig_params, sharded_params)
    
        # 创建另一个模型，将第一个模型的某个属性复制到第二个模型
        model = nn.Sequential(*[MLP(3, dim_multiplier=3) for _ in range(3)])
        model[0].in_proj = model[1].in_proj
        # 再次复制模型参数，以备后续比较
        orig_params = [param.detach().clone() for param in model.parameters()]
        # 对模型进行完全分片处理
        fully_shard(model)
        # 获取分片后的参数列表
        sharded_params = list(model.parameters())
        # 调用函数检查一维分片后的参数是否正确
        self._check_1d_sharded_parameters(orig_params, sharded_params)
    
    def _check_1d_sharded_parameters(
        self, orig_params: List[nn.Parameter], sharded_params: List[nn.Parameter]
    ):
        # 断言原始参数和分片参数的长度应该相等
        self.assertEqual(len(orig_params), len(sharded_params))
        # 初始化一个全局设备网格，使用 "cuda" 设备和指定的世界大小
        global_mesh = init_device_mesh("cuda", (self.world_size,))
        # 遍历原始参数和分片参数的对应项
        for orig_param, sharded_param in zip(orig_params, sharded_params):
            # 断言分片参数的类型是 DTensor
            self.assertIsInstance(sharded_param, DTensor)
            # 断言分片参数的设备网格与全局网格相同
            self.assertEqual(sharded_param.device_mesh, global_mesh)
            # 断言分片参数的大小与原始参数相同
            self.assertEqual(sharded_param.size(), orig_param.size())
            # 断言分片参数的步长与原始参数相同
            self.assertEqual(sharded_param.stride(), orig_param.stride())
            # 断言分片参数的放置位置是一个 Shard(0) 元组
            self.assertEqual(sharded_param._spec.placements, (Shard(0),))
            # 将原始参数在指定维度上分成 self.world_size 个块
            chunks = torch.chunk(orig_param, self.world_size, dim=0)
            # 断言分片参数的本地张量与当前进程的块匹配
            self.assertEqual(sharded_param._local_tensor, chunks[self.rank])
# 定义一个测试类 TestFullyShardShardedParameterDTensor，继承自 FSDPTestMultiThread
class TestFullyShardShardedParameterDTensor(FSDPTestMultiThread):
    
    # 返回当前测试的世界大小，这里返回固定值 4
    @property
    def world_size(self) -> int:
        return 4

    # 在不支持 CUDA 的情况下跳过测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义测试方法 test_shard_dtensor_parameters
    def test_shard_dtensor_parameters(self):
        # 根据世界大小确定 dp_size 的值，如果世界大小大于 2，则为 2，否则为 1
        dp_size = 2 if self.world_size > 2 else 1
        # 初始化设备网格 global_mesh，使用 CUDA 并指定维度和网格维度名称
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        # 获取 dp_mesh 和 tp_mesh
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        
        # 创建一个多层感知机模型，输入大小为 9，维度乘数为 3
        model = MLP(9, dim_multiplier=3)
        # 复制模型参数，作为原始参数
        orig_params = [param.detach().clone() for param in model.parameters()]
        # 获取模型中参数的名称列表
        orig_param_names = [param_name for param_name, _ in model.named_parameters()]
        
        # 并行化模块，根据 tp_mesh 的设备网格
        parallelize_module(
            model,
            tp_mesh,
            {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
        )
        
        # 完全分片模型，根据 dp_mesh 的设备网格
        fully_shard(model, mesh=dp_mesh)
        
        # 获取分片后的模型参数列表
        sharded_params = list(model.parameters())
        
        # 断言分片后的参数数量与原始参数数量相同
        self.assertEqual(len(orig_params), len(sharded_params))
        
        # 遍历比较每一个原始参数与分片后参数的特性
        for orig_param_name, orig_param, sharded_param in zip(
            orig_param_names, orig_params, sharded_params
        ):
            # 断言分片后的参数类型为 DTensor
            self.assertIsInstance(sharded_param, DTensor)
            # 断言分片后的参数设备网格与全局网格相同
            self.assertEqual(sharded_param.device_mesh, global_mesh)
            # 断言分片后的参数大小与原始参数大小相同
            self.assertEqual(sharded_param.size(), orig_param.size())
            # 断言分片后的参数步长与原始参数步长相同
            self.assertEqual(sharded_param.stride(), orig_param.stride())
            
            # 根据参数名确定预期的放置方式
            if "in_proj" in orig_param_name:
                expected_placements = (Shard(0), Shard(0))
            elif "out_proj" in orig_param_name and "weight" in orig_param_name:
                expected_placements = (Shard(0), Shard(1))
            else:
                expected_placements = (Shard(0), Replicate())
            
            # 断言分片后的参数的规格（spec）与预期放置方式相同
            self.assertEqual(sharded_param._spec.placements, expected_placements)


# 定义另一个测试类 TestFullyShardLazyInit，同样继承自 FSDPTestMultiThread
class TestFullyShardLazyInit(FSDPTestMultiThread):
    
    # 返回当前测试的世界大小，这里返回固定值 2
    @property
    def world_size(self) -> int:
        return 2

    # 在不支持 CUDA 的情况下跳过测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_is_root(self):
        """
        Tests that ``_is_root`` is set correctly after lazy initialization.

        FSDP(model(
            0: MLP(FSDP(in_proj), FSDP(out_proj)),
            1: MLP(in_proj, out_proj),
        ))
        """
        # 创建一个包含两个MLP模型的序列模型
        model = nn.Sequential(MLP(8), MLP(8))
        # 对模型的第一个MLP子模块的in_proj进行全分片
        fully_shard(model[0].in_proj)
        # 对模型的第一个MLP子模块的out_proj进行全分片
        fully_shard(model[0].out_proj)
        # 对整个模型进行全分片，根状态获得model[1]
        fully_shard(model)  # root gets `model[1]`
        # 获取根状态
        root_state = fully_shard.state(model)
        # 对根状态进行懒初始化
        root_state._lazy_init()

        # 获取model[0]的in_proj状态
        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        # 获取model[0]的out_proj状态
        model0_out_proj_state = fully_shard.state(model[0].out_proj)
        # 断言根状态的_is_root属性为True
        self.assertTrue(root_state._is_root)
        # 断言model[0]的in_proj状态的_is_root属性为False
        self.assertFalse(model0_in_proj_state._is_root)
        # 断言model[0]的out_proj状态的_is_root属性为False
        self.assertFalse(model0_out_proj_state._is_root)

        # 获取所有状态上下文的所有状态
        all_states = root_state._state_ctx.all_states
        # 断言所有状态的数量为3
        self.assertEqual(len(all_states), 3)
        # 断言所有状态列表的内容与预期一致
        self.assertEqual(
            all_states, [root_state, model0_in_proj_state, model0_out_proj_state]
        )
    def test_fully_shard_module_and_param_fqns(self):
        """
        Tests that the module and parameter FQNs are computed correctly after
        lazy initialization.

        FSDP(model(
            0: MLP(FSDP(in_proj), FSDP(out_proj)),
            1: MLP(in_proj, out_proj),
        ))
        """
        # 创建一个包含两个MLP模块的序列神经网络模型
        model = nn.Sequential(MLP(8), MLP(8))
        
        # 对model的第一个MLP模块的in_proj和out_proj进行全分片处理
        fully_shard(model[0].in_proj)
        fully_shard(model[0].out_proj)
        
        # 对整个模型进行全分片处理，根节点得到model[1]
        fully_shard(model)  # root gets `model[1]`
        
        # 获取根节点的状态对象并进行延迟初始化
        root_state = fully_shard.state(model)
        root_state._lazy_init()

        # 获取根节点参数组
        root_param_group = root_state._fsdp_param_group
        self.assertIsNotNone(root_param_group)
        # 根节点的模块FQN为空字符串
        self.assertEqual(root_param_group._module_fqn, "")
        
        # 获取根节点参数的FQN集合
        root_param_fqns = {
            fsdp_param._param_fqn for fsdp_param in root_param_group.fsdp_params
        }
        # 根节点参数的FQN集合应包含以下参数
        self.assertEqual(
            root_param_fqns,
            {
                "1.in_proj.weight",
                "1.in_proj.bias",
                "1.out_proj.weight",
                "1.out_proj.bias",
            },
        )

        # 获取model[0]的in_proj模块状态对象
        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        model0_in_proj_param_group = model0_in_proj_state._fsdp_param_group
        self.assertIsNotNone(model0_in_proj_param_group)
        # model[0]的in_proj模块的模块FQN为"0.in_proj"
        self.assertEqual(model0_in_proj_param_group._module_fqn, "0.in_proj")
        
        # 获取model[0]的in_proj模块参数的FQN集合
        model0_in_proj_param_fqns = {
            fsdp_param._param_fqn
            for fsdp_param in model0_in_proj_param_group.fsdp_params
        }
        # model[0]的in_proj模块参数的FQN集合应包含以下参数
        self.assertEqual(
            model0_in_proj_param_fqns, {"0.in_proj.weight", "0.in_proj.bias"}
        )

        # 获取model[0]的out_proj模块状态对象
        model0_out_proj_state = fully_shard.state(model[0].out_proj)
        model0_out_proj_param_group = model0_out_proj_state._fsdp_param_group
        self.assertIsNotNone(model0_out_proj_param_group)
        # model[0]的out_proj模块的模块FQN为"0.out_proj"
        self.assertEqual(model0_out_proj_param_group._module_fqn, "0.out_proj")
        
        # 获取model[0]的out_proj模块参数的FQN集合
        model0_out_proj_param_fqns = {
            fsdp_param._param_fqn
            for fsdp_param in model0_out_proj_param_group.fsdp_params
        }
        # model[0]的out_proj模块参数的FQN集合应包含以下参数
        self.assertEqual(
            model0_out_proj_param_fqns, {"0.out_proj.weight", "0.out_proj.bias"}
        )

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_double_lazy_init(self):
        # 创建一个包含两个MLP模块的序列神经网络模型
        model = nn.Sequential(MLP(8), MLP(8))
        
        # 对model的第一个MLP模块的in_proj和out_proj进行全分片处理
        fully_shard(model[0].in_proj)
        fully_shard(model[0].out_proj)
        
        # 对整个模型进行全分片处理
        fully_shard(model)
        
        # 获取根节点状态对象
        root_state = fully_shard.state(model)
        # 获取model[0]的in_proj模块状态对象
        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        # 对model[0]的in_proj模块状态对象进行延迟初始化
        model0_in_proj_state._lazy_init()
        
        # 准备用于测试的正则表达式信息
        regex = (
            "FSDP state has already been lazily initialized for 0.in_proj\n"
            "FSDP requires running forward through the root module first"
        )
        
        # 断言应抛出RuntimeError，并且异常消息应符合正则表达式的格式
        with self.assertRaisesRegex(RuntimeError, regex):
            root_state._lazy_init()
# 定义一个测试类，继承自 FSDPTestMultiThread 类
class TestFullyShardMetaDeviceInit(FSDPTestMultiThread):

    # 定义一个属性，返回并行处理的世界大小为 4
    @property
    def world_size(self) -> int:
        return 4

    # 根据是否支持 CUDA 进行条件判断，如果不支持 CUDA 则跳过测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义测试方法，测试元设备初始化
    def test_meta_device_1d_init(self):
        # 获取默认的分布式组
        default_pg = torch.distributed.distributed_c10d._get_default_group()
        # 初始化设备网格，使用 CUDA，网格形状为 (默认组大小,)
        mesh = init_device_mesh("cuda", mesh_shape=(default_pg.size(),))

        # 测试均匀分片（8）和不均匀分片（3）
        for mlp_dim in (8, 3):
            # 在元设备上下文中创建模型，包括带缓冲区的 MLP 和普通 MLP
            with torch.device("meta"):
                model = nn.Sequential(MLP(mlp_dim, with_buffer=True), MLP(mlp_dim))
                # 检查模型参数是否在元设备上
                for param in model.parameters():
                    self.assertEqual(param.device, torch.device("meta"))
                # 对模型的第一个和第二个层进行完全分片
                fully_shard(model[0], mesh=mesh)
                fully_shard(model[1], mesh=mesh)
                fully_shard(model, mesh=mesh)
            # 再次检查模型参数是否在元设备上
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            # 执行自定义的测试方法，将模型参数转换为空并重置，传入模型、网格和 mlp_dim
            self._test_to_empty_and_reset_parameters(model, mesh, mlp_dim)

        # 测试在元设备上下文中调用 `fully_shard` 并且 `init_device_mesh` 调用仍然有效
        mlp_dim = 8
        with torch.device("meta"):
            model = nn.Sequential(MLP(mlp_dim, with_buffer=True), MLP(mlp_dim))
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            # 对模型的所有模块进行完全分片
            for module in (model[0], model[1], model):
                fully_shard(module)
        # 再次检查模型参数是否在元设备上
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("meta"))
        # 执行自定义的测试方法，将模型参数转换为空并重置，传入模型、网格和 mlp_dim
        self._test_to_empty_and_reset_parameters(model, mesh, mlp_dim)
    # 测试 meta 设备的 2D 初始化函数
    def test_meta_device_2d_init(self):
        # 断言全局变量 self.world_size 至少为 4，如果不满足条件则抛出异常并显示当前 self.world_size 的值
        assert self.world_size >= 4, f"{self.world_size}"
        # 设置分布式并行处理的大小为 2
        dp_size = 2
        # 使用 init_device_mesh 函数初始化全局网格，使用 CUDA 设备，网格大小为 (dp_size, self.world_size // dp_size)，网格维度命名为 ("dp", "tp")
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        # 从全局网格中获取 dp 和 tp 网格
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]

        # 测试均匀分片 (8) 和不均匀分片 (3) 的情况
        for mlp_dim in (8, 3):
            # 使用 torch.device("meta") 设备上下文环境
            with torch.device("meta"):
                # 创建一个具有缓冲区的 MLP 模型
                model = MLP(mlp_dim, with_buffer=True)
                # 验证模型所有参数的设备是否为 torch.device("meta")
                for param in model.parameters():
                    self.assertEqual(param.device, torch.device("meta"))
                # 将模型并行化，使用 tp_mesh 网格，指定各自的并行化方式
                parallelize_module(
                    model,
                    tp_mesh,
                    {"in_proj": ColwiseParallel(), "out_proj": RowwiseParallel()},
                )
                # 再次验证模型所有参数的设备是否为 torch.device("meta")
                for param in model.parameters():
                    self.assertEqual(param.device, torch.device("meta"))
                # 对模型的 in_proj 和 out_proj 使用完全分片，使用 dp_mesh 网格
                fully_shard(model.in_proj, mesh=dp_mesh)
                fully_shard(model.out_proj, mesh=dp_mesh)
                # 对整个模型使用完全分片，使用 dp_mesh 网格
                fully_shard(model, mesh=dp_mesh)
            # 最后再次验证模型所有参数的设备是否为 torch.device("meta")
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            # 执行 _test_to_empty_and_reset_parameters 测试函数，测试模型在给定全局网格和 mlp_dim 下的行为
            self._test_to_empty_and_reset_parameters(model, global_mesh, mlp_dim)

    # 测试将模型参数清空并重置的函数
    def _test_to_empty_and_reset_parameters(
        self, model: nn.Module, mesh: DeviceMesh, mlp_dim: int
    ):
        # 检查是否能将模型材料化到 GPU 上，并使用空值
        device = torch.device("cuda", torch.cuda.current_device())
        model.to_empty(device=device)
        # 验证模型所有参数的设备是否为指定的 device
        for param in model.parameters():
            self.assertEqual(param.device, device)
        # 使用 Adam 优化器优化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 针对每个模块的参数和缓冲区，检查 `reset_parameters()` 方法是否正确初始化值
        const = 1337
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            tensor.detach().fill_(const)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        # 验证模型参数是否已经被正确重置
        for param in model.parameters():
            local_tensor = param.to_local()
            if local_tensor.numel() > 0:
                self.assertNotEqual(local_tensor, torch.ones_like(local_tensor) * const)
        # 验证模型缓冲区是否已经被正确重置
        for buffer in model.buffers():
            self.assertNotEqual(buffer, torch.ones_like(buffer) * const)

        # 检查是否能够正常运行一次迭代
        inp = torch.randn((4, mlp_dim), device="cuda")
        model(inp).sum().backward()
        optim.step()

    # 如果未开启 CUDA 测试，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义一个测试方法，用于测试无效的元设备初始化
    def test_invalid_meta_device_init(self):
        # 获取默认的分布式组
        default_pg = torch.distributed.distributed_c10d._get_default_group()
        # 使用 "cuda" 初始化设备网格，网格形状为分布式组的大小
        mesh = init_device_mesh("cuda", mesh_shape=(default_pg.size(),))
        # 定义 MLP 的维度
        mlp_dim = 8
        # 使用 "meta" 设备上下文创建一个包含两个 MLP 层的序列模型
        with torch.device("meta"):
            # 创建包含缓冲区的 MLP 和普通 MLP
            model = nn.Sequential(MLP(mlp_dim, with_buffer=True), MLP(mlp_dim))
            # 验证模型的每个参数都在 "meta" 设备上
            for param in model.parameters():
                self.assertEqual(param.device, torch.device("meta"))
            # 将模型的第一层和第二层完全分片，使用给定的设备网格
            fully_shard(model[0], mesh=mesh)
            fully_shard(model[1], mesh=mesh)
            # 对整个模型进行完全分片，使用给定的设备网格
            fully_shard(model, mesh=mesh)
        # 创建一个在 "cuda" 设备上的输入张量
        inp = torch.randn((4, mlp_dim), device="cuda")
        # 定义一个错误的正则表达式，用于捕获运行时错误，并指出哪些参数仍然在 "meta" 设备上
        error_regex = (
            "FSDP parameters should be materialized from meta device before training, "
            "but the following were still on meta device: "
            r"\['0.in_proj.weight', '0.in_proj.bias', '0.out_proj.weight', '0.out_proj.bias'\]"
        )
        # 验证在运行时捕获到 RuntimeError，并匹配特定的错误信息正则表达式
        with self.assertRaisesRegex(RuntimeError, error_regex):
            model(inp)

    # 根据是否支持 CUDA 测试跳过单元测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
# 定义一个测试类 TestFullyShardProcessGroupInit，继承自 FSDPTestMultiThread 类
class TestFullyShardProcessGroupInit(FSDPTestMultiThread):

    # 定义一个属性方法 world_size，返回整数值 4，表示进程组的总大小为 4
    @property
    def world_size(self) -> int:
        return 4

    # 使用 unittest 模块的 skipIf 装饰器，如果 TEST_CUDA 为 False，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义一个测试方法，用于初始化分布式处理组
    def test_1d_process_group_init(self):
        # 断言分布式环境中的世界大小为4，如果不是则抛出异常
        assert self.world_size == 4, f"{self.world_size}"

        # 为了方便起见，使用设备网格的基础设施来构造分布式处理组
        # （在实际应用中，训练器可能会手动通过 `new_group()` 方法进行）
        dp_size = 2
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        ref_dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]

        # 从参考的设备网格获取第一个组
        dp_pg = ref_dp_mesh.get_group(0)

        # 验证 `from_group()` 方法的正确性
        dp_mesh = DeviceMesh.from_group(dp_pg, "cuda", mesh_dim_names=("dp",))

        # 只比较网格张量，不比较 `DeviceMesh` 对象本身，
        # 因为参考网格有一个父网格，而 `from_group` 方法创建的网格没有
        self.assertEqual(dp_mesh.mesh, ref_dp_mesh.mesh)
        self.assertEqual(dp_mesh._coordinate_on_dim, ref_dp_mesh._coordinate_on_dim)
        self.assertEqual(dp_mesh._dim_group_infos, ref_dp_mesh._dim_group_infos)

        # 在 DP 网格上检查 1D FSDP（Fully Sharded Data Parallel）的前向和反向传播一致性
        # 注意: 这里不能使用基于 2D DTensor 的训练，因为从 `from_group` 方法得到的 DP 网格
        # 不遵循其父网格。
        torch.manual_seed(42)
        mlp_dim = 8
        ref_model = MLP(mlp_dim)

        # 将参考模型的参数广播到所有进程
        for param in ref_model.parameters():
            dist.broadcast(param.detach(), src=0)

        # 深拷贝参考模型
        model = copy.deepcopy(ref_model)

        # 使用参考 DP 网格并行化测试模型
        for module in (ref_model.in_proj, ref_model.out_proj, ref_model):
            fully_shard(module, mesh=ref_dp_mesh)

        # 使用来自分布式处理组的新 DP 网格并行化测试模型
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module, mesh=dp_mesh)

        # 确保 TP (Tensor Processor) 排名具有相同的输入
        inp = torch.randn((4, mlp_dim), device="cuda")
        if self.rank in (0, 1):
            dist.broadcast(inp, src=0, group=tp_mesh.get_group(0))
        elif self.rank in (2, 3):
            dist.broadcast(inp, src=2, group=tp_mesh.get_group(0))

        # 计算并反向传播参考模型的损失
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()

        # 计算并反向传播测试模型的损失
        loss = model(inp).sum()
        loss.backward()

        # 断言测试模型的损失与参考模型的损失相等
        self.assertEqual(loss, ref_loss)

        # 逐个比较测试模型和参考模型的参数和梯度
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            # 由于网格不同，不能直接比较 `DTensor`，因为参考参数的网格有一个父网格，
            # 而测试参数的网格没有
            self.assertEqual(param.to_local(), ref_param.to_local())
            self.assertEqual(param.device_mesh.mesh, ref_param.device_mesh.mesh)
            self.assertEqual(param.grad.to_local(), ref_param.grad.to_local())
            self.assertEqual(
                param.grad.device_mesh.mesh, ref_param.grad.device_mesh.mesh
            )
# 定义一个测试类 `TestFullyShardHSDPBroadcast`，继承自 `FSDPTestMultiThread` 类
class TestFullyShardHSDPBroadcast(FSDPTestMultiThread):
    # 定义一个属性方法 `world_size`，返回整数值 4，表示测试中使用的进程数
    @property
    def world_size(self) -> int:
        return 4

    # 使用 unittest 模块的装饰器 `skipIf`，在不满足条件 `not TEST_CUDA` 时跳过测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义一个测试方法，用于测试跨副本广播功能
    def test_hsdp_broadcast_across_replicas(self):
        # 定义每个副本中的分片数量和副本数量
        shard_size, replicate_size = 2, 2
        # 初始化设备网格，使用CUDA，指定副本和分片的维度名称
        mesh = init_device_mesh(
            "cuda", (replicate_size, shard_size), mesh_dim_names=("replicate", "shard")
        )
        # 创建模型参数对象
        model_args = ModelArgs()
        # 创建Transformer模型实例
        model = Transformer(model_args)
        
        # 添加一个缓冲区，用于展示此流程也适用于缓冲区
        model.register_buffer("buf", torch.randn((model_args.dim,)))
        
        # 遍历模型中的每个模块
        for module in model.modules():
            # 如果模块是TransformerBlock类型
            if isinstance(module, TransformerBlock):
                # 对模块进行完全分片，使用指定的设备网格
                fully_shard(module, mesh=mesh)
        
        # 对整个模型进行完全分片，使用指定的设备网格
        fully_shard(model, mesh=mesh)

        # 仅在副本网格的rank 0上保留模型状态
        if mesh.get_local_rank("replicate") > 0:
            # 对模型的参数和缓冲区进行detach和填充操作
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                tensor.detach().fill_(1337)

        # 检查副本是否不同
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            # 将张量转换为本地张量（如果是DTensor类型）
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            # 创建与副本大小相同的空张量列表
            local_tensor_list = [
                torch.empty_like(local_tensor) for _ in range(mesh["replicate"].size())
            ]
            # 使用分布式操作在副本网格的组内进行全局收集
            dist.all_gather(
                local_tensor_list, local_tensor, group=mesh.get_group("replicate")
            )
            # 检查所有副本的本地张量是否不相同
            for other_local_tensor in local_tensor_list[1:]:
                self.assertEqual(other_local_tensor.shape, local_tensor_list[0].shape)
                self.assertNotEqual(other_local_tensor, local_tensor_list[0])

        # 从副本网格的rank 0广播
        replicate_group = mesh.get_group("replicate")
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            # 获取广播源的rank
            src_rank = dist.get_process_group_ranks(replicate_group)[0]
            # 使用分布式操作广播张量
            torch.distributed.broadcast(
                tensor.to_local() if isinstance(tensor, DTensor) else tensor,
                src=src_rank,
                group=replicate_group,
            )

        # 检查所有副本的张量是否相同
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            # 将张量转换为本地张量（如果是DTensor类型）
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            # 创建与副本大小相同的空张量列表
            local_tensor_list = [
                torch.empty_like(local_tensor) for _ in range(mesh["replicate"].size())
            ]
            # 使用分布式操作在副本网格的组内进行全局收集
            dist.all_gather(
                local_tensor_list, local_tensor, group=mesh.get_group("replicate")
            )
            # 检查所有副本的本地张量是否相同
            for other_local_tensor in local_tensor_list[1:]:
                self.assertEqual(other_local_tensor, local_tensor_list[0])

        # 检查能否在不出错的情况下运行一个迭代
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        model(inp).sum().backward()
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```