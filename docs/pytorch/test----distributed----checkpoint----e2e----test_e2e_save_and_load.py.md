# `.\pytorch\test\distributed\checkpoint\e2e\test_e2e_save_and_load.py`

```py
# Owner(s): ["oncall: distributed"]

# 引入时间模块
import time
# 引入数据类相关模块
from dataclasses import dataclass, field
# 引入枚举相关模块
from enum import auto, Enum
# 引入偏函数相关模块
from functools import partial
# 引入字节流相关模块
from io import BytesIO
# 引入类型提示相关模块
from typing import Any, Dict, List

# 引入PyTorch主要模块
import torch
# 引入分布式相关模块
import torch.distributed as dist
# 引入分布式检查点相关模块
import torch.distributed.checkpoint as DCP
# 引入分布式检查点状态字典保存器相关模块
import torch.distributed.checkpoint.state_dict_saver as saver
# 引入神经网络相关模块
import torch.nn as nn
import torch.nn.functional as F
# 引入设备网格初始化相关模块
from torch.distributed._tensor.device_mesh import init_device_mesh
# 引入模型状态字典相关模块
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
)
# 引入从指定键加载状态字典相关模块
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
# 引入检查点异常相关模块
from torch.distributed.checkpoint.utils import CheckpointException
# 引入分布式c10d相关模块
from torch.distributed.distributed_c10d import ReduceOp
# 引入全分片数据并行相关模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# 引入分片策略相关模块
from torch.distributed.fsdp.api import ShardingStrategy
# 引入张量并行相关模块
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
# 引入分布式数据并行相关模块
from torch.nn.parallel import DistributedDataParallel

# 引入内部测试工具相关模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
# 引入分布式张量公共测试相关模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
# 引入临时目录测试工具相关模块
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
# 引入常见状态字典验证相关模块
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


# 简单且无聊的模型
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # 定义神经网络层
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    # 定义前向传播方法
    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    # 返回一个随机生成的8x8张量
    def get_input(self):
        return torch.rand(8, 8, device="cuda")


# 测试状态对象
class TestStatefulObj:
    def __init__(self):
        # 在CUDA设备上创建一个随机张量
        self.data = torch.rand(10, 10, device="cuda")

    # 返回状态字典，包含data字段
    def state_dict(self):
        return {"data": self.data}

    # 加载状态字典中的data字段
    def load_state_dict(self, state_dict):
        self.data = state_dict["data"]

    # 比较两个TestStatefulObj对象的data张量是否相等
    def __eq__(self, other):
        return torch.equal(self.data, other.data)


# 模型类型枚举
class ModelType(Enum):
    FSDP = auto()     # 全分片数据并行
    HSDP = auto()     # 半分片数据并行
    FSDP_TP = auto()  # 全分片数据并行（第三方）
    DDP = auto()      # 数据并行
    NONE = auto()     # 无并行化


# 测试训练状态数据类
@dataclass
class TestTrainState:
    step: int = 0               # 当前步数
    current_loss: float = -1    # 当前损失值
    losses: List[float] = field(default_factory=list)  # 损失列表
    # 返回当前对象的状态字典，包含当前步数、当前损失和损失历史记录
    def state_dict(self) -> Dict[str, Any]:
        # 创建一个字节流对象，用于保存损失历史记录
        loss_bytes = BytesIO()
        # 将损失历史记录保存到字节流中
        torch.save(self.losses, loss_bytes)
        # 返回包含当前状态信息的字典
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),  # 当前步数的张量表示
            "current_loss": torch.tensor(self.current_loss, dtype=torch.float32),  # 当前损失的张量表示
            "losses": loss_bytes,  # 损失历史记录的字节流对象
        }

    # 从给定的状态字典中加载状态，更新当前对象的属性
    def load_state_dict(self, state_dict) -> None:
        # 加载步数并转换为整数
        self.step = state_dict["step"].item()
        # 加载当前损失并转换为浮点数
        self.current_loss = state_dict["current_loss"].item()
        # 将损失历史记录的字节流对象定位到起始位置
        state_dict["losses"].seek(0)
        # 加载损失历史记录并更新对象属性
        self.losses = torch.load(state_dict["losses"])

    # 判断当前对象与另一个对象是否相等
    def __eq__(self, other):
        # 比较当前步数、当前损失和损失历史记录是否都相等
        return (
            self.step == other.step
            and self.current_loss == other.current_loss
            and self.losses == other.losses
        )
# 定义一个名为 `_train` 的函数，用于训练给定的模型。
def _train(model, optim, train_steps=1):
    # 设定随机种子为0，保证可重复性
    torch.manual_seed(0)
    # 初始化损失变量为 None
    loss = None

    # 创建一个测试训练状态对象
    train_state = TestTrainState()

    # 循环执行训练步骤
    for _ in range(train_steps):
        # 获取模型输入并计算损失的总和
        loss = model(model.get_input()).sum()
        # 反向传播计算梯度
        loss.backward()

        # 在实际训练中，通常会在多个分布式处理单元中同步损失。
        # 这里只是为了测试目的进行模拟。
        train_state.step += 1
        # 生成一个随机的当前损失值，并存入训练状态对象
        train_state.current_loss = torch.rand(1).item()
        train_state.losses.append(train_state.current_loss)

        # 根据优化器更新模型参数
        optim.step()
        # 清空梯度，准备处理下一个迭代的梯度
        optim.zero_grad()

    # 返回最终的损失值和训练状态对象
    return loss, train_state


# 定义一个测试类 TestE2ESaveAndLoad，继承自 DTensorTestBase 和 VerifyStateDictMixin
class TestE2ESaveAndLoad(DTensorTestBase, VerifyStateDictMixin):

    # 定义一个属性方法 backend，返回字符串 "cpu:gloo,cuda:nccl"
    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"

    # 定义一个内部方法 _create_model，用于创建模型
    def _create_model(self, compile, model_type, state_dict_options=None):
        # 创建一个使用 CUDA 的测试模型
        dummy_model = TestDummyModel().cuda()

        # 断言模型类型在 ModelType 枚举中，否则抛出异常
        assert model_type in ModelType, f"{model_type} is not supported."
        
        # 根据不同的模型类型进行模型初始化
        if model_type == ModelType.FSDP:
            # 初始化设备网格并创建 FSDP 模型
            device_mesh = init_device_mesh(self.device_type, (self.world_size,))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
            )
        elif model_type == ModelType.HSDP:
            # 初始化设备网格并创建 FSDP 模型，使用混合分片策略
            device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        elif model_type == ModelType.FSDP_TP:
            # 初始化设备网格，使用二维网格并并行化网络模块
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            tp_mesh = mesh_2d["tp"]
            dp_mesh = mesh_2d["dp"]
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            # 并行化模块并创建 FSDP 模型
            model = parallelize_module(dummy_model, tp_mesh, parallelize_plan)
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        elif model_type == ModelType.DDP:
            # 创建分布式数据并行模型
            model = DistributedDataParallel(dummy_model)
            model.get_input = partial(TestDummyModel.get_input, model)
        else:
            # 默认情况下使用普通的测试模型
            model = dummy_model

        # 如果需要编译模型，调用 torch.compile 进行模型编译
        if compile:
            # TODO: 当启用动态形状支持时，将 dynamic=True 设为 True
            # model = torch.compile(model)
            model = torch.compile(model, dynamic=False)

        # 根据模型获取相应的优化器
        optim = self._optim(model)
        
        # 如果模型类型不是 ModelType.NONE，调用 _patch_model_state_dict 和 _patch_optimizer_state_dict 进行模型状态字典的修补
        if model_type is not ModelType.NONE:
            _patch_model_state_dict(model, options=state_dict_options)
            _patch_optimizer_state_dict(
                model, optimizers=optim, options=state_dict_options
            )

        # 返回创建的模型和优化器
        return model, optim

    # 定义一个内部方法 _optim，用于创建优化器
    def _optim(self, model):
        # 返回一个 Adam 优化器，学习率设为 0.1
        return torch.optim.Adam(model.parameters(), lr=0.1)

    # 使用通信装饰器，定义一个测试方法
    @with_comms
    # 如果 GPU 数量小于 4，跳过该测试
    @skip_if_lt_x_gpu(4)
    # 使用临时目录装饰器，定义一个测试方法
    @with_temp_dir
    # 参数化测试方法，测试 compile 参数为 True 和 False 时的情况
    @parametrize("compile", [True, False])
    # 使用@parametrize装饰器，为test_e2e方法提供多组参数化测试，以覆盖不同的模型类型
    @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP, ModelType.DDP])
    def test_e2e(self, compile, model_type):
        # 调用_run_e2e_test方法执行端到端测试
        self._run_e2e_test(compile, model_type)

    # 使用@with_comms装饰器，确保测试方法中可以使用通信相关的功能
    # 使用@skip_if_lt_x_gpu(4)装饰器，仅当GPU数量大于等于4时才执行该测试
    @with_comms
    @skip_if_lt_x_gpu(4)
    # 使用@with_temp_dir装饰器，为测试方法提供临时目录
    # 使用@parametrize装饰器，为test_e2e_async_cached方法提供两组参数化测试，测试缓存阶段性状态字典的不同情况
    @parametrize("cache_staged_state_dict", [False, True])
    def test_e2e_async_cached(self, cache_staged_state_dict):
        # 调用_run_e2e_test方法执行端到端异步缓存测试
        self._run_e2e_test(
            compile=False,
            model_type=ModelType.FSDP,
            async_op=True,
            cache_staged_state_dict=cache_staged_state_dict,
        )

    # _run_e2e_test方法，执行端到端测试的核心逻辑
    def _run_e2e_test(
        self, compile, model_type, async_op=False, cache_staged_state_dict=False
    ):
        # 创建非分布式模型，进行简单的训练
        model, optim = self._create_model(compile, ModelType.NONE)
        _train(model, optim, train_steps=2)

        # 创建分布式模型，并进行训练，获取原始的训练状态
        dist_model, dist_optim = self._create_model(compile, model_type)
        _, original_train_state = _train(dist_model, dist_optim, train_steps=2)

        # 创建一个原始的可保存/加载的对象
        original_stateful_obj = TestStatefulObj()

        # 准备要保存的状态字典
        sd = {
            "model": dist_model,
            "optimizer": dist_optim,
            "s": original_stateful_obj,
            "train_state": original_train_state,
        }

        # 如果是异步操作，则使用文件系统写入器进行异步保存
        if async_op:
            writer = DCP.FileSystemWriter(
                self.temp_dir, cache_staged_state_dict=cache_staged_state_dict
            )
            f = saver.async_save(sd, storage_writer=writer)
            t = time.monotonic()
            # 等待保存操作完成
            while not f.done():
                time.sleep(1)
                print(f"still waiting... {time.monotonic() - t}")

            f.result()
        else:
            # 否则直接保存状态字典到指定的检查点目录
            DCP.save(sd, checkpoint_id=self.temp_dir)

        # 创建加载后的状态对象
        loaded_stateful_obj = TestStatefulObj()
        loaded_train_state = TestTrainState()
        dist_model, dist_optim = self._create_model(compile, model_type)

        # 加载保存的状态字典到分布式模型中
        DCP.load(
            state_dict={
                "model": dist_model,
                "optimizer": dist_optim,
                "s": loaded_stateful_obj,
                "train_state": loaded_train_state,
            },
            checkpoint_id=self.temp_dir,
        )

        # 断言加载后的状态对象与原始对象相等
        self.assertEqual(original_stateful_obj, loaded_stateful_obj)
        self.assertEqual(original_train_state, loaded_train_state)

        # 在两个模型上各训练一步，然后断言损失值相等
        loss, _ = _train(model, optim, train_steps=1)
        dist_loss, _ = _train(dist_model, dist_optim, train_steps=1)
        self.assertEqual(loss, dist_loss)

        # 获取模型和优化器的状态字典，并进行验证
        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        model_sd, optim_sd = get_state_dict(model, optim)

        # 使用_verify_msd方法验证模型状态字典
        self._verify_msd(model_sd, dist_msd)
        # 使用_verify_osd_by_load方法验证优化器状态字典
        self._verify_osd_by_load(model, optim, self._optim(model), dist_osd)

    # 使用@with_comms装饰器，确保测试方法中可以使用通信相关的功能
    # 使用@with_temp_dir装饰器，为测试方法提供临时目录
    @skip_if_lt_x_gpu(4)
    def test_different_ordered_state_dict_keys(self):
        """Tests that the order of keys in the state dict does not matter when loading
        If order was not accounted for, the following test would cause a deadlock.
        """

        world_size = self.world_size  # 获取当前测试环境的进程数

        class Foo:
            def state_dict(self):
                return {}  # 返回空字典，模拟对象状态的保存

            def load_state_dict(self, state_dict):
                tl = [
                    torch.ones(2, dtype=torch.int64, device="cuda")
                    for _ in range(world_size)
                ]  # 创建包含 world_size 个 CUDA 上的整数张量的列表

                t = (
                    torch.arange(2, dtype=torch.int64, device="cuda")
                    + 1
                    + 2 * dist.get_rank()
                )  # 在 CUDA 上创建一个整数张量 t，根据当前进程的排名调整值

                dist.all_gather(tl, t, async_op=False)  # 使用分布式通信收集所有进程上的张量数据

        class Bar:
            def state_dict(self):
                return {}  # 返回空字典，模拟对象状态的保存

            def load_state_dict(self, state_dict):
                tensor = (
                    torch.arange(2, dtype=torch.int64, device="cuda")
                    + 1
                    + 2 * dist.get_rank()
                )  # 在 CUDA 上创建一个整数张量 tensor，根据当前进程的排名调整值

                dist.all_reduce(tensor, op=ReduceOp.SUM)  # 使用分布式通信对所有进程上的张量进行求和操作

        if self.rank == 0:
            sd = {
                "A": Foo(),  # 创建包含 Foo 对象的字典，键为 "A"
                "B": Bar(),  # 创建包含 Bar 对象的字典，键为 "B"
            }
        else:
            sd = {
                "B": Bar(),  # 创建包含 Bar 对象的字典，键为 "B"
                "A": Foo(),  # 创建包含 Foo 对象的字典，键为 "A"
            }

        DCP.save(sd, checkpoint_id=self.temp_dir)  # 调用 DCP 对象的保存方法，保存状态字典 sd 到临时目录
        DCP.load(sd, checkpoint_id=self.temp_dir)  # 调用 DCP 对象的加载方法，从临时目录加载状态字典 sd

    @with_temp_dir
    def test_no_dist(self):
        # since comm's are not initialized in this method, `no_dist`
        # is assumed False
        DCP.save({}, checkpoint_id=self.temp_dir)  # 调用 DCP 对象的保存方法，保存空字典到临时目录
        DCP.load({}, checkpoint_id=self.temp_dir)  # 调用 DCP 对象的加载方法，从临时目录加载空字典

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    # 定义测试函数，测试部分加载功能
    def test_partial_load(self):
        # 创建模型和优化器，不进行编译，模型类型为ModelType.NONE
        model, optim = self._create_model(compile=False, model_type=ModelType.NONE)
        # 使用_train函数对模型和优化器进行训练两步
        _train(model, optim, train_steps=2)

        # 创建分布式模型和优化器，不进行编译，模型类型为ModelType.FSDP
        dist_model, dist_optim = self._create_model(
            compile=False, model_type=ModelType.FSDP
        )
        # 使用_train函数对分布式模型和优化器进行训练两步
        _train(dist_model, dist_optim, train_steps=2)

        # 将分布式模型和优化器保存到临时目录
        DCP.save(
            {"model": dist_model, "optimizer": dist_optim}, checkpoint_id=self.temp_dir
        )

        # 重新创建分布式模型，不进行编译，模型类型为ModelType.FSDP
        dist_model, _ = self._create_model(compile=False, model_type=ModelType.FSDP)
        # 从临时目录加载模型参数到dist_model
        DCP.load({"model": dist_model}, checkpoint_id=self.temp_dir)

        # 获取分布式模型的模型状态字典
        dist_msd = get_model_state_dict(dist_model)
        # 获取普通模型的模型状态字典
        model_sd = get_model_state_dict(model)
        # 验证两个模型状态字典是否相同
        self._verify_msd(model_sd, dist_msd)

        # 另一种方式加载模型参数，加载 "model" 键对应的状态字典，从临时目录
        loaded_model_sd = _load_state_dict_from_keys(
            "model", checkpoint_id=self.temp_dir
        )["model"]
        # 使用_verify_msd函数验证普通模型的状态字典和加载的状态字典是否相同，同时将结果下放到CPU
        self._verify_msd(model_sd, loaded_model_sd, offload_to_cpu=True)

        # 加载优化器状态，加载 "optimizer.state" 键对应的状态字典，从临时目录
        loaded_optim_state = _load_state_dict_from_keys(
            "optimizer.state", checkpoint_id=self.temp_dir
        )["optimizer"]["state"]
        # 验证加载的优化器状态字典中不包含 "param_groups" 键
        self.assertNotIn("param_groups", loaded_optim_state)
        # 遍历分布式优化器的状态字典中的每个键值对
        for k, v in dist_optim.state_dict()["state"].items():
            # 对比加载的优化器状态字典中每个键值对的 "exp_avg"、"exp_avg_sq" 和 "step" 字段的张量，
            # 并将结果下放到CPU
            for optim_key in ["exp_avg", "exp_avg_sq", "step"]:
                self._compare_tensor(
                    loaded_optim_state[k][optim_key], v[optim_key], offload_to_cpu=True
                )

    # 使用装饰器，配置通信环境
    @with_comms
    # 如果GPU数量小于4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 使用临时目录作为测试的上下文
    @with_temp_dir
    # 定义测试函数，测试覆盖写入
    def test_overwrite(self):
        # 创建两个大小为10的随机张量 t1 和 t2
        t1, t2 = torch.randn(10), torch.randn(10)
        # 将张量 t1 保存为 "random" 键对应的数据，到临时目录
        DCP.save({"random": t1}, checkpoint_id=self.temp_dir)
        # 使用覆盖写入模式，将张量 t2 保存为 "random" 键对应的数据，到临时目录
        DCP.save(
            {"random": t2},
            storage_writer=DCP.FileSystemWriter(self.temp_dir, overwrite=True),
        )

        # 创建一个包含 "random" 键的字典 sd，值为大小为10的零张量
        sd = {"random": torch.zeros(10)}
        # 从临时目录加载 "random" 键对应的数据到字典 sd
        DCP.load(sd, checkpoint_id=self.temp_dir)

        # 断言 sd["random"] 是否与张量 t2 全部接近
        self.assertTrue(torch.allclose(sd["random"], t2))

        # 使用断言验证，在覆盖写入模式下再次保存相同键的数据，会抛出 CheckpointException 异常
        with self.assertRaisesRegex(
            CheckpointException, ".*Checkpoint already exists.*"
        ):
            DCP.save(
                {"random": t2},
                storage_writer=DCP.FileSystemWriter(self.temp_dir, overwrite=False),
            )
# TestNoCPU 类，继承自 DTensorTestBase，用于测试不使用 CPU 的情况
class TestNoCPU(DTensorTestBase):

    # backend 属性，返回字符串 "nccl"
    @property
    def backend(self):
        return "nccl"

    # test_no_cpu 方法，使用装饰器 with_comms 包装
    @with_comms
    def test_no_cpu(self):
        # 使用 assertRaisesRegex 断言捕获 AssertionError 异常，并验证异常消息
        with self.assertRaisesRegex(
            AssertionError, r"A CPU backend must be enabled for async save;.*?"
        ):
            # 调用 saver.async_save 方法，传入空字典并获取结果
            f = saver.async_save({})
            f.result()


# TestInitStateDict 类，继承自 DTensorTestBase，用于测试初始化状态字典的功能
class TestInitStateDict(DTensorTestBase):

    # test_init_state_dict 方法，使用装饰器 with_temp_dir 包装
    @with_temp_dir
    def test_init_state_dict(self):
        # 获取临时目录的路径
        temp_dir = self.temp_dir
        # 创建 TestDummyModel 实例
        model = TestDummyModel()
        # 使用 Adam 优化器初始化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        # 准备要保存的状态字典，包括模型和优化器的状态
        state_dict_to_save = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optim),
        }
        # 使用 DCP.save 方法保存状态字典到指定的检查点目录
        DCP.save(state_dict_to_save, checkpoint_id=temp_dir)

        # 设置随机种子为 0，创建另一个 TestDummyModel 实例
        torch.manual_seed(0)
        model_2 = TestDummyModel()
        # 更改优化器的学习率，这里不是一个张量
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.2)

        # 获取第二个模型的状态字典和优化器状态字典
        msd = get_model_state_dict(model_2)
        osd = get_optimizer_state_dict(model_2, optim_2)

        # 准备要加载的状态字典
        state_dict_to_load = {"model": msd, "optimizer": osd}
        # 使用 DCP.load 方法加载状态字典到指定的检查点目录
        DCP.load(state_dict_to_load, checkpoint_id=temp_dir)

        # 验证两个变量指向相同的内存对象，以证明 DCP 是原地加载的
        self.assertTrue(msd is state_dict_to_load["model"])
        self.assertTrue(osd is state_dict_to_load["optimizer"])

        # set_state_dict 调用 load_state_dict 加载模型和优化器的状态字典
        # 预期 optim_2.param_groups 中的学习率现在应为 0.1 而不是 0.2
        set_state_dict(
            model_2,
            optim_2,
            model_state_dict=state_dict_to_load["model"],
            optim_state_dict=state_dict_to_load["optimizer"],
        )
        # 验证加载后的模型状态字典和优化器状态字典
        self.assertEqual(msd, get_model_state_dict(model_2))
        self.assertEqual(osd, get_optimizer_state_dict(model_2, optim_2))
        self.assertEqual(optim_2.param_groups[0]["lr"], 0.1)


# 实例化 TestE2ESaveAndLoad 类的参数化测试
instantiate_parametrized_tests(TestE2ESaveAndLoad)
# 如果当前文件作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```