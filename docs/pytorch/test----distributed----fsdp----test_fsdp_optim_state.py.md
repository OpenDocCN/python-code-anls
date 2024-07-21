# `.\pytorch\test\distributed\fsdp\test_fsdp_optim_state.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和库
import bisect  # 导入 bisect 模块，用于二分查找
import sys  # 导入 sys 模块，提供对解释器相关的操作访问
from copy import deepcopy  # 导入 deepcopy 函数，用于深拷贝对象
from enum import auto, Enum  # 导入 auto 和 Enum 类，用于创建枚举类型
from typing import Any, Callable, Dict, List, Optional, Tuple, Type  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch import distributed as dist  # 导入 torch.distributed 模块的 distributed 别名
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 导入 ShardedTensor 类
from torch.distributed._state_dict_utils import _gather_state_dict  # 导入 _gather_state_dict 函数
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,  # 导入 _CHECKPOINT_WRAPPED_MODULE 变量
    apply_activation_checkpointing,  # 导入 apply_activation_checkpointing 函数
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 FSDP 类
from torch.distributed.fsdp.api import ShardingStrategy  # 导入 ShardingStrategy 类
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,  # 导入 FullOptimStateDictConfig 类
    FullStateDictConfig,  # 导入 FullStateDictConfig 类
    OptimStateKeyType,  # 导入 OptimStateKeyType 类
    ShardedOptimStateDictConfig,  # 导入 ShardedOptimStateDictConfig 类
    ShardedStateDictConfig,  # 导入 ShardedStateDictConfig 类
    StateDictSettings,  # 导入 StateDictSettings 类
    StateDictType,  # 导入 StateDictType 类
)
from torch.distributed.optim import _NamedOptimizer  # 导入 _NamedOptimizer 类
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入 skip_if_lt_x_gpu 函数
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,  # 导入 CUDAInitMode 类
    FSDPInitMode,  # 导入 FSDPInitMode 类
    FSDPTest,  # 导入 FSDPTest 类
    TransformerWithSharedParams,  # 导入 TransformerWithSharedParams 类
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入 instantiate_parametrized_tests 函数
    parametrize,  # 导入 parametrize 函数
    run_tests,  # 导入 run_tests 函数
    TEST_WITH_DEV_DBG_ASAN,  # 导入 TEST_WITH_DEV_DBG_ASAN 变量
)

# 定义 STATE_DICT_TYPES 列表，包含 FULL_STATE_DICT 和 SHARDED_STATE_DICT 两种状态字典类型
STATE_DICT_TYPES = [StateDictType.FULL_STATE_DICT, StateDictType.SHARDED_STATE_DICT]

# 如果分布式不可用，则输出提示信息并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果 TEST_WITH_DEV_DBG_ASAN 变量为真，则输出提示信息并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class _OSDCommMethod(Enum):
    """Method for communicating the optimizer state dict for internal tests."""

    BROADCAST_OBJECT_LIST = auto()  # 枚举类型，用于广播对象列表通信方法
    SCATTER_FULL_OSD = auto()  # 枚举类型，用于分散完整优化状态字典方法
    FLATTEN_SHARDED_OSD = auto()  # 枚举类型，用于展开分片优化状态字典方法
    OPTIM_STATE_DICT = auto()  # 枚举类型，用于优化器状态字典方法


class _ModelClass(Enum):
    """Different model type to test."""

    NESTED = auto()  # 枚举类型，用于测试的嵌套模型类型
    TRANSFORMER = auto()  # 枚举类型，用于测试的 Transformer 模型类型


class Bias(torch.nn.Module):
    """This module applies a 1D additive bias with dimension ``dim``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim > 0
        torch.manual_seed(0)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))  # 创建维度为 dim 的随机初始化偏置参数

    def forward(self, x):
        return x + self.bias  # 返回输入 x 加上偏置参数的结果


class BlockA(torch.nn.Module):
    """
    Used to define interesting nested structure for FSDP wrapping.
    BlockA
        Bias0
            bias
        weight
        Bias1
            bias
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        assert all(v > 0 for v in (in_dim, out_dim))
        torch.manual_seed(0)
        self.bias_module0 = Bias(out_dim)  # 创建维度为 out_dim 的偏置模块 Bias0
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))  # 创建输入维度为 in_dim，输出维度为 out_dim 的权重参数
        self.bias_module1 = Bias(out_dim)  # 创建维度为 out_dim 的偏置模块 Bias1
        self.relu = torch.nn.ReLU()  # 创建 ReLU 激活函数模块
    # 定义一个前向传播方法，用于神经网络的前向计算
    def forward(self, x):
        # 矩阵乘法，计算输入 x 与权重 self.weight 的乘积
        x = x @ self.weight
        # 调用 bias_module0 方法，对 x 加上第一个偏置
        x = self.bias_module0(x)
        # 对 x 应用 ReLU 激活函数，以确保偏置具有不同的梯度
        x = self.relu(x)
        # 调用 bias_module1 方法，对 x 加上第二个偏置
        x = self.bias_module1(x)
        # 返回前向传播的结果 x
        return x
class BlockB(torch.nn.Module):
    """
    用于定义FSDP包装的有趣嵌套结构。
    BlockB
        weight - 权重
        Bias - 偏置模块
            bias - 偏置参数
        Bias - 偏置模块
            bias - 偏置参数
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        assert all(v > 0 for v in (in_dim, out_dim))  # 断言输入维度和输出维度均大于0
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))  # 初始化权重参数
        self.bias_module0 = Bias(out_dim)  # 初始化第一个偏置模块
        self.bias_module1 = Bias(out_dim)  # 初始化第二个偏置模块
        self.relu = torch.nn.ReLU()  # 定义ReLU激活函数

    def forward(self, x):
        x = x @ self.weight  # 矩阵乘法操作，应用权重
        x = self.bias_module0(x)  # 应用第一个偏置模块
        x = self.relu(x)  # 应用ReLU激活函数，确保偏置有不同的梯度
        x = self.bias_module1(x)  # 应用第二个偏置模块
        return x


class NestedModel(torch.nn.Module):
    """
    嵌套模型类，包含多个Block和其他组件。
    """

    def __init__(self) -> None:
        super().__init__()
        self.block0 = BlockB(5, 3)  # 初始化Block0
        self.block1 = BlockB(3, 7)  # 初始化Block1
        self.bias = torch.nn.Parameter(torch.randn((5,)))  # 初始化偏置参数
        self.block2 = torch.nn.Sequential(
            BlockA(7, 9),  # 添加BlockA子模块
            BlockA(9, 9),  # 添加BlockA子模块
            BlockB(9, 5),  # 添加BlockB子模块
        )
        self.relu = torch.nn.ReLU()  # 定义ReLU激活函数

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.block0(x))  # 应用ReLU激活函数到Block0
        x = self.relu(self.block1(x))  # 应用ReLU激活函数到Block1
        x = self.relu(self.block2(x))  # 应用ReLU激活函数到Block2
        x = x + self.bias  # 加上偏置参数
        return x

    def get_input(self, device):
        BATCH_SIZE = 8
        return (torch.randn((BATCH_SIZE, 5)).to(device),)  # 返回随机输入数据

    def get_loss(self, inp, output):
        return output.sum()  # 计算输出的总和作为损失函数值

    def run_backward(self, loss):
        loss.backward()  # 执行反向传播

    @staticmethod
    def wrap(
        model: torch.nn.Module,
        group: Optional[dist.ProcessGroup] = None,
        ignore_modules: bool = False,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        对模型进行包装，支持分布式训练相关参数。

        Args:
            model: 待包装的模型
            group: 可选，进程组对象用于分布式训练
            ignore_modules: 是否忽略特定模块
            fsdp_kwargs: FSDP的相关参数

        Returns:
            包装后的模型
        """
        # 在这里执行模型包装的具体逻辑，根据参数调整模型的结构或者功能
        pass  # 占位符，实际执行时需补充具体实现
    ) -> torch.nn.Module:
        # 定义一个静态方法 wrap_with_unmanaged_params，用于将模型和未管理的参数进行包装
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        # 将 model.block1.bias_module0 属性进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block1.bias_module0 = FSDP(
            model.block1.bias_module0,
            process_group=group,
            **fsdp_kwargs,
        )
        # 将整个 model.block1 模块进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block1 = FSDP(model.block1, process_group=group, **fsdp_kwargs)
        # 将 model.block2[1].bias_module0 属性进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block2[1].bias_module0 = FSDP(
            model.block2[1].bias_module0,
            process_group=group,
            **fsdp_kwargs,
        )
        # 将 model.block2[1].bias_module1 属性进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block2[1].bias_module1 = FSDP(
            model.block2[1].bias_module1,
            process_group=group,
            **fsdp_kwargs,
        )
        # 将整个 model.block2[1] 模块进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block2[1] = FSDP(model.block2[1], process_group=group, **fsdp_kwargs)
        # 将 model.block2[2] 模块进行 FSDP 封装，使用指定的 process_group、ignored_modules 和 fsdp_kwargs
        ignored_modules = [model.block2[2].bias_module0] if ignore_modules else None
        model.block2[2] = FSDP(
            model.block2[2],
            process_group=group,
            ignored_modules=ignored_modules,
            **fsdp_kwargs,
        )
        # 返回已经包装好的模型对象
        return model

    @staticmethod
    def wrap_alt(
        model: torch.nn.Module,
        group: Optional[dist.ProcessGroup] = None,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        # 定义一个静态方法 wrap_alt，用于对模型进行替代的 FSDP 包装
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        # 将 model.block0.bias_module0 属性进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block0.bias_module0 = FSDP(
            model.block0.bias_module0,
            process_group=group,
            **fsdp_kwargs,
        )
        # 将整个 model.block0 模块进行 FSDP 封装，使用指定的 process_group 和 fsdp_kwargs
        model.block0 = FSDP(model.block0, process_group=group, **fsdp_kwargs)
        # 返回已经包装好的模型对象
        return model

    @staticmethod
    def wrap_with_unmanaged_params(
        model,
        add_to_fsdp_module: bool,
        group=None,
    ) -> Tuple[torch.nn.Module, List[torch.nn.Parameter]]:
        """Registers unmanaged parameters before wrapping with :meth:`wrap`."""
        # 静态方法 wrap_with_unmanaged_params 用于在使用 wrap 方法之前注册未管理的参数
        device = next(model.parameters()).device
        # 创建一个随机的未管理参数 unmanaged_param，并将其注册到指定的模块（model.block2[2] 或 model）
        unmanaged_param = torch.nn.Parameter(torch.randn(5, 5, device=device))
        register_module = model.block2[2] if add_to_fsdp_module else model
        register_module.register_parameter(
            "unmanaged_param",
            unmanaged_param,
        )
        # 返回使用 wrap 方法包装后的模型对象和未管理的参数列表
        return NestedModel.wrap(model, group), [unmanaged_param]

    @staticmethod
    # 为给定的未管理参数 `unmanaged_param` 添加一个条目，假设使用 Adam 优化器和单一参数组。
    def add_unmanaged_param_entry(osd, unmanaged_param, step) -> None:
        """Adds an entry for the unmanaged parameter ``unmanaged_param``
        assuming Adam optimizer and a single parameter group."""
        
        # 未管理的参数应按照 `model.parameters()` 的顺序传递给该方法，因为它们的参数 ID 将按照跳过的 ID 的顺序分配
        # 为未管理的参数分配一个参数 ID
        unmanaged_param_id = -1
        param_ids = osd["param_groups"][0]["params"]
        for i in range(1, len(param_ids)):
            diff = param_ids[i] - param_ids[i - 1]
            if diff != 1:
                assert diff > 1, f"Invalid IDs: {param_ids[i - 1]} {param_ids[i]}"
                unmanaged_param_id = param_ids[i - 1] + 1
                break
        if unmanaged_param_id == -1:
            unmanaged_param_id = len(param_ids)  # 最后一个 ID 被跳过
        assert unmanaged_param_id >= 0, "应该跳过一个参数 ID"
        
        # 为未管理的参数添加一个状态条目
        state_device = next(iter(next(iter(osd["state"].values())).values())).device
        osd["state"][unmanaged_param_id] = {
            "step": torch.tensor(float(step), device=state_device),
            "exp_avg": torch.randn(unmanaged_param.shape, device=state_device),
            "exp_avg_sq": torch.randn(unmanaged_param.shape, device=state_device),
        }
        
        # 将 ID 按顺序插入到参数组中
        bisect.insort(osd["param_groups"][0]["params"], unmanaged_param_id)

    # 注意：我们从参数组中排除 `self.bias`，以便测试优化器输入不包括所有模型参数的情况
    def param_group0(self) -> List[torch.nn.Parameter]:
        # 使用 `block1` 的参数作为第一个参数组，以偏离 `model.parameters()` 的顺序
        return list(self.block1.parameters())

    def param_group1(self) -> List[torch.nn.Parameter]:
        # 进一步通过重新排列 `block2` 的参数在 `block0` 的参数之前，以偏离 `model.parameters()` 的顺序
        return list(self.block2.parameters()) + list(self.block0.parameters())
# 创建一个简单且普通的模型来测试接口和一些不需要复杂包装策略的边缘情况。
class TestDummyModel(torch.nn.Module):
    def __init__(self, no_grad: bool = False):
        super().__init__()
        torch.manual_seed(0)
        # 定义一个包含两层的神经网络模型，第一层为线性层，激活函数为ReLU
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        # 根据 `no_grad` 参数设置是否需要梯度计算
        self.net1[0].weight.requires_grad = not no_grad
        self.net1[0].bias.requires_grad = not no_grad
        # 第二层为线性层，激活函数为ReLU
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        # 第三层为线性层，输入维度为32，输出维度为64
        self.net3 = nn.Linear(32, 64)
        # 第四层为包含ReLU和线性层的序列
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # 前向传播函数，依次经过网络层net1、net2、net3和net4
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        # 返回一个随机生成的8x8的张量，位于CUDA设备上
        return torch.rand(8, 8, device="cuda")


class TestFSDPOptimState(FSDPTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化模型类别字典，映射到相应的初始化方法
        self._model_class = {
            _ModelClass.NESTED: self._init_nested_model,
            _ModelClass.TRANSFORMER: self._init_transformer_model,
        }

    def _init_nested_model(
        self,
        wrap: bool,
        wrap_alt: bool = False,  # 如果`wrap=False`，则忽略此参数
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        use_diff_optim_inputs: bool = False,
        fsdp_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 创建一个NestedModel实例，并将其放置在指定的设备上
        model = NestedModel().to(device)
        if wrap:
            # 如果wrap为True，则根据wrap_alt参数选择不同的包装方式
            model = (
                NestedModel.wrap_alt(model, group, fsdp_kwargs)
                if wrap_alt
                else NestedModel.wrap(model, group, fsdp_kwargs=fsdp_kwargs)
            )
        if not use_multiple_param_groups:
            # 如果不使用多个参数组，则将所有参数作为一个列表输入优化器
            optim_input = list(model.parameters())
        else:
            # 否则，根据模型的不同参数组形成优化器的输入列表
            optim_input = [
                {"params": model.param_group0()},
                {"params": model.param_group1(), "weight_decay": 0.9},
            ]
        # 如果使用不同的优化器输入，并且当前进程的排名是奇数，则反转参数顺序
        if use_diff_optim_inputs and self.rank % 2 == 1:
            if isinstance(optim_input[0], dict):
                for param_group in optim_input:
                    param_group["params"] = list(reversed(param_group["params"]))
            else:
                optim_input = list(reversed(optim_input))
        # 使用给定的优化器类和学习率0.01来创建优化器对象
        optim = optim_class(optim_input, lr=0.01)
        return model, optim, optim_input

    def _init_transformer_model(
        self,
        wrap: bool,
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        use_diff_optim_inputs: bool = False,
    ):
        if use_multiple_param_groups or use_diff_optim_inputs:
            # 如果使用多个参数组或者不同的优化器输入，则抛出未实现错误
            # 这些设置在顶层使用 FSDP 封装的 Transformer 中不被实现，
            # 因为只有一个扁平参数，这些布尔值没有意义
            raise NotImplementedError
        if group is None:
            # 如果没有指定组，则使用默认的分布式组
            group = dist.distributed_c10d._get_default_group()
        # 初始化 TransformerWithSharedParams 模型
        model = TransformerWithSharedParams.init(
            group,
            FSDPInitMode.RECURSIVE if wrap else FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        # 使用给定的优化器类初始化优化器
        optim = optim_class(model.parameters(), lr=0.01)
        return model, optim, None

    def _step_model(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device = torch.device("cuda"),
        num_iters: int = 1,
    ) -> List[float]:
        """执行模型的前向传播、反向传播和优化器步骤，
        重复 ``num_iters`` 次，并返回每次迭代的损失值列表。"""
        torch.manual_seed(0)  # 设置种子以保证确定性
        losses = []
        module = getattr(model, "module", model)
        for _ in range(num_iters):
            optim.zero_grad()  # 清除梯度
            inp = module.get_input(device)  # 获取输入数据
            output = model(*inp)  # 执行模型前向传播
            loss = module.get_loss(inp, output).to(device)  # 计算损失并将其移至指定设备
            losses.append(loss.item())  # 记录损失值
            module.run_backward(loss)  # 执行反向传播
            optim.step()  # 执行优化器步骤
        return losses

    def _broadcast_full_osd(self, full_osd: Dict[str, Any], group=None):
        """在所有进程中广播完整的优化器状态字典，
        而不是使用 ``torch.save()`` 和 ``torch.load()``。"""
        obj_list = [full_osd]
        dist.broadcast_object_list(
            obj_list,
            src=0,
            group=group,
        )
        full_osd = obj_list[0]  # 更新本地的完整优化器状态字典
        return full_osd

    def _are_equal_states(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> bool:
        """Checks if ``state1`` and ``state2`` contain the same mappings."""
        # 检查 state1 和 state2 的键集合是否相同，如果不同则返回 False
        if set(state1.keys()) != set(state2.keys()):
            return False
        # 遍历 state1 中的每个键值对
        for state_name, value1 in state1.items():
            # 获取 state2 中相同键的值
            value2 = state2[state_name]
            # 检查 value1 和 value2 的类型是否相同，如果不同则返回 False
            if type(value1) != type(value2):
                return False
            # 如果 value1 是 tensor 类型（torch.Tensor）
            if torch.is_tensor(value1):  # tensor state
                assert torch.is_tensor(value2)
                # 将 tensor 转移到 CPU 上，以保证设备无关性
                value1 = value1.cpu()
                value2 = value2.cpu()
                # 检查 tensor 的形状和值是否相等，如果不相等则返回 False
                if value1.shape != value2.shape or not torch.all(
                    torch.isclose(value1, value2)
                ):
                    return False
            else:  # 如果 value1 是非 tensor 类型
                # 检查 value1 和 value2 的值是否相等，如果不相等则返回 False
                if value1 != value2:
                    return False
        # 如果所有检查通过，则返回 True
        return True

    def _check_same_state(
        self,
        fsdp_osd,
        ref_osd,
        check_same_param_keys: bool,
    ):
        """Checks that ``full_osd`` and ``ref_osd`` have the same "state" part.
        If ``check_same_param_keys=True``, then checks that the parameter keys
        match (e.g. when both should be parameter names), and does not check
        the parameter keys otherwise."""
        # 确保 ``ref_osd`` 中包含 "state" 键
        assert "state" in ref_osd
        # 使用 self.assertTrue 确保 ``fsdp_osd`` 中也包含 "state" 键
        self.assertTrue("state" in fsdp_osd)
        # 获取 ``ref_osd`` 中的 "state" 部分
        ref_osd_state = ref_osd["state"]
        # 将 ``fsdp_osd`` 的 "state" 部分转换为字典形式，其中值为 _gather_state_dict(v) 的结果
        fsdp_osd_state = {
            k: _gather_state_dict(v) for k, v in fsdp_osd["state"].items()
        }

        if check_same_param_keys:
            # 如果需要检查参数键是否相同，则先比较参数键集合
            ref_osd_param_ids = set(ref_osd_state.keys())
            fsdp_osd_param_ids = set(fsdp_osd_state.keys())
            # 使用 self.assertTrue 确保两者的参数键集合相同，否则报错
            self.assertTrue(
                ref_osd_param_ids == fsdp_osd_param_ids,
                f"Rank {self.rank}: {(ref_osd_param_ids, fsdp_osd_param_ids)}",
            )
            # 检查状态值是否相同
            for param_id, param_state in fsdp_osd_state.items():
                for state_name, value in param_state.items():
                    # 使用 self.assertEqual 检查 ``fsdp_osd`` 和 ``ref_osd`` 的状态值是否相等
                    ref_value = ref_osd_state[param_id][state_name]
                    self.assertEqual(value, ref_value)
            return

        # 如果不需要检查参数键是否相同，则只要求参数键集合同构（即可映射，但不需要严格相等）
        ref_osd_states = list(ref_osd_state.values())
        fsdp_osd_states = list(fsdp_osd_state.values())
        # 使用 self.assertEqual 确保两者状态值列表长度相等
        self.assertEqual(len(ref_osd_states), len(fsdp_osd_states))
        # 使用暴力法进行二次比较，因为很难按值而不是按对象哈希张量
        for fsdp_osd_state in fsdp_osd_states:
            # 检查至少存在一项匹配，这样可以确保列表内容相等
            self.assertTrue(
                any(
                    self._are_equal_states(fsdp_osd_state, ref_osd_state)
                    for ref_osd_state in ref_osd_states
                )
            )

    def _check_same_param_groups(
        self,
        full_osd,
        ref_osd,
        check_same_param_keys: bool,
    ):
        """
        Checks that ``full_osd`` and ``ref_osd`` have the same
        "param_groups" part. If ``check_same_param_keys=True`, then checks that
        the parameter keys match (e.g. when both should be parameter names),
        and does not check the parameter keys otherwise.
        """
        # 断言参考对象的 OSD 中包含 "param_groups"
        assert "param_groups" in ref_osd
        # 使用断言确保完整对象的 OSD 中也包含 "param_groups"
        self.assertTrue("param_groups" in full_osd)
        # 获取参考对象中的 param_groups 和完整对象中的 param_groups
        ref_osd_param_groups = ref_osd["param_groups"]
        full_osd_param_groups = full_osd["param_groups"]
        # 使用断言确保两者的 param_groups 长度相等
        self.assertTrue(len(full_osd_param_groups), len(ref_osd_param_groups))
        # 遍历两个 param_groups 的列表，并比较其元素
        for full_osd_pg, ref_osd_pg in zip(
            full_osd_param_groups,
            ref_osd_param_groups,
        ):
            # 比较当前 param_group 的 keys 集合是否相同
            self.assertEqual(
                set(full_osd_pg.keys()),
                set(ref_osd_pg.keys()),
            )
            # 遍历当前 param_group 中的每个键值对，检查是否相等
            for name, full_osd_value in full_osd_pg.items():
                # 如果 name 是 "params" 并且 check_same_param_keys 为 False，则跳过
                if name == "params" and not check_same_param_keys:
                    continue
                # 使用断言比较完整对象中的值和参考对象中的值是否相等
                self.assertEqual(full_osd_value, ref_osd_pg[name])

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", STATE_DICT_TYPES)
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("rank0_only", [False, True])
    @parametrize("use_diff_optim_inputs", [False, True])
    def test_optim_state_dict_nested(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
        rank0_only: bool,
        use_diff_optim_inputs: bool,
    ) -> None:
        """
        Tests :meth:`full_optim_state_dict` and meth:`sharded_optim_state_dict`
        by comparing the returned dict for an FSDP-wrapped model with that of
        an equivalent non-wrapped model.

        The test checks the equivalence excluding the parameter keys since the
        FSDP and normal optimizer state dicts key by names and IDs,
        respectively. This means that the test can pass even if parameter keys
        are incorrectly mapped to values. Their correct mapping is tested in
        other tests that exercise the save/load workflow.
        """
        # 运行子测试，测试优化器状态字典的嵌套结构
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_optim_state_dict_nested,
            state_dict_type=state_dict_type,
            use_multiple_param_groups=use_multiple_param_groups,
            rank0_only=rank0_only,
            use_diff_optim_inputs=use_diff_optim_inputs,
        )

    def _test_optim_state_dict_nested(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
        rank0_only: bool,
        use_diff_optim_inputs: bool,
        use_optim_input: bool,
    # 定义方法，接收多个参数并且不返回任何内容
    ) -> None:
        # 如果只有 rank0 并且状态字典类型为 SHARDED_STATE_DICT，则不支持，直接返回
        if rank0_only and state_dict_type == StateDictType.SHARDED_STATE_DICT:
            return  # not supported
        
        # 设置常量 NUM_ITERS 为 3
        NUM_ITERS = 3
        
        # 初始化嵌套模型，返回模型、优化器和优化器输入
        model1, optim1, optim_input = self._init_nested_model(
            wrap=True,
            use_multiple_param_groups=use_multiple_param_groups,
            use_diff_optim_inputs=use_diff_optim_inputs,
        )
        
        # 在模型上执行多次步骤，返回损失值列表
        losses1 = self._step_model(model1, optim1, num_iters=NUM_ITERS)
        
        # 如果状态字典类型为 FULL_STATE_DICT
        if state_dict_type == StateDictType.FULL_STATE_DICT:
            # 如果使用优化器输入
            if use_optim_input:
                # 调用 FSDP 类的 full_optim_state_dict 方法，返回完整优化器状态字典
                fsdp_osd = FSDP.full_optim_state_dict(
                    model1,
                    optim1,
                    optim_input,
                    rank0_only=rank0_only,
                )
            else:
                # 调用 FSDP 类的 full_optim_state_dict 方法，返回完整优化器状态字典
                fsdp_osd = FSDP.full_optim_state_dict(
                    model1,
                    optim1,
                    rank0_only=rank0_only,
                )
        else:
            # 调用 FSDP 类的 sharded_optim_state_dict 方法，返回分片优化器状态字典
            fsdp_osd = FSDP.sharded_optim_state_dict(model1, optim1)
        
        # 对于非目标排名，返回空状态字典
        if rank0_only and self.rank != 0:
            self.assertEqual(len(fsdp_osd), 0)
            return
        
        # 初始化非嵌套模型，返回模型和优化器
        model2, optim2, _ = self._init_nested_model(
            wrap=False,
            use_multiple_param_groups=use_multiple_param_groups,
            use_diff_optim_inputs=use_diff_optim_inputs,
        )
        
        # 在模型上执行多次步骤，返回损失值列表
        losses2 = self._step_model(model2, optim2, num_iters=NUM_ITERS)
        
        # 获取优化器的状态字典作为参考
        ref_osd = optim2.state_dict()
        
        # 检查损失，以排除模型漂移引起的错误
        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert l1 == l2, f"Losses differ on iter {i}: {l1:.5f} {l2:.5f}"
        
        # 不检查参数键，因为完整/分片优化器状态字典使用参数名，而非包装的等效方法使用参数 ID
        check_same_param_keys = False
        
        # 检查是否具有相同的参数组
        self._check_same_param_groups(
            fsdp_osd,
            ref_osd,
            check_same_param_keys=check_same_param_keys,
        )
        
        # 检查是否具有相同的状态
        self._check_same_state(
            fsdp_osd,
            ref_osd,
            check_same_param_keys=check_same_param_keys,
        )

    # 如果 GPU 数量小于 2，则跳过此测试
    @skip_if_lt_x_gpu(2)
    def test_full_optim_state_dict_keys(self):
        """Tests that the parameter keys returned by
        :meth:`full_optim_state_dict` match those of :meth:`state_dict` with
        full ``state_dict_type`` for a non-FSDP-root model with nested FSDP
        instances and ignored modules."""
        
        device = torch.device("cuda")  # 设置使用 CUDA 设备
        
        model = NestedModel().to(device)  # 创建 NestedModel 实例并移到 CUDA 设备上
        wrapped_model = NestedModel.wrap(model, ignore_modules=True)  # 使用 NestedModel.wrap 包装模型，忽略部分模块
        
        # Add checkpointing to ensure optim_state_dict and state_dict strip out
        # checkpointing prefixes.
        apply_activation_checkpointing(
            model, check_fn=lambda module: isinstance(module, torch.nn.Sequential)
        )  # 应用激活检查点，用于确保 optim_state_dict 和 state_dict 去除检查点前缀
        
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)  # 创建 Adam 优化器对象
        self._step_model(model, optim, device)  # 调用 _step_model 方法对模型和优化器进行一步训练
        
        optim_state_dict = FSDP.full_optim_state_dict(
            wrapped_model, optim, rank0_only=False
        )  # 调用 FSDP.full_optim_state_dict 获取完整的优化器状态字典
        
        with FSDP.state_dict_type(wrapped_model, StateDictType.FULL_STATE_DICT):
            state_dict = wrapped_model.state_dict()  # 获取完整状态字典
        
        self.assertEqual(optim_state_dict["state"].keys(), state_dict.keys())  # 断言优化器状态字典的键与模型状态字典的键相同
        
        # Check that checkpointing prefix was indeed stripped.
        for key in optim_state_dict["state"]:
            self.assertNotIn(_CHECKPOINT_WRAPPED_MODULE, key)  # 验证确实已去除检查点前缀

    @skip_if_lt_x_gpu(2)
    def test_full_optim_state_dict_nested_invalid(self):
        """Tests that :meth:`full_optim_state_dict` raises an error when
        nonzero ranks are missing the optimizer state for parameters on rank
        0."""
        
        device = torch.device("cuda")  # 设置使用 CUDA 设备
        
        model = NestedModel.wrap(NestedModel().to(device), None)  # 使用 NestedModel.wrap 包装模型
        optim_input = list(model.parameters())  # 获取模型的参数列表
        
        if self.rank != 0:
            # Exclude a parameter so that nonzero ranks are missing state
            optim_input = optim_input[:-1]  # 如果不是 rank 0，则排除一个参数，使非零 rank 缺少状态
        
        optim = torch.optim.Adam(optim_input, lr=1e-3)  # 创建 Adam 优化器对象
        self._step_model(model, optim, num_iters=3)  # 调用 _step_model 方法对模型和优化器进行多次训练
        
        error_regex = (
            "FSDP currently requires each rank to have at least the "
            "optimizer states needed by rank 0's optimizer but some ranks "
            "are missing some of those states"
        )  # 定义错误信息的正则表达式
        
        with self.assertRaisesRegex(RuntimeError, error_regex):
            FSDP.full_optim_state_dict(model, optim)  # 断言调用 FSDP.full_optim_state_dict 会引发特定的 RuntimeError 异常

    @skip_if_lt_x_gpu(2)
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("wrap_alt", [False, True])
    @parametrize("use_diff_optim_inputs", [False, True])
    def test_shard_full_optim_state_dict_nested(
        self,
        use_multiple_param_groups: bool,
        wrap_alt: bool,
        use_diff_optim_inputs: bool,
    ):
        """Tests sharding of :meth:`full_optim_state_dict` for nested models."""
        
        # 测试嵌套模型的 full_optim_state_dict 的分片功能
    ):
        """
        Tests :meth:`shard_full_optim_state_dict` for a non-FSDP-root model
        with nested FSDP instances.
        """
        # 运行子测试，测试不是根 FSDP 模型的情况下，具有嵌套 FSDP 实例的 `shard_full_optim_state_dict` 方法
        self.run_subtests(
            # 设置 `use_optim_input` 参数为 False 和 True 进行测试
            {"use_optim_input": [False, True]},
            # 调用 `_test_load_optim_state` 方法进行优化器状态加载测试
            self._test_load_optim_state,
            # 指定模型类为 `_ModelClass.NESTED`
            model_class=_ModelClass.NESTED,
            # 使用多个参数组进行测试，根据 `use_multiple_param_groups` 参数设置
            use_multiple_param_groups=use_multiple_param_groups,
            # 不减半世界大小
            halve_world_size=False,
            # 使用广播对象列表作为 OSD 通信方法
            osd_comm_method=_OSDCommMethod.BROADCAST_OBJECT_LIST,
            # 根据 `use_diff_optim_inputs` 参数设置是否使用不同的优化器输入进行测试
            use_diff_optim_inputs=use_diff_optim_inputs,
            # 根据 `wrap_alt` 参数设置是否使用备用包装进行测试
            wrap_alt=wrap_alt,
            # 指定迭代次数为 3
            num_iters=3,
        )

        # 调用 `_test_load_optim_state_with_optim_state_dict` 方法，测试加载带有优化器状态字典的情况
        self._test_load_optim_state_with_optim_state_dict(
            # 指定模型类为 `_ModelClass.NESTED`
            _ModelClass.NESTED,
            # 设置状态字典的配置，包括完整状态字典和完整优化器状态字典的设置
            state_dict_settings=StateDictSettings(
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(),
                FullOptimStateDictConfig(),
            ),
            # 使用单个参数组进行测试，根据 `use_multiple_param_groups` 参数设置
            use_multiple_param_groups=False,
            # 不减半世界大小
            halve_world_size=False,
            # 根据 `use_diff_optim_inputs` 参数设置是否使用不同的优化器输入进行测试
            use_diff_optim_inputs=use_diff_optim_inputs,
            # 根据 `wrap_alt` 参数设置是否使用备用包装进行测试
            wrap_alt=wrap_alt,
            # 指定迭代次数为 3
            num_iters=3,
        )

    @skip_if_lt_x_gpu(2)
    def test_shard_full_optim_state_dict_nested_halve_world_size(self):
        """
        Tests :meth:`shard_full_optim_state_dict` for a non-FSDP-root model
        with nested FSDP instances when loading into a new process group with
        halved world size.
        """
        # 为了节省 CI 成本，使用 "更难" 的设置进行测试：
        use_multiple_param_groups = True
        use_diff_optim_inputs = True
        wrap_alt = True

        # 运行子测试，测试不是根 FSDP 模型的情况下，具有嵌套 FSDP 实例的 `shard_full_optim_state_dict` 方法
        self.run_subtests(
            # 设置 `use_optim_input` 参数为 False 和 True 进行测试
            {"use_optim_input": [False, True]},
            # 调用 `_test_load_optim_state` 方法进行优化器状态加载测试
            self._test_load_optim_state,
            # 指定模型类为 `_ModelClass.NESTED`
            model_class=_ModelClass.NESTED,
            # 使用多个参数组进行测试，根据 `use_multiple_param_groups` 参数设置
            use_multiple_param_groups=use_multiple_param_groups,
            # 减半世界大小
            halve_world_size=True,
            # 使用广播对象列表作为 OSD 通信方法
            osd_comm_method=_OSDCommMethod.BROADCAST_OBJECT_LIST,
            # 根据 `use_diff_optim_inputs` 参数设置是否使用不同的优化器输入进行测试
            use_diff_optim_inputs=use_diff_optim_inputs,
            # 根据 `wrap_alt` 参数设置是否使用备用包装进行测试
            wrap_alt=wrap_alt,
            # 指定迭代次数为 3
            num_iters=3,
        )

        # 调用 `_test_load_optim_state_with_optim_state_dict` 方法，测试加载带有优化器状态字典的情况
        self._test_load_optim_state_with_optim_state_dict(
            # 指定模型类为 `_ModelClass.NESTED`
            _ModelClass.NESTED,
            # 设置状态字典的配置，包括完整状态字典和完整优化器状态字典的设置
            state_dict_settings=StateDictSettings(
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(),
                FullOptimStateDictConfig(),
            ),
            # 使用多个参数组进行测试，根据 `use_multiple_param_groups` 参数设置
            use_multiple_param_groups=use_multiple_param_groups,
            # 减半世界大小
            halve_world_size=True,
            # 根据 `use_diff_optim_inputs` 参数设置是否使用不同的优化器输入进行测试
            use_diff_optim_inputs=use_diff_optim_inputs,
            # 根据 `wrap_alt` 参数设置是否使用备用包装进行测试
            wrap_alt=wrap_alt,
            # 指定迭代次数为 3
            num_iters=3,
        )
    def test_shard_full_optim_state_dict_transformer(self) -> None:
        """Tests :meth:`shard_full_optim_state_dict` for an FSDP-root
        transformer model with shared parameters."""
        # 运行子测试，测试具有共享参数的 FSDP-root transformer 模型的 `shard_full_optim_state_dict` 方法
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.TRANSFORMER,
            use_multiple_param_groups=False,
            halve_world_size=True,
            osd_comm_method=_OSDCommMethod.BROADCAST_OBJECT_LIST,
            use_diff_optim_inputs=False,
            num_iters=3,
        )

        # 使用指定的优化器状态字典设置，测试加载优化器状态
        self._test_load_optim_state_with_optim_state_dict(
            _ModelClass.TRANSFORMER,
            state_dict_settings=StateDictSettings(
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(),
                FullOptimStateDictConfig(),
            ),
            use_multiple_param_groups=False,
            halve_world_size=True,
            use_diff_optim_inputs=False,
            num_iters=3,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("wrap_alt", [False, True])
    @parametrize("use_diff_optim_inputs", [False, True])
    def test_scatter_full_optim_state_dict_nested(
        self,
        use_multiple_param_groups: bool,
        wrap_alt: bool,
        use_diff_optim_inputs: bool,
    ):
        """Tests :meth:`scatter_full_optim_state_dict` for a non-FSDP-root
        model with nested FSDP instances."""
        # 运行子测试，测试具有嵌套 FSDP 实例的非 FSDP-root 模型的 `scatter_full_optim_state_dict` 方法
        self.run_subtests(
            {"use_optim_input": [False, True]},
            self._test_load_optim_state,
            model_class=_ModelClass.NESTED,
            use_multiple_param_groups=use_multiple_param_groups,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.SCATTER_FULL_OSD,
            use_diff_optim_inputs=use_diff_optim_inputs,
            wrap_alt=wrap_alt,
            num_iters=3,
        )

        # 使用指定的优化器状态字典设置，测试加载优化器状态
        self._test_load_optim_state_with_optim_state_dict(
            _ModelClass.NESTED,
            state_dict_settings=StateDictSettings(
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(),
                FullOptimStateDictConfig(rank0_only=True),
            ),
            use_multiple_param_groups=use_multiple_param_groups,
            halve_world_size=False,
            use_diff_optim_inputs=use_diff_optim_inputs,
            wrap_alt=wrap_alt,
            num_iters=3,
        )

    @skip_if_lt_x_gpu(2)
    def test_scatter_full_optim_state_dict_nested_halve_world_size(self):
        """Tests :meth:`scatter_full_optim_state_dict` for a non-FSDP-root
        model with nested FSDP instances when loading into a new process group
        with halved world size."""
        # To save CI costs, we test with the "harder" settings:
        use_multiple_param_groups = True  # 设置是否使用多个参数组
        use_diff_optim_inputs = True  # 设置是否使用不同的优化器输入
        wrap_alt = True  # 设置是否使用备用包装器
        self.run_subtests(  # 运行子测试
            {"use_optim_input": [False, True]},  # 测试不同的优化器输入选项
            self._test_load_optim_state,  # 调用加载优化器状态的测试方法
            model_class=_ModelClass.NESTED,  # 模型类别为嵌套类型
            use_multiple_param_groups=use_multiple_param_groups,  # 传入是否使用多个参数组的设置
            halve_world_size=True,  # 设置为使用减半的世界大小
            osd_comm_method=_OSDCommMethod.SCATTER_FULL_OSD,  # 使用全分散优化器状态字典的通信方法
            use_diff_optim_inputs=use_diff_optim_inputs,  # 传入是否使用不同的优化器输入的设置
            wrap_alt=wrap_alt,  # 传入是否使用备用包装器的设置
            num_iters=3,  # 设置迭代次数为3
        )

        self._test_load_optim_state_with_optim_state_dict(
            _ModelClass.NESTED,  # 加载优化器状态时的模型类别为嵌套类型
            state_dict_settings=StateDictSettings(
                StateDictType.FULL_STATE_DICT,  # 使用完整的状态字典
                FullStateDictConfig(),  # 使用默认的完整状态字典配置
                FullOptimStateDictConfig(rank0_only=True),  # 仅在rank0上使用完整的优化器状态字典配置
            ),
            use_multiple_param_groups=use_multiple_param_groups,  # 传入是否使用多个参数组的设置
            halve_world_size=True,  # 设置为使用减半的世界大小
            use_diff_optim_inputs=use_diff_optim_inputs,  # 传入是否使用不同的优化器输入的设置
            wrap_alt=wrap_alt,  # 传入是否使用备用包装器的设置
            num_iters=3,  # 设置迭代次数为3
        )

    @skip_if_lt_x_gpu(2)
    def test_scatter_full_optim_state_dict_transformer(self) -> None:
        """Tests :meth:`scatter_full_optim_state_dict` for an FSDP-root
        transformer model with shared parameters."""
        self.run_subtests(  # 运行子测试
            {"use_optim_input": [False, True]},  # 测试不同的优化器输入选项
            self._test_load_optim_state,  # 调用加载优化器状态的测试方法
            model_class=_ModelClass.TRANSFORMER,  # 模型类别为Transformer类型
            use_multiple_param_groups=False,  # 不使用多个参数组
            halve_world_size=True,  # 设置为使用减半的世界大小
            osd_comm_method=_OSDCommMethod.SCATTER_FULL_OSD,  # 使用全分散优化器状态字典的通信方法
            use_diff_optim_inputs=False,  # 不使用不同的优化器输入
            num_iters=3,  # 设置迭代次数为3
        )

        self._test_load_optim_state_with_optim_state_dict(
            _ModelClass.TRANSFORMER,  # 加载优化器状态时的模型类别为Transformer类型
            state_dict_settings=StateDictSettings(
                StateDictType.FULL_STATE_DICT,  # 使用完整的状态字典
                FullStateDictConfig(),  # 使用默认的完整状态字典配置
                FullOptimStateDictConfig(rank0_only=True),  # 仅在rank0上使用完整的优化器状态字典配置
            ),
            use_multiple_param_groups=False,  # 不使用多个参数组
            halve_world_size=True,  # 设置为使用减半的世界大小
            use_diff_optim_inputs=False,  # 不使用不同的优化器输入
            num_iters=3,  # 设置迭代次数为3
        )

    @skip_if_lt_x_gpu(2)
    @skip_if_lt_x_gpu(2)
    def test_flatten_sharded_optim_state_dict_nested(self) -> None:
        """Defines a test method for flattening sharded optimizer state dicts
        in a nested model under the FSDP context.

        This method tests loading optimizer states for a nested model,
        ensuring compatibility with sharded optimizer state dict flattening.
        """
        self._test_load_optim_state(
            _ModelClass.NESTED,
            use_multiple_param_groups=False,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.FLATTEN_SHARDED_OSD,
            use_diff_optim_inputs=False,
            use_optim_input=False,
            wrap_alt=True,
            num_iters=3,
        )

        self._test_load_optim_state_with_optim_state_dict(
            _ModelClass.NESTED,
            state_dict_settings=StateDictSettings(
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(),
                ShardedOptimStateDictConfig(),
            ),
            use_multiple_param_groups=False,
            halve_world_size=False,
            use_diff_optim_inputs=False,
            wrap_alt=True,
            num_iters=3,
        )

    @skip_if_lt_x_gpu(2)
    def test_flatten_sharded_optim_state_dict_transformer(self) -> None:
        """Defines a test method for flattening sharded optimizer state dicts
        in a transformer model under the FSDP context.

        This method tests loading optimizer states for a transformer model,
        ensuring compatibility with sharded optimizer state dict flattening.
        """
        self._test_load_optim_state(
            _ModelClass.TRANSFORMER,
            use_multiple_param_groups=False,
            halve_world_size=False,
            osd_comm_method=_OSDCommMethod.FLATTEN_SHARDED_OSD,
            use_diff_optim_inputs=False,
            use_optim_input=False,
            num_iters=3,
        )

        self._test_load_optim_state_with_optim_state_dict(
            _ModelClass.TRANSFORMER,
            state_dict_settings=StateDictSettings(
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(),
                ShardedOptimStateDictConfig(),
            ),
            use_multiple_param_groups=False,
            halve_world_size=False,
            use_diff_optim_inputs=False,
            num_iters=3,
        )

    @skip_if_lt_x_gpu(2)
    # 定义测试方法，用于测试 `optim_state_dict` 方法，针对一个嵌套模型的 FSDP 根模型进行测试
    def test_use_orig_params(self) -> None:
        """Tests :meth:`optim_state_dict` for an FSDP-root nested model."""

        # 调用 `run_subtests` 方法，执行多个子测试
        self.run_subtests(
            {
                "halve_world_size": [True, False],  # 测试参数：减半世界大小，值为 True 和 False
                "wrap_alt": [True, False],  # 测试参数：wrap_alt，值为 True 和 False
            },
            self._test_load_optim_state_with_optim_state_dict,  # 要运行的测试函数
            model_class=_ModelClass.NESTED,  # 模型类别为嵌套模型
            state_dict_settings=StateDictSettings(  # 设置状态字典选项
                StateDictType.FULL_STATE_DICT,  # 使用完整状态字典
                FullStateDictConfig(),  # 完整状态字典的配置
                FullOptimStateDictConfig(),  # 完整优化器状态字典的配置
            ),
            use_multiple_param_groups=False,  # 不使用多个参数组
            use_diff_optim_inputs=False,  # 不使用不同的优化器输入
            num_iters=3,  # 迭代次数为 3
            fsdp_kwargs={"use_orig_params": True},  # FSDP 的关键字参数，使用原始参数为 True
        )

        # 再次调用 `run_subtests` 方法，执行另一组子测试，参数设置与上面基本相同，仅状态字典设置有所不同
        self.run_subtests(
            {
                "halve_world_size": [True, False],  # 测试参数：减半世界大小，值为 True 和 False
                "wrap_alt": [True, False],  # 测试参数：wrap_alt，值为 True 和 False
            },
            self._test_load_optim_state_with_optim_state_dict,  # 要运行的测试函数
            model_class=_ModelClass.NESTED,  # 模型类别为嵌套模型
            state_dict_settings=StateDictSettings(  # 设置状态字典选项
                StateDictType.FULL_STATE_DICT,  # 使用完整状态字典
                FullStateDictConfig(),  # 完整状态字典的配置
                FullOptimStateDictConfig(rank0_only=True),  # 仅在 rank0 下使用的完整优化器状态字典配置
            ),
            use_multiple_param_groups=False,  # 不使用多个参数组
            use_diff_optim_inputs=False,  # 不使用不同的优化器输入
            num_iters=3,  # 迭代次数为 3
            fsdp_kwargs={"use_orig_params": True},  # FSDP 的关键字参数，使用原始参数为 True
        )

        # 第三个子测试，测试不同的参数组合，这里省略了一些参数，仅展示了部分设置
        self.run_subtests(
            {
                "wrap_alt": [True, False],  # 测试参数：wrap_alt，值为 True 和 False
            },
            self._test_load_optim_state_with_optim_state_dict,  # 要运行的测试函数
            model_class=_ModelClass.NESTED,  # 模型类别为嵌套模型
            state_dict_settings=StateDictSettings(  # 设置状态字典选项
                StateDictType.SHARDED_STATE_DICT,  # 使用分片状态字典
                ShardedStateDictConfig(),  # 分片状态字典的配置
                ShardedOptimStateDictConfig(),  # 分片优化器状态字典的配置
            ),
            use_multiple_param_groups=False,  # 不使用多个参数组
            halve_world_size=False,  # 不减半世界大小
            use_diff_optim_inputs=False,  # 不使用不同的优化器输入
            num_iters=3,  # 迭代次数为 3
            fsdp_kwargs={"use_orig_params": True},  # FSDP 的关键字参数，使用原始参数为 True
        )

    # 测试方法的参数设置与子测试方法有所不同，这里展示了多个参数化的测试用例
    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", STATE_DICT_TYPES)  # 参数化测试：状态字典类型从 STATE_DICT_TYPES 中取值
    @parametrize("add_to_fsdp_module", [False, True])  # 参数化测试：是否将其添加到 FSDP 模块，值为 False 和 True
    def test_shard_full_optim_state_dict_unmanaged_params(
        self,
        state_dict_type: StateDictType,
        add_to_fsdp_module: bool,
        # 更多参数在这里省略，由 `**new_model_kwargs` 接收
    ):
        """
        Tests :meth:`shard_full_optim_state_dict` when there are unmanaged
        parameters.
          - If ``add_to_fsdp_module=True``, then the unmanaged parameters are
          added to a module to be wrapped with FSDP, in which case there should
          be an error since we require that all unflattened parameter
          comprising a flat parameter have the same scalar state (e.g. Adam
          "step") but the added parameter is missing its entry.
          - If ``add_to_fsdp_module=False``, then the unmanaged parameters are
          added to a module not to be wrapped with FSDP, in which case there
          should be no error (emulating model parallel use cases where some
          parameters may be managed externally to FSDP).
        We do not separately test unmanaged parameters for
        :meth:`scatter_full_optim_state_dict` and `flatten_sharded_optim_state_dict`
        to save CI cost since it call into the same subroutine
        :meth:`_flatten_optim_state_dict`.
        """
        # 根据不同的 state_dict_type 设置 use_optim_input 的值，用于测试
        if state_dict_type == StateDictType.SHARDED_STATE_DICT:
            use_optim_input = [False]
        else:
            use_optim_input = [False, True]
        # 运行子测试，测试 _test_shard_full_optim_state_dict_unmanaged_params 方法
        self.run_subtests(
            {"use_optim_input": use_optim_input},
            self._test_shard_full_optim_state_dict_unmanaged_params,
            state_dict_type=state_dict_type,
            add_to_fsdp_module=add_to_fsdp_module,
        )

    def _test_shard_full_optim_state_dict_unmanaged_params(
        self,
        state_dict_type: StateDictType,
        add_to_fsdp_module: bool,
        use_optim_input: bool,
    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", STATE_DICT_TYPES)
    @parametrize("use_multiple_param_groups", [False, True])
    def test_rekey_optim_state_dict_to_ids(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
    ):
        """Tests :meth:`rekey_optim_state_dict` with the new keys being
        parameter IDs by checking that a wrapped model (i.e. with FSDP modules)
        can rekey its optimizer state dict to match that of an equivalent
        non-wrapped model (i.e. without FSDP modules)."""
        # 根据不同的 state_dict_type 设置 use_optim_input 的值，用于测试
        if state_dict_type == StateDictType.SHARDED_STATE_DICT:
            use_optim_input = [False]
        else:
            use_optim_input = [False, True]
        # 运行子测试，测试 _test_rekey_optim_state_dict_to_ids 方法
        self.run_subtests(
            {"use_optim_input": use_optim_input},
            self._test_rekey_optim_state_dict_to_ids,
            state_dict_type=state_dict_type,
            use_multiple_param_groups=use_multiple_param_groups,
        )

    @skip_if_lt_x_gpu(2)
    def _test_rekey_optim_state_dict_to_ids(
        self,
        state_dict_type: StateDictType,
        use_multiple_param_groups: bool,
        use_optim_input: bool,
        ):
            # 设置迭代次数为3次
            NUM_ITERS = 3
            # 运行一个包装模型进行几次迭代
            model1, optim1, optim_input1 = self._init_nested_model(
                wrap=True,
                use_multiple_param_groups=use_multiple_param_groups,
            )
            self._step_model(model1, optim1, num_iters=NUM_ITERS)
            # 如果要保存完整的状态字典
            if state_dict_type == StateDictType.FULL_STATE_DICT:
                # 使用FSDP库函数获取完整的优化器状态字典，包括优化器输入
                fsdp_osd = (
                    FSDP.full_optim_state_dict(model1, optim1, optim_input1)
                    if use_optim_input
                    else FSDP.full_optim_state_dict(model1, optim1)
                )
                # 使用广播方式而不是torch.save()/torch.load()来确保所有进程都有完整的状态字典
                fsdp_osd = self._broadcast_full_osd(fsdp_osd)
            else:
                # 使用FSDP库函数获取分片的优化器状态字典
                fsdp_osd = FSDP.sharded_optim_state_dict(model1, optim1)
            # 运行一个非包装模型进行几次迭代
            model2, optim2, optim_input2 = self._init_nested_model(
                wrap=False,
                use_multiple_param_groups=use_multiple_param_groups,
            )
            self._step_model(model2, optim2, num_iters=NUM_ITERS)
            # 使用非包装模型的参数ID重新调整包装模型的优化器状态字典
            rekeyed_osd = (
                FSDP.rekey_optim_state_dict(
                    fsdp_osd,
                    OptimStateKeyType.PARAM_ID,
                    model2,
                    optim_input=optim_input2,
                )
                if use_optim_input
                else FSDP.rekey_optim_state_dict(
                    fsdp_osd,
                    OptimStateKeyType.PARAM_ID,
                    model2,
                    optim=optim2,
                )
            )
            # 检查重新调整后的字典和实际字典是否相同
            osd = optim2.state_dict()
            check_same_param_keys = True
            self._check_same_param_groups(
                rekeyed_osd,
                osd,
                check_same_param_keys=check_same_param_keys,
            )
            self._check_same_state(
                rekeyed_osd,
                osd,
                check_same_param_keys=check_same_param_keys,
            )
            # 作为健全性检查，检查我们是否可以加载并运行几次迭代
            if state_dict_type != StateDictType.SHARDED_STATE_DICT:
                optim2.load_state_dict(rekeyed_osd)
                self._step_model(model2, optim2, num_iters=NUM_ITERS)

    @skip_if_lt_x_gpu(2)
    # 定义测试方法 test_rekey_optim_state_dict_to_names，用于测试 rekey_optim_state_dict 方法
    def test_rekey_optim_state_dict_to_names(self):
        """Tests :meth:`rekey_optim_state_dict` with the new keys being
        parameter names by checking that a non-wrapped model (i.e. without FSDP
        modules) can rekey its optimizer state dict to match the expected
        output of :meth:`full_optim_state_dict`, hence be sharded using
        :meth:`shard_full_optim_state_dict`, and finally match the per-rank
        optimizer state dict of a wrapped model (i.e. with FSDP modules)."""
        
        # 运行子测试，参数包括 use_optim_input 和 _test_rekey_optim_state_dict_to_names 方法
        self.run_subtests(
            {"use_optim_input": [False, True]},  # 提供 use_optim_input 参数的值 [False, True]
            self._test_rekey_optim_state_dict_to_names,  # 调用 _test_rekey_optim_state_dict_to_names 方法进行测试
            use_multiple_param_groups=False,  # 设置 use_multiple_param_groups 参数为 False
        )

    # 定义私有测试方法 _test_rekey_optim_state_dict_to_names
    def _test_rekey_optim_state_dict_to_names(
        self,
        use_multiple_param_groups: bool,  # 参数：是否使用多个参数组
        use_optim_input: bool,  # 参数：是否使用优化器输入
    ):
        NUM_ITERS = 3
        # Run a wrapped model for a few iterations
        # 初始化一个嵌套模型，并运行几次迭代
        model1, optim1, optim_input1 = self._init_nested_model(
            wrap=True,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model1, optim1, num_iters=NUM_ITERS)
        # Run a non-wrapped model for a few iterations
        # 初始化一个未包装的模型，并运行几次迭代
        model2, optim2, optim_input2 = self._init_nested_model(
            wrap=False,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model2, optim2, num_iters=NUM_ITERS)
        # Re-key the non-wrapped model's optimizer state dict using parameter
        # names (still according to itself)
        # 使用参数名重新对未包装模型的优化器状态字典进行重键，仍然根据自身定义
        osd2 = optim2.state_dict()
        rekeyed_osd = (
            FSDP.rekey_optim_state_dict(
                osd2,
                OptimStateKeyType.PARAM_NAME,
                model2,
                optim_input=optim_input2,
            )
            if use_optim_input
            else FSDP.rekey_optim_state_dict(
                osd2,
                OptimStateKeyType.PARAM_NAME,
                model2,
                optim=optim2,
            )
        )
        # Shard the non-wrapped model's re-keyed optimizer state dict, which
        # maps back to (flattened) parameter IDs
        # 对未包装模型的重键优化器状态字典进行分片，该字典映射回（扁平化的）参数ID
        sharded_osd = (
            FSDP.shard_full_optim_state_dict(
                rekeyed_osd,
                model1,
                optim_input=optim_input1,
            )
            if use_optim_input
            else FSDP.shard_full_optim_state_dict(
                rekeyed_osd,
                model1,
                optim=optim1,
            )
        )
        # Check that this sharded optimizer state dict matches the wrapped
        # model's per-rank optimizer state dict
        # 检查这个分片的优化器状态字典是否与包装模型的每个等级的优化器状态字典匹配
        osd1 = optim1.state_dict()
        check_same_param_keys = True
        self._check_same_param_groups(
            sharded_osd,
            osd1,
            check_same_param_keys=check_same_param_keys,
        )
        self._check_same_state(
            sharded_osd,
            osd1,
            check_same_param_keys=check_same_param_keys,
        )
        # As a sanity check, check that we can load and run a few iterations
        # 作为健全性检查，检查我们是否可以加载并运行几次迭代
        optim1.load_state_dict(sharded_osd)
        self._step_model(model1, optim1, num_iters=NUM_ITERS)
    def test_optim_input_warning(self):
        """Tests that passing the ``optim_input`` argument into optimizer state
        checkpointing APIs issues a warning."""
        
        def should_check_method(method_name: str):
            # 检查每个方法，因为它们都接受 `optim_input` 参数
            return method_name not in (
                "sharded_optim_state_dict",
                "flatten_sharded_optim_state_dict",
            )

        def get_warning_context():
            # 返回一个上下文函数，用于检查警告
            warning_regex = "`optim_input` argument is deprecated"
            return self.assertWarnsRegex(
                expected_warning=FutureWarning, expected_regex=warning_regex
            )

        self._run_on_all_optim_state_apis(
            should_check_method, get_warning_context, fsdp_kwargs=None
        )

    def _run_on_all_optim_state_apis(
        self,
        should_check_method_fn: Callable[[str], bool],
        context_fn: Callable,
        fsdp_kwargs: Optional[Dict[str, Any]],
    ):
        """Runs a test on all optimizer state APIs."""
        # 跳过如果 GPU 数量小于 2 的情况
        @skip_if_lt_x_gpu(2)
        # 参数化测试，对于各种状态字典类型进行测试
        @parametrize("state_dict_type", STATE_DICT_TYPES)
    # 定义一个测试函数，用于测试在第一个参数没有优化器状态时（例如未使用或冻结），
    # 保存和加载 Adam 优化器状态字典的情况。
    def test_save_load_without_0th_param_state(self, state_dict_type: StateDictType):
        """
        Tests saving and loading an optim state dict for Adam optimizer (i.e.
        any optimizer with a "step" key in its state) when the first parameter
        does not have optimizer state (e.g. unused or frozen).
        """

        # 定义一个简单的神经网络模型
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(5, 5)
                self.lin2 = nn.Linear(5, 5)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 不使用 `lin1`，这是传递给优化器并检查是否具有 "step" 状态的参数
                return self.relu(self.lin2(x))

        # 创建一个 CUDA 上的模型实例
        model = Model().cuda()
        # 将模型的某些部分包装为 FSDP（Fully Sharded Data Parallelism）格式
        model.lin1 = FSDP(model.lin1)
        model.lin2 = FSDP(model.lin2)
        fsdp_model = FSDP(model)
        # 使用 Adam 优化器对模型参数进行优化
        optim = torch.optim.Adam(
            fsdp_model.parameters(), lr=1e-2
        )  # 或者任何带有 "step" 的优化器

        # 运行一次迭代以构造优化器的状态
        device = torch.device("cuda")
        inp = torch.randn((2, 5), device=device)
        loss = fsdp_model(inp).sum()
        loss.backward()
        optim.step()

        # 检查保存和加载状态字典不会出错
        if state_dict_type == StateDictType.FULL_STATE_DICT:
            # 获取完整的优化器状态字典
            fsdp_osd = FSDP.full_optim_state_dict(fsdp_model, optim, rank0_only=False)
            # 对完整的优化器状态字典进行分片
            flattened_osd = FSDP.shard_full_optim_state_dict(fsdp_osd, fsdp_model)
        elif state_dict_type == StateDictType.SHARDED_STATE_DICT:
            # 获取分片的优化器状态字典
            fsdp_osd = FSDP.sharded_optim_state_dict(fsdp_model, optim)
            # 将分片的优化器状态字典展开
            flattened_osd = FSDP.flatten_sharded_optim_state_dict(
                fsdp_osd, fsdp_model, optim
            )
        # 加载优化器的状态字典
        optim.load_state_dict(flattened_osd)
        
        # `__setstate__()` 方法会检查第一个参数，以确定是否 "step" 表示为张量或浮点数，
        # 因此其状态必须非空。
        
        # 再次运行一次迭代作为健全性检查
        inp = torch.randn((2, 5), device=device)
        loss = fsdp_model(inp).sum()
        loss.backward()
        optim.step()
    def test_optim_state_without_param_groups(self):
        # 定义一个简单的神经网络模型类
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                # 创建一个包含线性层和ReLU激活函数的序列模型
                self.net1 = nn.Sequential(nn.Linear(2, 4), nn.ReLU())

            def forward(self, x):
                return self.net1(x)

        # 创建一个包含FSDP的模型，并移至GPU上
        model = FSDP(SimpleModel().cuda())
        # 使用Adam优化器，将模型参数传入
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 训练一步，保存原始优化器状态字典和原始优化器参数组
        batch = torch.rand(3, 2, device=torch.device("cuda"))
        for param in model.parameters():
            if param.requires_grad:
                t = torch.zeros_like(param)
                param.grad = torch.autograd.Variable(t)
        optim.step()
        loss = model(batch).sum()
        loss.backward()

        # 深拷贝原始的优化器状态字典
        original_osd = deepcopy(optim.state_dict())
        original_osd_no_param_groups = deepcopy(original_osd)
        # 从优化器状态字典中手动移除param_groups
        original_param_groups = deepcopy(
            original_osd_no_param_groups.pop("param_groups")
        )
        # 将不包含param_groups的优化器状态字典传递给FSDP
        original_fsdp_optim_state_dict = deepcopy(
            FSDP.optim_state_dict(
                model, optim, optim_state_dict=original_osd_no_param_groups
            )
        )
        # 检查由FSDP分片的state_dict不包含param_groups
        self.assertEqual(None, original_fsdp_optim_state_dict.get("param_groups"))

        # 再训练一步，使优化器处于不同状态
        for param in model.parameters():
            if param.requires_grad:
                t = torch.zeros_like(param)
                param.grad = torch.autograd.Variable(t)
        optim.step()
        loss = model(batch).sum()
        loss.backward()

        # 准备用于加载优化器状态的state_dict_to_load
        state_dict_to_load = FSDP.optim_state_dict_to_load(
            model, optim, original_fsdp_optim_state_dict
        )
        # 在加载优化器状态之前，手动添加param_groups到state_dict_to_load
        state_dict_to_load["param_groups"] = original_param_groups
        optim.load_state_dict(state_dict_to_load)
        # 断言优化器状态加载后与原始状态一致
        self.assertEqual(original_osd, optim.state_dict())

        # 获取FSDP的优化器状态字典
        fsdp_optim_state = FSDP.optim_state_dict(model, optim)
        # 检查FSDP优化器状态字典与原始的FSDP状态字典是否一致，包括参数键的检查
        self._check_same_state(
            original_fsdp_optim_state_dict, fsdp_optim_state, check_same_param_keys=True
        )
        # 断言优化器状态字典中的param_groups与原始param_groups一致
        self.assertEqual(original_param_groups, optim.state_dict()["param_groups"])

    @skip_if_lt_x_gpu(2)
    def test_with_empty_optimizer_state(self):
        # 创建一个包含FSDP的模型，并移至GPU上
        model = FSDP(TestDummyModel().cuda())
        # 使用Adam优化器，将模型参数传入
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 获取优化器的状态字典
        state_dict = optim.state_dict()
        # 获取由FSDP分片的优化器状态字典
        gathered_state_dict = FSDP.optim_state_dict(model, optim)
        # 断言由FSDP分片的状态字典中的state与原始状态字典中的state一致
        self.assertEqual(gathered_state_dict["state"], state_dict["state"])
    # 定义一个方法，用于测试加载优化器状态字典的函数，并使用给定的参数进行配置
    def _test_load_optim_state_with_optim_state_dict(
        self,
        model_class: _ModelClass,
        state_dict_settings: StateDictSettings,
        use_multiple_param_groups: bool,
        halve_world_size: bool,
        use_diff_optim_inputs: bool,
        num_iters: int,
        **new_model_kwargs,
    ):
        # 如果 GPU 小于两个则跳过测试
        @skip_if_lt_x_gpu(2)
        def test_interface_arguments(self):
            # 创建一个使用 FSDP（Fully Sharded Data Parallel）封装的模型
            model = FSDP(TestDummyModel().cuda())
            # 使用 Adam 优化器来优化模型参数，学习率为 0.01
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)

            # 定义一个执行模型前向、反向传播和优化步骤的函数
            def step():
                loss = model(model.get_input())
                loss.backward(loss)
                optim.step()

            # 执行一次优化步骤
            step()
            # 深度复制当前优化器的状态字典
            original_osd = deepcopy(optim.state_dict())
            # 使用 FSDP 封装的方法获取优化器状态字典
            osd = FSDP.optim_state_dict(model, optim, optim_state_dict=original_osd)
            # 检查两个优化器状态字典是否相同
            self._check_same_state(
                FSDP.optim_state_dict(model, optim), osd, check_same_param_keys=True
            )
            # 再执行一次优化步骤
            step()
            # 获取用于加载的优化器状态字典
            osd_to_load = FSDP.optim_state_dict_to_load(
                model, optim, osd, load_directly=True
            )
            # 检查当前优化器状态字典和最初的状态字典是否相同
            self._check_same_state(
                optim.state_dict(), original_osd, check_same_param_keys=True
            )

            # 测试默认设置
            osd = FSDP.optim_state_dict(model, optim, optim_state_dict=original_osd)
            # 检查是否有分片张量存在于优化器状态字典中，并且不在 CUDA 上
            for state in osd["state"].values():
                for s in state.values():
                    self.assertFalse(isinstance(s, ShardedTensor))
                    self.assertFalse(s.is_cuda)

            # 测试带有分片状态字典但不使用 offload_to_cpu
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(),
                ShardedOptimStateDictConfig(offload_to_cpu=False),
            ):
                osd = FSDP.optim_state_dict(model, optim, optim_state_dict=original_osd)
                # 检查是否所有的分片张量都被正确识别，并且第一个本地分片在 CUDA 上
                for state in osd["state"].values():
                    for s in state.values():
                        if s.dim() == 0:
                            continue
                        self.assertTrue(isinstance(s, ShardedTensor))
                        if s._local_shards[0]:
                            self.assertTrue(s._local_shards[0].tensor.is_cuda)

            # 测试全状态字典模式并且仅使用 rank0_only
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(),
                FullOptimStateDictConfig(
                    offload_to_cpu=True,
                    rank0_only=True,
                ),
            ):
                osd = FSDP.optim_state_dict(model, optim, optim_state_dict=original_osd)
                # 如果当前进程不是主进程（rank > 0），则期望优化器状态字典为空
                if dist.get_rank() > 0:
                    self.assertEqual(osd, {})
                else:
                    # 检查是否所有张量都不在 CUDA 上，并且不是分片张量
                    for state in osd["state"].values():
                        for s in state.values():
                            if s.dim() == 0:
                                continue
                            self.assertFalse(s.is_cuda)
                            self.assertFalse(isinstance(s, ShardedTensor))
    def test_state_dict_with_none_tensor_state(self):
        # 定义测试函数，用于测试在存在或不存在张量状态的情况下的模型状态字典操作
        def _run_test(use_orig_params, optimizer_has_tensor_state):
            # 创建 FSDP 模型对象，将其移动到 GPU 上，并根据参数决定是否使用原始参数
            model = FSDP(TestDummyModel().cuda(), use_orig_params=use_orig_params)
            # 根据是否存在张量状态选择优化器类（Adam 或 SGD）
            optimizer_cls = (
                torch.optim.Adam if optimizer_has_tensor_state else torch.optim.SGD
            )
            # 初始化优化器，传入模型参数和学习率
            optim = optimizer_cls(model.parameters(), lr=1e-2)

            # 定义模型执行步骤的函数
            def step():
                # 计算模型输出并计算损失
                loss = model(model.get_input())
                # 反向传播损失
                loss.backward(loss)
                # 优化器执行一步更新
                optim.step()

            # 执行模型步骤
            step()
            # 深拷贝当前优化器状态字典
            original_osd = deepcopy(optim.state_dict())
            # 遍历原始优化器状态字典中的每一个状态
            for state in original_osd["state"].values():
                # 向每个状态中添加自定义值
                state["value1"] = 2.74
                state["value2"] = None

            # 获取 FSDP 下的优化器状态字典
            osd = FSDP.optim_state_dict(model, optim, optim_state_dict=original_osd)
            # 将优化器状态字典转换为加载的格式
            osd_to_load = FSDP.optim_state_dict_to_load(model, optim, osd)
            # 遍历待加载的优化器状态字典中的每一个状态
            for state in osd_to_load["state"].values():
                # 断言每个状态中的值是否正确加载
                self.assertEqual(state["value1"], 2.74)
                self.assertEqual(state["value2"], None)

        # 运行子测试，测试不同的参数组合
        self.run_subtests(
            {
                "use_orig_params": [False, True],
                "optimizer_has_tensor_state": [False, True],
            },
            _run_test,
        )

    @skip_if_lt_x_gpu(2)
    def test_with_no_shard(self):
        # 测试在没有分片策略的情况下的模型状态字典操作
        def _run_test(use_orig_params: bool) -> None:
            # 创建 FSDP 模型对象，将其移动到 GPU 上，设置为无分片策略，并根据参数决定是否使用原始参数
            model = FSDP(
                TestDummyModel().cuda(),
                sharding_strategy=ShardingStrategy.NO_SHARD,
                use_orig_params=use_orig_params,
            )
            # 初始化 Adam 优化器，传入模型参数和学习率
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)

            # 定义模型执行步骤的函数
            def step():
                # 计算模型输出并计算损失
                loss = model(model.get_input())
                # 反向传播损失
                loss.backward(loss)
                # 优化器执行一步更新
                optim.step()

            # 执行模型步骤
            step()

            # 深拷贝当前优化器状态字典
            original_osd = deepcopy(optim.state_dict())

            # 获取 FSDP 下的优化器状态字典
            osd = FSDP.optim_state_dict(model, optim)
            # 将优化器状态字典转换为加载的格式
            osd_to_load = FSDP.optim_state_dict_to_load(model, optim, osd)
            # 加载转换后的优化器状态字典到优化器中
            optim.load_state_dict(osd_to_load)

            # 获取加载后的新优化器状态字典
            new_osd = optim.state_dict()

            # 断言加载前后状态字典的一致性
            self.assertEqual(original_osd, new_osd)

        # 运行子测试，测试不同的参数组合
        self.run_subtests({"use_orig_params": [False, True]}, _run_test)

    @skip_if_lt_x_gpu(2)
    # 定义测试方法，测试在没有梯度的情况下的模型行为
    def test_no_grad(self):
        # 创建一个带有 no_grad 标志的测试模型，并移至 GPU
        model = TestDummyModel(no_grad=True).cuda()
        # 使用深拷贝创建 FSDP 模型，使用原始参数
        fsdp_model = FSDP(deepcopy(model), use_orig_params=True)
        # 使用 Adam 优化器初始化 FSDP 模型的参数
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)

        # 进行5次迭代
        for i in range(5):
            # 如果迭代次数是奇数
            if i % 2 == 1:
                # 允许 net1 的第一个层的权重和偏置的梯度计算
                fsdp_model.net1[0].weight.requires_grad = True
                fsdp_model.net1[0].bias.requires_grad = True
            else:
                # 禁止 net1 的第一个层的权重和偏置的梯度计算
                fsdp_model.net1[0].weight.requires_grad = False
                fsdp_model.net1[0].bias.requires_grad = False
            # 获取模型的输入批次
            batch = fsdp_model.get_input()
            # 计算模型输出的总损失
            loss = fsdp_model(batch).sum()
            # 执行反向传播计算梯度
            loss.backward()
            # 在优化器上执行一步优化
            fsdp_optim.step()
            # 深拷贝当前优化器状态字典
            orig_state_dict = deepcopy(fsdp_optim.state_dict())
            # 获取当前 FSDP 模型和优化器的状态字典
            optim_state_dict = FSDP.optim_state_dict(fsdp_model, fsdp_optim)
            # 将优化器状态字典加载到模型和优化器中
            FSDP.optim_state_dict_to_load(
                fsdp_model,
                fsdp_optim,
                FSDP.optim_state_dict(fsdp_model, fsdp_optim),
                load_directly=True,
            )

            # 检查当前优化器状态字典和原始状态字典是否相同
            self._check_same_state(
                fsdp_optim.state_dict(),
                orig_state_dict,
                check_same_param_keys=True,
            )
# 调用一个参数化测试类的实例化函数，并传入 TestFSDPOptimState 类作为参数
instantiate_parametrized_tests(TestFSDPOptimState)

# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 如果是主程序，则执行测试函数
    run_tests()
```