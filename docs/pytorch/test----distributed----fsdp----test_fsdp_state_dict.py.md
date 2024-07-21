# `.\pytorch\test\distributed\fsdp\test_fsdp_state_dict.py`

```
# Owner(s): ["oncall: distributed"]

# 导入所需的模块和库
import io  # 提供 I/O 操作
import itertools  # 提供迭代工具
import sys  # 提供系统相关的功能
from contextlib import nullcontext  # 提供上下文管理工具，用于创建空上下文
from copy import deepcopy  # 提供深拷贝功能
from functools import partial  # 支持创建部分函数
from typing import Any, Dict  # 提供类型提示功能

import torch  # PyTorch 主库
import torch.nn as nn  # PyTorch 神经网络模块
from torch import distributed as dist  # PyTorch 分布式功能
from torch.distributed._shard.sharded_tensor import (  # 分布式张量相关功能
    init_from_local_shards,  # 从本地碎片初始化张量
    Shard,  # 分片对象
    ShardedTensor,  # 分布式张量对象
)
from torch.distributed._state_dict_utils import (  # 状态字典工具
    _all_gather_sharded_tensor,  # 收集所有分布式张量
    _gather_state_dict,  # 收集状态字典
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # 检查点包装器
    apply_activation_checkpointing,  # 应用激活检查点
    checkpoint_wrapper,  # 检查点包装器
    CheckpointImpl,  # 检查点实现
)
from torch.distributed.fsdp import (  # FSDP 功能模块
    CPUOffload,  # CPU 卸载
    FullStateDictConfig,  # 完整状态字典配置
    FullyShardedDataParallel as FSDP,  # 全部分片数据并行
    LocalStateDictConfig,  # 本地状态字典配置
    MixedPrecision,  # 混合精度
    ShardedStateDictConfig,  # 分片状态字典配置
    StateDictType,  # 状态字典类型
)
from torch.distributed.fsdp._common_utils import FSDP_PREFIX  # FSDP 前缀
from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM  # 非分片参数
from torch.distributed.fsdp.wrap import (  # FSDP 包装相关
    enable_wrap,  # 启用包装
    ModuleWrapPolicy,  # 模块包装策略
    wrap,  # 包装
)
from torch.nn import (  # PyTorch 神经网络模块
    Linear,  # 线性层
    Module,  # 模块基类
    TransformerDecoderLayer,  # Transformer 解码器层
    TransformerEncoderLayer,  # Transformer 编码器层
)
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.optim import SGD  # SGD 优化器
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 分布式测试工具
from torch.testing._internal.common_fsdp import (  # FSDP 测试工具
    _assert_module_states,  # 断言模块状态
    _broadcast_state_dict,  # 广播状态字典
    _get_state_dict,  # 获取状态字典
    _zero_model,  # 模型参数清零
    CUDAInitMode,  # CUDA 初始化模式
    FSDPInitMode,  # FSDP 初始化模式
    FSDPTest,  # FSDP 测试
    get_full_params,  # 获取完整参数
    SkipModel,  # 跳过模型
    TransformerWithSharedParams,  # 具有共享参数的 Transformer
)
from torch.testing._internal.common_utils import (  # 通用测试工具
    instantiate_parametrized_tests,  # 实例化参数化测试
    parametrize,  # 参数化装饰器
    run_tests,  # 运行测试
    TEST_WITH_DEV_DBG_ASAN,  # 是否使用 dev-asan 调试
)

# 如果分布式不可用，则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用 dev-asan 调试，则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义一些常量
INNER_SHAPE = [4, 4]  # 内部形状
OUTER_SHAPE = [4, 5]  # 外部形状
BUFFER_SHAPE = [5, 5]  # 缓冲区形状

NON_ROOT_FSDP_PREFIX = "non_fsdp_lin"  # 非根 FSDP 前缀

# 支持的状态字典实现类型
_UNFLATTENED_STATE_DICT_IMPLS = ["state_dict", "sharded_state_dict"]  # 非扁平化状态字典实现
_FLATTENED_STATE_DICT_IMPLS = ["local_state_dict"]  # 扁平化状态字典实现
_SUPPORTED_STATE_DICT_IMPLS = (
    _UNFLATTENED_STATE_DICT_IMPLS + _FLATTENED_STATE_DICT_IMPLS
)  # 支持的状态字典实现类型列表

# 状态字典映射关系
STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,  # 完整状态字典类型
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,  # 本地状态字典类型
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,  # 分片状态字典类型
}


class Model(Module):
    """
    示例模型类，继承自 nn.Module
    """

    def __init__(
        self,
        wrap_fsdp,
        register_buffers=False,
        ignore_inner=False,
        mixed_precision=False,
        process_group=None,
        # 以下省略部分构造函数参数
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个内部的线性层，其形状由INNER_SHAPE定义
        self.inner = Linear(*INNER_SHAPE)
        # 如果需要注册缓冲区
        if register_buffers:
            # 为内部线性层注册一个名为"buffer"的缓冲区，内容为随机生成的张量
            self.inner.register_buffer("buffer", torch.randn(BUFFER_SHAPE))
            # 为内部线性层注册一个名为"non_persistent_buffer"的非持久化缓冲区，内容为随机生成的张量
            self.inner.register_buffer(
                "non_persistent_buffer", torch.randn(BUFFER_SHAPE), persistent=False
            )
        # 如果需要使用FSDP进行包装
        if wrap_fsdp:
            # 将内部线性层用FSDP进行封装，可能忽略内部线性层，可能使用混合精度计算
            self.inner = FSDP(
                self.inner,
                ignored_modules=([self.inner] if ignore_inner else []),
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
                if mixed_precision
                else None,
                process_group=process_group,
            )
        # 创建一个外部的线性层，其形状由OUTER_SHAPE定义
        self.outer = Linear(*OUTER_SHAPE)
        # 如果需要注册缓冲区
        if register_buffers:
            # 为外部线性层注册一个名为"buffer"的缓冲区，内容为随机生成的张量
            self.outer.register_buffer("buffer", torch.randn(BUFFER_SHAPE))
            # 为外部线性层注册一个名为"non_persistent_buffer"的非持久化缓冲区，内容为随机生成的张量
            self.outer.register_buffer(
                "non_persistent_buffer", torch.randn(BUFFER_SHAPE), persistent=False
            )

    def forward(self, x):
        # 前向传播两次
        # 对输入x分别通过内部线性层进行计算得到i和j
        i = self.inner(x)
        j = self.inner(x)
        # 将i和j相加后，通过外部线性层进行计算，并返回结果
        return self.outer(i + j)
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 设定随机种子为0，确保结果可重复
        torch.manual_seed(0)
        # 定义神经网络结构 net1，包括线性层和ReLU激活函数
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        # 定义神经网络结构 net2，包括线性层和ReLU激活函数
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        # 将 net2 赋值给 net3，即 net3 和 net2 指向同一个对象
        self.net3 = self.net2
        # 创建一个随机可学习的参数
        self.random_parameter = nn.Parameter(torch.Tensor(10))
        # 将 random_parameter 赋值给 shared_parameter，即 shared_parameter 和 random_parameter 指向同一个对象
        self.shared_parameter = self.random_parameter

    def forward(self, x):
        # 执行前向传播，先经过 net1，再经过 net2，最后经过 net3
        return self.net3(self.net2(self.net1(x)))

    def get_input(self):
        # 返回一个在 CUDA 设备上生成的随机张量作为输入
        return torch.rand(8, 8, device="cuda")


class TestFSDPStateDict(FSDPTest):
    @property
    def world_size(self):
        # 返回当前 CUDA 设备数量和2中的较小值
        return min(torch.cuda.device_count(), 2)

    def _broadcast_state_dict(self, model, state_dict):
        # TODO (rohan-varma): remove model
        # 调用全局函数 _broadcast_state_dict，传递当前进程的排名和状态字典，返回广播后的状态字典
        return _broadcast_state_dict(self.rank, state_dict)

    def _state_compare(self, model, model_new, assert_fn, state_generator="parameters"):
        # 获取 model 和 model_new 的指定状态（默认为参数）列表
        state_base = list(getattr(model, state_generator)())
        state_new = list(getattr(model_new, state_generator)())
        # 断言两个模型状态列表的长度相等
        self.assertEqual(len(state_base), len(state_new))
        # 调用传入的 assert_fn 函数比较两个状态列表
        assert_fn(state_base, state_new)

    def _compare_models(
        self, model, model_new, assert_fn, check_fp16=False, check_buffers=True
    ):
        # 断言 assert_fn 必须是 self.assertEqual 或 self.assertNotEqual 中的一个
        assert assert_fn in (self.assertEqual, self.assertNotEqual)
        # 使用 FSDP.summon_full_params 包装 model 和 model_new，确保全参数模式
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_new):
                # 比较 model 和 model_new 的状态（默认为参数）
                self._state_compare(model, model_new, assert_fn)
                if check_buffers:
                    # 检查模型是否有缓冲区
                    has_buffers = any(
                        len(list(m.buffers())) for m in (model, model_new)
                    )
                    if has_buffers:
                        # 如果有缓冲区，比较模型的缓冲区状态
                        self._state_compare(
                            model, model_new, assert_fn, state_generator="buffers"
                        )
                if check_fp16:
                    # 检查模型的参数是否为 torch.float16 类型
                    for tensor in model_new.parameters():
                        self.assertEqual(tensor.dtype, torch.float16)

    def _get_simple_nested_model(
        self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs
    ):
        if wrap:
            # 创建两个线性层，并将其移动到 CUDA 设备上
            lin1 = nn.Linear(10, 10, bias=False).cuda()
            lin2 = nn.Linear(10, 10, bias=False).cuda()
            if checkpoint_wrap:
                # 如果需要，对每个线性层进行检查点包装
                lin1 = checkpoint_wrapper(lin1)
                lin2 = checkpoint_wrapper(lin2)
            # 创建一个包含 FSDP 的序列模型 seq
            seq = nn.Sequential(FSDP(lin1, *fsdp_args, **fsdp_kwargs), lin2)
            if checkpoint_wrap:
                # 如果需要，对整个 seq 模型进行检查点包装
                seq = checkpoint_wrapper(seq)
            # 创建一个完全分布式的模型，包含 FSDP
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            # 创建一个简单的序列模型，包含两个线性层
            model = nn.Sequential(
                nn.Linear(10, 10, bias=False).cuda(),
                nn.Linear(10, 10, bias=False).cuda(),
            )
        # 返回创建的模型
        return model
    # 创建一个简单的模型，可能包装了 FSDP，根据条件决定是否使用检查点包装
    def _get_simple_model(self, *fsdp_args, checkpoint_wrap=False, **fsdp_kwargs):
        # 创建一个在 GPU 上运行的没有偏置的线性层
        lin = nn.Linear(10, 10, bias=False).cuda()
        # 如果指定了需要检查点包装，则对线性层进行检查点包装
        if checkpoint_wrap:
            lin = checkpoint_wrapper(lin)
        # 使用 FSDP 封装线性层创建模型
        model = FSDP(lin, *fsdp_args, **fsdp_kwargs)
        # 返回创建的模型
        return model

    # 创建一个嵌套多缓冲区的模型，可能包装了 FSDP，根据条件决定是否使用检查点包装
    def _get_multibuffer_nested_model(
        self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs
    ):
        # 定义全精度类型为 torch.float32
        full_p = torch.float32
        # 获取混合精度设置，如果定义了则弹出
        lin_mp = fsdp_kwargs.pop("mixed_precision", None)
        # 根据混合精度设置创建相应的 BatchNorm 的混合精度对象
        bn_mp = (
            MixedPrecision(param_dtype=full_p, reduce_dtype=full_p, buffer_dtype=full_p)
            if lin_mp
            else None
        )
        # 如果需要包装模型
        if wrap:
            # 创建两个在 GPU 上运行的没有偏置的线性层和 BatchNorm 层
            lin1 = nn.Linear(10, 10, bias=False).cuda()
            bn1 = nn.BatchNorm1d(10).cuda()
            lin2 = nn.Linear(10, 10, bias=False).cuda()
            # 如果指定了需要检查点包装，则对线性层和 BatchNorm 层进行检查点包装
            if checkpoint_wrap:
                lin1 = checkpoint_wrapper(lin1)
                bn1 = checkpoint_wrapper(bn1)
                lin2 = checkpoint_wrapper(lin2)
            # 创建一个序列模型，其中包含 FSDP 封装的线性层和 BatchNorm 层，以及未封装的线性层
            seq = nn.Sequential(
                FSDP(lin1, *fsdp_args, mixed_precision=lin_mp, **fsdp_kwargs),
                FSDP(bn1, *fsdp_args, mixed_precision=bn_mp, **fsdp_kwargs),
                lin2,
            )
            # 如果指定了需要检查点包装，则对整个序列模型进行检查点包装
            if checkpoint_wrap:
                seq = checkpoint_wrapper(seq)
            # 使用 FSDP 封装序列模型创建最终的模型
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            # 如果不需要包装模型，则创建一个简单的序列模型，包含两个线性层和一个 BatchNorm 层
            model = nn.Sequential(
                nn.Linear(10, 10, bias=False).cuda(),
                nn.BatchNorm1d(10).cuda(),
                nn.Linear(10, 10, bias=False).cuda(),
            )
        # 返回创建的模型
        return model

    # 创建一个不使用 FSDP 封装的根模块，可能包装了简单的嵌套模型
    def _get_non_fsdp_root_module(self, *fsdp_args, wrap=True, **fsdp_kwargs):
        # 定义一个内部类 FSDPContainer，继承自 nn.Module，包含一个非 FSDP 封装的线性层和两个 FSDP 封装的模型
        class FSDPContainer(nn.Module):
            def __init__(self, fsdp_1, fsdp_2):
                super().__init__()
                # 创建一个在 GPU 上运行的没有偏置的线性层
                self.non_fsdp_lin = nn.Linear(10, 10, bias=False).cuda()
                self.fsdp_1 = fsdp_1
                self.fsdp_2 = fsdp_2

            def forward(self, x):
                # 在 forward 方法中应用非 FSDP 封装的线性层和两个 FSDP 封装的模型
                x = self.non_fsdp_lin(x)
                x = self.fsdp_1(x)
                x = self.fsdp_2(x)
                return x

        # 返回创建的 FSDPContainer 类的实例，包含两个简单的嵌套模型
        return FSDPContainer(
            self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs),
            self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs),
        )

    # 获取状态字典管理器，接受模型、状态字典类型、是否在 rank0 和离线状态字典
    ):
        _state_dict_type = STATE_DICT_MAPPING[state_dict_type]
        # 根据 state_dict_type 获取相应的配置对象
        if state_dict_type == "state_dict":
            # 如果 state_dict_type 是 "state_dict"，创建 FullStateDictConfig 配置对象
            config = FullStateDictConfig(
                rank0_only=state_dict_rank0_and_offload,
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        elif state_dict_type == "local_state_dict":
            # 如果 state_dict_type 是 "local_state_dict"，创建 LocalStateDictConfig 配置对象
            config = LocalStateDictConfig(
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        elif state_dict_type == "sharded_state_dict":
            # 如果 state_dict_type 是 "sharded_state_dict"，创建 ShardedStateDictConfig 配置对象
            config = ShardedStateDictConfig(
                offload_to_cpu=state_dict_rank0_and_offload,
            )
        else:
            # 如果 state_dict_type 不是上述三者之一，则抛出异常
            raise ValueError("Unsupported state_dict_type")
        # 调用 FSDP 的 state_dict_type 方法，传入模型、状态字典类型和配置对象，并返回结果
        return FSDP.state_dict_type(model, _state_dict_type, config)

    def _validate_state_dict_contents(
        self, model, fsdp_state_dict, state_dict_rank0_and_offload, ignore_keys=None
    ):
        # 如果 state_dict_rank0_and_offload 为真
        if state_dict_rank0_and_offload:
            # 如果当前进程的 rank 是 0
            if self.rank == 0:
                # 断言 fsdp_state_dict 不为空字典
                self.assertNotEqual(fsdp_state_dict, {})
                # 遍历 fsdp_state_dict 中的每个键值对
                for key, tensor in fsdp_state_dict.items():
                    # 如果 ignore_keys 存在且当前键在 ignore_keys 中，跳过当前循环
                    if ignore_keys and key in ignore_keys:
                        continue
                    # 断言 tensor 的设备为 CPU 设备
                    self.assertEqual(
                        tensor.device,
                        torch.device("cpu"),
                        f"{key} is unexpectedly on device {tensor.device}",
                    )
            else:
                # 对于非 FSDP 根进程，非 FSDP 部分可能仍然在 rank 0 上有参数，因此暂时绕过此检查
                if isinstance(model, FSDP):
                    # 断言 fsdp_state_dict 为空字典
                    self.assertEqual(
                        fsdp_state_dict,
                        {},
                        f"Expected empty state_dict but got {fsdp_state_dict} on rank {dist.get_rank()}",
                    )

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize(
        "checkpoint_wrap",
        ["source", "dest", "both", "source_after_wrap", "both_after_wrap"],
    )
    @parametrize("rank0_only_and_offload", [False, True])
    # 测试函数，测试 FSDP 状态字典与激活检查点
    def test_fsdp_state_dict_with_activation_checkpoint(
        self, state_dict_type, checkpoint_wrap, rank0_only_and_offload
    ):
        """
        Tests saving the state dict, zeroing a target model's parameters, and
        loading the state dict, where the source and target models may have a
        checkpoint wrapper.
        """

        def apply_ac_to_linears(model) -> None:
            # 创建一个部分应用了checkpoint_wrapper的函数，设置不允许重入和不转移到CPU
            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            # 应用激活检查点到模型的线性层
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: isinstance(submodule, nn.Linear),
            )

        # 针对两个模型获取函数，依次测试
        for model_call in [
            partial(self._get_simple_model),  # 获取简单模型函数的部分应用
            partial(self._get_simple_nested_model),  # 获取简单嵌套模型函数的部分应用
        ]:
            # 根据checkpoint_wrap参数获取模型，并测试
            model = model_call(checkpoint_wrap=(checkpoint_wrap in ("source", "both")))
            # 若checkpoint_wrap为source_after_wrap或both_after_wrap，则应用激活检查点到线性层
            if checkpoint_wrap in ("source_after_wrap", "both_after_wrap"):
                apply_ac_to_linears(model)
            # 使用self._get_state_dict_mgr函数管理模型的状态字典
            with self._get_state_dict_mgr(
                model, state_dict_type, rank0_only_and_offload
            ):
                # 收集模型的状态字典
                state_dict = _gather_state_dict(_get_state_dict(model, False, False))
                # 可能将新模型包装在激活检查点包装器中，以测试保存/加载
                model_new = model_call(
                    checkpoint_wrap=(checkpoint_wrap in ("dest", "both"))
                )
                # 若checkpoint_wrap为both_after_wrap，则应用激活检查点到新模型的线性层
                if checkpoint_wrap == "both_after_wrap":
                    apply_ac_to_linears(model_new)
                # 将新模型的参数置零
                _zero_model(model_new)
                # 比较原模型和新模型，使用self.assertNotEqual断言
                self._compare_models(model, model_new, self.assertNotEqual)
                # 若rank0_only_and_offload为True，则广播模型的状态字典
                if rank0_only_and_offload:
                    state_dict = self._broadcast_state_dict(model, state_dict)
                # 使用模型的load_state_dict方法加载状态字典，使用严格模式
                model_new.load_state_dict(state_dict, strict=True)
                # 比较原模型和新模型，使用self.assertEqual断言
                self._compare_models(model, model_new, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("rank0_only_and_offload", [False, True])
    def test_state_dict_with_manual_ac_wrapper(
        self,
        state_dict_type: str,
        rank0_only_and_offload: bool,
    ):
        """
        Tests saving and loading a state dict for a model manually wrapped with
        ``FSDP(CheckpointWrapper(module))``, where the ``CheckpointWrapper`` is
        wrapped before FSDP.

        TODO: Investigate why the test above does not cover everything in this
        test and de-duplicate afterwards.
        """
        # 如果 state_dict_type 是 "sharded_state_dict" 并且 rank0_only_and_offload 为 True，则不支持，直接返回
        if state_dict_type == "sharded_state_dict" and rank0_only_and_offload:
            return  # not supported
        
        # 初始化一个没有 AC 的 TransformerWithSharedParams 模型
        model_ac = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
        )
        
        # 对没有 AC 的模型进行手动包装成 FSDP
        model_no_ac = deepcopy(model_ac)
        for i, layer in enumerate(model_no_ac.transformer.encoder.layers):
            model_no_ac.transformer.encoder.layers[i] = FSDP(layer)
        for i, layer in enumerate(model_no_ac.transformer.decoder.layers):
            model_no_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_no_ac.transformer = FSDP(model_no_ac.transformer)

        # 对有 AC 的模型进行手动包装成 `FSDP(CheckpointWrapper(module))`
        for i, layer in enumerate(model_ac.transformer.encoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.encoder.layers[i] = FSDP(layer)
        for i, layer in enumerate(model_ac.transformer.decoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_ac.transformer = FSDP(model_ac.transformer)

        # 保存、加载和比较这两个模型的状态字典
        with self._get_state_dict_mgr(
            model_no_ac, state_dict_type, rank0_only_and_offload
        ):
            state_dict_no_ac = model_no_ac.state_dict()
        with self._get_state_dict_mgr(
            model_ac, state_dict_type, rank0_only_and_offload
        ):
            state_dict_ac = model_ac.state_dict()
        
        # 断言两个模型的状态字典的键相同
        self.assertEqual(state_dict_ac.keys(), state_dict_no_ac.keys())
        
        # 如果 rank0_only_and_offload 为 True，则将状态字典广播到所有设备上
        if rank0_only_and_offload:
            state_dict_no_ac = self._broadcast_state_dict(model_no_ac, state_dict_no_ac)
            state_dict_ac = self._broadcast_state_dict(model_ac, state_dict_ac)
        
        # 加载状态字典到模型中并比较两个模型
        with self._get_state_dict_mgr(
            model_no_ac, state_dict_type, rank0_only_and_offload
        ):
            model_no_ac.load_state_dict(state_dict_no_ac)
        with self._get_state_dict_mgr(
            model_ac, state_dict_type, rank0_only_and_offload
        ):
            model_ac.load_state_dict(state_dict_ac)
        self._compare_models(model_ac, model_no_ac, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    # 定义测试函数，用于测试具有共享参数的状态字典
    def test_state_dict_with_shared_parameters(self, state_dict_type):
        # 定义自动封装策略，包括TransformerEncoderLayer和TransformerDecoderLayer
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        # 部分函数初始化模型创建器，部分参数通过偏函数固定，用于创建带有共享参数的Transformer模型
        model_creator = partial(
            TransformerWithSharedParams.init,
            self.process_group,  # 使用给定的进程组
            FSDPInitMode.RECURSIVE,  # 递归初始化FSDP
            CUDAInitMode.CUDA_BEFORE,  # 在CUDA模式下初始化
            {"auto_wrap_policy": auto_wrap_policy},  # 传递自动封装策略作为参数
        )

        # 使用模型创建器创建一个带有FSDP的模型
        fsdp_model = model_creator()
        # 使用状态字典管理器获取当前模型的状态字典，不覆盖元数据
        with self._get_state_dict_mgr(fsdp_model, state_dict_type, False):
            state_dict = fsdp_model.state_dict()

        # 使用模型创建器创建一个新的模型
        new_model = model_creator()
        # 将新模型的所有缓冲区置零
        _zero_model(new_model, zero_buffers=True)
        # 使用状态字典管理器获取新模型的状态字典，不覆盖元数据
        with self._get_state_dict_mgr(new_model, state_dict_type, False):
            # 加载之前保存的状态字典到新模型中
            new_model.load_state_dict(state_dict)

    # 如果GPU数量小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试，测试是否使用原始参数
    @parametrize("use_orig_params", [False, True])
    # 定义一个测试方法，用于测试在只有 rank 0 保存模型检查点并且仅在 rank 0 加载它的流程
    # 使用 `sync_module_states=True` 来模拟工作流程，以避免冗余的 CPU 内存使用
    def test_state_dict_rank0_offload_save_load_flow(self, use_orig_params: bool):
        """Tests saving a model checkpoint only on rank 0 and loading it only
        on rank 0 with ``sync_module_states=True`` to emulate the workflow to
        avoid redundant CPU memory usage."""
        # 自动包装策略，用于 TransformerEncoderLayer 和 TransformerDecoderLayer
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        # FSDP 的相关参数
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": use_orig_params,
        }
        # 使用 FSDP 初始化模型，指定初始化模式和 CUDA 初始化模式
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs,
        )
        # 强制模型参数和缓冲区非零
        with FSDP.summon_full_params(fsdp_model):
            for tensor in itertools.chain(
                fsdp_model.parameters(), fsdp_model.buffers()
            ):
                if torch.count_nonzero(tensor) == 0:
                    with torch.no_grad():
                        tensor.add_(torch.ones_like(tensor))
        # 获取当前模型的状态字典，并深拷贝
        with self._get_state_dict_mgr(fsdp_model, "state_dict", True):
            state_dict = deepcopy(_get_state_dict(fsdp_model))
        # 在所有 rank 上初始化一个非包装模型
        new_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
        )
        # 将新模型的参数和缓冲区全部置零
        _zero_model(new_model, zero_buffers=True)
        # 只在 rank 0 上加载检查点
        if self.rank == 0:
            new_model.load_state_dict(state_dict, strict=True)
        # 使用 `_assert_module_states` 断言模型状态
        _assert_module_states(
            new_model,
            process_group=self.process_group,
            assert_fn=self.assertNotEqual,
        )
        # 使用 `sync_module_states=True` 从 rank 0 广播模块状态
        new_fsdp_model = FSDP(
            new_model,
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=auto_wrap_policy,
            sync_module_states=True,
        )
        # 断言各个 rank 上的 FSDP 模型是否相等
        with FSDP.summon_full_params(new_fsdp_model):
            _assert_module_states(
                new_fsdp_model,
                process_group=self.process_group,
                assert_fn=self.assertEqual,
            )
        # 检查 FSDP 模型是否正确加载了检查点
        with FSDP.summon_full_params(fsdp_model):
            with FSDP.summon_full_params(new_fsdp_model):
                params = list(fsdp_model.parameters())
                params_new = list(new_fsdp_model.parameters())
                self.assertEqual(params, params_new)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("fp16", [True, False])
    # 使用参数化测试框架，测试保存和加载状态字典的基本功能
    @parametrize("state_dict_rank0_and_offload", [True, False])
    @parametrize("use_orig_params", [True, False])
    def test_basic_save_and_load_state_dict(
        self,
        state_dict_type: str,
        cpu_offload: bool,
        fp16: bool,
        state_dict_rank0_and_offload: bool,
        use_orig_params: bool,
    ):
        """
        Tests that we can save a state_dict and load it for modules with persistent buffers, including
        in the context of non-default mixed precision, different ``state_dict_type`` s and CPU offloading.
        """
        
        # 检查是否跳过测试，如果条件不满足则跳过
        if (state_dict_rank0_and_offload and state_dict_type != "state_dict") or (
            use_orig_params and state_dict_type not in _UNFLATTENED_STATE_DICT_IMPLS
        ):
            return  # not supported
        
        # 根据 mixed_precision 参数设置 MixedPrecision 对象，用于不同精度的模型参数
        mixed_precision = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision
            else None
        )
        
        # 通过部分应用函数创建模型
        model_call = partial(
            self._get_multibuffer_nested_model,
            cpu_offload=cpu_offload,
            use_orig_params=use_orig_params,
            mixed_precision=mixed_precision,
        )
        
        # 调用 model_call() 获取模型
        model = model_call()
        
        # 获取状态字典管理器
        ctx = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        
        # 在上下文中获取带有 offload 参数的状态字典
        with ctx:
            fsdp_state_dict = _get_state_dict(model, cpu_offload.offload_params, False)

        # 验证状态字典的内容
        self._validate_state_dict_contents(
            model, fsdp_state_dict, state_dict_rank0_and_offload
        )

        # 重新调用 model_call() 创建新模型
        model_new = model_call()
        
        # 如果不使用 offload 参数，则将模型移到 GPU 上
        if not cpu_offload.offload_params:
            model_new = model_new.cuda()

        # 将模型置零以确保参数不同
        _zero_model(model_new, zero_buffers=True)
        
        # 比较原始模型和新模型，使用不相等的断言
        self._compare_models(model, model_new, self.assertNotEqual)

        # 如果 state_dict_rank0_and_offload 为真，则广播状态字典
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)
        
        # 在 FSDP 上下文中，根据 state_dict_type 加载状态字典到新模型
        with FSDP.state_dict_type(model_new, STATE_DICT_MAPPING[state_dict_type]):
            model_new.load_state_dict(fsdp_state_dict, strict=True)

        # 比较原始模型和新模型，使用相等的断言
        self._compare_models(model, model_new, self.assertEqual)

    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 使用 @parametrize 装饰器多次运行该测试方法，测试不同的参数组合
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize("mixed_precision", [True, False])
    @parametrize("state_dict_rank0_and_offload", [True, False])
    # 定义测试方法：验证在一些训练后保存模型状态正常更新的情况
    def test_save_and_load_after_forward_state_dict(
        self, state_dict_type, mixed_precision, state_dict_rank0_and_offload
    ):
        """
        Test that saving after some training results in params being updated as
        expected.
        """
        # 如果 state_dict_rank0_and_offload 为 True，并且 state_dict_type 不是 "state_dict"，则退出测试
        if state_dict_rank0_and_offload and state_dict_type != "state_dict":
            return
        
        # 设置当前 CUDA 设备
        torch.cuda.set_device(self.rank)
        
        # 根据 mixed_precision 参数初始化混合精度对象，或者为 None
        mixed_precision = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision
            else None
        )
        
        # 根据 mixed_precision 创建简单嵌套模型
        model = self._get_simple_nested_model(mixed_precision=mixed_precision)
        
        # 使用 SGD 优化器优化模型参数
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # 获取初始模型参数
        initial_params = get_full_params(model)
        
        # 进行 6 次训练迭代
        for _ in range(6):
            inp = torch.randn(1, 10, device=torch.cuda.current_device())
            output = model(*inp)
            loss = output.sum()
            expected_dtype = torch.float32 if mixed_precision is None else torch.float16
            # 断言损失的数据类型符合预期
            self.assertEqual(expected_dtype, loss.dtype)
            loss.backward()
            optim.step()
        
        # 获取训练后的模型参数
        trained_params = get_full_params(model)
        
        # 确保进行了一定量的训练，初始参数不等于训练后的参数
        self.assertNotEqual(initial_params, trained_params)
        
        # 保存状态字典的副本
        fsd_mgr = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with fsd_mgr:
            state_dict = model.state_dict()
            # 如果 state_dict_type 是 "state_dict"，则为状态字典的每个值创建克隆
            if state_dict_type == "state_dict":
                state_dict = {k: v.clone() for k, v in state_dict.items()}
            else:
                # 否则，对于每个分片张量，将第一个本地分片克隆并分离
                for sharded_tensor in state_dict.values():
                    shard = sharded_tensor._local_shards[0]
                    shard.tensor = shard.tensor.clone().detach_()
        
        # 验证状态字典的内容是否正确
        self._validate_state_dict_contents(
            model, state_dict, state_dict_rank0_and_offload
        )
        
        # 将模型置零
        _zero_model(model)
        
        # 确保检查点参数具有完整的参数数据类型
        for tensor in state_dict.values():
            self.assertEqual(tensor.dtype, torch.float32)
        
        # 如果 state_dict_rank0_and_offload 为 True，则广播状态字典到模型
        if state_dict_rank0_and_offload:
            state_dict = self._broadcast_state_dict(model, state_dict)
        
        # 使用 FSDP 将状态字典类型与指定的映射加载到模型中
        with FSDP.state_dict_type(model, STATE_DICT_MAPPING[state_dict_type]):
            model.load_state_dict(state_dict, strict=True)
        
        # 获取加载后的模型参数
        loaded_params = get_full_params(model)
        
        # 断言加载后的参数与训练后的参数相等
        self.assertEqual(loaded_params, trained_params)

    # 初始化模型的私有方法
    def _initialize_model(
        self,
        wrap_fsdp: bool,
        wrap_ddp: bool = True,
        register_buffers: bool = False,
    ):
        # 保持输入数据的确定性
        torch.manual_seed(0)

        # 创建模型对象，使用 FSDP 封装，注册缓冲区
        model = Model(wrap_fsdp, register_buffers=register_buffers).cuda()
        # 如果 wrap_fsdp 为 True，则再次用 FSDP 包装模型
        if wrap_fsdp:
            model = FSDP(model)
        # 如果 wrap_ddp 为 True，则使用 DistributedDataParallel 处理模型
        elif wrap_ddp:
            model = DistributedDataParallel(model, device_ids=[self.rank])
        # 返回创建好的模型对象
        return model

    @staticmethod
    def _state_dict(model: Module, state_dict_type: str):
        try:
            # 获取指定 state_dict_type 对应的枚举值
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError as e:
            # 如果找不到对应的 state_dict_type，则抛出异常
            raise ValueError(f"No state_dict type for {state_dict_type}") from e

        # 使用 FSDP 提供的 state_dict_type 上下文管理器来获取模型的状态字典
        with FSDP.state_dict_type(model, enum_val):
            return model.state_dict()

    @staticmethod
    def _load_state_dict(
        model: Module, state_dict_type: str, state_dict: Dict[str, Any]
    ):
        try:
            # 获取指定 state_dict_type 对应的枚举值
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError as e:
            # 如果找不到对应的 state_dict_type，则抛出异常
            raise ValueError(f"No state_dict for {state_dict_type}") from e

        # 使用 FSDP 提供的 state_dict_type 上下文管理器来加载模型的状态字典
        with FSDP.state_dict_type(model, enum_val):
            return model.load_state_dict(state_dict, strict=True)

    def _dist_train(
        self, wrap_fsdp: bool, state_dict_type: str = "", move_to_cpu: bool = False
    ):
        # TODO: Move this test to common_fsdp.
        # 初始化模型
        model = self._initialize_model(wrap_fsdp)
        # 使用 SGD 优化器初始化优化器对象
        optim = SGD(model.parameters(), lr=0.1)

        # 生成随机输入数据，64x4 的张量，要求计算梯度，使用 CUDA 加速
        in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        # 进行 3 次迭代训练
        for _ in range(3):
            out = model(in_data)  # 模型前向计算
            out.sum().backward()  # 计算梯度
            optim.step()  # 更新模型参数
            optim.zero_grad()  # 梯度清零

        # 如果 wrap_fsdp 为 True，则创建空白的 FSDP 模型
        if wrap_fsdp:
            blank_model = FSDP(Model(True).cuda())
            _zero_model(blank_model)
            # 获取当前模型的状态字典
            state_dict = self._state_dict(model, state_dict_type)
            # 如果需要将模型参数移到 CPU
            if move_to_cpu:
                for key in list(state_dict.keys()):
                    tensor = state_dict[key]
                    if isinstance(tensor, torch.Tensor):
                        state_dict[key] = tensor.cpu()
                    else:
                        shards = tensor.local_shards()
                        if shards:
                            shards[0].tensor = shards[0].tensor.cpu()

            # 加载状态字典到空白模型中
            self._load_state_dict(blank_model, state_dict_type, state_dict)
            # 返回完整的空白模型参数
            return get_full_params(blank_model)
        else:
            # 如果 wrap_fsdp 为 False，则返回模型的参数列表
            return list(model.parameters())

    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2，则跳过测试
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)  # 参数化测试，使用支持的 state_dict 实现类型
    def test_state_dict_save_load_flow(self, state_dict_type):
        # 运行子测试
        self.run_subtests(
            {"move_to_cpu": [True, False]},  # 参数化 move_to_cpu 的测试
            self._test_state_dict_save_load_flow,
            state_dict_type=state_dict_type,  # 传递 state_dict_type 参数
        )
    # 定义测试方法，用于测试状态字典的保存和加载流程
    def _test_state_dict_save_load_flow(self, state_dict_type, move_to_cpu):
        # 在分布式训练环境中执行训练，使用 FSDP（Fully Sharded Data Parallelism）参数
        fsdp_params = self._dist_train(
            wrap_fsdp=True,
            state_dict_type=state_dict_type,
            move_to_cpu=move_to_cpu,
        )
        # 在分布式训练环境中执行训练，不使用 FSDP 参数
        ddp_params = self._dist_train(wrap_fsdp=False)
        # 断言 FSDP 参数与 DDP 参数相等
        self.assertEqual(ddp_params, fsdp_params)

    # 如果 GPU 数量小于 2，则跳过此测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试，state_dict_type 取值为 _SUPPORTED_STATE_DICT_IMPLS 中的元素
    def test_fsdp_state_dict_keys(self, state_dict_type):
        # 获取模型的状态字典
        state_dict = self._state_dict(self._initialize_model(True), state_dict_type)
        if state_dict_type == "local_state_dict":
            # 断言状态字典的键包含 FLAT_PARAM 和 "inner.FLAT_PARAM"
            self.assertEqual({FLAT_PARAM, f"inner.{FLAT_PARAM}"}, state_dict.keys())
        elif state_dict_type in ("state_dict", "sharded_state_dict"):
            # 键应与本地模型的状态字典键匹配
            local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False)
            local_keys = local_model.state_dict().keys()
            self.assertEqual(state_dict.keys(), local_keys)
        else:
            # 抛出未实现的异常，提示未对 state_dict_type 进行测试
            raise NotImplementedError(f"No test for {state_dict_type}!")

    # 如果 GPU 数量小于 2，则跳过此测试
    @skip_if_lt_x_gpu(2)
    # 参数化测试，state_dict_type 取值为 _UNFLATTENED_STATE_DICT_IMPLS 中的元素
    # 参数化测试，state_dict_rank0_and_offload 取值为 [True, False]
    # 参数化测试，fsdp_root 取值为 [True, False]
    def test_state_dict_load_into_local_module(
        self,
        state_dict_type,
        state_dict_rank0_and_offload,
        fsdp_root,
    ):
        """
        Tests that FSDP's state_dict can be loaded into a local model.
        """
        # 如果 state_dict_rank0_and_offload 为 True 并且 state_dict_type 不是 "state_dict"，则直接返回
        if state_dict_rank0_and_offload and state_dict_type != "state_dict":
            return
        # 如果没有指定 fsdp_root，则获取一个非 FSDP 根模块
        if not fsdp_root:
            model = self._get_non_fsdp_root_module()
        else:
            # 否则，初始化一个包含 FSDP 的模型，并注册缓冲区
            model = self._initialize_model(wrap_fsdp=True, register_buffers=True)
        # 使用 SGD 优化器，学习率为 0.1
        optim = SGD(model.parameters(), lr=0.1)
        # 如果没有指定 fsdp_root，则生成一个在 CUDA 设备上生成随机数据的张量
        if not fsdp_root:
            in_data = torch.randn(
                1, 10, requires_grad=True, device=torch.device("cuda")
            )
        else:
            # 否则，生成一个形状为 (64, 4) 的张量，并要求梯度，放在 CUDA 设备上
            in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        # 循环执行三次
        for _ in range(3):
            # 将输入数据传入模型，计算输出
            out = model(in_data)
            # 计算输出的总和并反向传播
            out.sum().backward()
            # 执行优化器的一步优化
            optim.step()
            # 清空梯度
            optim.zero_grad()

        # 使用 FSDP.summon_full_params 将模型参数转化为全参数模式
        with FSDP.summon_full_params(model):
            # 深拷贝模型参数列表
            fsdp_params = deepcopy(list(model.parameters()))

        # 获取 FSDP 的 state_dict，注意默认情况下返回 full_state_dict
        sd_mgr = self._get_state_dict_mgr(
            model, state_dict_type, state_dict_rank0_and_offload
        )
        with sd_mgr:
            # 获取模型的 state_dict
            fsdp_state_dict = model.state_dict()

        # 创建忽略的键列表，这些键以 NON_ROOT_FSDP_PREFIX 开头
        ignore_keys = [k for k in fsdp_state_dict.keys() if NON_ROOT_FSDP_PREFIX in k]
        # 验证状态字典内容
        self._validate_state_dict_contents(
            model,
            fsdp_state_dict,
            state_dict_rank0_and_offload,
            ignore_keys=ignore_keys,
        )
        # 创建全零的本地模型
        if not fsdp_root:
            blank_local_model = self._get_non_fsdp_root_module(wrap=False)
        else:
            # 否则，初始化一个不包含 FSDP 的模型，并且不使用 DDP，也注册缓冲区
            blank_local_model = self._initialize_model(
                wrap_fsdp=False, wrap_ddp=False, register_buffers=True
            )

        # 验证空白本地模型中没有任何 FSDP
        for mod in blank_local_model.modules():
            self.assertFalse(isinstance(mod, FSDP))

        # 将所有本地模型参数置零
        for param in blank_local_model.parameters():
            with torch.no_grad():
                param.zero_()

        # 收集 FSDP 状态字典
        fsdp_state_dict = _gather_state_dict(fsdp_state_dict)

        # 如果 state_dict_rank0_and_offload 为 True，则广播 FSDP 的状态字典
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)

        # 加载 FSDP 的全状态字典到本地模型，并验证参数是否符合预期
        blank_local_model.load_state_dict(fsdp_state_dict, strict=True)
        local_params = list(blank_local_model.parameters())
        for fsdp_param, local_param in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize("double_nest", [True])
    @skip_if_lt_x_gpu(2)
    def test_wrong_state_dict_config(self):
        # 创建一个 FSDP 模型对象，包装在 CUDA 上
        model = FSDP(Model(wrap_fsdp=True).cuda())
        # 断言捕获 RuntimeError，并检查错误信息是否包含特定文本
        with self.assertRaisesRegex(RuntimeError, "Expected state_dict_config of type"):
            # 设置 state_dict_type，并执行相关代码块
            with model.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, LocalStateDictConfig()
            ):
                pass

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize("prefix", [True, False])
    @parametrize("ignore_inner", [True, False])
    @parametrize("mixed_precision", [True, False])
    def test_state_dict_with_ignored_modules(
        self, state_dict_type, prefix, ignore_inner, mixed_precision
    ):
        # 测试针对忽略模块的 state_dict_type 的不同组合
        ...

    @skip_if_lt_x_gpu(2)
    def test_state_dict_type(self):
        # 创建一个 SkipModel 模块，双重嵌套设置为 True
        module = SkipModel(double_nest=True)
        # 使用 enable_wrap 包装模块，并进入上下文管理器
        with enable_wrap(wrapper_cls=FSDP):
            fsdp = wrap(module)
        # 设置 FSDP 模型的 state_dict_type 为 LOCAL_STATE_DICT
        with FSDP.state_dict_type(fsdp, StateDictType.LOCAL_STATE_DICT):
            pass
        # 遍历 FSDP 模块中的所有模块，检查其 _state_dict_type 是否为 FULL_STATE_DICT
        for module in FSDP.fsdp_modules(fsdp):
            self.assertEqual(module._state_dict_type, StateDictType.FULL_STATE_DICT)

    @skip_if_lt_x_gpu(2)
    def test_local_state_dict_with_empty_ranks(self):
        # 定义一个简单的 PyTorch 模型，包含一个张量和一个参数
        class Model(Module):
            def __init__(self):
                super().__init__()
                self.my_tensor = torch.full((1,), 3.1415926)
                self.my_parameter = nn.Parameter(self.my_tensor)

            def forward(self, x):
                return self.my_parameter

        model = FSDP(Model().cuda())
        # 设置 FSDP 模型的 state_dict_type 为 LOCAL_STATE_DICT
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            out = model(None)
            out.backward()

            # 深拷贝模型的状态字典
            state_dict = deepcopy(model.state_dict())
            # 在没有梯度的情况下，使用全参数召唤器进入上下文管理器
            with torch.no_grad():
                with FSDP.summon_full_params(model):
                    # 断言模型参数的值是否为初始值 3.1415926
                    self.assertEqual(model.my_parameter.item(), 3.1415926)
                    # 将模型参数复制为新值 1.75，并断言其值是否正确
                    model.my_parameter.copy_(torch.full((1,), 1.75).cuda())
                    self.assertEqual(model.my_parameter.item(), 1.75)
            # 恢复模型状态字典
            model.load_state_dict(state_dict)
            # 使用全参数召唤器再次进入上下文管理器，断言模型参数是否恢复到初始值
            with FSDP.summon_full_params(model):
                self.assertEqual(model.my_parameter.item(), 3.1415926)

    @skip_if_lt_x_gpu(2)
    def test_torch_save_load(self):
        # 创建一个带有 FSDP 封装的模型对象，并移到 CUDA 上
        model = Model(wrap_fsdp=True).cuda()
        # 设置 FSDP 模型的 state_dict_type 为 LOCAL_STATE_DICT
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            # 获取模型的状态字典
            state_dict = model.state_dict()
            # 创建一个字节流对象作为检查点
            checkpoint = io.BytesIO()
            # 将模型的状态字典保存到检查点中
            torch.save(state_dict, checkpoint)
            checkpoint.seek(0)
            # 从检查点中加载保存的状态字典
            state_dict_saved = torch.load(checkpoint)
            # 检查加载后的状态字典是否与原始状态字典匹配
            for k, v in state_dict_saved.items():
                if isinstance(v, ShardedTensor):
                    self.assertEqual(
                        v._local_shards[0].tensor, state_dict[k]._local_shards[0].tensor
                    )
                else:
                    self.assertEqual(v, state_dict[k])
    def test_shared_module_and_shared_parameter(self):
        # 创建一个 FSDP 模型，包装一个在 GPU 上的测试模型
        model = FSDP(TestDummyModel().cuda())
        # 使用 FULL_STATE_DICT 模式获取模型的状态字典
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            # 获取模型的状态字典
            state_dict = model.state_dict()
            # 断言随机参数和共享参数相等
            self.assertEqual(
                state_dict["random_parameter"], state_dict["shared_parameter"]
            )
            # 断言 net2.0 的偏置与 net3.0 的偏置相等
            self.assertEqual(state_dict["net2.0.bias"], state_dict["net3.0.bias"])
            # 断言 net2.0 的权重与 net3.0 的权重相等
            self.assertEqual(state_dict["net2.0.weight"], state_dict["net3.0.weight"])

    @skip_if_lt_x_gpu(2)
    def test_full_state_dict_missing_unexpected_keys_cleaned(self):
        # 获取一个简单的嵌套模型
        model = self._get_simple_nested_model()
        # 获取模型的状态字典
        sd = model.state_dict()
        # 移除一个随机的键
        sd.pop(next(iter(sd.keys())))
        # 添加一个意外的键
        sd["unexpected"] = torch.ones(1)
        # 加载状态字典并检查缺失和意外的键
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert len(missing) == 1
        assert len(unexpected) == 1
        # 断言缺失的键和意外的键不包含 FSDP_PREFIX
        self.assertTrue(FSDP_PREFIX not in missing[0])
        self.assertTrue(FSDP_PREFIX not in unexpected[0])

    @skip_if_lt_x_gpu(2)
    def test_sharded_load_multi_backend_pg(self):
        # 定义自动包装策略
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        # 定义 FSDP 参数
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "use_orig_params": True,
        }
        # 循环测试加载到 CPU 和 GPU
        for load_cpu in [True, False]:
            with self.subTest(load_cpu=load_cpu):
                # 创建一个新的分组，使用多个后端（cpu:gloo,cuda:nccl）
                pg = dist.new_group(backend="cpu:gloo,cuda:nccl")
                # 初始化具有共享参数的 Transformer 模型
                fsdp_model = TransformerWithSharedParams.init(
                    pg,
                    FSDPInitMode.RECURSIVE,
                    CUDAInitMode.CUDA_BEFORE,
                    fsdp_kwargs,
                )
                # 设置 FSDP 模型的状态字典类型为 SHARDED_STATE_DICT
                FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)
                # 获取 FSDP 模型的分片状态字典
                sharded = fsdp_model.state_dict()
                # 复制模型参数到 param_copy
                param_copy = [t.clone().detach_() for t in fsdp_model.parameters()]
                # 将模型参数置零
                with torch.no_grad():
                    for p in fsdp_model.parameters():
                        p.zero_()

                if load_cpu:
                    # 如果加载到 CPU，将分片状态字典转移到 CPU
                    for k, v in sharded.items():
                        sharded[k] = v.cpu()

                # 加载分片状态字典到 FSDP 模型
                fsdp_model.load_state_dict(sharded)
                # 检查加载后的模型参数与原始复制的参数是否相等
                for p1, p2 in zip(param_copy, fsdp_model.parameters()):
                    self.assertEqual(p1, p2, f"not equal: {p1.sum()} vs {p2.sum()}")

    @skip_if_lt_x_gpu(2)
    # 定义一个单元测试函数，用于测试世界大小为1的情况
    def test_world_size_one(self):
        # 初始化一个变量 my_pg，用于存储当前进程的分组
        my_pg = None
        # 遍历世界大小的范围
        for i in range(self.world_size):
            # 创建一个新的分组，仅包含当前索引 i
            pg = dist.new_group(ranks=[i])
            # 如果当前索引 i 等于当前进程的排名 self.rank
            if i == self.rank:
                # 将当前分组赋值给 my_pg
                my_pg = pg

        # 使用 TransformerWithSharedParams 类的静态方法 init 初始化模型
        model = TransformerWithSharedParams.init(
            my_pg,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
        )
        # 使用 FSDP 类的 state_dict_type 方法，将模型的状态字典类型设置为 SHARDED_STATE_DICT
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            # 获取模型的状态字典
            state_dict = model.state_dict()
            # 加载模型的状态字典
            model.load_state_dict(state_dict)

        # 同步所有进程，确保所有进程已经执行完前面的代码，再继续执行后面的代码
        dist.barrier()
# 定义一个测试类，继承自 FSDPTest
class TestFSDPStateDict4GPUs(FSDPTest):

    # 返回当前 CUDA 设备数量作为 world_size 属性
    @property
    def world_size(self):
        return torch.cuda.device_count()

    # 装饰器函数，如果当前 GPU 数量小于 4，则跳过测试
    @skip_if_lt_x_gpu(4)
    def test_local_state_dict_reshard(self):
        """
        This test demonstrates the ability to do resharding when using
        local_state_dict. Although we do not recommend users to use
        local_state_dict, there are still some corner cases that
        using local_state_dict is a better solution.
        """
        # 在 CUDA 上创建 FSDP 模型，使用 wrap_fsdp=True 封装模型
        model = FSDP(Model(wrap_fsdp=True)).cuda()
        # 使用 SGD 优化器优化模型参数
        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        # 创建一个在当前 CUDA 设备上的 4x4 随机张量作为输入 batch
        batch = torch.randn(4, 4, device=torch.cuda.current_device())
        # 将 batch 输入模型得到输出
        output = model(batch)
        # 计算输出的和作为损失
        loss = output.sum()
        # 反向传播损失
        loss.backward()
        # 优化器执行一步优化
        optim.step()

        # 使用 FSDP 的 state_dict_type 方法切换模型的状态字典类型为 LOCAL_STATE_DICT
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()

        # 获取当前进程的分布式训练的排名
        rank = dist.get_rank()
        # 创建一个新的分组，包括排名为 0 和 1 的进程
        new_pg = dist.new_group(ranks=[0, 1])
        resharded_state_dict = {}

        # 模拟从 4 个 GPU 到 2 个 GPU 的 resharding 过程
        for key, value in state_dict.items():
            if isinstance(value, ShardedTensor):
                # 如果值是 ShardedTensor 类型，则进行 resharding
                full_flat_param = _all_gather_sharded_tensor(value)
                if rank < 2:
                    full_numel = full_flat_param.size()
                    chunks = full_flat_param.chunk(2)
                    flat_param = chunks[rank]
                    shard_offset = 0 if rank == 0 else chunks[0].numel()
                    local_shards = [
                        Shard.from_tensor_and_offsets(flat_param, [shard_offset], rank)
                    ]
                    # 从本地 shards 初始化 ShardedTensor
                    sharded_tensor = init_from_local_shards(
                        local_shards, full_numel, process_group=new_pg
                    )
                    resharded_state_dict[key] = sharded_tensor
            else:
                # 如果值不是 ShardedTensor 类型，则直接复制到 resharded_state_dict 中
                if rank < 2:
                    resharded_state_dict[key] = value

        # 如果当前进程的排名小于 2，则继续操作
        if rank < 2:
            # 在新的进程组上创建新的 FSDP 模型 model2
            model2 = FSDP(
                Model(wrap_fsdp=True, process_group=new_pg), process_group=new_pg
            ).cuda()
            # 使用 FSDP 的 state_dict_type 方法切换模型2的状态字典类型为 LOCAL_STATE_DICT
            with FSDP.state_dict_type(model2, StateDictType.LOCAL_STATE_DICT):
                # 加载 resharded_state_dict 到 model2
                model2.load_state_dict(resharded_state_dict)

        # 使用 FSDP 的 state_dict_type 方法切换模型的状态字典类型为 FULL_STATE_DICT
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            full_state_dict1 = model.state_dict()

        # 如果当前进程的排名小于 2，则继续操作
        if rank < 2:
            # 使用 FSDP 的 state_dict_type 方法切换模型2的状态字典类型为 FULL_STATE_DICT
            with FSDP.state_dict_type(model2, StateDictType.FULL_STATE_DICT):
                # 获取模型2的状态字典
                full_state_dict2 = model2.state_dict()
            # 断言模型1的 FULL_STATE_DICT 等于模型2的 FULL_STATE_DICT
            self.assertEqual(full_state_dict1, full_state_dict2)


# 实例化参数化测试 TestFSDPStateDict
instantiate_parametrized_tests(TestFSDPStateDict)

# 如果当前脚本为主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```