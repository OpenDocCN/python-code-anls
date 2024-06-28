# `.\trainer.py`

```
# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# Integrations must be imported before ML frameworks:
# isort: off
from .integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
)

# Importing integration modules and utilities for reporting and hyperparameters
from .integrations import get_reporting_integration_callbacks, hp_params

# Importing utility functions and classes from the huggingface_hub library
import huggingface_hub.utils as hf_hub_utils

# Importing essential modules from Python's standard library
import numpy as np  # NumPy for numerical computing
import torch  # PyTorch for deep learning framework
import torch.distributed as dist  # PyTorch distributed for parallel computing

# Importing specific functions and classes from huggingface_hub library
from huggingface_hub import ModelCard, create_repo, upload_folder

# Importing version comparison utility from packaging module
from packaging import version

# Importing neural network related modules from PyTorch
from torch import nn  # Neural network module
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler  # Data handling utilities

# Importing essential internal modules from Transformers library
from . import __version__  # Current version of Transformers library
from .configuration_utils import PretrainedConfig  # Configuration utilities for pretrained models
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator  # Data collation utilities
from .debug_utils import DebugOption, DebugUnderflowOverflow  # Debugging utilities
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend  # Hyperparameter search utilities
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available  # DeepSpeed integration utilities
from .integrations.tpu import tpu_spmd_dataloader  # TPU integration for data loading
from .modelcard import TrainingSummary  # ModelCard for model documentation
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model  # Model utilities
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES  # Auto model mapping names
from .optimization import Adafactor, get_scheduler  # Optimization utilities
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13  # PyTorch utility functions
from .tokenization_utils_base import PreTrainedTokenizerBase  # Base class for tokenization utilities
from .trainer_callback import CallbackHandler, DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerCallback, TrainerControl, TrainerState  # Trainer callback utilities
from .trainer_pt_utils import DistributedTensorGatherer, IterableDatasetShard, LabelSmoother, LayerWiseDummyOptimizer, LengthGroupedSampler, SequentialDistributedSampler, distributed_broadcast_scalars, distributed_concat, find_batch_size, get_dataloader_sampler, get_model_param_count  # PyTorch specific trainer utilities
    # 从给定的名称中获取模块类
    get_module_class_from_name,
    
    # 获取参数的名称列表
    get_parameter_names,
    
    # 递归连接（concatenate）操作，可能是连接嵌套结构的函数
    nested_concat,
    
    # 递归分离（detach）操作，可能是将梯度信息分离出来的函数
    nested_detach,
    
    # 将嵌套结构转换为 NumPy 数组的函数
    nested_numpify,
    
    # 对 XLA 网格进行递归降维（reduce）的函数
    nested_xla_mesh_reduce,
    
    # 重新发出 PyTorch 警告信息的函数
    reissue_pt_warnings,
    
    # 移除虚拟检查点的函数
    remove_dummy_checkpoint,
# 从 `trainer_utils` 模块中导入多个工具类和函数
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,  # 导入检查点目录前缀
    BestRun,  # 导入最佳运行结果类
    EvalLoopOutput,  # 导入评估循环输出类
    EvalPrediction,  # 导入评估预测类
    HPSearchBackend,  # 导入超参数搜索后端类
    HubStrategy,  # 导入Hub策略类
    IntervalStrategy,  # 导入间隔策略类
    PredictionOutput,  # 导入预测输出类
    RemoveColumnsCollator,  # 导入移除列的集合类
    TrainerMemoryTracker,  # 导入训练器内存追踪类
    TrainOutput,  # 导入训练输出类
    check_target_module_exists,  # 导入检查目标模块是否存在的函数
    default_compute_objective,  # 导入默认计算目标的函数
    denumpify_detensorize,  # 导入去除NumPy array或tensor化的函数
    enable_full_determinism,  # 导入启用完全确定性的函数
    find_executable_batch_size,  # 导入查找可执行批量大小的函数
    get_last_checkpoint,  # 导入获取最后一个检查点的函数
    has_length,  # 导入判断对象是否具有长度的函数
    neftune_post_forward_hook,  # 导入Neftune后向钩子函数
    number_of_arguments,  # 导入获取参数个数的函数
    seed_worker,  # 导入种子工作器函数
    set_seed,  # 导入设置种子的函数
    speed_metrics,  # 导入速度度量指标函数
)

# 从 `training_args` 模块中导入优化器名称、并行模式、训练参数类
from .training_args import OptimizerNames, ParallelMode, TrainingArguments

# 从 `utils` 模块中导入多个常量、类和函数
from .utils import (
    ADAPTER_CONFIG_NAME,  # 导入适配器配置名称
    ADAPTER_SAFE_WEIGHTS_NAME,  # 导入适配器安全权重名称
    ADAPTER_WEIGHTS_NAME,  # 导入适配器权重名称
    CONFIG_NAME,  # 导入配置名称
    SAFE_WEIGHTS_INDEX_NAME,  # 导入安全权重索引名称
    SAFE_WEIGHTS_NAME,  # 导入安全权重名称
    WEIGHTS_INDEX_NAME,  # 导入权重索引名称
    WEIGHTS_NAME,  # 导入权重名称
    PushInProgress,  # 导入推送进行中类
    PushToHubMixin,  # 导入推送到Hub混合类
    can_return_loss,  # 导入能否返回损失的函数
    find_labels,  # 导入查找标签的函数
    is_accelerate_available,  # 导入加速库是否可用的函数
    is_apex_available,  # 导入APEX是否可用的函数
    is_bitsandbytes_available,  # 导入BitsAndBytes是否可用的函数
    is_datasets_available,  # 导入数据集是否可用的函数
    is_galore_torch_available,  # 导入Galore Torch是否可用的函数
    is_in_notebook,  # 导入是否在笔记本中的函数
    is_ipex_available,  # 导入IPEx是否可用的函数
    is_peft_available,  # 导入PEFT是否可用的函数
    is_safetensors_available,  # 导入安全张量是否可用的函数
    is_sagemaker_dp_enabled,  # 导入SageMaker分布式训练是否启用的函数
    is_sagemaker_mp_enabled,  # 导入SageMaker模型并行是否启用的函数
    is_torch_compile_available,  # 导入Torch编译是否可用的函数
    is_torch_neuroncore_available,  # 导入Torch NeuronCore是否可用的函数
    is_torch_npu_available,  # 导入Torch NPU是否可用的函数
    is_torch_xla_available,  # 导入Torch XLA是否可用的函数
    logging,  # 导入日志功能
    strtobool,  # 导入字符串转布尔值的函数
)

# 从 `utils.quantization_config` 模块中导入量化方法
from .utils.quantization_config import QuantizationMethod

# 默认回调函数列表
DEFAULT_CALLBACKS = [DefaultFlowCallback]

# 默认进度回调函数
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# 如果在笔记本中，则导入笔记本进度回调函数
if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback
    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# 如果APEX可用，则导入AMP
if is_apex_available():
    from apex import amp

# 如果数据集可用，则导入datasets模块
if is_datasets_available():
    import datasets

# 如果Torch XLA可用，则导入相关模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

# 如果SageMaker模型并行可用，则导入相关模块和版本检查
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# 如果安全张量可用，则导入相关模块
if is_safetensors_available():
    import safetensors.torch

# 如果PEFT可用，则导入PeftModel
if is_peft_available():
    from peft import PeftModel

# 如果加速库可用，则导入加速库相关模块和函数
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]  # 数据采样器列表
    # 检查加速库的版本是否大于 "0.23.0"
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        # 如果满足条件，导入 SeedableRandomSampler 类从 accelerate.data_loader 模块
        from accelerate.data_loader import SeedableRandomSampler
    
        # 将 SeedableRandomSampler 类加入到 DATA_SAMPLERS 列表中
        DATA_SAMPLERS += [SeedableRandomSampler]
    
    # 检查是否存在 DeepSpeed 库
    if is_deepspeed_available():
        # 如果 DeepSpeed 可用，从 accelerate.utils 模块导入 DeepSpeedSchedulerWrapper 类
        from accelerate.utils import DeepSpeedSchedulerWrapper
# 检查给定的模型是否属于 PEFT 模型类或其混合类
def _is_peft_model(model):
    # 检查是否安装了 PEFT
    if is_peft_available():
        # 如果安装了 PEFT，先将基础类设置为 PeftModel
        classes_to_check = (PeftModel,)
        # 检查 PEFT 的版本是否大于等于 0.7.0
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            # 如果版本允许，添加 PeftMixedModel 到待检查类列表
            classes_to_check = (*classes_to_check, PeftMixedModel)
        # 返回模型是否属于待检查类列表中的任何一个
        return isinstance(model, classes_to_check)
    # 如果未安装 PEFT，则返回 False
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada 在下一个 `accelerate` 发布中，使用版本检查替换此检查
    # 检查是否安装了 accelerate 并且 save_fsdp_model 的参数列表中包含 "adapter_only"
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        # 如果条件成立，返回适合 FSDP 检查点的参数字典
        return {"adapter_only": True}
    else:
        # 否则返回空字典
        return {}


if TYPE_CHECKING:
    import optuna  # 导入类型检查时所需的 optuna 模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# 用于保存检查点文件的文件名常量
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class Trainer:
    """
    Trainer 是一个简单但功能齐全的 PyTorch 训练和评估循环，专为 🤗 Transformers 优化。

    重要属性:

        - **model** -- 始终指向核心模型。如果使用的是 transformers 模型，它将是 [`PreTrainedModel`] 的子类。
        - **model_wrapped** -- 始终指向最外层的模型。如果使用 `DeepSpeed`，内部模型会被包装成 `DeepSpeed` 和 `torch.nn.DistributedDataParallel`。
          如果内部模型没有被包装，则 `self.model_wrapped` 与 `self.model` 相同。
        - **is_model_parallel** -- 是否将模型切换到模型并行模式（不同于数据并行，意味着一些模型层在不同的 GPU 上拆分）。
        - **place_model_on_device** -- 是否自动将模型放置在设备上。如果使用模型并行或 DeepSpeed，或者默认的 `TrainingArguments.place_model_on_device`
          被覆盖为返回 `False`，它将设置为 `False`。
        - **is_in_train** -- 当前模型是否正在执行 `train`（例如，在 `train` 运行时调用 `evaluate`）。

    """

    # 下面的方法是 Trainer 的示例方法，保存在 trainer_pt_utils 中。
    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        初始化一个 `Trainer` 对象，用于模型训练和评估。

        Args:
            model (Union[PreTrainedModel, nn.Module], optional): 要训练的模型。
            args (TrainingArguments, optional): 训练和评估的参数设置。
            data_collator (Optional[DataCollator], optional): 用于批处理数据的数据收集器。
            train_dataset (Optional[Dataset], optional): 训练数据集。
            eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]], optional): 评估数据集。
            tokenizer (Optional[PreTrainedTokenizerBase], optional): 用于处理输入数据的分词器。
            model_init (Optional[Callable[[], PreTrainedModel]], optional): 初始化模型的函数。
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]], optional): 用于计算评估指标的函数。
            callbacks (Optional[List[TrainerCallback]], optional): 训练过程中使用的回调函数列表。
            optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR], optional): 优化器和学习率调度器的元组。
            preprocess_logits_for_metrics (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): 对预测结果进行预处理的函数。
        """

    def _activate_neftune(self, model):
        r"""
        激活 NEFTune 方法，参考代码和论文：
        https://github.com/neelsjain/NEFTune
        https://arxiv.org/abs/2310.05914
        """
        unwrapped_model = unwrap_model(model)

        if _is_peft_model(unwrapped_model):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        del unwrapped_model

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model

    def _deactivate_neftune(self, model):
        """
        停用 NEFTune 方法。确保先调用 `_activate_neftune`。
        """
        if not hasattr(self, "neftune_hook_handle"):
            raise ValueError("Neftune is not activated make sure to call `trainer._activate_neftune()` first")

        unwrapped_model = unwrap_model(model)

        if _is_peft_model(unwrapped_model):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        self.neftune_hook_handle.remove()
        del embeddings.neftune_noise_alpha, unwrapped_model

    def add_callback(self, callback):
        """
        向当前的 [`~transformers.TrainerCallback`] 列表中添加一个回调函数。

        Args:
           callback (type or [`~transformers.TrainerCallback`]):
               [`~transformers.TrainerCallback`] 类或其实例。如果是类，则实例化一个该类的成员。
        """
    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        # 调用回调处理器的方法，从当前回调列表中弹出并返回指定的回调对象
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        # 调用回调处理器的方法，从当前回调列表中移除指定的回调对象
        self.callback_handler.remove_callback(callback)

    def _move_model_to_device(self, model, device):
        # 将模型移动到指定的设备上
        model = model.to(device)
        # 如果当前使用的是TPU并且模型具有"tie_weights"方法，则需要重新连接权重
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # 检查模型的前向方法签名，仅保留模型接受的参数列
            model_to_inspect = self.model
            # 如果模型是PEFT模型，则调整要检查的基础模型
            if _is_peft_model(self.model):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # 对于PeftMixedModel，没有提供"get_base_model"方法，因此需要直接访问base_model.model
                    model_to_inspect = self.model.base_model.model
            # 获取模型前向方法的参数签名，并将参数名称添加到签名列列表中
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # 标签可能命名为label或label_ids，使用默认数据拼接器来处理这些情况
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        # 如果不需要移除未使用的列，则直接返回原始数据集
        if not self.args.remove_unused_columns:
            return dataset
        # 根据需要设置签名列
        self._set_signature_columns_if_needed()
        # 获取签名列
        signature_columns = self._signature_columns

        # 找出数据集中被忽略的列（即不在签名列中的列）
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        # 如果有被忽略的列，则生成描述信息
        dset_description = "" if description is None else f"in the {description} set"
        # 记录日志，指出被忽略的列
        logger.info(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
            " you can safely ignore this message."
        )

        # 筛选出在签名列中且存在于数据集列名中的列
        columns = [k for k in signature_columns if k in dataset.column_names]

        # 根据 datasets 库的版本进行不同处理
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            # 设置数据集的格式，保留指定的列
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            # 移除数据集中的被忽略列
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        # 如果不需要移除未使用的列，则直接返回原始的数据 collator
        if not self.args.remove_unused_columns:
            return data_collator
        # 根据需要设置签名列
        self._set_signature_columns_if_needed()
        # 获取签名列
        signature_columns = self._signature_columns

        # 创建一个移除未使用列的数据 collator 对象
        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # 构建数据采样器。
        if self.args.group_by_length:
            # 如果数据集支持 datasets 库并且是 datasets.Dataset 类型，则获取长度信息
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            # 获取模型输入的名称，通常是第一个模型输入的名称
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            # 返回一个 LengthGroupedSampler 对象，用于按长度分组的数据采样
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            # 如果不按长度分组，则返回一个随机采样器 RandomSampler
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        返回训练数据加载器 [`~torch.utils.data.DataLoader`]。

        如果 `train_dataset` 不实现 `__len__`，则不使用采样器；否则使用随机采样器（适应分布式训练）。

        如果需要注入自定义行为，请子类化并重写此方法。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # 如果支持 datasets 库并且 train_dataset 是 datasets.Dataset 类型，则移除未使用的列
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            # 否则，使用移除了未使用列的数据集收集器
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # 设置数据加载器的参数
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # 如果 train_dataset 不是 IterableDataset 类型，则设置采样器等参数
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # 使用加速器准备数据加载器并返回
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    # 返回用于评估数据集的采样器，根据不同条件返回不同的采样器或者 None
    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        # 如果使用了遗留的预测循环方式
        if self.args.use_legacy_prediction_loop:
            # 如果当前环境支持 Torch XLA
            if is_torch_xla_available():
                # 返回一个按顺序分布的分布式采样器，用于 Torch XLA 环境
                return SequentialDistributedSampler(
                    eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            # 如果当前环境支持 SageMaker 的多进程
            elif is_sagemaker_mp_enabled():
                # 返回一个按顺序分布的分布式采样器，用于 SageMaker 多进程环境
                return SequentialDistributedSampler(
                    eval_dataset,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            else:
                # 返回一个按顺序的采样器
                return SequentialSampler(eval_dataset)

        # 如果设备的数量小于等于 1，返回一个按顺序的采样器
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            # 否则返回 None，表示不使用任何特定的采样器
            return None
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation `torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a `datasets.Dataset`, columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        # 检查是否没有提供 eval_dataset 且 self.eval_dataset 也没有设置，如果是则抛出数值错误
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # 如果已经存在 _eval_dataloader，并且设置了 dataloader_persistent_workers，则通过加速器准备 _eval_dataloader 并返回
        if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloader)

        # 如果 eval_dataset 未提供，则使用 self.eval_dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        # 如果使用了 datasets 库，并且 eval_dataset 是 datasets.Dataset 类型，则调用 _remove_unused_columns 方法删除不被 model.forward() 方法接受的列
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            # 否则调用 _get_collator_with_removed_columns 方法更新数据集的数据收集器，以移除不需要的列
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        # 设置数据加载器参数
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # 如果 eval_dataset 不是 IterableDataset 类型，则设置 sampler、drop_last 和 prefetch_factor 参数
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # 创建评估数据加载器 DataLoader 对象
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

        # 如果设置了 dataloader_persistent_workers，则将 eval_dataloader 赋值给 self._eval_dataloader
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        # 返回通过加速器准备后的 eval_dataloader
        return self.accelerator.prepare(eval_dataloader)
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        # 获取数据收集器
        data_collator = self.data_collator

        # 如果datasets库可用且test_dataset是datasets.Dataset类型，则移除模型不接受的列
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            # 否则，使用移除特定列的数据收集器
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        # 设置DataLoader的参数
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # 如果test_dataset不是IterableDataset类型，则配置采样器和其他参数
        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # 使用加速器准备DataLoader并返回
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        # 创建优化器
        self.create_optimizer()
        
        # 根据条件选择优化器
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # 如果使用的SageMaker版本 >= 1.10 并且启用了fp16，解包优化器
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        
        # 创建学习率调度器
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        # 获取所有需要应用权重衰减的参数名称
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        
        # 过滤掉包含"bias"的参数名
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        return decay_parameters
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # 根据条件选择要优化的模型
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        # 如果没有指定优化器，则根据模型参数进行分组设置
        if self.optimizer is None:
            # 获取需要进行权重衰减的参数名列表
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # 设置优化器的分组参数
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # 获取优化器类和参数
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # 如果 optimizer_kwargs 中包含 'params' 键，则使用它覆盖之前设置的 optimizer_grouped_parameters
            # 例如，适用于 GaLore 优化器
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # 如果 optimizer_kwargs 中包含 'optimizer_dict' 键，则使用它覆盖 optimizer_grouped_parameters
            # 避免参数冲突问题，适用于逐层的虚拟优化器
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            # 创建优化器实例
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # 如果使用的优化器是 'Adam8bit'，则执行特定处理
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                # 获取全局优化管理器的实例
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                # 注册所有嵌入层模块并指定优化参数的精度为 32 位浮点数
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        # 如果启用了 SageMaker 多进程，则使用分布式优化器
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        # 返回设置好的优化器
        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ):
        """
        Helper function to retrieve the optimizer class and its keyword arguments based on training arguments and model.
        """
        # 省略，用于获取优化器类和参数的辅助函数
        pass
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        设置调度器。在调用此方法之前，训练器的优化器必须已经设置好，或者作为参数传递进来。

        Args:
            num_training_steps (int): 要执行的训练步数。
        """
        # 如果当前没有设置学习率调度器，则根据参数设置一个新的调度器
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            # 标记已经创建了学习率调度器
            self._created_lr_scheduler = True
        # 返回当前的学习率调度器对象
        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        辅助函数，通过访问其数据集来获取 [`~torch.utils.data.DataLoader`] 中的样本数量。
        当 dataloader.dataset 不存在或长度为零时，尽可能进行估计。
        """
        try:
            dataset = dataloader.dataset
            # 对于 IterableDatasetShard，需要进一步深入获取其数据集的长度
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            # 返回 DataLoader 中数据集的长度
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # 如果没有数据集或长度信息，通过 DataLoader 的长度估算
            # 通过 DataLoader 的长度估算样本数，乘以每设备训练批次大小
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        辅助函数，通过枚举数据加载器来获取 [`~torch.utils.data.DataLoader`] 中的令牌数量。
        """
        train_tokens = 0
        try:
            # 枚举训练数据加载器的步骤和批次
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()  # 获取当前批次中 "input_ids" 的令牌数量
                if max_steps is not None:
                    return tokens * max_steps  # 如果指定了最大步数，则返回按最大步数估算的令牌总数
                train_tokens += tokens  # 累加当前批次的令牌数量到训练令牌总数
            return train_tokens  # 返回训练数据加载器中的总令牌数量
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")  # 日志警告，无法从数据加载器获取令牌数量信息
            return train_tokens  # 返回当前已经累积的训练令牌总数
    # 设置超参数搜索的初始化代码
    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """HP search setup code"""
        # 将试验对象保存到实例属性中
        self._trial = trial

        # 如果超参数搜索后端为空或试验对象为空，则直接返回
        if self.hp_search_backend is None or trial is None:
            return

        # 根据选择的超参数搜索后端进行参数初始化
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            # 使用 Optuna 提供的超参数空间生成参数字典
            params = self.hp_space(trial)
        elif self.hp_search_backend == HPSearchBackend.RAY:
            # 如果使用 Ray 提供的试验参数，排除其中的 WandB 相关项
            params = trial
            params.pop("wandb", None)
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            # 如果使用 SigOpt 提供的试验分配，将字符串形式的值转换为整数
            params = {k: int(v) if isinstance(v, str) else v for k, v in trial.assignments.items()}
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            # 如果使用 WandB 提供的试验参数
            params = trial

        # 根据参数字典更新实例属性中的参数
        for key, value in params.items():
            # 如果参数在 self.args 中不存在，则发出警告
            if not hasattr(self.args, key):
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in"
                    " `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            # 将值转换为与旧属性相同类型的值
            if old_attr is not None:
                value = type(old_attr)(value)

            # 更新 self.args 中的参数值
            setattr(self.args, key, value)

        # 根据不同的超参数搜索后端记录日志信息
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info(f"Trial: {trial.params}")
        if self.hp_search_backend == HPSearchBackend.SIGOPT:
            logger.info(f"SigOpt Assignments: {trial.assignments}")
        if self.hp_search_backend == HPSearchBackend.WANDB:
            logger.info(f"W&B Sweep parameters: {trial}")

        # 如果启用了 DeepSpeed 加速，并且未设置 args.deepspeed，则引发异常
        if self.is_deepspeed_enabled:
            if self.args.deepspeed is None:
                raise ValueError("For sweeps with deepspeed, `args.deepspeed` must be set")

            # 重新构建 DeepSpeed 配置，以反映更新后的训练参数
            from accelerate.utils import DeepSpeedPlugin
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            # 创建 HfTrainerDeepSpeedConfig 实例
            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            # 处理 trainer 配置过程
            self.args.hf_deepspeed_config.trainer_config_process(self.args)
            # 创建 DeepSpeed 插件
            self.args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.args.hf_deepspeed_config)

        # 创建加速器并进行后处理
        self.create_accelerator_and_postprocess()
    # 将试验结果报告给超参数搜索后端（例如Optuna或Ray）
    def _report_to_hp_search(self, trial: Union["optuna.Trial", Dict[str, Any]], step: int, metrics: Dict[str, float]):
        # 如果超参数搜索后端未设置或试验对象为空，则直接返回
        if self.hp_search_backend is None or trial is None:
            return
        
        # 复制metrics，以免修改原始数据
        metrics = metrics.copy()
        
        # 计算当前目标值
        self.objective = self.compute_objective(metrics)
        
        # 如果使用Optuna作为超参数搜索后端
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna
            
            # 如果试验不是多目标优化
            if not trial.study._is_multi_objective():
                # 向Optuna试验报告当前目标值和步数
                trial.report(self.objective, step)
                
                # 如果试验应该被剪枝，则结束训练并抛出TrialPruned异常
                if trial.should_prune():
                    self.callback_handler.on_train_end(self.args, self.state, self.control)
                    raise optuna.TrialPruned()
        
        # 如果使用Ray作为超参数搜索后端
        elif self.hp_search_backend == HPSearchBackend.RAY:
            import ray.train
            
            # 使用临时目录保存检查点
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                # 如果需要保存模型检查点
                if self.control.should_save:
                    # 保存当前模型检查点到临时目录
                    self._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                    checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                
                # 将目标值添加到metrics中
                metrics["objective"] = self.objective
                
                # 向Ray报告metrics和检查点
                ray.train.report(metrics, checkpoint=checkpoint)

    # 保存当前训练状态的检查点
    def _tune_save_checkpoint(self, checkpoint_dir: str):
        # 设置输出目录为临时检查点目录下的指定全局步数的目录
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        
        # 将模型保存到指定的输出目录
        self.save_model(output_dir, _internal_call=True)
        
        # 如果需要保存参数
        if self.args.should_save:
            # 将当前训练状态保存为JSON文件
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            
            # 保存优化器状态字典
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            
            # 保存学习率调度器状态字典
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    # 调用模型初始化函数并返回初始化后的模型
    def call_model_init(self, trial=None):
        # 获取model_init函数的参数个数
        model_init_argcount = number_of_arguments(self.model_init)
        
        # 根据model_init函数的参数个数调用相应的初始化方式
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")
        
        # 如果模型初始化结果为None，则抛出异常
        if model is None:
            raise RuntimeError("model_init should not return None.")
        
        # 返回初始化后的模型
        return model
    # 定义一个方法用于在 PyTorch 中评估 JIT 模式下的模型
    def torch_jit_model_eval(self, model, dataloader, training=False):
        # 如果不是训练模式
        if not training:
            # 如果数据加载器为 None，则记录警告并返回原始模型
            if dataloader is None:
                logger.warning("failed to use PyTorch jit mode due to current dataloader is none.")
                return model
            # 获取一个示例批次数据
            example_batch = next(iter(dataloader))
            # 准备输入数据
            example_batch = self._prepare_inputs(example_batch)
            try:
                # 复制模型用于 JIT 编译
                jit_model = copy.copy(model)
                # 将模型设置为评估模式
                jit_model.eval()
                # 提取模型中的原始 forward 方法
                original_forward = jit_model.__dict__.pop("_original_forward", None)
                # 如果存在原始 forward 方法，则恢复
                if original_forward:
                    jit_model.forward = original_forward
                # 使用加速器自动混合精度，并关闭缓存，同时不计算梯度
                with self.accelerator.autocast(cache_enabled=False), torch.no_grad():
                    # 根据 PyTorch 版本选择不同的 JIT 编译方式
                    if version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.0.0"):
                        # 如果示例批次是字典，则使用关键字参数输入 JIT 编译
                        if isinstance(example_batch, dict):
                            jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                        # 否则，构建适合的关键字参数输入
                        else:
                            jit_model = torch.jit.trace(
                                jit_model,
                                example_kwarg_inputs={key: example_batch[key] for key in example_batch},
                                strict=False,
                            )
                    else:
                        # 构建适合的位置参数输入
                        jit_inputs = []
                        for key in example_batch:
                            example_tensor = torch.ones_like(example_batch[key])
                            jit_inputs.append(example_tensor)
                        jit_inputs = tuple(jit_inputs)
                        jit_model = torch.jit.trace(jit_model, jit_inputs, strict=False)
                # 冻结 JIT 编译后的模型，以提高性能
                jit_model = torch.jit.freeze(jit_model)
                # 使用 JIT 模型执行两次示例批次数据
                with torch.no_grad():
                    jit_model(**example_batch)
                    jit_model(**example_batch)
                # 更新模型为 JIT 编译后的模型，并关闭 CPU 自动混合精度优化
                model = jit_model
                self.use_cpu_amp = False
            # 捕获可能的异常并记录警告
            except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
                logger.warning(f"failed to use PyTorch jit mode due to: {e}.")

        # 返回最终的模型
        return model
    # 使用 IPEX 优化模型，如果 IPEX 不可用则抛出 ImportError 异常
    def ipex_optimize_model(self, model, training=False, dtype=torch.float32):
        if not is_ipex_available():
            raise ImportError(
                "Using IPEX but IPEX is not installed or IPEX's version does not match current PyTorch, please refer"
                " to https://github.com/intel/intel-extension-for-pytorch."
            )

        import intel_extension_for_pytorch as ipex

        if not training:
            # 如果不是训练模式，设置模型为评估模式
            model.eval()
            # 根据条件设置 dtype，如果不在训练中且启用了 bf16 全量评估，则使用 torch.bfloat16
            dtype = torch.bfloat16 if not self.is_in_train and self.args.bf16_full_eval else dtype
            # 对模型进行优化，设置数据类型和优化级别 O1，同时禁用 conv_bn_folding 来避免符号跟踪中的问题
            model = ipex.optimize(model, dtype=dtype, level="O1", conv_bn_folding=False, inplace=not self.is_in_train)
        else:
            if not model.training:
                # 如果模型不处于训练状态，则设置为训练状态
                model.train()
            # 对模型进行优化，设置数据类型、优化器和优化级别 O1，同时进行原地操作
            model, self.optimizer = ipex.optimize(
                model, dtype=dtype, optimizer=self.optimizer, inplace=True, level="O1"
            )

        return model

    # 训练方法，支持从检查点恢复和使用 Optuna 进行超参数搜索
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
    
    # 内部训练循环方法，接受批处理大小、参数、从检查点恢复的标志和超参数搜索实例作为输入
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
    
    # 根据超参数搜索后端和试验对象确定输出目录
    def _get_output_dir(self, trial):
        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                import ray.train
                run_id = ray.train.get_context().get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb
                run_id = wandb.run.id
            # 如果设置了超参数名称生成函数，则使用该函数；否则使用默认名称格式
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            # 组合运行目录，基于设定的输出目录和生成的运行名称
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            # 如果未设置超参数搜索后端或没有试验对象，则直接使用默认的输出目录
            run_dir = self.args.output_dir
        return run_dir

    # 在加载模型后发出警告，根据加载结果中的丢失和意外键发出相应的警告信息
    def _issue_warnings_after_load(self, load_result):
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                # 如果加载结果中的丢失键与保存时忽略的键匹配，则进行权重绑定
                self.model.tie_weights()
            else:
                # 否则发出丢失键的警告
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            # 如果加载结果中存在意外键，则发出相应的警告
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
        # 检查是否应记录日志，并且全局步数大于上次记录的步数
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # 如果使用 Torch XLA，标记当前步骤
            if is_torch_xla_available():
                xm.mark_step()

            # 初始化日志字典
            logs: Dict[str, float] = {}

            # 使用 all_gather + mean() 计算所有进程的平均损失
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # 将训练损失重置为零
            tr_loss -= tr_loss

            # 计算并记录平均损失
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            
            # 如果存在梯度范数，记录梯度范数值
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            
            # 记录当前学习率
            logs["learning_rate"] = self._get_learning_rate()

            # 累加总损失
            self._total_loss_scalar += tr_loss_scalar

            # 更新上次记录的全局步数
            self._globalstep_last_logged = self.state.global_step

            # 存储 FLOPs（浮点运算次数）
            self.store_flos()

            # 记录日志
            self.log(logs)

        metrics = None
        # 如果需要进行评估
        if self.control.should_evaluate:
            # 执行评估并获取指标
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # 报告给超参数搜索器当前的全局步数和指标
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # 如果使用 ReduceLROnPlateau 类型的学习率调度器
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 获取用于最佳模型选择的指标名称
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                # 调用学习率调度器的步进方法，根据最新的指标值更新学习率
                self.lr_scheduler.step(metrics[metric_to_check])

        # 如果需要保存模型
        if self.control.should_save:
            # 保存模型检查点
            self._save_checkpoint(model, trial, metrics=metrics)
            # 调用保存模型时的回调处理
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    # 从检查点 `checkpoint` 中加载 RNG 状态
    def _load_rng_state(self, checkpoint):
        # 如果检查点为 None，则直接返回
        if checkpoint is None:
            return
        
        # 如果使用多进程（分布式训练），根据进程索引读取相应的 RNG 状态文件
        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            # 如果对应的 RNG 文件不存在，则记录警告信息并返回
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            # 如果是单进程，则读取默认的 RNG 状态文件
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            # 如果默认的 RNG 文件不存在，则记录警告信息并返回
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        # 加载检查点中的 RNG 状态
        checkpoint_rng_state = torch.load(rng_file)
        # 恢复 Python 内置的随机数生成器状态
        random.setstate(checkpoint_rng_state["python"])
        # 恢复 NumPy 随机数生成器状态
        np.random.set_state(checkpoint_rng_state["numpy"])
        # 恢复 PyTorch CPU 随机数生成器状态
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        
        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 如果是分布式训练且使用并行模式
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # 恢复所有 GPU 的 CUDA 随机数生成器状态
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    # 恢复当前 GPU 的 CUDA 随机数生成器状态
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
        
        # 如果使用了 Torch XLA
        if is_torch_xla_available():
            # 恢复 Torch XLA 的随机数生成器状态
            xm.set_rng_state(checkpoint_rng_state["xla"])
        
        # 如果使用了 Torch NPU
        if is_torch_npu_available():
            # 如果是分布式训练且使用并行模式
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # 恢复所有 NPU 的随机数生成器状态
                torch.npu.random.set_rng_state_all(checkpoint_rng_state["npu"])
            else:
                try:
                    # 恢复当前 NPU 的随机数生成器状态
                    torch.npu.random.set_rng_state(checkpoint_rng_state["npu"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the NPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
    # 定义一个保存检查点的方法，用于保存模型及其相关状态和参数
    def _save_checkpoint(self, model, trial, metrics=None):
        # 在所有情况下，包括使用 ddp/dp/deepspeed，self.model 总是指向我们想要保存的模型的引用，除了 FullyShardedDDP。
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # 保存模型检查点的文件夹名称，包含全局步数信息
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # 如果未进行超参数搜索并且没有试验信息，则存储 FLOPS
        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        # 获取运行输出目录，根据试验信息创建检查点输出目录
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        # 调用保存模型的方法，并指明这是内部调用
        self.save_model(output_dir, _internal_call=True)

        # 如果不仅仅保存模型，则继续保存优化器和调度器
        if not self.args.save_only_model:
            # 保存优化器和调度器状态
            self._save_optimizer_and_scheduler(output_dir)
            # 保存随机数生成器状态
            self._save_rng_state(output_dir)

        # 确定新的最佳指标和最佳模型检查点
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            # 如果当前的指标值更好，则更新最佳指标和最佳模型检查点路径
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # 如果需要保存 Trainer 的状态信息，则将其保存为 JSON 文件
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # 如果指定了推送到 Hub，则从当前检查点路径进行推送
        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # 可能会删除一些旧的检查点
        if self.args.should_save:
            # 仅依赖于数字化的检查点 id 进行旋转管理
            # 在某些云环境中，特别是一些 fuse 文件系统中，mtime 并不可靠
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)
    # 在非分布式训练中保存随机数生成器状态
    rng_states = {
        "python": random.getstate(),  # 获取 Python 内置随机数生成器的状态
        "numpy": np.random.get_state(),  # 获取 NumPy 随机数生成器的状态
        "cpu": torch.random.get_rng_state(),  # 获取 PyTorch CPU 随机数生成器的状态
    }
    # 如果 CUDA 可用
    if torch.cuda.is_available():
        # 如果在分布式模式下
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # 在非分布式情况下，保存全局 CUDA 随机数生成器的状态（会考虑到 DataParallel）
            rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
        else:
            # 获取当前 CUDA 设备上的随机数生成器的状态
            rng_states["cuda"] = torch.cuda.random.get_rng_state()

    # 如果 PyTorch XLA 可用
    if is_torch_xla_available():
        # 获取当前 XLA 设备上的随机数生成器的状态
        rng_states["xla"] = xm.get_rng_state()

    # 如果 PyTorch NPU 可用
    if is_torch_npu_available():
        # 如果在分布式模式下
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # 获取所有 NPU 设备上的随机数生成器的状态
            rng_states["npu"] = torch.npu.random.get_rng_state_all()
        else:
            # 获取当前 NPU 设备上的随机数生成器的状态
            rng_states["npu"] = torch.npu.random.get_rng_state()

    # 在保存模型之前，确保输出目录已经存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)

    # 根据当前进程数决定保存的文件名
    if self.args.world_size <= 1:
        # 如果只有一个进程，则保存为 rng_state.pth
        torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
    else:
        # 如果有多个进程，则保存为 rng_state_{进程编号}.pth
        torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))
    # 保存优化器和调度器状态到指定的输出目录
    def _save_optimizer_and_scheduler(self, output_dir):
        # 如果支持 Torch XLA 加速
        if is_torch_xla_available():
            # 使用 XM 进行同步，保存优化器状态字典到指定路径
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            # 使用 XM 保存学习率调度器状态字典到指定路径，并记录警告信息
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        # 如果启用了 SageMaker 分布式训练
        elif is_sagemaker_mp_enabled():
            # 获取本地优化器状态字典，并进行屏障同步
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            # 如果当前进程是第一个或者配置要求分片优化器状态，保存优化器状态到指定路径
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        # 如果启用了 DeepSpeed
        elif self.is_deepspeed_enabled:
            # 根据条件判断是否排除冻结参数并保存模型检查点到指定路径
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                self.model_wrapped.save_checkpoint(output_dir)
        # 如果启用了 FSDP（Fully Sharded Data Parallelism）
        elif self.is_fsdp_enabled:
            # 保存 FSDP 特定的模型检查点和优化器状态到指定路径
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **_get_fsdp_ckpt_kwargs()
            )
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        # 如果需要保存模型
        elif self.args.should_save:
            # 仅保存优化器状态字典到指定路径
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # 保存学习率调度器状态字典到指定路径，如果满足保存条件
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_xla_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            # 处理保存过程中的警告信息
            reissue_pt_warnings(caught_warnings)
    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: Union[str, List[str]] = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ):
        """
        Perform hyperparameter search using Optuna.

        Args:
            hp_space (Optional[Callable[["optuna.Trial"], Dict[str, float]]]):
                Function defining the hyperparameter search space.
            compute_objective (Optional[Callable[[Dict[str, float]], float]]):
                Function to compute the objective given a set of hyperparameters.
            n_trials (int):
                Number of trials (hyperparameter combinations) to evaluate.
            direction (Union[str, List[str]]):
                Direction to optimize the objective, either 'minimize' or 'maximize'.
            backend (Optional[Union[str, HPSearchBackend]]):
                Backend for hyperparameter search.
            hp_name (Optional[Callable[["optuna.Trial"], str]]):
                Function to generate a name for the hyperparameter set.
            **kwargs:
                Additional keyword arguments passed to the hyperparameter search.

        Returns:
            None
        """
        pass

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.

        Args:
            data (Union[torch.Tensor, Any]):
                The input data to prepare.

        Returns:
            Union[torch.Tensor, Any]:
                Prepared data ready to be fed into the model.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # Adjust dtype for deepspeed enabled models
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        # 调用内部方法，准备输入数据，确保转换为张量（如果尚未），同时处理潜在的状态
        inputs = self._prepare_input(inputs)
        
        # 如果输入数据为空，抛出数值错误，防止模型无法训练
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        
        # 如果设置了历史索引且存在历史数据，则将历史数据添加到输入中
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        
        # 返回准备好的输入数据字典
        return inputs

    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        # 调用自动混合精度上下文管理器的帮助器包装器
        return self.autocast_smart_context_manager()

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        # 根据是否启用 CPU 自动混合精度，创建相应的上下文管理器
        if self.use_cpu_amp:
            ctx_manager = torch.cpu.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)
        else:
            ctx_manager = contextlib.nullcontext()

        return ctx_manager

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # 设置模型为训练模式
        model.train()
        
        # 准备输入数据
        inputs = self._prepare_inputs(inputs)

        # 如果启用 SageMaker 多进程训练，则调用相应的前向-反向传播函数
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # 使用上下文管理器计算损失
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # 如果有多个 GPU，则对损失进行平均，用于多 GPU 并行训练
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # 如果使用 Apex 混合精度训练，则使用 Amp 应用梯度缩放
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # 返回分离后的损失值，除以梯度累积步骤数
        return loss.detach() / self.args.gradient_accumulation_steps
    # 定义一个方法，计算模型的损失值。
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # 如果存在标签平滑器且输入中包含标签，则将标签从输入中弹出
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # 使用模型处理输入数据，获取模型输出
        outputs = model(**inputs)
        # 如果定义了保存过去状态的索引
        if self.args.past_index >= 0:
            # 将输出中对应索引的内容保存到 _past 属性中
            self._past = outputs[self.args.past_index]

        # 如果存在标签，则执行以下逻辑
        if labels is not None:
            # 获取未封装的模型（去除所有包装器）
            unwrapped_model = unwrap_model(model)
            # 判断模型是否为 PEFT 模型
            if _is_peft_model(unwrapped_model):
                # 获取模型的名称
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # 如果模型名称存在于 causal LM 映射名称中
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                # 使用标签平滑器计算损失（带标签偏移）
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                # 使用标签平滑器计算损失
                loss = self.label_smoother(outputs, labels)
        else:
            # 如果模型输出是字典且没有包含损失键
            if isinstance(outputs, dict) and "loss" not in outputs:
                # 抛出数值错误，说明模型未从输入中返回损失值
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # 从 outputs 中获取损失值（可能是元组形式的 ModelOutput）
            # 这里不使用 .loss 是因为模型可能返回元组而不是 ModelOutput
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 如果需要返回输出内容，则同时返回损失和模型输出
        return (loss, outputs) if return_outputs else loss

    # 定义一个方法，用于判断当前进程是否是本地的主进程（在分布式训练中，一台机器上的主进程）
    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    # 定义一个方法，用于判断当前进程是否是全局的主进程（在分布式训练中，只有一个进程会返回 True）
    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # 对于 SageMaker ModelParallel 的特殊情况，进程索引为 dp_process_index 而不是全局进程索引
        if is_sagemaker_mp_enabled():
            return smp.rank() == 0
        else:
            # 判断当前进程索引是否为 0
            return self.args.process_index == 0
    # 定义保存模型的方法，可以指定输出目录和是否内部调用标志
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        # 如果未指定输出目录，则使用默认的输出目录
        if output_dir is None:
            output_dir = self.args.output_dir

        # 如果当前环境支持 Torch XLA，保存模型到指定目录
        if is_torch_xla_available():
            self._save_tpu(output_dir)
        
        # 如果是在 SageMaker 多进程环境下，创建输出目录，并保存模型状态字典
        elif is_sagemaker_mp_enabled():
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            # 如果应当保存模型，则执行保存操作
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            # 如果是 SageMaker MP 版本大于等于 1.10，则创建一个标志文件指示模型状态字典已保存
            if IS_SAGEMAKER_MP_POST_1_10:
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        
        # 如果启用了 FSDP（Fully Sharded Data Parallelism），保存模型状态字典
        elif self.is_fsdp_enabled:
            # 检查是否完整状态字典，并且加速库版本大于 0.24.1
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.model)
                # 如果应当保存模型，则执行保存操作
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        
        # 如果启用了 DeepSpeed，尝试获取 DeepSpeed 的状态字典并保存
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                # 如果应当保存模型，则执行保存操作
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                # 如果出现异常，则警告并保存空的状态字典
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # 移除虚拟的状态字典
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                # 使用包装后的模型保存完整的检查点
                self.model_wrapped.save_checkpoint(output_dir)
        
        # 如果应当保存模型，则执行保存操作
        elif self.args.should_save:
            self._save(output_dir)

        # 当用户调用 `save_model` 且 `push_to_hub` 为真时，推送模型到 Hub
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
    # 定义一个方法用于保存模型到指定目录，可以选择性地指定输出目录，否则使用默认输出目录
    def _save_tpu(self, output_dir: Optional[str] = None):
        # 如果未提供输出目录，则使用 self.args.output_dir
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # 打印日志，指示正在保存模型检查点到指定的输出目录
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 获取模型对象
        model = self.model

        # 在TPU上标记当前步骤
        xm.mark_step()

        # 将模型移动到CPU上保存
        model.to("cpu")

        # 如果是主节点（master ordinal），创建输出目录（如果不存在），并保存训练参数
        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # 保存训练好的模型和配置，使用 `save_pretrained()` 方法
        # 可以通过 `from_pretrained()` 方法重新加载
        supported_classes = (PushToHubMixin,)
        xm.rendezvous("saving_checkpoint")

        # 如果模型不是支持的类别，则尝试解开模型再保存其状态字典
        if not isinstance(model, supported_classes):
            if isinstance(unwrap_model(model), supported_classes):
                unwrap_model(model).save_pretrained(
                    output_dir,
                    is_main_process=self.args.should_save,
                    state_dict=model.state_dict(),
                    save_function=xm.save,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = model.state_dict()
                xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # 如果模型属于支持的类别，则直接调用其 save_pretrained 方法保存
            model.save_pretrained(
                output_dir,
                is_main_process=self.args.should_save,
                save_function=xm.save,
                safe_serialization=self.args.save_safetensors,
            )

        # 如果存在 tokenizer 并且应该保存，则保存 tokenizer 到输出目录
        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)

        # 将模型从 CPU 移回到指定设备（通常是 GPU 或 TPU），确保后续计算可以继续正常运行
        model.to(self.args.device)
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 如果指定了输出目录，则使用指定的输出目录，否则使用默认的输出目录 self.args.output_dir
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # 创建输出目录，如果目录已存在则不报错
        os.makedirs(output_dir, exist_ok=True)
        # 记录日志，指示正在将模型检查点保存到哪个目录
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 定义支持的模型类别，如果没有 PEFT 可用，则只支持 PreTrainedModel；否则还支持 PeftModel
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # 保存训练好的模型和配置，使用 `save_pretrained()` 方法保存
        # 可以使用 `from_pretrained()` 方法重新加载
        if not isinstance(self.model, supported_classes):
            # 如果没有提供状态字典，则获取当前模型的状态字典
            if state_dict is None:
                state_dict = self.model.state_dict()

            # 如果模型属于支持的类别，则调用其 save_pretrained 方法保存模型和状态字典
            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                # 如果模型不是 `PreTrainedModel` 类型，则只保存其状态字典
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                # 如果设置了保存安全张量，则使用 safetensors 库保存文件
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    # 否则使用 PyTorch 自带的 torch.save 方法保存状态字典
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # 如果模型属于支持的类别，则直接调用其 save_pretrained 方法保存模型和状态字典
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        # 如果存在 tokenizer 对象，则保存其预训练模型
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 最佳实践：将训练参数与训练好的模型一起保存
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def store_flos(self):
        # 存储模型中所用的浮点运算总数
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # 如果并行模式为 DISTRIBUTED，则将当前计算的 FLOPS 分布式广播到所有设备并求和
            self.state.total_flos += (
                distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
            )
            # 清零当前计算的 FLOPS
            self.current_flos = 0
        else:
            # 否则直接将当前计算的 FLOPS 加到总数中，并清零当前计算的 FLOPS
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ):
        # 该函数暂未提供具体实现，在这里只做函数声明
    # 返回已排序的检查点路径列表，按照文件修改时间或者文件名中的数字排序
    def _sorted_checkpoints(self, use_mtime=False, output_dir=None) -> List[str]:
        ordering_and_checkpoint_path = []

        # 获取匹配指定前缀的所有检查点文件夹路径
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        # 遍历每个检查点文件夹路径
        for path in glob_checkpoints:
            if use_mtime:
                # 如果使用文件修改时间作为排序依据，将时间戳和路径添加到排序列表中
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                # 否则，尝试从文件名中提取数字，将数字和路径添加到排序列表中
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        # 按照时间戳或者文件名中的数字排序检查点路径
        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        # 仅保留排序后的检查点路径，不包含排序依据
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

        # 确保最佳模型不会被删除，将其移动到检查点列表的前部
        if (
            self.state.best_model_checkpoint is not None
            and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]

        # 返回排序后的检查点路径列表
        return checkpoints_sorted

    # 根据保存的总数限制旋转检查点，删除多余的检查点
    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # 获取排序后的检查点路径列表
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        # 如果检查点数量小于或等于保存总数限制，则不需要删除
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # 根据特定条件调整保存总数限制
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        # 计算需要删除的检查点数量，并获取待删除的检查点路径列表
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

        # 删除待删除的检查点文件夹
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)
        """
        # 运行预测并返回预测结果和可能的指标。

        # 根据数据集和使用情况，测试数据集可能包含标签。在这种情况下，该方法还会返回指标，例如在 `evaluate()` 中一样。

        Args:
            test_dataset (`Dataset`):
                要运行预测的数据集。如果是 `datasets.Dataset`，则会自动删除模型 `forward()` 方法不接受的列。必须实现 `__len__` 方法。
            ignore_keys (`List[str]`, *可选*):
                在模型输出中应忽略的键列表（如果是字典）。
            metric_key_prefix (`str`, *可选*, 默认为 `"test"`):
                用作指标键前缀的可选前缀。例如，如果前缀是 "test"（默认），则指标 "bleu" 将命名为 "test_bleu"。

        <Tip>

        如果您的预测或标签具有不同的序列长度（例如，因为您在标记分类任务中进行动态填充），则会对预测进行填充（在右侧），以允许串联到一个数组中。填充索引为 -100。

        </Tip>

        Returns: *NamedTuple* 具有以下键的命名元组:

            - predictions (`np.ndarray`): 对 `test_dataset` 的预测。
            - label_ids (`np.ndarray`, *可选*): 标签（如果数据集包含）。
            - metrics (`Dict[str, float]`, *可选*): 可能包含标签的字典。

        """
        # 内存指标 - 必须尽早设置
        self._memory_tracker.start()

        # 获取测试数据加载器
        test_dataloader = self.get_test_dataloader(test_dataset)
        
        # 记录开始时间
        start_time = time.time()

        # 选择使用旧版预测循环还是评估循环
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        
        # 执行预测/评估循环
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        
        # 计算总批量大小
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        
        # 如果存在编译时间指标，则调整开始时间
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        
        # 更新速度指标
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # 调用预测的回调处理器
        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        
        # 停止内存跟踪器并更新指标
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # 返回预测输出
        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Runs evaluation loop over `dataloader`.

        Args:
            dataloader (DataLoader): The data loader for evaluation.
            description (str): Description of the evaluation loop.
            prediction_loss_only (Optional[bool], optional): Whether to compute only prediction loss. Defaults to None.
            ignore_keys (Optional[List[str]], optional): List of keys to ignore during evaluation. Defaults to None.
            metric_key_prefix (str, optional): Prefix for metric keys. Defaults to "eval".
        """
        # Implementation of evaluation loop code...
        

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`.
        """
        if tensors is None:
            return
        # Check if running on TPU (Tensor Processing Unit)
        if is_torch_xla_available():
            if name is None:
                name = "nested_gather"
            tensors = nested_xla_mesh_reduce(tensors, name)
        # Check if running on SageMaker's multi-processing
        elif is_sagemaker_mp_enabled():
            tensors = smp_gather(tensors)
        # Check if running in distributed setting
        elif (self.args.distributed_state is not None and self.args.distributed_state.distributed_type != "NO") or (
            self.args.distributed_state is None and self.args.local_rank != -1
        ):
            tensors = distributed_concat(tensors)
        return tensors

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Perform a prediction step using `model` on `inputs`.

        Args:
            model (nn.Module): The model for prediction.
            inputs (Dict[str, Union[torch.Tensor, Any]]): Dictionary of inputs for the model.
            prediction_loss_only (bool): Whether to compute only prediction loss.
            ignore_keys (Optional[List[str]], optional): List of keys to ignore during prediction.

        Returns:
            Depends on the model's prediction step implementation.
        """
        # Implementation of prediction step code...
        

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        Computes the number of floating point operations for the model.

        Args:
            inputs (Dict[str, Union[torch.Tensor, Any]]): The inputs and targets of the model.

        Returns:
            int: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def init_hf_repo(self):
        """
        Initializes a git repository in `self.args.hub_model_id`.
        """
        # Only proceed if current process is the zeroth process
        if not self.is_world_process_zero():
            return

        # Determine repository name
        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id

        # Create or access the repository
        repo_url = create_repo(repo_name, token=self.args.hub_token, private=self.args.hub_private_repo, exist_ok=True)
        self.hub_model_id = repo_url.repo_id
        self.push_in_progress = None
    `
        # 定义一个方法 `create_model_card`，用于创建模型卡片文档
        def create_model_card(
            self,
            # 可选参数：指定语言，用于描述模型卡片的语言环境
            language: Optional[str] = None,
            # 可选参数：指定许可证信息，用于描述模型的许可条件
            license: Optional[str] = None,
            # 可选参数：模型标签，可以是单个标签字符串或标签列表，用于标记模型特性
            tags: Union[str, List[str], None] = None,
            # 可选参数：模型名称，用于标识模型的名称
            model_name: Optional[str] = None,
            # 可选参数：模型微调自哪个模型，用于记录模型的微调来源
            finetuned_from: Optional[str] = None,
            # 可选参数：任务类型，可以是单个任务字符串或任务列表，描述模型适用的任务类型
            tasks: Union[str, List[str], None] = None,
            # 可选参数：数据集标签，可以是单个标签字符串或标签列表，描述模型所用的数据集标签
            dataset_tags: Union[str, List[str], None] = None,
            # 可选参数：数据集名称或标识，可以是单个名称字符串或标识可以是字符串或字符串列表
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `List[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            model_name (`str`, *optional*):
                The name of the model.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `List[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `List[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `List[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `List[str]`, *optional*):
                One or several dataset arguments, to be included in the metadata of the model card.
        """
        # 检查当前进程是否是主进程，如果不是，则直接返回，不执行后续操作
        if not self.is_world_process_zero():
            return

        # 模型卡片文件的保存路径
        model_card_filepath = os.path.join(self.args.output_dir, "README.md")

        # 判断模型卡片文件是否已存在
        is_peft_library = False
        if os.path.exists(model_card_filepath):
            # 如果文件存在，加载模型卡片数据并获取其中的库名称
            library_name = ModelCard.load(model_card_filepath).data.get("library_name")
            # 判断加载的模型卡片是否来自于 PEFT 库
            is_peft_library = library_name == "peft"

            # 如果有指定 tags，并且已存在的 tags 不为空，则将新的 tags 添加到现有 tags 中
            existing_tags = ModelCard.load(model_card_filepath).data.tags
            if tags is not None and existing_tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in existing_tags:
                    if tag not in tags:
                        tags.append(tag)

        # 根据 Trainer 中的信息生成训练摘要
        training_summary = TrainingSummary.from_trainer(
            self,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        # 将训练摘要转换为模型卡片格式
        model_card = training_summary.to_model_card()
        # 将模型卡片写入到指定路径的 README.md 文件中
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

        # 如果是 PEFT 库，则调用特定函数更新或创建模型卡片
        if is_peft_library:
            unwrap_model(self.model).create_or_update_model_card(self.args.output_dir)
    # 仅在当前节点是主节点时执行推送操作
    if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
        return
    # 如果上次推送未完成且未设置 args.hub_always_push=True，则不执行当前推送操作
    if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
        return

    output_dir = self.args.output_dir
    # 为避免重新同步所有模型权重，从检查点文件夹中复制指定的模型文件到输出目录
    modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
    # 如果可用，添加适配器相关文件
    if is_peft_available():
        modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
    # 遍历需要复制的模型文件列表
    for modeling_file in modeling_files:
        # 如果文件存在于检查点文件夹中，则复制到输出目录
        if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
            shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
    # 如果存在 tokenizer 对象，则保存其当前状态到输出目录
    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)
    # 同样地，保存训练参数对象到输出目录
    torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # 根据保存策略生成提交消息
    if self.args.save_strategy == IntervalStrategy.STEPS:
        commit_message = f"Training in progress, step {self.state.global_step}"
    else:
        commit_message = f"Training in progress, epoch {int(self.state.epoch)}"

    # 上传整个输出目录到指定的模型库仓库，作为一个新版本的提交
    model_push_job = upload_folder(
        repo_id=self.hub_model_id,
        folder_path=output_dir,
        commit_message=commit_message,
        token=self.args.hub_token,
        run_as_future=True,
        ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
    )

    push_jobs = [model_push_job]

    # 如果指定了保存策略为 CHECKPOINT 或 ALL_CHECKPOINTS，则将检查点文件夹也上传到模型库仓库
    if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
        # 确定在仓库中的路径名称，如果策略是 CHECKPOINT 则使用 "last-checkpoint"，否则使用检查点文件夹的名称
        path_in_repo = (
            "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
        )
        # 创建一个上传任务，将检查点文件夹中的内容作为一个检查点版本提交
        checkpoint_push = upload_folder(
            repo_id=self.hub_model_id,
            folder_path=checkpoint_folder,
            path_in_repo=path_in_repo,
            commit_message=commit_message + ", checkpoint",
            token=self.args.hub_token,
            run_as_future=True,
        )
        push_jobs.append(checkpoint_push)

    # 如果当前没有进行中的推送任务或已完成的任务，创建新的推送任务
    if self.push_in_progress is None or self.push_in_progress.is_done():
        self.push_in_progress = PushInProgress(push_jobs)
    else:
        # 否则，将当前生成的推送任务添加到已有的推送任务列表中
        self.push_in_progress.jobs.extend(push_jobs)
    # 检查是否存在属性 "push_in_progress"，如果不存在则直接返回，不进行后续操作
    if not hasattr(self, "push_in_progress"):
        return
    
    # 检查当前推送操作是否正在进行，并且还未完成
    if self.push_in_progress is not None and not self.push_in_progress.is_done():
        # 记录日志，提示当前正在等待检查点推送操作完成，可能需要几分钟时间
        logger.info("Waiting for the current checkpoint push to be finished, this might take a couple of minutes.")
        
        # 等待推送操作完成，直到完成为止
        self.push_in_progress.wait_until_done()
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Upload `self.model` and `self.tokenizer` to the 🤗 model hub on the repo `self.args.hub_model_id`.

        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to [`~Trainer.create_model_card`].

        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.
        """
        model_name = kwargs.pop("model_name", None)
        # 如果未指定模型名称且应该保存模型
        if model_name is None and self.args.should_save:
            # 如果没有指定模型在模型输出目录中使用目录名称作为模型名称
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                # 否则使用 hub_model_id 中的最后一个部分作为模型名称
                model_name = self.args.hub_model_id.split("/")[-1]

        # 如果未初始化 hub_model_id，则初始化
        if self.hub_model_id is None:
            self.init_hf_repo()

        # 需要在所有进程上执行以支持 TPU 训练，但仅在 self.args.should_save 确定的进程上保存
        self.save_model(_internal_call=True)

        # 只在一个节点上执行推送操作
        if not self.is_world_process_zero():
            return

        # 如果模型已有标签并且用户传递了 "tags" 参数，则添加额外的标签以处理内部标签
        if getattr(self.model, "model_tags", None) is not None:
            if "tags" not in kwargs:
                kwargs["tags"] = []

            # 如果 tags 是字符串，转换为列表
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            # 将模型的每个标签添加到 kwargs["tags"] 中
            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        # 创建模型卡片
        self.create_model_card(model_name=model_name, **kwargs)

        # 等待当前上传完成
        self._finish_current_push()
        # 返回上传文件夹的结果
        return upload_folder(
            repo_id=self.hub_model_id,
            folder_path=self.args.output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
        )

    #
    # Deprecated code
    #
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Perform a prediction loop over the given data loader.

        Args:
            dataloader (DataLoader): The data loader containing the data to predict on.
            description (str): Description of the prediction loop.
            prediction_loss_only (Optional[bool], optional): Whether to calculate only prediction loss.
            ignore_keys (Optional[List[str]], optional): Keys to ignore during prediction.
            metric_key_prefix (str, optional): Prefix for metric keys.

        Returns:
            None
        """

    def _gather_and_numpify(self, tensors, name):
        """
        Gather values of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy arrays.

        Args:
            tensors: Tensor or list/tuple of nested tensors to gather and convert.
            name: Name associated with the gathering operation.

        Returns:
            numpy.ndarray or list/tuple/nested structure of numpy arrays corresponding to `tensors`.
        """
        if tensors is None:
            return

        # If using Torch XLA, perform mesh reduction
        if is_torch_xla_available():
            tensors = nested_xla_mesh_reduce(tensors, name)
        # If using SageMaker multi-processing, gather tensors
        elif is_sagemaker_mp_enabled():
            tensors = smp_gather(tensors)
        # If in distributed training mode, concatenate tensors
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            tensors = distributed_concat(tensors)

        # Convert gathered tensors to numpy arrays
        return nested_numpify(tensors)

    def _add_sm_patterns_to_gitignore(self) -> None:
        """
        Add SageMaker Checkpointing patterns to .gitignore file if running on the main process.

        Returns:
            None
        """
        # Ensure only the main process performs this operation
        if not self.is_world_process_zero():
            return

        # Patterns to be added to .gitignore
        patterns = ["*.sagemaker-uploading", "*.sagemaker-uploaded"]

        # Read current .gitignore content if it exists
        if os.path.exists(os.path.join(self.repo.local_dir, ".gitignore")):
            with open(os.path.join(self.repo.local_dir, ".gitignore"), "r") as f:
                current_content = f.read()
        else:
            current_content = ""

        # Prepare content to write, appending patterns if not already present
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if content.endswith("\n"):
                    content += pattern
                else:
                    content += f"\n{pattern}"

        # Write to .gitignore if there were changes
        if content != current_content:
            with open(os.path.join(self.repo.local_dir, ".gitignore"), "w") as f:
                logger.debug(f"Writing .gitignore file. Content: {content}")
                f.write(content)

        # Add .gitignore to the git index
        self.repo.git_add(".gitignore")

        # Ensure there's no race condition with git status
        time.sleep(0.5)

        # Commit changes if repository is not clean
        if not self.repo.is_repo_clean():
            self.repo.git_commit("Add *.sagemaker patterns to .gitignore.")
            self.repo.git_push()
    # 定义一个方法，用于创建加速器对象并进行后处理
    def create_accelerator_and_postprocess(self):
        # 设置梯度累积插件的参数字典
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        # 设置不与数据加载器同步
        grad_acc_kwargs["sync_with_dataloader"] = False
        # 创建梯度累积插件对象
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        # 创建加速器对象
        self.accelerator = Accelerator(
            deepspeed_plugin=self.args.deepspeed_plugin,  # 指定深度加速插件
            gradient_accumulation_plugin=gradient_accumulation_plugin,  # 指定梯度累积插件
            **self.args.accelerator_config.to_dict(),  # 使用加速器配置的参数
        )
        # 某些 Trainer 类需要使用 `gather` 而不是 `gather_for_metrics`，因此存储一个标志
        self.gather_function = self.accelerator.gather_for_metrics

        # 检查是否启用了 DeepSpeed 插件
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        # 检查是否启用了 FSDP 插件
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # 加速器创建后的设置
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            # 设置 FSDP 插件的所有 gather 限制
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            # 如果加速器版本兼容，则设置激活检查点功能
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                # 如果同时设置了激活检查点和梯度检查点，则抛出错误
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        # 如果启用了 DeepSpeed，并且未提供 hf_deepspeed_config 参数，则传播参数到 DeepSpeed
        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        # 如果设置了 `save_only_model` 并且同时使用了 DeepSpeed 或 FSDP 以及 `load_best_model_at_end`，则抛出错误
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # 如果使用了 DeepSpeed 或 FSDP，并且设置了 `auto_find_batch_size`，则抛出未实现错误
        if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.auto_find_batch_size:
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")
    # 将 Trainer 参数传播到 DeepSpeed 插件中
    def propagate_args_to_deepspeed(self, auto_find_batch_size=False):
        """
        Sets values in the deepspeed plugin based on the Trainer args
        根据 Trainer 参数设置 DeepSpeed 插件中的数值
        """
        # 导入 DeepSpeed 配置类
        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

        # 获取当前加速器状态中的 DeepSpeed 插件
        ds_plugin = self.accelerator.state.deepspeed_plugin

        # 使用 Trainer args 配置一个 HfTrainerDeepSpeedConfig 对象
        ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
        # 将 DeepSpeed 插件的配置设置为新创建的 HfTrainerDeepSpeedConfig 对象的配置
        ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
        # 根据 Trainer 参数进一步处理 DeepSpeed 配置
        ds_plugin.hf_ds_config.trainer_config_process(self.args, auto_find_batch_size)

    # 更新 FSDP 插件中的 QLoRa 相关设置
    def _fsdp_qlora_plugin_updates(self):
        """
        Updates the FSDP plugin with QLoRa related settings if applicable
        如果适用，更新 FSDP 插件的 QLoRa 相关设置
        """
        # 检查是否启用了 FSDP 并且模型是 PEFT 模型
        if self.is_fsdp_enabled and _is_peft_model(self.model):
            # 导入 PEFT 配置和 FSDP 自动包装策略工具
            from peft import LoraConfig
            from peft.utils.other import fsdp_auto_wrap_policy

            # 如果模型的 active_peft_config 是 LoraConfig 类型
            if isinstance(self.model.active_peft_config, LoraConfig):
                # 获取加速器状态中的 FSDP 插件
                fsdp_plugin = self.accelerator.state.fsdp_plugin
                # 设置 FSDP 插件的自动包装策略
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.model)

            # 如果模型的量化方法是 BITS_AND_BYTES，且量化配置是浮点数，并且加速器版本高于 "0.27.0"
            if (
                getattr(self.model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
                and self.model.hf_quantizer.quantization_config.bnb_4bit_quant_storage.is_floating_point
                and version.parse(accelerate_version) > version.parse("0.27.0")
            ):
                # 获取加速器状态中的 FSDP 插件并设置混合精度
                fsdp_plugin.set_mixed_precision(
                    self.model.hf_quantizer.quantization_config.bnb_4bit_quant_storage, override=True
                )
```