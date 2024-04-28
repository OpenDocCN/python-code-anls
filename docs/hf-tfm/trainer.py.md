# `.\transformers\trainer.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，你可以在遵守许可证的情况下使用此文件
# 你可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于特定用途的适用性保证
# 请查看许可证以获取有关权限和限制的详细信息

"""
Trainer 类，用于轻松从头开始训练或在新任务上微调 🤗 Transformers 模型。
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

# 在导入 ML 框架之前必须导入集成:
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
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
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
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    # 调用 nested_xla_mesh_reduce 函数，执行某种 XLA 网格归约操作
    
    reissue_pt_warnings,
    # 调用 reissue_pt_warnings 函数，重新发出 PyTorch 的警告
    
    remove_dummy_checkpoint,
    # 调用 remove_dummy_checkpoint 函数，移除虚拟检查点
# 导入模块中的函数和变量
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
# 导入模块中的类和函数
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
# 导入模块中的变量和函数
from .utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
# 导入模块中的类
from .utils.quantization_config import QuantizationMethod

# 默认回调函数列表
DEFAULT_CALLBACKS = [DefaultFlowCallback]
# 默认进度回调函数
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# 如果在笔记本中
if is_in_notebook():
    # 导入笔记本进度回调函数
    from .utils.notebook import NotebookProgressCallback
    # 设置默认进度回调函数为笔记本进度回调函数
    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# 如果安装了 Apex 库
if is_apex_available():
    # 导入 Apex 库的 amp 模块
    from apex import amp

# 如果安装了 datasets 库
if is_datasets_available():
    # 导入 datasets 库
    import datasets

# 如果 Torch TPU 可用
if is_torch_tpu_available(check_device=False):
    # 导入 Torch TPU 模块
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

# 如果启用了 SageMaker MP
if is_sagemaker_mp_enabled():
    # 导入 SageMaker MP 库
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    # 判断是否为 SageMaker MP 1.10 之后的版本
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    # 导入训练器 PT 工具
    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# 如果安装了 SafeTensors 库
if is_safetensors_available():
    # 导入 SafeTensors 库
    import safetensors.torch

# 如果安装了 PEFT 库
if is_peft_available():
    # 导入 PEFT 模型
    from peft import PeftModel

# 如果安装了 Accelerate 库
if is_accelerate_available():
    # 导入 Accelerate 库的相关模块和函数
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )
    # 默认数据采样器列表
    DATA_SAMPLERS = [RandomSampler]
    # 如果 Accelerate 版本大于 0.23.0
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        # 导入 SeedableRandomSampler 类
        from accelerate.data_loader import SeedableRandomSampler
        # 添加 SeedableRandomSampler 到数据采样器列表
        DATA_SAMPLERS += [SeedableRandomSampler]
    # 如果 DeepSpeed 可用
    if is_deepspeed_available():
        # 从 accelerate.utils 中导入 DeepSpeedSchedulerWrapper 工具
        from accelerate.utils import DeepSpeedSchedulerWrapper
# 判断给定的模型是否是 PeftModel 类型，需要满足 PeftModel 可用且给定模型确实是 PeftModel 类型
def _is_peft_model(model):
    # 检查 PeftModel 是否可用，并且给定模型是否是 PeftModel 的实例
    return is_peft_available() and isinstance(model, PeftModel)


# 如果是类型检查阶段，导入 optuna 模块
if TYPE_CHECKING:
    import optuna


# 导入 logging 模块并获取 logger 对象
logger = logging.get_logger(__name__)


# 用于检查点保存的文件名
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


# Trainer 类定义，提供了一个简单但功能齐全的 PyTorch 训练和评估循环，针对 🤗 Transformers 进行了优化
class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    # Those are used as methods of the Trainer in examples.
    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    # Trainer 类的构造函数
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,  # 模型参数，默认为 None
        args: TrainingArguments = None,  # 训练参数，默认为 None
        data_collator: Optional[DataCollator] = None,  # 数据收集器，默认为 None
        train_dataset: Optional[Dataset] = None,  # 训练数据集，默认为 None
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,  # 评估数据集，默认为 None
        tokenizer: Optional[PreTrainedTokenizerBase] = None,  # 分词器，默认为 None
        model_init: Optional[Callable[[], PreTrainedModel]] = None,  # 模型初始化函数，默认为 None
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,  # 计算指标的函数，默认为 None
        callbacks: Optional[List[TrainerCallback]] = None,  # 回调函数列表，默认为 None
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  # 优化器元组，默认为 (None, None)
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,  # 为指标预处理逻辑的函数，默认为 None
    # 激活neftune方法，根据给定的模型
    def _activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper:
        https://arxiv.org/abs/2310.05914
        """
        # 解包模型
        unwrapped_model = unwrap_model(model)

        # 检查是否是peft模型，获取对应的嵌入层
        if _is_peft_model(unwrapped_model):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        # 删除解包后的模型
        del unwrapped_model

        # 设置neftune噪声alpha值
        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        # 注册前向钩子
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model

    # 关闭neftune方法
    def _deactivate_neftune(self, model):
        """
        Deactivates the neftune method. Make sure to call `_activate_neftune` first.
        """
        # 如果没有neftune钩子句柄，则抛出异常
        if not hasattr(self, "neftune_hook_handle"):
            raise ValueError("Neftune is not activated make sure to call `trainer._activate_neftune()` first")

        # 解包模型
        unwrapped_model = unwrap_model(model)

        # 检查是否是peft模型，获取对应的嵌入层
        if _is_peft_model(unwrapped_model):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        # 移除neftune钩子句柄
        self.neftune_hook_handle.remove()
        # 删除neftune噪声alpha值和解包后的模型
        del embeddings.neftune_noise_alpha, unwrapped_model

    # 添加回调函数到当前的回调列表中
    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback`]):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    # 从当前的回调列表中移除回调函数并返回
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
        return self.callback_handler.pop_callback(callback)
    # 从当前的 [`~transformers.TrainerCallback`] 列表中移除一个回调函数
    def remove_callback(self, callback):
        self.callback_handler.remove_callback(callback)

    # 将模型移动到指定设备上
    def _move_model_to_device(self, model, device):
        # 使用 model.to(device) 方法将模型移动到指定设备
        model = model.to(device)
        # 如果模型并行模式为 TPU，并且模型有 tie_weights 方法，则重新绑定权重
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    # 如果需要，设置模型的输入列标签
    def _set_signature_columns_if_needed(self):
        # 如果未设置模型的输入列标签
        if self._signature_columns is None:
            # 检查模型的前向函数签名，仅保留其接受的参数
            model_to_inspect = self.model
            if _is_peft_model(self.model):
                model_to_inspect = self.model.get_base_model()
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # 标签可能命名为 label 或 label_ids，默认的数据收集器会处理这个问题
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    # 移除数据集中未使用的列
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        # 如果未设置删除未使用列的参数，则直接返回数据集
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        # 找出未使用的列
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        # 从数据集中移除未使用的列
        columns = [k for k in signature_columns if k in dataset.column_names]
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    # 获取移除了指定列的数据收集器
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    # 定义一个函数，将数据收集器包装在一个可调用对象中，移除未使用的列
    def wrap_data_collator(self, data_collator) -> Callable:
        # 如果不需要移除未使用的列，则直接返回数据收集器
        if not self.args.remove_unused_columns:
            return data_collator
        # 如果需要移除未使用的列，则设置签名列（如果需要的话）
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        # 创建一个移除列的数据收集器
        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    # 获取训练数据采样器
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # 如果训练数据集为空或者没有长度信息，则返回空
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # 构建采样器
        if self.args.group_by_length:
            # 如果启用按长度分组，并且训练数据集是 datasets.Dataset 类型
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                # 获取长度信息
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            # 获取模型输入名称
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            # 返回一个按长度分组的采样器
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            # 返回一个随机采样器
            return RandomSampler(self.train_dataset)
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        # 检查是否存在训练数据集
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # 获取训练数据集和数据收集器
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # 如果支持 datasets 库且训练数据集是 datasets.Dataset 类型，则移除未使用的列
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            # 否则，获取移除未使用列后的数据收集器
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # 设置数据加载器参数
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # 如果训练数据集不是 torch.utils.data.IterableDataset 类型，则设置采样器和是否丢弃最后一个批次
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        # 准备数据加载器并返回
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        # Deprecated code
        # 如果使用旧的预测循环
        if self.args.use_legacy_prediction_loop:
            # 如果是在 Torch TPU 环境下
            if is_torch_tpu_available():
                return SequentialDistributedSampler(
                    eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            # 如果是在 SageMaker MP 环境下
            elif is_sagemaker_mp_enabled():
                return SequentialDistributedSampler(
                    eval_dataset,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            else:
                return SequentialSampler(eval_dataset)

        # 如果 world_size 小于等于 1，则返回顺序采样器；否则返回 None
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        # 检查是否提供了评估数据集，如果没有则抛出异常
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        # 如果提供了评估数据集，则使用提供的数据集，否则使用默认的self.eval_dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        # 如果datasets库可用且评估数据集是datasets.Dataset类型，则移除不被model.forward()方法接受的列
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            # 否则，使用_get_collator_with_removed_columns方法移除不需要的列
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        # 设置DataLoader的参数
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        # 如果评估数据集不是torch.utils.data.IterableDataset类型，则设置sampler和drop_last参数
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        # 使用加速器准备DataLoader并返回
        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        返回测试[`~torch.utils.data.DataLoader`]。

        如果您希望注入一些自定义行为，请子类化并重写此方法。

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                要使用的测试数据集。如果它是一个[`~datasets.Dataset`]，则会自动删除`model.forward()`方法不接受的列。它必须实现`__len__`。
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            # 如果可用，并且测试数据集是datasets.Dataset的实例，则删除未使用的列
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            # 否则，根据测试数据集创建数据收集器
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            # 如果测试数据集不是可迭代数据集，则设置采样器和是否丢弃最后一个批次
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        # 我们使用与评估相同的批量大小。
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        设置优化器和学习率调度器。

        我们提供了一个良好的默认值。如果您想使用其他内容，可以通过Trainer的init传递一个元组到`optimizers`，或者在子类中重写此方法（或`create_optimizer`和/或
        `create_scheduler`）。
        """
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # 如果Sagemaker版本大于等于1.10并且启用了fp16，则解开优化器
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        获取将应用权重衰减的所有参数名称

        注意，某些模型实现了自己的layernorm而不是调用nn.LayerNorm，因此这些模块仍然可能会应用权重衰减，因为此函数仅过滤出nn.LayerNorm的实例
        """
        # 获取将应用权重衰减的所有参数名称
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        # 过滤掉包含“bias”的参数名称
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # 如果启用了 SageMaker Model Parallelism，则使用包装后的模型，否则使用原始模型
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        # 如果未指定优化器，则根据模型参数设置默认优化器
        if self.optimizer is None:
            # 获取需要进行权重衰减的参数名
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # 分组模型参数，根据是否需要衰减将参数分为两组
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

            # 获取优化器类和初始化参数
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # 根据优化器类和参数初始化优化器
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            # 如果优化器是 Adam8bit，则进行特定的配置
            if optimizer_cls.__name__ == "Adam8bit":
                # 导入 bitsandbytes 模块
                import bitsandbytes

                # 获取全局优化管理器实例
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                # 初始化跳过参数计数器
                skipped = 0
                # 遍历模型的所有模块
                for module in opt_model.modules():
                    # 如果模块是 nn.Embedding 类型
                    if isinstance(module, nn.Embedding):
                        # 统计跳过的参数数量
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        # 记录跳过的参数信息
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        # 注册模块覆盖以进行优化
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        # 记录调试信息
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                # 记录总共跳过的参数数量
                logger.info(f"skipped: {skipped/2**20}M params")

        # 如果启用了 SageMaker Model Parallelism，则使用分布式优化器
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        # 返回优化器
        return self.optimizer

    @staticmethod
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        设置调度器。训练器的优化器必须在调用此方法之前设置好，或者作为参数传递。

        Args:
            num_training_steps (int): 要执行的训练步数。
        """
        # 如果调度器尚未设置，则根据参数设置调度器
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        # 返回设置好的调度器
        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        通过访问其数据集来获取 [`~torch.utils.data.DataLoader`] 中样本数量的辅助函数。当dataloader.dataset不存在或没有长度时，尽最大努力估计。
        """
        try:
            dataset = dataloader.dataset
            # 对于IterableDatasetShard的特殊情况，需要深入挖掘
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # 没有数据集或长度，通过dataloader的长度估计
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        通过枚举dataloader来获取 [`~torch.utils.data.DataLoader`] 中标记数量的辅助函数。
        """
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("无法从dataloader获取标记数量")
            return train_tokens
    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """HP search setup code"""
        # 设置试验参数
        self._trial = trial

        # 如果超参数搜索后端未指定或者试验为空，则返回
        if self.hp_search_backend is None or trial is None:
            return
        # 如果超参数搜索后端为 OPTUNA
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            # 从试验中获取超参数空间
            params = self.hp_space(trial)
        # 如果超参数搜索后端为 RAY
        elif self.hp_search_backend == HPSearchBackend.RAY:
            # 直接从试验中获取参数，同时移除 'wandb' 键
            params = trial
            params.pop("wandb", None)
        # 如果超参数搜索后端为 SIGOPT
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            # 将试验分配的参数转换成字典形式
            params = {k: int(v) if isinstance(v, str) else v for k, v in trial.assignments.items()}
        # 如果超参数搜索后端为 WANDB
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            # 直接使用试验参数
            params = trial

        # 遍历参数字典，将参数设置到 `TrainingArguments` 中
        for key, value in params.items():
            if not hasattr(self.args, key):
                # 若在 `TrainingArguments` 中不存在对应的属性，则发出警告
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in"
                    " `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            # 将参数值转换为正确的类型
            if old_attr is not None:
                value = type(old_attr)(value)

            setattr(self.args, key, value)
        
        # 打印相应的日志，显示超参数设置信息
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info(f"Trial: {trial.params}")
        if self.hp_search_backend == HPSearchBackend.SIGOPT:
            logger.info(f"SigOpt Assignments: {trial.assignments}")
        if self.hp_search_backend == HPSearchBackend.WANDB:
            logger.info(f"W&B Sweep parameters: {trial}")
        
        # 如果启用了 DeepSpeed，则重新构建 DeepSpeed 配置以反映更新的训练参数
        if self.is_deepspeed_enabled:
            if self.args.deepspeed is None:
                raise ValueError("For sweeps with deepspeed, `args.deepspeed` must be set")
            # 重新构建 DeepSpeed 配置
            from accelerate.utils import DeepSpeedPlugin
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            self.args.hf_deepspeed_config.trainer_config_process(self.args)
            self.args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.args.hf_deepspeed_config)

        # 创建加速器并进行后处理
        self.create_accelerator_and_postprocess()
    # 将训练指标报告给超参数搜索后端
    def _report_to_hp_search(self, trial: Union["optuna.Trial", Dict[str, Any]], step: int, metrics: Dict[str, float]):
        # 如果超参数搜索后端为空或试验为空，则返回
        if self.hp_search_backend is None or trial is None:
            return
        # 复制指标字典
        metrics = metrics.copy()
        # 计算目标值
        self.objective = self.compute_objective(metrics)
        # 如果使用的是 Optuna 超参数搜索后端
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna

            # 如果试验不是多目标
            if not trial.study._is_multi_objective():
                # 报告目标值
                trial.report(self.objective, step)
                # 如果试验应该剪枝
                if trial.should_prune():
                    # 在训练结束时调用回调处理程序
                    self.callback_handler.on_train_end(self.args, self.state, self.control)
                    # 抛出试验被剪枝的异常
                    raise optuna.TrialPruned()
        # 如果使用的是 Ray 超参数搜索后端
        elif self.hp_search_backend == HPSearchBackend.RAY:
            import ray.train

            # 使用临时目录
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                # 如果应该保存
                if self.control.should_save:
                    # 保存检查点
                    self._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                    checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                # 添加目标值到指标字典
                metrics["objective"] = self.objective
                # 报告指标
                ray.train.report(metrics, checkpoint=checkpoint)

    # 保存检查点
    def _tune_save_checkpoint(self, checkpoint_dir: str):
        # 设置输出目录
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        # 保存模型
        self.save_model(output_dir, _internal_call=True)
        # 如果应该保存
        if self.args.should_save:
            # 保存状态到 JSON 文件
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            # 保存优化器状态
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            # 保存学习率调度器状态
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    # 调用模型初始化函数
    def call_model_init(self, trial=None):
        # 获取模型初始化函数的参数数量
        model_init_argcount = number_of_arguments(self.model_init)
        # 如果参数数量为 0
        if model_init_argcount == 0:
            # 调用模型初始化函数
            model = self.model_init()
        # 如果参数数量为 1
        elif model_init_argcount == 1:
            # 使用试验参数调用模型初始化函数
            model = self.model_init(trial)
        else:
            # 抛出异常，模型初始化函数应该有 0 或 1 个参数
            raise RuntimeError("model_init should have 0 or 1 argument.")

        # 如果模型为空
        if model is None:
            # 抛出异常，模型初始化函数不应返回 None
            raise RuntimeError("model_init should not return None.")

        # 返回模型
        return model
    # 使用 Torch JIT 模式对模型进行评估
    def torch_jit_model_eval(self, model, dataloader, training=False):
        # 如果不是训练模式
        if not training:
            # 如果数据加载器为空
            if dataloader is None:
                logger.warning("failed to use PyTorch jit mode due to current dataloader is none.")
                return model
            # 获取一个示例批次数据并准备输入
            example_batch = next(iter(dataloader))
            example_batch = self._prepare_inputs(example_batch)
            try:
                # 复制模型并设置为评估模式
                jit_model = copy.copy(model)
                jit_model.eval()
                # 保存原始的前向传播函数
                original_forward = jit_model.__dict__.pop("_original_forward", None)
                # 从模型中移除混合精度钩子
                if original_forward:
                    jit_model.forward = original_forward
                # 禁用自动缓存加速器，关闭梯度计算
                with self.accelerator.autocast(cache_enabled=False), torch.no_grad():
                    # 根据 Torch 版本选择不同的 JIT 跟踪方式
                    if version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.0.0"):
                        if isinstance(example_batch, dict):
                            jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                        else:
                            jit_model = torch.jit.trace(
                                jit_model,
                                example_kwarg_inputs={key: example_batch[key] for key in example_batch},
                                strict=False,
                            )
                    else:
                        # 创建 JIT 输入
                        jit_inputs = []
                        for key in example_batch:
                            example_tensor = torch.ones_like(example_batch[key])
                            jit_inputs.append(example_tensor)
                        jit_inputs = tuple(jit_inputs)
                        jit_model = torch.jit.trace(jit_model, jit_inputs, strict=False)
                # 冻结 JIT 模型
                jit_model = torch.jit.freeze(jit_model)
                # 使用 JIT 模型进行推理
                with torch.no_grad():
                    jit_model(**example_batch)
                    jit_model(**example_batch)
                # 更新模型为 JIT 模型
                model = jit_model
                # 禁用 CPU 自动混合精度
                self.use_cpu_amp = False
            except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
                logger.warning(f"failed to use PyTorch jit mode due to: {e}.")

        return model
    # 对模型进行 IPEX 优化，支持在训练或推断时进行，可以指定数据类型
    def ipex_optimize_model(self, model, training=False, dtype=torch.float32):
        # 如果没有安装 IPEX，则抛出 ImportError
        if not is_ipex_available():
            raise ImportError(
                "Using IPEX but IPEX is not installed or IPEX's version does not match current PyTorch, please refer"
                " to https://github.com/intel/intel-extension-for-pytorch."
            )

        # 导入 IPEX 模块
        import intel_extension_for_pytorch as ipex

        # 如果不处于训练状态
        if not training:
            # 将模型设置为评估模式
            model.eval()
            # 如果不处于训练中且参数中指定了完全 BF16 评估，则将数据类型设置为 BF16
            dtype = torch.bfloat16 if not self.is_in_train and self.args.bf16_full_eval else dtype
            # 对模型进行优化，设置数据类型，优化级别为 O1，关闭卷积 BN 折叠功能，inplace 参数根据是否处于训练状态确定
            model = ipex.optimize(model, dtype=dtype, level="O1", conv_bn_folding=False, inplace=not self.is_in_train)
        else:
            # 如果模型不处于训练状态，则设置为训练模式
            if not model.training:
                model.train()
            # 对模型进行优化，设置数据类型，优化器为当前优化器，inplace 参数设置为 True，优化级别为 O1
            model, self.optimizer = ipex.optimize(
                model, dtype=dtype, optimizer=self.optimizer, inplace=True, level="O1"
            )

        # 返回优化后的模型
        return model

    # 训练函数，支持从检查点恢复、使用 Optuna 等参数优化工具、指定忽略评估的键等
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    # 内部训练循环函数，用于执行实际的训练过程
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # 获取输出目录函数，根据超参数搜索后端和试验对象确定输出目录
    def _get_output_dir(self, trial):
        # 如果启用了超参数搜索且存在试验对象
        if self.hp_search_backend is not None and trial is not None:
            # 如果超参数搜索后端为 OPTUNA，则使用试验编号作为运行编号
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            # 如果超参数搜索后端为 RAY，则使用 Ray 提供的运行上下文获取试验 ID
            elif self.hp_search_backend == HPSearchBackend.RAY:
                import ray.train

                run_id = ray.train.get_context().get_trial_id()
            # 如果超参数搜索后端为 SIGOPT，则使用试验 ID 作为运行编号
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            # 如果超参数搜索后端为 WANDB，则使用 WandB 提供的运行 ID
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            # 如果指定了超参数名称函数，则使用该函数生成运行名称，否则使用默认名称
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            # 拼接运行目录路径
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            # 否则直接使用指定的输出目录
            run_dir = self.args.output_dir
        # 返回最终的运行目录路径
        return run_dir

    # 加载模型后发出警告函数，用于检测加载模型时出现的键缺失或键不匹配情况并发出警告
    def _issue_warnings_after_load(self, load_result):
        # 如果加载结果中存在缺失的键
        if len(load_result.missing_keys) != 0:
            # 如果模型定义了在保存时忽略的键且缺失的键与之匹配，则尝试绑定权重
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                # 否则发出缺失键的警告
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        # 如果加载结果中存在不匹配的键
        if len(load_result.unexpected_keys) != 0:
            # 发出不匹配键的警告
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
    # 检查是否应该记录和评估模型，在全局步骤大于上次记录的全局步骤时执行
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # 如果是在 Torch TPU 上运行，标记步骤
            if is_torch_tpu_available():
                xm.mark_step()

            # 创建一个空字典用于存储日志信息
            logs: Dict[str, float] = {}

            # 使用 all_gather + mean() 获取所有进程的平均损失
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # 将损失重置为零
            tr_loss -= tr_loss

            # 计算并记录平均损失
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # 记录学习率
            logs["learning_rate"] = self._get_learning_rate()

            # 更新总损失
            self._total_loss_scalar += tr_loss_scalar
            # 更新上次记录的全局步骤
            self._globalstep_last_logged = self.state.global_step
            # 存储 FLOPs
            self.store_flos()

            # 记录日志
            self.log(logs)

        # 初始化评估指标为空
        metrics = None
        # 如果应该评估模型
        if self.control.should_evaluate:
            # 进行模型评估
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # 向超参数搜索报告结果
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # 如果使用 ReduceLROnPlateau 调度器，则现在运行延迟的 LR 调度器，因为此时指标已填充
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 获取用于最佳模型的指标名称
                metric_to_check = self.args.metric_for_best_model
                # 如果指标名称不是以 "eval_" 开头，则添加 "eval_" 前缀
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                # 根据指标更新学习率
                self.lr_scheduler.step(metrics[metric_to_check])

        # 如果应该保存模型
        if self.control.should_save:
            # 保存检查点
            self._save_checkpoint(model, trial, metrics=metrics)
            # 调用保存时的回调函数
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    # 加载检查点中的随机数生成器状态
    def _load_rng_state(self, checkpoint):
        # 如果检查点为空，则直接返回
        if checkpoint is None:
            return

        # 如果运行在多个进程中
        if self.args.world_size > 1:
            # 获取当前进程索引
            process_index = self.args.process_index
            # 构建当前进程的 RNG 文件路径
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            # 如果 RNG 文件不存在
            if not os.path.isfile(rng_file):
                # 提示未找到当前进程的 RNG 文件，并说明可能导致的不确定性
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                # 直接返回，不进行状态加载
                return
        # 如果只运行在单个进程中
        else:
            # 构建通用的 RNG 文件路径
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            # 如果 RNG 文件不存在
            if not os.path.isfile(rng_file):
                # 提示未找到 RNG 文件，并说明可能导致的不确定性
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                # 直接返回，不进行状态加载
                return

        # 加载检查点中保存的 RNG 状态
        checkpoint_rng_state = torch.load(rng_file)
        # 恢复 Python 内置的随机数生成器状态
        random.setstate(checkpoint_rng_state["python"])
        # 恢复 NumPy 随机数生成器状态
        np.random.set_state(checkpoint_rng_state["numpy"])
        # 恢复 PyTorch CPU 随机数生成器状态
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 如果当前是分布式并行模式
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # 恢复所有 GPU 的随机数生成器状态
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                # 尝试恢复单个 GPU 的随机数生成器状态
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                # 如果出现异常
                except Exception as e:
                    # 提示无法恢复 GPU 的 RNG 状态，并说明可能导致的不确定性
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
        # 如果是 Torch TPU 可用
        if is_torch_tpu_available():
            # 恢复 Torch XLA 的 RNG 状态
            xm.set_rng_state(checkpoint_rng_state["xla"])
        # 如果是 Torch NPU 可用
        if is_torch_npu_available():
            # 如果当前是分布式并行模式
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # 恢复所有 NPU 的 RNG 状态
                torch.npu.random.set_rng_state_all(checkpoint_rng_state["npu"])
            else:
                # 尝试恢复单个 NPU 的 RNG 状态
                try:
                    torch.npu.random.set_rng_state(checkpoint_rng_state["npu"])
                # 如果出现异常
                except Exception as e:
                    # 提示无法恢复 NPU 的 RNG 状态，并说明可能导致的不确定性
                    logger.info(
                        f"Didn't manage to set back the RNG states of the NPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )
    # 在非分布式训练中保存 RNG 状态
    def _save_rng_state(self, output_dir):
        # 创建包含不同 RNG 状态的字典
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 如果是分布式训练模式
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # 在非分布式情况下，保存全局 CUDA RNG 状态（会处理 DataParallel）
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # 如果是 Torch TPU 可用
        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # 如果是 Torch NPU 可用
        if is_torch_npu_available():
            # 如果是分布式训练模式
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["npu"] = torch.npu.random.get_rng_state_all()
            else:
                rng_states["npu"] = torch.npu.random.get_rng_state()

        # 一个进程可能在进程 0 有机会保存模型之前到达这里，此时 output_dir 可能还不存在
        # 创建目录，如果目录已存在则不报错
        os.makedirs(output_dir, exist_ok=True)

        # 如果进程数小于等于 1
        if self.args.world_size <= 1:
            # 保存 RNG 状态到文件 rng_state.pth
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            # 保存 RNG 状态到文件 rng_state_{self.args.process_index}.pth
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))
    # 定义保存优化器和学习率调度器的方法，将参数输出到指定目录
    def _save_optimizer_and_scheduler(self, output_dir):
        # 如果是在 Torch TPU 上可用
        if is_torch_tpu_available():
            # 使用 Torch XLA 等待进程，确保保存优化器状态
            xm.rendezvous("saving_optimizer_states")
            # 保存优化器状态字典到指定目录下的文件中
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            # 捕获可能的警告信息
            with warnings.catch_warnings(record=True) as caught_warnings:
                # 保存学习率调度器状态字典到指定目录下的文件中
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                # 重新触发捕获的警告信息
                reissue_pt_warnings(caught_warnings)
        # 如果启用了 SageMaker 模型并行训练
        elif is_sagemaker_mp_enabled():
            # 获取优化器的本地状态字典，不收集分片信息
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            # 在 SageMaker 模型并行训练中进行同步
            smp.barrier()
            # 如果是第一个进程或者配置中指定要分片优化器状态
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                # 部分地保存优化器状态字典到指定目录下的文件中
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        # 如果启用了 DeepSpeed
        elif self.is_deepspeed_enabled:
            # 在 zero3 模型中，模型文件本身不会保存，除非 DeepSpeed 配置中 `stage3_gather_16bit_weights_on_model_save` 为 True
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            # 如果接受排除冻结参数，并且模型是 PEFT 模型
            if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                # 保存检查点，排除冻结参数
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                # 保存检查点
                self.model_wrapped.save_checkpoint(output_dir)
        # 如果启用了 FSDP（Fully Sharded Data Parallelism）
        elif self.is_fsdp_enabled:
            # 保存 FSDP 特定的检查点以便从检查点中恢复
            save_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir)
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        # 如果需要保存检查点
        elif self.args.should_save:
            # 在上述条件不满足时，保存优化器状态字典到指定目录下的文件中
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # 保存学习率调度器和标量
        # 如果不是 DeepSpeed 或者不是 DeepSpeed 自定义调度器，并且不是在 Torch TPU 上可用
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            # 捕获可能的警告信息
            with warnings.catch_warnings(record=True) as caught_warnings:
                # 保存学习率调度器状态字典到指定目录下的文件中
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            # 重新触发捕获的警告信息
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
                A function defining the hyperparameter search space.
            compute_objective (Optional[Callable[[Dict[str, float]], float]]):
                A function to compute the objective value given hyperparameters.
            n_trials (int):
                Number of trials for hyperparameter search.
            direction (Union[str, List[str]]):
                Direction to optimize the objective, either 'minimize' or 'maximize'.
            backend (Optional[Union[str, HPSearchBackend]]):
                Backend for hyperparameter search.
            hp_name (Optional[Callable[["optuna.Trial"], str]]):
                A function to define the hyperparameter's name.

        **kwargs:
            Additional keyword arguments.

        """
        # 实现超参数搜索的函数
        if hp_space is None or compute_objective is None:
            raise ValueError("Both hp_space and compute_objective must be provided for hyperparameter search.")
        
        # 如果未指定优化方向，则默认为最小化
        if isinstance(direction, str):
            directions = [direction]
        else:
            directions = direction
        
        # 调用 Optuna 进行超参数搜索
        optuna_search(
            hp_space=hp_space,
            compute_objective=compute_objective,
            n_trials=n_trials,
            directions=directions,
            backend=backend,
            hp_name=hp_name,
            **kwargs,
        )

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # 如果当前处于某个 epoch，则将该 epoch 的值记录到日志中
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        # 如果设置了包含输入令牌数，则将其记录到日志中
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
        
        # 将全局步数记录到日志中
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        
        # 调用回调处理程序的 on_log 方法
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_input(self,  Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        # 如果数据是字典，则递归调用_prepare_input对其进行准备
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        # 如果数据是元组或列表，则递归调用_prepare_input对其中每个元素进行准备
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        # 如果数据是张量，则将其移到适当的设备上，并根据情况进行深度加速处理
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP 模型的输入是 int/uint，会调整为正确的嵌入 dtype。而其他模型（例如 wav2vec2）的输入已经是 float，
                # 因此可能需要特殊处理以匹配模型的 dtype
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        # 调用内部方法_prepare_input对输入进行准备处理
        inputs = self._prepare_input(inputs)
        # 如果输入为空，则抛出异常
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        # 如果模型设置了`past_index`且过去状态不为空，则将过去状态存入输入中的"mems"键
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        # 返回一个包含所需上下文管理器的辅助包装器
        return self.autocast_smart_context_manager()

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        # 如果使用 CPU AMP，则创建相应的上下文管理器
        if self.use_cpu_amp:
            ctx_manager = torch.cpu.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)
        # 如果未使用 CPU AMP，则创建一个空的上下文管理器
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
        # 将模型设置为训练模式
        model.train()
        # 准备输入数据
        inputs = self._prepare_inputs(inputs)

        # 如果使用 SageMaker 多进程训练，则调用相应的函数进行前向和后向传播
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # 使用上下文管理器计算损失
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # 如果使用多个 GPU，则对损失进行平均
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # 如果使用 Apex，则使用 amp 对损失进行缩放
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        # 否则，使用加速器对损失进行反向传播
        else:
            self.accelerator.backward(loss)

        # 返回经过处理的损失
        return loss.detach() / self.args.gradient_accumulation_steps
    # 计算损失函数的方法，由 Trainer 调用。默认情况下，所有模型都在第一个元素中返回损失值。
    # 子类可以重写此方法以实现自定义行为。
    def compute_loss(self, model, inputs, return_outputs=False):
        # 如果存在标签平滑器且输入中包含 "labels" 键
        if self.label_smoother is not None and "labels" in inputs:
            # 从输入中弹出 "labels" 键对应的值作为标签
            labels = inputs.pop("labels")
        else:
            labels = None
        # 使用模型处理输入数据，得到模型输出
        outputs = model(**inputs)
        # 如果存在过去状态，保存过去状态
        # TODO: 后续需要修复并优化此部分代码。
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # 如果存在标签，则根据模型类型选择标签平滑器处理损失
        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            # 如果模型输出是字典且不包含 "loss" 键，则抛出异常
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # 由于模型可能返回元组而不是 ModelOutput，因此这里不使用 .loss
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 如果需要返回模型输出，则返回损失值和模型输出；否则只返回损失值
        return (loss, outputs) if return_outputs else loss

    # 判断当前进程是否为本地主进程（例如，在多台机器上进行分布式训练时，本地主进程为 0 号进程）
    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    # 判断当前进程是否为全局主进程（在多台机器上进行分布式训练时，只有一个进程会返回 True）
    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # 对于 SageMaker ModelParallel，进程索引为 dp_process_index，而不是全局进程索引
        if is_sagemaker_mp_enabled():
            return smp.rank() == 0
        else:
            return self.args.process_index == 0
    # 定义保存模型的方法，允许指定输出目录，默认为 None，_internal_call 用于内部调用
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        # 如果输出目录为 None，则使用 self.args.output_dir
        if output_dir is None:
            output_dir = self.args.output_dir

        # 如果在 Torch TPU 可用的情况下，调用 _save_tpu 方法保存模型
        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        # 如果启用了 SageMaker Model Parallelism，则在所有进程上调用 state_dict 方法
        elif is_sagemaker_mp_enabled():
            # 调用 state_dict 需要在包装模型和所有进程上进行
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            # 如果应该保存，则保存模型
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            # 如果使用的 SageMaker Model Parallelism 版本大于 1.10，则创建一个 'user_content.pt' 文件作为标记
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' 表示使用 smp >= 1.10 保存的模型 state_dict
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        # 如果启用了 Fully Sharded Data Parallelism（FSDP），并且版本大于 0.24.1，则保存 FSDP 模型的状态字典
        elif self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        # 如果启用了 DeepSpeed，则保存 DeepSpeed 模型的状态字典
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            # 如果出现 ValueError，则表示 stage3_gather_16bit_weights_on_model_save=false，保存完整的检查点
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # 移除虚拟的状态字典
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                # 保存包装模型的检查点
                self.model_wrapped.save_checkpoint(output_dir)

        # 如果应该保存，则保存模型
        elif self.args.should_save:
            self._save(output_dir)

        # 当用户调用 save_model 时，推送到 Hub
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
    # 定义一个私有方法用于保存模型到TPU，可指定输出目录，默认为self.args.output_dir
    def _save_tpu(self, output_dir: Optional[str] = None):
        # 如果未指定输出目录，则使用self.args.output_dir作为输出目录
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # 打印提示信息，指示正在保存模型检查点到指定目录
        logger.info(f"Saving model checkpoint to {output_dir}")
        # 获取模型对象
        model = self.model
        # 将模型转移到CPU
        model.to("cpu")

        # 如果当前进程是主进程
        if xm.is_master_ordinal():
            # 创建输出目录，如果目录已存在则不会覆盖
            os.makedirs(output_dir, exist_ok=True)
            # 保存训练参数到输出目录下的TRAINING_ARGS_NAME文件中
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # 等待所有进程都到达此处，然后继续执行
        xm.rendezvous("saving_checkpoint")
        # 如果模型不是PreTrainedModel类型
        if not isinstance(model, PreTrainedModel):
            # 如果模型的包装模型是PreTrainedModel类型
            if isinstance(unwrap_model(model), PreTrainedModel):
                # 调用PreTrainedModel的save_pretrained方法保存模型和配置
                unwrap_model(model).save_pretrained(
                    output_dir,
                    is_main_process=self.args.should_save,
                    state_dict=model.state_dict(),
                    save_function=xm.save,
                )
            else:
                # 打印提示信息，说明Trainer.model不是PreTrainedModel类型，只保存其状态字典
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                # 获取模型的状态字典
                state_dict = model.state_dict()
                # 将状态字典保存到输出目录下的WEIGHTS_NAME文件中
                xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # 调用PreTrainedModel的save_pretrained方法保存模型和配置
            model.save_pretrained(output_dir, is_main_process=self.args.should_save, save_function=xm.save)
        # 如果存在分词器对象且应该保存，则保存分词器到输出目录
        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)

        # 将模型从CPU移回到设备上，以便后续计算可以正常进行
        model.to(self.args.device)
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 如果执行此函数，我们是进程零，所以不需要检查
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # 创建输出目录，如果不存在则创建
        os.makedirs(output_dir, exist_ok=True)
        # 记录日志，指示正在将模型检查点保存到output_dir目录
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 支持的类，如果没有安装 Peft 库，则仅支持 PreTrainedModel 类
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # 保存已训练的模型和配置使用 `save_pretrained()`。
        # 然后可以使用 `from_pretrained()` 重新加载它们
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                # 保存模型和状态字典到output_dir
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                # 如果Trainer.model不是`PreTrainedModel`，则只保存其状态字典
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    # 以安全的方式保存状态字典
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    # 保存状态字典
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # 保存模型和状态字典到output_dir
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        # 如果存在tokenizer，保存其配置到output_dir
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # 好的做法：将训练参数与训练的模型一起保存
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def store_flos(self):
        # 存储用于模型的浮点运算数量
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # 如果是分布式并行模式，将当前flos广播到所有设备上并计算总和
            self.state.total_flos += (
                distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
            )
            self.current_flos = 0
        else:
            # 否则直接将当前flos添加到总flos中
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    # 定义函数_sorted_checkpoints，返回按照时间或者文件名排序的模型检查点路径列表
    def _sorted_checkpoints(self, use_mtime=False, output_dir=None) -> List[str]:
        # 初始化排序后的检查点路径列表
        ordering_and_checkpoint_path = []

        # 获取输出目录下以指定前缀开头的所有文件夹的路径列表
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        # 遍历所有检查点路径
        for path in glob_checkpoints:
            # 如果使用修改时间排序
            if use_mtime:
                # 将修改时间和路径添加到排序列表中
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                # 否则，使用正则表达式匹配文件名中的数字部分作为排序依据
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                # 如果匹配成功
                if regex_match is not None and regex_match.groups() is not None:
                    # 将数字部分转换为整数，并将其与路径添加到排序列表中
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        # 对排序后的检查点路径列表进行排序
        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        # 提取排序后的检查点路径列表中的路径部分
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        
        # 确保不删除最佳模型
        if (
            self.state.best_model_checkpoint is not None
            and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            # 获取最佳模型在排序后的检查点路径列表中的索引
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            # 将最佳模型移动到排序后的检查点路径列表的倒数第二个位置
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        
        # 返回排序后的检查点路径列表
        return checkpoints_sorted

    # 定义_rotate_checkpoints函数，用于轮换和删除模型检查点
    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        # 如果未设置保存限制或保存限制小于等于0，则直接返回
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # 获取按照时间或者文件名排序的检查点路径列表
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        # 如果检查点数量不超过保存限制，则直接返回
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # 处理在最后一个检查点是最佳模型但不是唯一检查点的情况
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        # 计算需要删除的检查点数量
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        # 获取需要删除的检查点列表
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        # 遍历并删除需要删除的检查点
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    # 定义evaluate函数，用于评估模型性能
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    # 定义predict函数，用于模型推断
    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # 获取测试数据加载器
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # 根据使用的预测循环方法选择相应的循环函数
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        # 运行预测循环，获取输出结果
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        # 如果输出中包含即时编译时间，则更新开始时间
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        # 更新输出中的速度指标
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # 在预测之前调用回调函数
        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        # 停止内存追踪并更新指标
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # 返回预测结果
        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)
    # 定义评估循环函数，用于评估模型在给定数据集上的性能
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    # 内部函数，用于收集并汇总分布式环境中的张量数据
    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        # 如果张量为空，则直接返回
        if tensors is None:
            return
        # 如果是在 Torch TPU 环境中，进行张量的汇总操作
        if is_torch_tpu_available():
            # 如果未指定名称，则使用默认名称
            if name is None:
                name = "nested_gather"
            # 使用 XLA 网格减少函数对张量进行汇总
            tensors = nested_xla_mesh_reduce(tensors, name)
        # 如果是在 SageMaker 多进程环境中，使用 SageMaker 提供的汇总函数
        elif is_sagemaker_mp_enabled():
            tensors = smp_gather(tensors)
        # 如果是在分布式环境中，使用分布式环境提供的汇总函数
        elif (self.args.distributed_state is not None and self.args.distributed_state.distributed_type != "NO") or (
            self.args.distributed_state is None and self.args.local_rank != -1
        ):
            tensors = distributed_concat(tensors)
        # 返回汇总后的张量
        return tensors

    # 预测步骤函数，用于执行模型的前向推理
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    # 浮点运算函数，用于计算每个前向推理步骤的浮点运算数
    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        # 如果模型具有计算浮点运算数的方法，则调用该方法
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        # 否则返回0，表示没有浮点运算
        else:
            return 0

    # 初始化模型在 Hugging Face 仓库中的信息
    def init_hf_repo(self):
        """
        Initializes a git repo in `self.args.hub_model_id`.
        """
        # 只在主进程中执行以下代码
        if not self.is_world_process_zero():
            return

        # 如果未提供 Hugging Face 仓库模型 ID，则使用输出目录的名称作为仓库名称
        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id

        # 创建仓库，并获取仓库 URL
        repo_url = create_repo(repo_name, token=self.args.hub_token, private=self.args.hub_private_repo, exist_ok=True)
        # 设置模型在 Hugging Face 仓库中的 ID
        self.hub_model_id = repo_url.repo_id
        # 初始化推送状态
        self.push_in_progress = None
    # 创建模型卡片的方法
    def create_model_card(
        # 语言参数，可选
        self,
        language: Optional[str] = None,
        # 许可证参数，可选
        license: Optional[str] = None,
        # 标签参数，可选，可以是字符串或字符串列表
        tags: Union[str, List[str], None] = None,
        # 模型名称参数，可选
        model_name: Optional[str] = None,
        # 微调自参数，可选
        finetuned_from: Optional[str] = None,
        # 任务参数，可选，可以是字符串或字符串列表
        tasks: Union[str, List[str], None] = None,
        # 数据集标签参数，可选，可以是字符串或字符串列表
        dataset_tags: Union[str, List[str], None] = None,
        # 数据集参数，可选，可以是字符串或字符串列表
        dataset: Union[str, List[str], None] = None,
        # 数据集参数参数，可选，可以是字符串或字符串列表
        dataset_args: Union[str, List[str], None] = None,
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
        # 检查当前进程是否为主进程，如果不是则返回
        if not self.is_world_process_zero():
            return

        # 拼接模型卡片文件路径
        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        is_peft_library = False
        # 如果模型卡片文件存在
        if os.path.exists(model_card_filepath):
            # 加载模型卡片文件，获取库名称
            library_name = ModelCard.load(model_card_filepath).data.get("library_name")
            is_peft_library = library_name == "peft"

            # 追加现有的标签到 `tags`
            existing_tags = ModelCard.load(model_card_filepath).data.tags
            if tags is not None and existing_tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in existing_tags:
                    if tag not in tags:
                        tags.append(tag)

        # 从 Trainer 中获取训练摘要信息
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
        # 将训练摘要信息转换为模型卡片
        model_card = training_summary.to_model_card()
        # 将模型卡片写入文件
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

        # 如果是 peft 库，则创建或更新模型卡片
        if is_peft_library:
            unwrap_model(self.model).create_or_update_model_card(self.args.output_dir)
    # 从检查点文件夹中推送模型文件到指定的 Hub
    def _push_from_checkpoint(self, checkpoint_folder):
        # 只有一个节点执行推送操作
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # 如果上次推送还未完成，并且未设置 args.hub_always_push=True，则不执行此次推送
        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        output_dir = self.args.output_dir
        # 避免重新同步所有模型权重，直接从检查点文件夹复制文件
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        if is_peft_available():
            modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # 保存分词器很快，不确定可能生成多少文件，因此重新保存以确保
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # 保存训练参数同样很快，保存到指定路径
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.save_strategy == IntervalStrategy.STEPS:
            commit_message = f"Training in progress, step {self.state.global_step}"
        else:
            commit_message = f"Training in progress, epoch {int(self.state.epoch)}"

        # 上传模型文件夹到 Hub，并返回上传任务
        model_push_job = upload_folder(
            repo_id=self.hub_model_id,
            folder_path=output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=True,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
        )

        push_jobs = [model_push_job]

        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            path_in_repo = (
                "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
            )
            # 上传检查点文件夹到 Hub，并返回上传任务
            checkpoint_push = upload_folder(
                repo_id=self.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message + ", checkpoint",
                token=self.args.hub_token,
                run_as_future=True,
            )
            push_jobs.append(checkpoint_push)

        # 如果没有推送任务进行中或已完成，则创建新的推送任务
        if self.push_in_progress is None or self.push_in_progress.is_done():
            self.push_in_progress = PushInProgress(push_jobs)
        else:
            self.push_in_progress.jobs.extend(push_jobs)
    # 定义一个方法 `_finish_current_push`，用于完成当前的推送操作
    def _finish_current_push(self):
        # 如果对象没有属性 `push_in_progress`，则直接返回，不执行后续操作
        if not hasattr(self, "push_in_progress"):
            return
        # 如果当前推送操作不为空且尚未完成
        if self.push_in_progress is not None and not self.push_in_progress.is_done():
            # 记录日志，提示当前正在等待当前检查点推送完成，可能需要几分钟时间
            logger.info("Waiting for the current checkpoint push to be finished, this might take a couple of minutes.")
            # 等待当前推送操作完成
            self.push_in_progress.wait_until_done()
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        将 `self.model` 和 `self.tokenizer` 上传到 🤗 模型中心，存储在 `self.args.hub_model_id` 指定的仓库中。

        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                提交时的消息。
            blocking (`bool`, *optional*, defaults to `True`):
                是否在 `git push` 完成后返回。
            kwargs (`Dict[str, Any]`, *optional*):
                传递给 [`~Trainer.create_model_card`] 的额外关键字参数。

        Returns:
            如果 `blocking=False`，则返回模型上传的仓库 URL；如果 `blocking=True`，则返回跟踪提交进度的 `Future` 对象。
        """
        model_name = kwargs.pop("model_name", None)
        # 如果 `model_name` 未指定且应保存模型，则根据情况设定默认 `model_name`
        if model_name is None and self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]

        # 如果 `self.hub_model_id` 为空，则初始化模型中心的仓库
        if self.hub_model_id is None:
            self.init_hf_repo()

        # 在所有进程上执行以便于 TPU 训练，但只在由 `self.args.should_save` 决定的进程上保存
        self.save_model(_internal_call=True)

        # 只从一个节点推送
        if not self.is_world_process_zero():
            return

        # 如果模型已经具有某些标签且用户传递了 "tags" 参数给 `push_to_hub`，则自动处理所有模型的内部标签
        # 由于 Trainer 不调用 `model.push_to_hub`，所以需要添加额外的标签
        if "tags" in kwargs and getattr(self.model, "model_tags", None) is not None:
            # 如果 `tags` 是字符串，则转换为列表
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        # 创建模型卡
        self.create_model_card(model_name=model_name, **kwargs)

        # 等待当前上传完成
        self._finish_current_push()
        # 上传模型文件夹
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
```  
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction loop for generating model predictions on a dataset.

        Args:
            dataloader (DataLoader): DataLoader containing the dataset.
            description (str): Description of the prediction loop.
            prediction_loss_only (Optional[bool], optional): Whether to calculate only the prediction loss. Defaults to None.
            ignore_keys (Optional[List[str]], optional): List of keys to ignore when calculating predictions. Defaults to None.
            metric_key_prefix (str, optional): Prefix for metric keys. Defaults to "eval".
        """

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if is_torch_tpu_available():
            # If using TPU, perform a nested XLA mesh reduce operation on tensors
            tensors = nested_xla_mesh_reduce(tensors, name)
        elif is_sagemaker_mp_enabled():
            # If using SageMaker multi-processing, gather tensors
            tensors = smp_gather(tensors)
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            # If using distributed training, concatenate tensors
            tensors = distributed_concat(tensors)

        # Convert gathered tensors to numpy arrays
        return nested_numpify(tensors)

    def _add_sm_patterns_to_gitignore(self) -> None:
        """Add SageMaker Checkpointing patterns to .gitignore file."""
        # Ensure this function is only executed on the main process
        if not self.is_world_process_zero():
            return

        # Define SageMaker checkpointing patterns
        patterns = ["*.sagemaker-uploading", "*.sagemaker-uploaded"]

        # Get current content of .gitignore file
        if os.path.exists(os.path.join(self.repo.local_dir, ".gitignore")):
            with open(os.path.join(self.repo.local_dir, ".gitignore"), "r") as f:
                current_content = f.read()
        else:
            current_content = ""

        # Add SageMaker patterns to .gitignore if not already present
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if content.endswith("\n"):
                    content += pattern
                else:
                    content += f"\n{pattern}"

        # Write updated .gitignore file if changes were made
        if content != current_content:
            with open(os.path.join(self.repo.local_dir, ".gitignore"), "w") as f:
                logger.debug(f"Writing .gitignore file. Content: {content}")
                f.write(content)

        # Add .gitignore to git staging area
        self.repo.git_add(".gitignore")

        # Avoid race condition with git status
        time.sleep(0.5)

        # Commit changes to git repository if repository is not clean
        if not self.repo.is_repo_clean():
            self.repo.git_commit("Add *.sagemaker patterns to .gitignore.")
            self.repo.git_push()
    # 创建梯度累积插件的参数字典
    grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
    grad_acc_kwargs["sync_with_dataloader"] = False
    # 创建梯度累积插件对象
    gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

    # 创建加速器对象
    self.accelerator = Accelerator(
        dispatch_batches=self.args.dispatch_batches,
        split_batches=self.args.split_batches,
        deepspeed_plugin=self.args.deepspeed_plugin,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
    )
    # 一些 Trainer 类需要使用 `gather` 而不是 `gather_for_metrics`，因此存储一个标志
    self.gather_function = self.accelerator.gather_for_metrics

    # 检查是否启用了 deepspeed 和 accelerate，包括 Trainer 参数和 accelerate 启动器
    self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
    self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

    # 加速器创建后的设置
    if self.is_fsdp_enabled:
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
            "limit_all_gathers", fsdp_plugin.limit_all_gathers
        )
        if is_accelerate_available("0.23.0"):
            fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                "activation_checkpointing", fsdp_plugin.activation_checkpointing
            )
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

    # 如果启用了 deepspeed 并且没有 hf_deepspeed_config 参数，则将参数传播到 deepspeed
    if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
        self.propagate_args_to_deepspeed()

def propagate_args_to_deepspeed(self, auto_find_batch_size=False):
    """
    Sets values in the deepspeed plugin based on the Trainer args
    """
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    ds_plugin = self.accelerator.state.deepspeed_plugin

    # 设置 deepspeed 插件的值基于 Trainer 参数
    ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
    ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
    ds_plugin.hf_ds_config.trainer_config_process(self.args, auto_find_batch_size)
```