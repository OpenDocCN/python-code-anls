# `.\trainer_callback.py`

```
# coding=utf-8
# 定义编码格式为UTF-8

# Copyright 2020-present the HuggingFace Inc. team.
# 版权声明，指出代码的版权归HuggingFace Inc.团队所有，年份为2020至今

# Licensed under the Apache License, Version 2.0 (the "License");
# 使用Apache License 2.0许可协议，允许在符合条件下使用本代码

# you may not use this file except in compliance with the License.
# 除非遵守许可证规定，否则不得使用本文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# 除非法律要求或书面同意，否则不得使用软件

# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据许可证的规定，软件按"原样"提供，不提供任何形式的保证或条件，无论明示或暗示

# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解特定语言的许可和限制

"""
Callbacks to use with the Trainer class and customize the training loop.
"""
# 导入必要的模块和库
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# 导入NumPy库，用于数值计算
import numpy as np

# 导入tqdm库中的自动模块，用于显示进度条
from tqdm.auto import tqdm

# 导入本地模块
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging

# 获取logger对象用于日志记录
logger = logging.get_logger(__name__)

@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>
    """
    # TrainerState类，包含Trainer内部状态，当进行检查点保存时，将连同模型和优化器一起保存，并传递给TrainerCallback
    # 提示信息，解释在这个类中，一步（step）理解为一次更新步骤。当使用梯度累积时，一个更新步骤可能需要多次前向和反向传播。
    # 如果使用gradient_accumulation_steps=n，则一个更新步骤需要通过n批次数据进行。
    # 可选参数：训练时的当前 epoch（小数部分表示当前 epoch 完成的百分比）
    epoch: Optional[float] = None
    # 可选参数：已完成的全局训练步数
    global_step: int = 0
    # 可选参数：当前训练的总步数
    max_steps: int = 0
    # 可选参数：每隔多少步更新时记录日志
    logging_steps: int = 500
    # 可选参数：每隔多少步运行一次评估
    eval_steps: int = 500
    # 可选参数：每隔多少步保存一次检查点
    save_steps: int = 500
    # 可选参数：训练数据加载器的批次大小，仅在使用 `auto_find_batch_size` 时需要设置
    train_batch_size: int = None
    # 可选参数：训练过程中已见的输入标记数（训练时输入的标记数，而不是预测标记数）
    num_input_tokens_seen: int = 0
    # 可选参数：模型自训练开始以来执行的浮点操作总数（存储为浮点数以避免溢出）
    total_flos: float = 0
    # 可选参数：自训练开始以来已记录的日志列表
    log_history: List[Dict[str, float]] = None
    # 可选参数：迄今为止遇到的最佳指标值（用于跟踪最佳模型时）
    best_metric: Optional[float] = None
    # 可选参数：迄今为止遇到的最佳模型的检查点名称（用于跟踪最佳模型时）
    best_model_checkpoint: Optional[str] = None
    # 可选参数：当前进程是否为本地主进程（例如，在多机分布式训练中的一台机器上）
    is_local_process_zero: bool = True
    # 可选参数：当前进程是否为全局主进程（在多机分布式训练中，仅有一个进程会为全局主进程）
    is_world_process_zero: bool = True
    # 可选参数：是否处于超参数搜索过程中（影响 TensorBoard 中数据记录的方式）
    is_hyper_param_search: bool = False
    # 定义一个布尔类型的属性，表示是否为全局进程的第一个进程
    is_world_process_zero: bool = True
    # 定义一个布尔类型的属性，表示是否进行超参数搜索
    is_hyper_param_search: bool = False
    # 定义一个字符串类型的属性，表示试验名称，默认为 None
    trial_name: str = None
    # 定义一个字典类型的属性，表示试验参数，键为字符串，值可以是字符串、浮点数、整数或布尔值，默认为 None
    trial_params: Dict[str, Union[str, float, int, bool]] = None

    def __post_init__(self):
        # 如果实例化时没有指定 log_history 属性，则初始化为空列表
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """将实例内容以 JSON 格式保存到指定的 `json_path` 文件中。"""
        # 将实例转换为字典，然后转换为 JSON 格式的字符串，并写入到文件中
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """从指定的 `json_path` 文件加载内容并创建一个类实例。"""
        # 打开指定路径的 JSON 文件，读取内容，并转换为对象初始化参数
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 使用加载的 JSON 文本内容来实例化当前类并返回
        return cls(**json.loads(text))
@dataclass
class TrainerControl:
    """
    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        # 重置 `should_training_stop` 变量，以准备进行新的训练
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        # 重置 `should_epoch_stop` 变量，以准备进行新的 epoch
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        # 重置 `should_save`, `should_evaluate`, `should_log` 变量，以准备进行新的步骤
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    # no-format
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:
    """
    Args:
        args ([`TrainingArguments`]):
            用于实例化 [`Trainer`] 的训练参数。
        state ([`TrainerState`]):
            [`Trainer`] 的当前状态。
        control ([`TrainerControl`]):
            返回给 [`Trainer`] 的对象，用于做出某些决策。
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            正在训练的模型。
        tokenizer ([`PreTrainedTokenizer`]):
            用于对数据进行编码的分词器。
        optimizer (`torch.optim.Optimizer`):
            训练步骤中使用的优化器。
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            用于设置学习率的调度器。
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            用于训练的当前数据加载器。
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            用于评估的当前数据加载器。
        metrics (`Dict[str, float]`):
            上次评估阶段计算的指标。

            只能在 `on_evaluate` 事件中访问。
        logs  (`Dict[str, float]`):
            要记录的值。

            只能在 `on_log` 事件中访问。

    `control` 对象是唯一可以被回调函数更改的对象，在更改它的事件中应返回修改后的版本。

    参数 `args`, `state` 和 `control` 对于所有事件都是位置参数，其余的参数在 `kwargs` 中分组。
    您可以在事件的签名中解包需要使用的参数。例如，查看简单 [`~transformers.PrinterCallback`] 的代码示例。

    示例：

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```
    # 当一个 epoch 结束时调用的事件
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    # 当训练步骤开始时调用的事件。如果使用梯度累积，则一个训练步骤可能包含多个输入。
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    # 在梯度累积期间每个子步骤结束时调用的事件
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    # 当训练步骤结束时调用的事件。如果使用梯度累积，则一个训练步骤可能包含多个输入。
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    # 在评估阶段结束后调用的事件
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    # 在成功预测后调用的事件
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        pass

    # 在检查点保存后调用的事件
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    # 在记录最后日志后调用的事件
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    # 在预测步骤后调用的事件
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass
# 定义一个继承自 TrainerCallback 的内部类，用于按顺序调用一组回调函数。
class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        # 初始化函数，接收一组回调函数、模型、分词器、优化器和学习率调度器作为参数
        self.callbacks = []
        # 遍历传入的回调函数列表，逐个添加到 callbacks 列表中
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model  # 存储模型对象
        self.tokenizer = tokenizer  # 存储分词器对象
        self.optimizer = optimizer  # 存储优化器对象
        self.lr_scheduler = lr_scheduler  # 存储学习率调度器对象
        self.train_dataloader = None  # 初始化训练数据加载器为 None
        self.eval_dataloader = None  # 初始化评估数据加载器为 None

        # 如果 callbacks 列表中没有 DefaultFlowCallback 类型的回调函数，发出警告
        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        # 添加回调函数到 callbacks 列表中
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        # 如果已经存在相同类型的回调函数，则发出警告
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        # 从 callbacks 列表中弹出指定类型或实例的回调函数
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        # 从 callbacks 列表中移除指定类型或实例的回调函数
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        # 返回 callbacks 列表中每个回调函数的类名组成的字符串
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用所有回调函数的 on_init_end 方法，并返回结果
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 设置控制对象的 should_training_stop 属性为 False，调用所有回调函数的 on_train_begin 方法，并返回结果
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用所有回调函数的 on_train_end 方法，并返回结果
        return self.call_event("on_train_end", args, state, control)
    # 当每个 epoch 开始时调用的方法，设置控制参数，标志不停止当前 epoch
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_epoch_begin", args, state, control)

    # 当每个 epoch 结束时调用的方法
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_epoch_end", args, state, control)

    # 当每个训练步骤开始时调用的方法，设置控制参数，标志不记录日志、不评估、不保存模型
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_step_begin", args, state, control)

    # 当每个训练步骤的子步骤结束时调用的方法
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_substep_end", args, state, control)

    # 当每个训练步骤结束时调用的方法
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_step_end", args, state, control)

    # 当执行评估（evaluation）时调用的方法，设置控制参数，标志不执行评估
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    # 当执行预测（prediction）时调用的方法
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    # 当执行保存模型时调用的方法，设置控制参数，标志不执行保存
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_save", args, state, control)

    # 当记录日志时调用的方法，设置控制参数，标志不记录日志
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_log", args, state, control, logs=logs)

    # 当执行预测步骤时调用的方法
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，并返回其结果
        return self.call_event("on_prediction_step", args, state, control)

    # 调用指定事件的处理函数，并将参数传递给回调函数，返回最终的控制参数
    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            # 调用回调函数的指定事件，并传递相关参数和关键字参数
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # 如果回调函数返回的结果不为空，则更新控制参数
            if result is not None:
                control = result
        # 返回最终的控制参数
        return control
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    # 当每个训练步骤结束时触发的回调函数
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 日志记录
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        # 根据步骤间隔策略记录日志
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # 评估模型
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # 保存模型
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # 结束训练
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    # 当每个 epoch 结束时触发的回调函数
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 根据 epoch 间隔策略记录日志
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # 根据 epoch 间隔策略评估模型
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # 根据 epoch 间隔策略保存模型
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    # 在训练开始时触发的回调函数
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            # 如果是主进程，创建并显示训练进度条
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    # 当每个训练步骤结束时触发的回调函数
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            # 如果是主进程，更新训练进度条
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    # 在预测步骤时触发的回调函数
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                # 如果预测进度条尚未创建，创建并显示预测进度条
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            # 更新预测进度条
            self.prediction_bar.update(1)
    # 当世界进程号为零时执行以下代码（即主进程执行）
    def on_evaluate(self, args, state, control, **kwargs):
        # 检查是否为主进程
        if state.is_world_process_zero:
            # 如果预测进度条对象存在，则关闭它
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            # 将预测进度条对象设为None，表示已关闭
            self.prediction_bar = None

    # 当世界进程号为零时执行以下代码（即主进程执行）
    def on_predict(self, args, state, control, **kwargs):
        # 检查是否为主进程
        if state.is_world_process_zero:
            # 如果预测进度条对象存在，则关闭它
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            # 将预测进度条对象设为None，表示已关闭
            self.prediction_bar = None

    # 当世界进程号为零且训练进度条对象存在时执行以下代码（即主进程执行）
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 检查是否为主进程且训练进度条对象存在
        if state.is_world_process_zero and self.training_bar is not None:
            # 移除日志中的"total_flos"项（如果存在）
            _ = logs.pop("total_flos", None)
            # 将日志信息转换为字符串并写入训练进度条
            self.training_bar.write(str(logs))

    # 当世界进程号为零时执行以下代码（即主进程执行）
    def on_train_end(self, args, state, control, **kwargs):
        # 检查是否为主进程
        if state.is_world_process_zero:
            # 关闭训练进度条对象
            self.training_bar.close()
            # 将训练进度条对象设为None，表示已关闭
            self.training_bar = None
class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Pop the "total_flos" key from logs if it exists and assign to _
        _ = logs.pop("total_flos", None)
        # If the current process is the local process zero, print the logs
        if state.is_local_process_zero:
            print(logs)


class EarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        # Initialize with the provided early_stopping_patience and early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # Initialize early_stopping_patience_counter to track failed metric improvements
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # Determine the comparison operator based on whether greater_is_better is True or False
        operator = np.greater if args.greater_is_better else np.less
        # Check if metric_value is better than the current best_metric with respect to the operator and threshold
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            # Reset early_stopping_patience_counter if metric improves significantly
            self.early_stopping_patience_counter = 0
        else:
            # Increment early_stopping_patience_counter if metric does not improve
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        # Ensure that load_best_model_at_end is set to True for this callback
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        # Ensure that metric_for_best_model is defined to determine the best metric for early stopping
        assert args.metric_for_best_model is not None, "EarlyStoppingCallback requires metric_for_best_model is defined"
        # Ensure that evaluation_strategy is not set to NO, as this callback depends on evaluation intervals
        assert args.evaluation_strategy != IntervalStrategy.NO, "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
    # 定义一个方法 `on_evaluate`，用于评估模型的性能
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 获取用于最佳模型选择的指标名称
        metric_to_check = args.metric_for_best_model
        # 如果指标名称不以 "eval_" 开头，则添加前缀 "eval_"
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        # 从 metrics 字典中获取指定指标的值
        metric_value = metrics.get(metric_to_check)

        # 如果指标值为 None，则记录警告信息并禁用早停功能
        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        # 检查当前指标值是否符合早停条件
        self.check_metric_value(args, state, control, metric_value)
        # 如果早停计数器大于等于早停阈值，则设置控制器的训练停止标志为 True
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
```