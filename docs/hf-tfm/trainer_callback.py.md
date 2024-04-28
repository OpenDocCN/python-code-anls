# `.\transformers\trainer_callback.py`

```py
# 导入所需的库和模块
import dataclasses  # 用于定义数据类的装饰器
import json  # 用于 JSON 数据的编解码
from dataclasses import dataclass  # 导入 dataclass 装饰器
from typing import Dict, List, Optional, Union  # 导入类型提示所需的类型

import numpy as np  # 导入 NumPy 库
from tqdm.auto import tqdm  # 导入 tqdm 库中的自动模式

from .trainer_utils import IntervalStrategy, has_length  # 导入训练器工具中的一些函数和类
from .training_args import TrainingArguments  # 导入训练参数类
from .utils import logging  # 导入日志记录工具

# 获取或创建日志记录器对象
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
    # 一个包含 [`Trainer`] 内部状态的类，当进行检查点保存时，该状态将与模型和优化器一起保存，并传递给 [`TrainerCallback`]。

    # <Tip> 提示信息开始
    # 在这个类中，一个步骤被理解为一个更新步骤。当使用梯度累积时，一个更新步骤可能需要多个前向和后向传播步骤：
    # 如果您使用 `gradient_accumulation_steps=n`，那么一个更新步骤需要通过 *n* 个批次。
    # <Tip> 提示信息结束
    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            The number of tokens seen during training (number of input tokens, not the number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
        is_hyper_param_search (`bool`, *optional*, defaults to `False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
    """

    # Current epoch, representing the epoch the training is at (with the decimal part being the percentage of the current epoch completed)
    epoch: Optional[float] = None
    # Number of update steps completed during training
    global_step: int = 0
    # Number of update steps to do during the current training
    max_steps: int = 0
    # Log every X updates steps
    logging_steps: int = 500
    # Run an evaluation every X steps
    eval_steps: int = 500
    # Save checkpoint every X updates steps
    save_steps: int = 500
    # The batch size for the training dataloader (only needed when auto_find_batch_size has been used)
    train_batch_size: int = None
    # Total number of training epochs
    num_train_epochs: int = 0
    # Number of tokens seen during training (input tokens, not prediction tokens)
    num_input_tokens_seen: int = 0
    # Total number of floating operations done by the model since the beginning of training (stored as floats to avoid overflow)
    total_flos: float = 0
    # List of logs done since the beginning of training
    log_history: List[Dict[str, float]] = None
    # Value of the best metric encountered so far when tracking the best model
    best_metric: Optional[float] = None
    # Name of the checkpoint for the best model encountered so far when tracking the best model
    best_model_checkpoint: Optional[str] = None
    # Whether this process is the local main process (e.g., on one machine if training in a distributed fashion on several machines)
    is_local_process_zero: bool = True
    # 是否是全局进程的第一个进程，默认为True
    is_world_process_zero: bool = True
    # 是否进行超参数搜索，默认为False
    is_hyper_param_search: bool = False
    # 试验名称，默认为None
    trial_name: str = None
    # 试验参数，字典类型，键为字符串，值可以是字符串、浮点数、整数或布尔值，默认为None
    trial_params: Dict[str, Union[str, float, int, bool]] = None
    
    def __post_init__(self):
        # 如果日志历史为空，则初始化为一个空列表
        if self.log_history is None:
            self.log_history = []
    
    def save_to_json(self, json_path: str):
        """将该实例的内容以JSON格式保存到`json_path`中。"""
        # 将实例转换为字典形式，并转换成JSON字符串，缩进为2，按键排序
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        # 将JSON字符串写入指定路径的文件中
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
    
    @classmethod
    def load_from_json(cls, json_path: str):
        """从`json_path`中的内容创建一个实例。"""
        # 从指定路径的文件中读取内容
        with open(json_path, "r", encoding="utf-8") as f:
            # 将内容读取为文本
            text = f.read()
        # 使用读取的文本内容创建一个类实例，并返回
        return cls(**json.loads(text))
# 导入 `dataclass` 装饰器，用于定义数据类
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

    # 定义了一些控制训练流程的变量，默认值都为 `False`
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    # 内部方法，用于重置新的训练
    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    # 内部方法，用于重置新的轮次
    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    # 内部方法，用于重置新的步骤
    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        # 重置一些用于保存、评估和记录日志的变量为 `False`
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


# 定义了一个用于检查训练循环状态并做出决策的对象的类
class TrainerCallback:
    # no-format
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:
    Args:
        args ([`TrainingArguments`]):
            用于实例化[`Trainer`]的训练参数。
        state ([`TrainerState`]):
            [`Trainer`]的当前状态。
        control ([`TrainerControl`]):
            返回给[`Trainer`]的对象，可用于做出一些决策。
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            正在训练的模型。
        tokenizer ([`PreTrainedTokenizer`]):
            用于编码数据的分词器。
        optimizer (`torch.optim.Optimizer`):
            用于训练步骤的优化器。
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            用于设置学习率的调度器。
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            用于训练的当前数据加载器。
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            用于训练的当前数据加载器。
        metrics (`Dict[str, float]`):
            上次评估阶段计算的指标。

            这些只能在事件`on_evaluate`中访问。
        logs  (`Dict[str, float]`):
            要记录的值。

            这些只能在事件`on_log`中访问。

    `control`对象是唯一可以被回调函数更改的对象，在这种情况下更改它的事件应返回修改后的版本。

    参数`args`、`state`和`control`对于所有事件都是位置参数，所有其他参数都被分组在`kwargs`中。
    您可以在事件的签名中解包您需要的参数。例如，请参阅简单的[`~transformers.PrinterCallback`]的代码。

    示例：

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```py"""

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在[`Trainer`]初始化结束时调用的事件。
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在训练开始时调用的事件。
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在训练结束时调用的事件。
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在一个epoch开始时调用的事件。
        """
        pass
    # 在每个 epoch 结束时触发的事件
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    # 在每个训练步骤开始时触发的事件。如果使用梯度累积，一个训练步骤可能需要多个输入。
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    # 在梯度累积期间每个子步骤结束时触发的事件。
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    # 在每个训练步骤结束时触发的事件。如果使用梯度累积，一个训练步骤可能需要多个输入。
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    # 在评估阶段之后触发的事件。
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    # 在成功预测之后触发的事件。
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        pass

    # 在检查点保存之后触发的事件。
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    # 在记录最后日志之后触发的事件。
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    # 在预测步骤之后触发的事件。
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass
# 定义一个名为 CallbackHandler 的类，继承自 TrainerCallback
class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    # 类的初始化方法
    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        # 初始化 callbacks 列表为空
        self.callbacks = []
        # 遍历传入的 callbacks 列表
        for cb in callbacks:
            # 调用 add_callback 方法添加每个 callback 到 callbacks 列表中
            self.add_callback(cb)
        # 设置类的属性
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        # 如果 callbacks 列表中没有 DefaultFlowCallback 类型的 callback，则发出警告
        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    # 添加一个 callback 到 callbacks 列表中
    def add_callback(self, callback):
        # 如果传入的 callback 是一个类，则实例化一个对象
        cb = callback() if isinstance(callback, type) else callback
        # 获取 callback 的类
        cb_class = callback if isinstance(callback, type) else callback.__class__
        # 如果 callbacks 列表中已经有了该类的 callback，则发出警告
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        # 将 callback 添加到 callbacks 列表中
        self.callbacks.append(cb)

    # 从 callbacks 列表中移除一个 callback
    def pop_callback(self, callback):
        # 如果传入的 callback 是一个类
        if isinstance(callback, type):
            # 遍历 callbacks 列表
            for cb in self.callbacks:
                # 如果找到了同类型的 callback，则移除并返回
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        # 如果传入的 callback 是一个对象
        else:
            # 遍历 callbacks 列表
            for cb in self.callbacks:
                # 如果找到了相同的 callback，则移除并返回
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    # 从 callbacks 列表中移除一个 callback
    def remove_callback(self, callback):
        # 如果传入的 callback 是一个类
        if isinstance(callback, type):
            # 遍历 callbacks 列表
            for cb in self.callbacks:
                # 如果找到了相同类型的 callback，则移除并返回
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        # 如果传入的 callback 是一个对象
        else:
            # 直接从 callbacks 列表中移除该 callback
            self.callbacks.remove(callback)

    # 返回 callbacks 列表中所有 callback 的类名的字符串形式
    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    # 在初始化结束时触发事件
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    # 在训练开始时触发事件
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 设置控制器的 should_training_stop 属性为 False
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    # 在训练结束时触发事件
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)
    # 当每个 epoch 开始时调用的方法，设置控制器中的 should_epoch_stop 为 False，表示不应该停止 epoch
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_epoch_begin", args, state, control)

    # 当每个 epoch 结束时调用的方法
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_epoch_end", args, state, control)

    # 当每个步骤开始时调用的方法
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 设置控制器中的 should_log、should_evaluate 和 should_save 为 False
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_step_begin", args, state, control)

    # 当每个子步骤结束时调用的方法
    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_substep_end", args, state, control)

    # 当每个步骤结束时调用的方法
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_step_end", args, state, control)

    # 当评估时调用的方法
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        # 设置控制器中的 should_evaluate 为 False
        control.should_evaluate = False
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    # 当预测时调用的方法
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    # 当保存时调用的方法
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 设置控制器中的 should_save 为 False
        control.should_save = False
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_save", args, state, control)

    # 当记录日志时调用的方法
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        # 设置控制器中的 should_log 为 False
        control.should_log = False
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_log", args, state, control, logs=logs)

    # 当预测步骤时调用的方法
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        # 调用事件处理函数，传递参数并返回控制器
        return self.call_event("on_prediction_step", args, state, control)

    # 调用事件处理函数，传递参数并返回控制器
    def call_event(self, event, args, state, control, **kwargs):
        # 遍历所有回调函数，依次调用指定事件的处理方法
        for callback in self.callbacks:
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
            # 如果某个回调函数的处理方法返回结果不为空，则更新控制器
            if result is not None:
                control = result
        # 返回最终的控制器
        return control
class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 如果是第一步且启用了 logging_first_step，则应记录日志
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        # 如果 logging_strategy 为 IntervalStrategy.STEPS 并且当前步数是 logging_steps 的倍数，则应记录日志
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # 评估
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # 保存
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # 训练结束
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 记录日志
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # 评估
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # 保存
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

    def on_train_begin(self, args, state, control, **kwargs):
        # 如果是全局进程的主进程
        if state.is_world_process_zero:
            # 创建进度条
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        # 如果是全局进程的主进程
        if state.is_world_process_zero:
            # 更新训练进度条
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # 如果是全局进程的主进程，并且评估数据加载器有长度
        if state.is_world_process_zero and has_length(eval_dataloader):
            # 如果预测进度条不存在，则创建一个
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            # 更新预测进度条
            self.prediction_bar.update(1)
    # 在评估过程中调用的回调函数，接受参数 args, state, control 和任意关键字参数 kwargs
    def on_evaluate(self, args, state, control, **kwargs):
        # 如果当前进程是世界进程的第一个进程
        if state.is_world_process_zero:
            # 如果预测进度条对象存在，则关闭它
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            # 将预测进度条对象设为 None
            self.prediction_bar = None

    # 在预测过程中调用的回调函数，接受参数 args, state, control 和任意关键字参数 kwargs
    def on_predict(self, args, state, control, **kwargs):
        # 如果当前进程是世界进程的第一个进程
        if state.is_world_process_zero:
            # 如果预测进度条对象存在，则关闭它
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            # 将预测进度条对象设为 None
            self.prediction_bar = None

    # 在记录日志时调用的回调函数，接受参数 args, state, control, logs 和任意关键字参数 kwargs
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 如果当前进程是世界进程的第一个进程，并且训练进度条对象存在
        if state.is_world_process_zero and self.training_bar is not None:
            # 从日志中移除 "total_flos" 键，并将其赋值给变量 _
            _ = logs.pop("total_flos", None)
            # 将日志信息转换为字符串并写入训练进度条对象
            self.training_bar.write(str(logs))

    # 在训练结束时调用的回调函数，接受参数 args, state, control 和任意关键字参数 kwargs
    def on_train_end(self, args, state, control, **kwargs):
        # 如果当前进程是世界进程的第一个进程
        if state.is_world_process_zero:
            # 关闭训练进度条对象
            self.training_bar.close()
            # 将训练进度条对象设为 None
            self.training_bar = None
class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    # 在日志回调时打印日志
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 移除日志中的 "total_flos" 项
        _ = logs.pop("total_flos", None)
        # 如果当前进程是本地的第一个进程
        if state.is_local_process_zero:
            # 打印日志
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

    # 初始化早停回调对象
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        # 设置早停的耐心程度
        self.early_stopping_patience = early_stopping_patience
        # 设置早停的阈值
        self.early_stopping_threshold = early_stopping_threshold
        # 用于记录验证指标未改善的次数
        self.early_stopping_patience_counter = 0

    # 检查指标值是否满足早停条件
    def check_metric_value(self, args, state, control, metric_value):
        # 根据是否更大来确定运算符
        operator = np.greater if args.greater_is_better else np.less
        # 如果最佳指标为空或者当前指标比最佳指标好且改善超过阈值
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            # 重置早停耐心计数器
            self.early_stopping_patience_counter = 0
        else:
            # 增加早停耐心计数器
            self.early_stopping_patience_counter += 1

    # 在训练开始时执行的操作
    def on_train_begin(self, args, state, control, **kwargs):
        # 确保参数设置为在结束时加载最佳模型
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        # 确保最佳模型指标已定义
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        # 确保评估策略不为 NO
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"
    # 在评估模型时调用的方法，接受参数、状态、控制信息、指标等参数
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 获取用于最佳模型的指标名称
        metric_to_check = args.metric_for_best_model
        # 如果指标名称不以"eval_"开头，则添加前缀"eval_"
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        # 获取指标值
        metric_value = metrics.get(metric_to_check)

        # 如果指标值为None，则打印警告信息并禁用提前停止
        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        # 检查指标值是否满足停止条件
        self.check_metric_value(args, state, control, metric_value)
        # 如果提前停止的耐心计数器达到了指定的耐心值，则设置控制信息，停止训练
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
```