# `.\transformers\integrations\deepspeed.py`

```py
# 导入模块
import importlib.metadata as importlib_metadata
import importlib.util
# 弱引用，用于对象被垃圾回收时发出通知
import weakref
# 导入partialmethod，用于创建新的partial对象
from functools import partialmethod
# 从依赖版本检查中导入函数
from ..dependency_versions_check import dep_version_check
# 从工具模块中导入加速和torch是否可用的检查函数以及日志记录函数
from ..utils import is_accelerate_available, is_torch_available, logging

# 如果torch可用
if is_torch_available():
    # 导入torch模块
    import torch
    # 从优化模块中导入获取调度器函数
    from ..optimization import get_scheduler

# 获取日志记录器
logger = logging.get_logger(__name__)

# 检查deepspeed是否可用
def is_deepspeed_available():
    # 检查是否存在deepspeed包
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # 如果存在deepspeed包
    if package_exists:
        try:
            # 尝试获取deepspeed的元数据
            _ = importlib_metadata.metadata("deepspeed")
            # 返回True表示deepspeed可用
            return True
        except importlib_metadata.PackageNotFoundError:
            # 返回False表示deepspeed不可用
            return False

# 如果加速和deepspeed均可用
if is_accelerate_available() and is_deepspeed_available():
    # 从加速模块的deepspeed工具中导入HfDeepSpeedConfig类作为DeepSpeedConfig
    from accelerate.utils.deepspeed import HfDeepSpeedConfig as DeepSpeedConfig
else:
    # 如果加速不可用，则从内置模块中导入object作为DeepSpeedConfig
    # 这样可以确保Python成功导入此文件，即使加速不可用
    from builtins import object as DeepSpeedConfig

# 定义HfDeepSpeedConfig类，继承自DeepSpeedConfig
class HfDeepSpeedConfig(DeepSpeedConfig):
    """
    这个对象包含一个DeepSpeed配置字典，并且可以快速查询像零阶这样的东西。

    一个这个对象的`weakref`存储在模块的全局变量中，可以从没有Trainer对象可用的地方访问配置（例如`from_pretrained`和`_get_resized_embeddings`）。因此，
    在程序仍在运行时，这个对象保持存活是很重要的。

    [`Trainer`] 使用 `HfTrainerDeepSpeedConfig` 子类。该子类具有逻辑，通过替换特殊的占位符值：`"auto"`，将配置与 [`TrainingArguments`] 的值同步。
    如果没有这种特殊逻辑，DeepSpeed 配置不会以任何方式被修改。

    参数:
        config_file_or_dict (`Union[str, Dict]`): DeepSpeed配置文件路径或字典。
    """
    # 初始化函数，用于初始化对象
    def __init__(self, config_file_or_dict):
        # 设置全局弱引用对象，将当前对象作为参数传入
        set_hf_deepspeed_config(self)
        # 检查加速库 "accelerate" 的依赖版本
        dep_version_check("accelerate")
        # 检查深度加速库 "deepspeed" 的依赖版本
        dep_version_check("deepspeed")
        # 调用父类的初始化方法，传入配置文件或字典
        super().__init__(config_file_or_dict)
class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    `HfTrainerDeepSpeedConfig`对象在创建`TrainingArguments`对象时创建，具有相同的生命周期。
    """

    def __init__(self, config_file_or_dict):
        # 调用父类的构造函数
        super().__init__(config_file_or_dict)
        # 初始化_dtype为None
        self._dtype = None
        # 初始化mismatches列表为空
        self.mismatches = []

    def dtype(self):
        # 如果_dtype为None，则抛出异常
        if self._dtype is None:
            raise ValueError("trainer_config_process()尚未调用，无法获取dtype")
        return self._dtype

    def is_auto(self, ds_key_long):
        # 获取ds_key_long对应的值
        val = self.get_value(ds_key_long)
        # 如果值为None，则返回False
        if val is None:
            return False
        else:
            return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        一个实用方法，用于处理配置文件，并可选择验证值是否匹配。

        1. 用`TrainingArguments`的值替换"auto"值。

        2. 如果不是"auto"且`must_match`为True，则检查DS配置是否与Trainer配置值匹配，如果不匹配，则将条目添加到`self.mismatched`中 - 将在`trainer_config_finalize`期间断言一个或多个不匹配。
        """
        # 查找配置节点
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        # 如果配置中的值为"auto"，则将其替换为hf_val
        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        # 如果不需要匹配，则直接返回
        if not must_match:
            return

        # 获取配置中的值
        ds_val = config.get(ds_key)
        # 如果ds_val不为None且不等于hf_val，则将不匹配的信息添加到mismatches列表中
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    # fill_only方法是fill_match方法的一个部分应用，用于处理不需要匹配的情况
    fill_only = partialmethod(fill_match, must_match=False)
    # 定义一个方法，用于在获得模型和训练步数之后运行的配置最终化阶段
    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        # zero

        # 处理使用 `auto` 值并依赖模型的 hidden_size 的配置键
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        # 从 hidden_size_based_keys 中筛选出使用 auto 值的键
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]

        # 如果存在使用 auto 值的键
        if len(hidden_size_auto_keys) > 0:
            # 如果模型配置中有 hidden_size 属性
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            # 如果模型配置中有 hidden_sizes 属性
            elif hasattr(model.config, "hidden_sizes"):
                # 如果有多个 hidden_size，选择最大的一个
                hidden_size = max(model.config.hidden_sizes)
            else:
                # 如果模型配置文件既没有 `hidden_size` 也没有 `hidden_sizes` 条目，则抛出 ValueError
                raise ValueError(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    "`auto` values for these keys with an integer value of your choice."
                )

            # 填充 zero_optimization.reduce_bucket_size 的值为 hidden_size * hidden_size
            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            # 如果使用 Zero3 零优化
            if self.is_zero3():
                # 根据模型配置自动分配最优的配置值
                self.fill_only("zero_optimization.stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
                self.fill_only("zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)

        # scheduler
        # 填充 scheduler.params.total_num_steps 的值为 num_training_steps
        self.fill_match("scheduler.params.total_num_steps", num_training_steps, "num_training_steps (calculated)")
        # 填充 scheduler.params.warmup_num_steps 的值为 args.get_warmup_steps(num_training_steps)
        self.fill_match("scheduler.params.warmup_num_steps", args.get_warmup_steps(num_training_steps), "warmup_steps")

        # 如果存在不匹配的配置值
        if len(self.mismatches) > 0:
            # 将不匹配的配置值组成字符串
            mismatches = "\n".join(self.mismatches)
            # 抛出 ValueError，提醒用户校正不匹配的 DeepSpeed 配置值
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch TrainingArguments"
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )
# 将配置对象全局保持，以便在 TrainingArguments 生命周期中的任何地方访问它
_hf_deepspeed_config_weak_ref = None

def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    # 这是一个特殊的弱引用全局对象，允许我们从没有简单方法在 Trainer 领域之外获取到 Deepspeed 配置的 API 中获取到 Deepspeed 配置
    global _hf_deepspeed_config_weak_ref
    # 当 HfDeepSpeedConfig 被销毁时（当 TrainingArguments 被销毁时），将自动消失
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)

def unset_hf_deepspeed_config():
    # 有助于单元测试以确保全局状态不会泄漏 - 从 `tearDown` 方法中调用
    global _hf_deepspeed_config_weak_ref
    _hf_deepspeed_config_weak_ref = None

def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False

def deepspeed_config():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().config
    else:
        return None

def deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps, model_parameters):
    """
    一个方便的包装器，处理优化器和学习率调度器配置。
    """
    from accelerate.utils import DummyOptim, DummyScheduler

    config = hf_deepspeed_config.config

    # 混合和匹配 DS 调度器和优化器是支持的，除非启用了 Offload，情况如下：
    # 1. DS 调度器 + DS 优化器：是
    # 2. HF 调度器 + HF 优化器：大多数情况*
    # 3. DS 调度器 + HF 优化器：大多数情况*
    # 4. HF 调度器 + DS 优化器：是
    #
    # 大多数情况*：所有具有 CPU 和 GPU 实现的非原生 DeepSpeed 优化器应该可以工作（除了 LAMB）

    optimizer = None
    if "optimizer" in config:
        if args.adafactor:
            raise ValueError(
                "--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. "
                "Only one optimizer can be configured."
            )
        optimizer = DummyOptim(params=model_parameters)
    else:
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the"
                " custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        # ds 支持 Adam、OneBitAdam 和 Lamb 优化器，并且可以从 torch 导入其他优化器。
        # 但是 trainer 默认使用 AdamW。
        optimizer = trainer.create_optimizer()
        # 要使用其他优化器需要使用: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    # 如果配置中存在名为 "scheduler" 的项
    if "scheduler" in config:
        # 创建一个虚拟的调度器对象，用于模拟学习率的调度
        lr_scheduler = DummyScheduler(optimizer)
    else:
        # 如果优化器是 DummyOptim 的实例
        if isinstance(optimizer, DummyOptim):

            # 定义一个函数，该函数返回一个调度器对象
            def _lr_scheduler_callable(optimizer):
                # 调用 get_scheduler 函数获取特定类型的调度器
                return get_scheduler(
                    # 从训练器中获取学习率调度器类型
                    trainer.args.lr_scheduler_type,
                    optimizer=optimizer,
                    # 获取学习率的热身步数
                    num_warmup_steps=trainer.args.get_warmup_steps(num_training_steps),
                    # 获取训练步数
                    num_training_steps=num_training_steps,
                )

            # 创建一个虚拟的调度器对象，传入一个可调用的函数作为学习率调度器
            lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)
        else:
            # 如果优化器不是 DummyOptim 的实例，则通过训练器创建调度器对象
            lr_scheduler = trainer.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    # 返回优化器和对应的学习率调度器
    return optimizer, lr_scheduler
# 初始化 DeepSpeed，更新 DeepSpeed 配置并尝试从之前保存的检查点恢复
def deepspeed_init(trainer, num_training_steps, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler


# 加载 DeepSpeed 引擎的检查点
def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path):
    # it's possible that the user is trying to resume from model_path, which doesn't necessarily
    # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
    # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
    # path contains what looks like a deepspeed checkpoint
    import glob
    # 列出以 'checkpoint_path' 开头的全局步骤文件夹列表，并按名称排序
    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))
    
    # 如果存在深度学习训练引擎的检查点文件夹
    if len(deepspeed_checkpoint_dirs) > 0:
        # 输出日志信息，尝试从指定检查点路径恢复训练
        logger.info(f"Attempting to resume from {checkpoint_path}")
        # 使用深度学习训练引擎对象加载检查点，并同时加载优化器和学习率调度器的状态
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path, load_optimizer_states=True, load_lr_scheduler_states=True
        )
        # 如果加载路径为空，抛出值错误异常，说明加载失败
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
    # 如果没有找到有效的检查点路径
    else:
        # 抛出值错误异常，指示找不到有效的检查点路径
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")
```