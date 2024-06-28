# `.\integrations\deepspeed.py`

```
"""
Integration with Deepspeed
"""
# 引入必要的模块和函数
import copy  # 导入深拷贝函数
import importlib.metadata as importlib_metadata  # 导入元数据模块
import importlib.util  # 导入模块加载工具
import weakref  # 导入弱引用模块
from functools import partialmethod  # 导入partialmethod函数

# 导入依赖版本检查和工具函数
from ..dependency_versions_check import dep_version_check  
from ..utils import is_accelerate_available, is_torch_available, logging  

# 如果torch可用，则导入torch模块
if is_torch_available():
    import torch  

# 获取日志记录器对象
logger = logging.get_logger(__name__)


# 检查是否存在DeepSpeed库
def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # 检查确保导入的是DeepSpeed库，而非其他内容，同时尝试获取其版本信息和作者信息进行验证
    if package_exists:
        try:
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


# 如果同时安装了accelerate和deepspeed，则导入DeepSpeedConfig类
if is_accelerate_available() and is_deepspeed_available():
    from accelerate.utils.deepspeed import HfDeepSpeedConfig as DeepSpeedConfig
else:
    # 如果accelerate不可用，则继承自dummy `object`，以确保可以导入本文件
    from builtins import object as DeepSpeedConfig


class HfDeepSpeedConfig(DeepSpeedConfig):
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """
    def __init__(self, config_file_or_dict):
        # 设置全局的弱引用对象
        set_hf_deepspeed_config(self)
        # 检查加速库 "accelerate" 的依赖版本
        dep_version_check("accelerate")
        # 检查深度加速库 "deepspeed" 的依赖版本
        dep_version_check("deepspeed")
        # 调用父类的初始化方法，传入配置文件或字典参数
        super().__init__(config_file_or_dict)
class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        # 调用父类的初始化方法，传入配置文件或字典
        super().__init__(config_file_or_dict)
        # 初始化私有变量 _dtype 为 None
        self._dtype = None
        # 初始化 mismatches 列表为空
        self.mismatches = []

    def dtype(self):
        # 如果 _dtype 为 None，则抛出数值错误异常
        if self._dtype is None:
            raise ValueError("trainer_config_process() wasn't called yet to tell dtype")
        # 返回 _dtype 的值
        return self._dtype

    def is_auto(self, ds_key_long):
        # 获取指定长键名 ds_key_long 对应的值
        val = self.get_value(ds_key_long)
        # 如果值为 None，则返回 False
        if val is None:
            return False
        else:
            # 否则返回值是否为 "auto" 的布尔结果
            return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.
        """
        # 查找指定长键名 ds_key_long 对应的配置节点和键名
        config, ds_key = self.find_config_node(ds_key_long)
        # 如果配置节点不存在，则直接返回
        if config is None:
            return

        # 如果配置值为 "auto"，则用 hf_val 替换它
        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        # 如果不需要匹配，则直接返回
        if not must_match:
            return

        # 否则，获取当前配置值和传入的 hf_val 进行比较
        ds_val = config.get(ds_key)
        # 如果值存在且与 hf_val 不匹配，则将不匹配信息添加到 self.mismatches 列表中
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    # 定义 fill_only 方法为 fill_match 的偏函数，关闭 must_match 参数
    fill_only = partialmethod(fill_match, must_match=False)
    # 处理 `auto` 值的配置键，并依赖于模型的隐藏大小
    hidden_size_based_keys = [
        "zero_optimization.reduce_bucket_size",
        "zero_optimization.stage3_prefetch_bucket_size",
        "zero_optimization.stage3_param_persistence_threshold",
    ]
    # 筛选出需要使用 `auto` 值的配置键列表
    hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]

    # 如果存在需要使用 `auto` 值的配置键
    if len(hidden_size_auto_keys) > 0:
        # 检查模型配置中是否有 `hidden_size` 属性
        if hasattr(model.config, "hidden_size"):
            hidden_size = model.config.hidden_size
        # 如果没有 `hidden_size` 属性，但有 `hidden_sizes` 属性，则选择最大的隐藏大小
        elif hasattr(model.config, "hidden_sizes"):
            hidden_size = max(model.config.hidden_sizes)
        else:
            # 如果模型配置文件既没有 `hidden_size` 也没有 `hidden_sizes` 条目，则引发错误
            raise ValueError(
                "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                "therefore it's not possible to automatically fill out the following `auto` entries "
                f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                "`auto` values for these keys with an integer value of your choice."
            )

        # 使用隐藏大小填充指定的配置键
        self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
        if self.is_zero3():
            # 如果是 Zero3 模式，根据模型配置自动分配优化配置值
            self.fill_only(
                "zero_optimization.stage3_prefetch_bucket_size",
                0.9 * hidden_size * hidden_size,
            )
            self.fill_only(
                "zero_optimization.stage3_param_persistence_threshold",
                10 * hidden_size,
            )

    # 填充调度器相关的参数值，匹配训练总步数和预热步数
    self.fill_match(
        "scheduler.params.total_num_steps",
        num_training_steps,
        "num_training_steps (calculated)",
    )
    self.fill_match(
        "scheduler.params.warmup_num_steps",
        args.get_warmup_steps(num_training_steps),
        "warmup_steps",
    )

    # 如果存在配置值不匹配的情况，引发 ValueError 异常
    if len(self.mismatches) > 0:
        mismatches = "\n".join(self.mismatches)
        raise ValueError(
            "Please correct the following DeepSpeed config values that mismatch TrainingArguments"
            f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
        )
# 将全局的 `_hf_deepspeed_config_weak_ref` 对象保持为全局状态，以便在 `TrainingArguments` 生命周期中的任何地方访问它。
_hf_deepspeed_config_weak_ref = None


def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    # 这是一个特殊的弱引用全局对象，允许我们从没有简单方式获取 Deepspeed 配置的 API 中获取 Deepspeed 配置。
    # 当 `HfDeepSpeedConfig` 销毁时（即 `TrainingArguments` 销毁时），它会自动消失。
    global _hf_deepspeed_config_weak_ref
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)


def unset_hf_deepspeed_config():
    # 有助于单元测试确保全局状态不会泄露 - 从 `tearDown` 方法中调用。
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
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    from accelerate.utils import DummyOptim, DummyScheduler

    config = hf_deepspeed_config.config

    # 如果在配置中发现了 `optimizer` 字段
    if "optimizer" in config:
        # 如果传递了 `--adafactor` 参数，则抛出值错误异常，因为 DeepSpeed 配置中只能配置一个优化器。
        if args.adafactor:
            raise ValueError(
                "--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. "
                "Only one optimizer can be configured."
            )
        # 创建一个虚拟的优化器对象 `DummyOptim`，用于占位模型参数
        optimizer = DummyOptim(params=model_parameters)
    else:
        # 如果 DeepSpeed 配置开启了 Offload
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the"
                " custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        # 默认情况下，trainer 使用 AdamW 优化器
        # 创建一个优化器对象，根据需要可以使用其它优化器，但会使 `zero_allow_untested_optimizer` 无效。
        optimizer = trainer.create_optimizer()
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    # 检查配置中是否存在 "scheduler" 键
    if "scheduler" in config:
        # 如果存在，则创建一个 DummyScheduler 的实例，使用给定的 optimizer
        lr_scheduler = DummyScheduler(optimizer)
    else:
        # 如果不存在 "scheduler" 键，则进入 else 分支
        # 检查 optimizer 是否是 DummyOptim 的实例
        if isinstance(optimizer, DummyOptim):

            # 定义一个内部函数 _lr_scheduler_callable，用于创建一个新的 lr_scheduler
            def _lr_scheduler_callable(optimizer):
                # 首先创建 trainer 的浅拷贝，以防后续修改影响原始的 trainer
                trainer_copy = copy.copy(trainer)
                # 在调用 _lr_scheduler_callable 时，trainer.lr_scheduler 已经被设置
                # 将其更新为 None，以便可以重新创建新的 scheduler
                trainer_copy.lr_scheduler = None
                # 使用 trainer_copy 创建一个新的 scheduler，并返回
                lr_scheduler = trainer_copy.create_scheduler(
                    num_training_steps=num_training_steps, optimizer=optimizer
                )
                return lr_scheduler

            # 创建一个 DummyScheduler 的实例，同时指定 lr_scheduler_callable 为 _lr_scheduler_callable 函数
            lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)
        else:
            # 如果 optimizer 不是 DummyOptim 的实例，则调用 trainer 的 create_scheduler 方法创建一个新的 scheduler
            lr_scheduler = trainer.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    # 返回 optimizer 和相应的 lr_scheduler
    return optimizer, lr_scheduler
# 初始化 DeepSpeed，根据 Trainer 的参数更新 DeepSpeed 配置。
# 如果指定了 resume_from_checkpoint，则尝试从先前保存的检查点恢复。
# Args:
#   trainer: Trainer 对象
#   num_training_steps: 每个单 GPU 的训练步数
#   inference: 是否启动推断模式（无优化器和学习率调度器）
# Returns:
#   optimizer, lr_scheduler：优化器和学习率调度器实例

def deepspeed_init(trainer, num_training_steps, inference=False):
    from deepspeed.utils import logger as ds_logger

    model = trainer.model  # 获取 Trainer 对象中的模型
    args = trainer.args  # 获取 Trainer 对象中的参数

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config  # 获取 DeepSpeed 插件的配置

    # 更新 DeepSpeed 配置的 trainer 部分，包括 args、model 和 num_training_steps
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # 设置 DeepSpeed 日志级别与 Trainer 一致
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # 推断模式下仅支持 ZeRO Stage 3
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference 只适用于 ZeRO Stage 3，请调整配置")

        # 清除 optimizer 和 lr_scheduler 配置，因为推断模式下不需要
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # 重要：在重新初始化时将 optimizer 设为 None
        # 获取所有需要梯度更新的模型参数
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # 使用 DeepSpeed 提供的优化器和学习率调度器函数初始化 optimizer 和 lr_scheduler
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # 保留以便快速调试：
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler


# 加载 DeepSpeed 引擎的检查点
# 检查指定的 checkpoint_path 是否包含 DeepSpeed 的检查点文件
# Args:
#   deepspeed_engine: DeepSpeed 引擎对象
#   checkpoint_path: 检查点路径
#   load_module_strict: 是否严格加载模块（默认为 True）
def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path, load_module_strict=True):
    # 用户可能试图从 model_path 恢复，该路径不一定包含 DeepSpeed 检查点，
    # 例如，示例只检查目录是否存在，并假定是恢复检查点而不是本地预训练权重。
    # 因此，这里我们检查路径是否包含类似 DeepSpeed 检查点的内容。
    import glob  # 导入 glob 模块，用于文件路径的通配符匹配
    
    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))
    # 使用 glob 模块匹配符合模式 `{checkpoint_path}/global_step*` 的文件路径，并按字母顺序排序后存储在列表中
    
    if len(deepspeed_checkpoint_dirs) > 0:
        logger.info(f"Attempting to resume from {checkpoint_path}")
        # 如果找到符合条件的检查点目录，则记录信息尝试从指定路径 {checkpoint_path} 恢复训练
    
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path,
            load_module_strict=load_module_strict,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        # 调用 deepspeed_engine 的 load_checkpoint 方法加载检查点文件，更新优化器和学习率调度器的状态
    
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
            # 如果加载路径为 None，则抛出值错误，指示未能从指定的检查点路径 {checkpoint_path} 恢复训练
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")
        # 如果未找到符合条件的检查点目录，则抛出值错误，指示在指定路径 {checkpoint_path} 下找不到有效的检查点
```