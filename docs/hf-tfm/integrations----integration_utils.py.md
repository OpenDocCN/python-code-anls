# `.\integrations\integration_utils.py`

```
"""
Integrations with other Python libraries.
"""
# functools 模块提供了创建和使用偏函数（partial function）的工具
import functools
# importlib.metadata 提供了从安装的包中读取元数据的功能
import importlib.metadata
# importlib.util 提供了高级导入支持
import importlib.util
# json 是 Python 的 JSON 编解码器
import json
# numbers 包含 Python 中的数字抽象基类
import numbers
# os 提供了与操作系统交互的功能
import os
# pickle 实现了基于 Python 对象的序列化和反序列化
import pickle
# shutil 提供了高级文件操作
import shutil
# sys 提供了访问与解释器交互的变量和函数
import sys
# tempfile 提供了生成临时文件和目录的功能
import tempfile
# dataclasses 提供了用于定义数据类的工具
from dataclasses import asdict, fields
# pathlib 提供了面向对象的文件系统路径操作
from pathlib import Path
# typing 提供了类型提示支持
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

# numpy 是科学计算的核心库
import numpy as np
# packaging.version 提供了版本管理功能
import packaging.version

# 导入本地模块中的 __version__，作为当前模块的版本号
from .. import __version__ as version
# 导入本地模块中的一些实用函数
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 如果 Torch 可用，则导入 Torch 模块
if is_torch_available():
    import torch

# 检查是否安装了 comet_ml，并且未禁用 COMET_MODE
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    try:
        import comet_ml  # noqa: F401

        # 检查是否设置了 COMET_API_KEY
        if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
            _has_comet = True
        else:
            # 如果未设置 COMET_API_KEY，则发出警告
            if os.getenv("COMET_MODE", "").upper() != "DISABLED":
                logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
            _has_comet = False
    except (ImportError, ValueError):
        _has_comet = False

# 检查是否安装了 neptune 或 neptune-client
_has_neptune = (
    importlib.util.find_spec("neptune") is not None or importlib.util.find_spec("neptune-client") is not None
)

# 如果是类型检查模式且安装了 Neptune，记录 Neptune 的版本信息
if TYPE_CHECKING and _has_neptune:
    try:
        _neptune_version = importlib.metadata.version("neptune")
        logger.info(f"Neptune version {_neptune_version} available.")
    except importlib.metadata.PackageNotFoundError:
        try:
            _neptune_version = importlib.metadata.version("neptune-client")
            logger.info(f"Neptune-client version {_neptune_version} available.")
        except importlib.metadata.PackageNotFoundError:
            _has_neptune = False

# 导入本地模块中的一些回调函数和实用类
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_xla_available  # noqa: E402

# Integration functions:

def is_wandb_available():
    # 检查是否安装了 WandB，任何非空值的 WANDB_DISABLED 变量都将禁用 WandB
    # 检查环境变量中是否定义了"WANDB_DISABLED"且其值在指定的真值列表中
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        # 如果条件成立，输出警告日志，提醒使用者停止使用"WANDB_DISABLED"环境变量，因为它将在v5版本中被移除，并建议使用"--report_to"标志来控制日志结果的集成方式（例如使用"--report_to none"来禁用集成）。
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        # 返回False，表示WANDB（Weights and Biases）被禁用
        return False
    # 检查是否能够找到"wandb"模块的规范（spec）
    return importlib.util.find_spec("wandb") is not None
# 检查是否安装了 ClearML（formerly Trains），返回 True 或 False
def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


# 检查是否安装了 Comet，返回 _has_comet 的值
def is_comet_available():
    return _has_comet


# 检查是否安装了 TensorBoard 或 TensorBoardX，返回 True 或 False
def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


# 检查是否安装了 Optuna，返回 True 或 False
def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


# 检查是否安装了 Ray，返回 True 或 False
def is_ray_available():
    return importlib.util.find_spec("ray") is not None


# 检查是否安装了 Ray Tune，返回 True 或 False
def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


# 检查是否安装了 SigOpt，返回 True 或 False
def is_sigopt_available():
    return importlib.util.find_spec("sigopt") is not None


# 检查是否安装了 Azure ML，返回 True 或 False
def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


# 检查是否启用了 MLflow 并安装了相关依赖，返回 True 或 False
def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


# 检查是否同时安装了 Dagshub 和 MLflow，返回 True 或 False
def is_dagshub_available():
    return None not in [importlib.util.find_spec("dagshub"), importlib.util.find_spec("mlflow")]


# 检查是否安装了 Neptune，返回 _has_neptune 的值
def is_neptune_available():
    return _has_neptune


# 检查是否安装了 CodeCarbon，返回 True 或 False
def is_codecarbon_available():
    return importlib.util.find_spec("codecarbon") is not None


# 检查是否安装了 Flytekit，返回 True 或 False
def is_flytekit_available():
    return importlib.util.find_spec("flytekit") is not None


# 检查是否安装了 Flyte Deck Standard，返回 True 或 False
def is_flyte_deck_standard_available():
    if not is_flytekit_available():
        return False
    return importlib.util.find_spec("flytekitplugins.deck") is not None


# 检查是否安装了 DVC Live，返回 True 或 False
def is_dvclive_available():
    return importlib.util.find_spec("dvclive") is not None


# 根据提供的试验对象（trial）返回超参数字典，可能从 Optuna、Ray Tune、SigOpt 或 W&B 中获取
def hp_params(trial):
    if is_optuna_available():
        import optuna
        if isinstance(trial, optuna.Trial):
            return trial.params
    if is_ray_tune_available():
        if isinstance(trial, dict):
            return trial
    if is_sigopt_available():
        if isinstance(trial, dict):
            return trial
    if is_wandb_available():  # Assuming is_wandb_available function is defined elsewhere
        if isinstance(trial, dict):
            return trial
    raise RuntimeError(f"Unknown type for trial {trial.__class__}")


# 使用 Optuna 进行超参数搜索，返回 BestRun 对象
def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna
    # 检查当前进程索引是否为0，只有主进程执行以下逻辑
    if trainer.args.process_index == 0:

        # 定义内部函数 _objective，用于定义优化目标函数
        def _objective(trial, checkpoint_dir=None):
            # 初始化 checkpoint 为 None
            checkpoint = None
            # 如果提供了 checkpoint_dir，则查找其中以 PREFIX_CHECKPOINT_DIR 开头的子目录
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            
            # 将 trainer 的 objective 属性设为 None
            trainer.objective = None
            
            # 如果运行环境的 world_size 大于1，则进入分布式训练模式
            if trainer.args.world_size > 1:
                # 检查并确保当前并行模式为 ParallelMode.DISTRIBUTED
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
                
                # 初始化分布式超参搜索
                trainer._hp_search_setup(trial)
                
                # 使用 torch.distributed 广播序列化后的 trainer.args 到所有进程
                torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                
                # 开始训练，从 checkpoint 恢复
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                # 单机模式下，开始训练，从 checkpoint 恢复，传入 trial 对象
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            
            # 如果在训练过程中没有进行评估，则执行评估过程
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
            
            # 返回训练的目标值
            return trainer.objective

        # 从 kwargs 中弹出 timeout 和 n_jobs 参数，分别设置默认值为 None 和 1
        timeout = kwargs.pop("timeout", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        
        # 如果 direction 是 list 类型，则设置 directions 为 direction，否则为 None
        directions = direction if isinstance(direction, list) else None
        direction = None if directions is not None else direction
        
        # 创建一个新的 Optuna Study 对象 study，根据参数设置方向和其他参数
        study = optuna.create_study(direction=direction, directions=directions, **kwargs)
        
        # 使用 study 对象优化 _objective 函数，执行 n_trials 次优化，支持并行数为 n_jobs
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        
        # 如果 study 不是多目标优化，则返回最佳运行结果
        if not study._is_multi_objective():
            best_trial = study.best_trial
            return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
        else:
            # 如果是多目标优化，则返回多个最佳运行结果列表
            best_trials = study.best_trials
            return [BestRun(str(best.number), best.values, best.params) for best in best_trials]
    
    else:
        # 对于非主进程，执行以下逻辑，循环 n_trials 次
        for i in range(n_trials):
            # 将 trainer 的 objective 属性设为 None
            trainer.objective = None
            
            # 序列化 trainer.args 到 args_main_rank 列表
            args_main_rank = list(pickle.dumps(trainer.args))
            
            # 检查并确保当前并行模式为 ParallelMode.DISTRIBUTED
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
            
            # 使用 torch.distributed 广播 args_main_rank 到所有进程
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            
            # 将 args_main_rank 反序列化为 args 对象
            args = pickle.loads(bytes(args_main_rank))
            
            # 遍历 args 的属性，将除了 "local_rank" 外的键值对设置为 trainer.args 的属性
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            
            # 开始训练，从 checkpoint=None 恢复
            trainer.train(resume_from_checkpoint=None)
            
            # 如果在训练过程中没有进行评估，则执行评估过程
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        
        # 非主进程返回 None
        return None
def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    # 导入必要的库
    import ray
    import ray.train

    def _objective(trial: dict, local_trainer):
        try:
            # 尝试导入笔记本进度回调类
            from transformers.utils.notebook import NotebookProgressCallback

            # 如果存在 NotebookProgressCallback，则从 local_trainer 中移除并添加 ProgressCallback
            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            # 如果模块未找到，则忽略
            pass

        # 将 local_trainer 的 objective 属性设置为 None
        local_trainer.objective = None

        # 获取 ray.train 的检查点对象
        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            # 如果有检查点，说明是恢复训练状态
            # 重置 local_trainer 的 objective 属性为 "objective"，以解决训练完成后额外触发检查点的问题
            local_trainer.objective = "objective"

            # 获取检查点目录下的第一个检查点路径
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = next(Path(checkpoint_dir).glob(f"{PREFIX_CHECKPOINT_DIR}*")).as_posix()
                # 从检查点路径恢复训练
                local_trainer.train(resume_from_checkpoint=checkpoint_path, trial=trial)
        else:
            # 如果没有检查点，则直接开始训练
            local_trainer.train(trial=trial)

        # 如果训练过程中未进行评估
        if getattr(local_trainer, "objective", None) is None:
            # 进行评估，并计算目标指标
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)

            # 更新 metrics，并标记为完成
            metrics.update({"objective": local_trainer.objective, "done": True})

            # 使用临时目录保存检查点，并创建 ray.train.Checkpoint 对象
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                local_trainer._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                # 报告评估结果和检查点
                ray.train.report(metrics, checkpoint=checkpoint)

    # 如果 trainer 的内存追踪未跳过内存指标的记录，则警告并设置为跳过
    if not trainer._memory_tracker.skip_memory_metrics:
        from ..trainer_utils import TrainerMemoryTracker

        logger.warning(
            "Memory tracking for your Trainer is currently "
            "enabled. Automatically disabling the memory tracker "
            "since the memory tracker is not serializable."
        )
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)

    # 在进行 ray 超参数搜索期间，模型和 TensorBoard writer 无法序列化，因此需要移除它们
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None

    # 设置默认的 `resources_per_trial`。
    # 检查是否在 `kwargs` 参数中存在 `resources_per_trial` 键
    if "resources_per_trial" not in kwargs:
        # 如果不存在，则设置默认值为每个试验分配 1 个 CPU 和（如果可用）1 个 GPU
        kwargs["resources_per_trial"] = {"cpu": 1}
        # 如果训练器有 GPU，则将 GPU 数量设置为 1
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        # 生成资源信息字符串，用于日志记录
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        # 记录日志，说明未传递 `resources_per_trial` 参数，使用默认值
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )

    # 确保每个训练器实例只使用根据试验分配的 GPU
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # 设置默认的进度报告器 `progress_reporter`
    if "progress_reporter" not in kwargs:
        # 导入所需的 CLIReporter 类
        from ray.tune import CLIReporter
        # 如果未指定 `progress_reporter`，则设置为 CLIReporter，并指定度量列为 "objective"
        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])

    # 如果 `kwargs` 中包含 `scheduler` 参数
    if "scheduler" in kwargs:
        # 导入可能需要中间报告的调度器类
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # 检查调度器是否需要中间报告，并且检查是否开启了评估
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            # 抛出运行时错误，要求开启评估以便调度器能够使用中间结果
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    # 使用 `ray.tune.with_parameters` 将 `_objective` 和本地训练器 `trainer` 组合成可调用的函数
    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    # 使用 `functools.wraps` 装饰 `trainable` 函数，以保留其元数据
    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        # 检查是否存在 datasets 模块
        if is_datasets_available():
            # 导入 datasets.load 模块
            import datasets.load

            # 初始化动态模块的路径
            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), "__init__.py")
            # 从路径加载动态模块
            spec = importlib.util.spec_from_file_location("datasets_modules", dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            # 将加载的模块添加到系统模块列表中
            sys.modules[spec.name] = datasets_modules
            # 执行加载的模块
            spec.loader.exec_module(datasets_modules)
        
        # 返回通过 tune.with_parameters 调用的 trainable 函数的结果
        return trainable(*args, **kwargs)

    # 检查 trainable 函数是否具有特殊属性 __mixins__
    if hasattr(trainable, "__mixins__"):
        # 如果有，将 dynamic_modules_import_trainable 函数的 __mixins__ 属性设置为 trainable 函数的 __mixins__
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__

    # 运行 ray.tune 的分布式调参任务
    analysis = ray.tune.run(
        dynamic_modules_import_trainable,
        config=trainer.hp_space(None),  # 使用 trainer.hp_space(None) 获取超参数空间配置
        num_samples=n_trials,  # 设置试验的样本数为 n_trials
        **kwargs,  # 其他传递的关键字参数
    )
    
    # 获取最佳试验的信息，基于指定的度量标准和方向
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3], scope=trainer.args.ray_scope)
    
    # 构造最佳运行的对象，并传递相关信息
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config, analysis)
    
    # 如果存在 _tb_writer 对象，则将其作为回调添加到 trainer 中
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    
    # 返回表示最佳运行的对象
    return best_run
def run_hp_search_wandb(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    # 导入检查Wandb是否可用的函数
    from ..integrations import is_wandb_available

    # 如果Wandb不可用，则抛出 ImportError 异常
    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")
    # 导入Wandb
    import wandb

    # 检查是否已经添加了 WandbCallback 到 trainer 的回调列表中
    reporting_to_wandb = False
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            reporting_to_wandb = True
            break
    # 如果没有添加，则将 WandbCallback 添加到 trainer 的回调列表中
    if not reporting_to_wandb:
        trainer.add_callback(WandbCallback())

    # 设置 trainer 的 report_to 属性为 ["wandb"]，表明报告结果到 Wandb
    trainer.args.report_to = ["wandb"]

    # 初始化最佳试验的信息
    best_trial = {"run_id": None, "objective": None, "hyperparameters": None}

    # 从 kwargs 中获取 sweep_id、project、name、entity 等参数
    sweep_id = kwargs.pop("sweep_id", None)
    project = kwargs.pop("project", None)
    name = kwargs.pop("name", None)
    entity = kwargs.pop("entity", None)
    # 从 kwargs 中获取 metric 参数，默认为 "eval/loss"
    metric = kwargs.pop("metric", "eval/loss")

    # 从 trainer 获取超参数空间配置
    sweep_config = trainer.hp_space(None)

    # 设置超参数空间配置的优化目标和指标名称
    sweep_config["metric"]["goal"] = direction
    sweep_config["metric"]["name"] = metric

    # 如果提供了 name 参数，则设置超参数空间配置的名称
    if name:
        sweep_config["name"] = name
    # 定义一个名为 _objective 的函数
    def _objective():
        # 如果 wandb.run 存在，则使用当前运行的 wandb.run，否则初始化一个新的 wandb.run
        run = wandb.run if wandb.run else wandb.init()
        # 将训练器的试验名称设置为当前运行的名称
        trainer.state.trial_name = run.name
        # 更新配置，包括 "assignments": {} 和指定的度量标准 metric
        run.config.update({"assignments": {}, "metric": metric})
        # 获取当前的配置
        config = wandb.config

        # 将训练器的 objective 属性设置为 None
        trainer.objective = None

        # 开始训练过程，resume_from_checkpoint=None，并传递配置项作为试验的参数
        trainer.train(resume_from_checkpoint=None, trial=vars(config)["_items"])
        
        # 如果在训练循环中没有进行任何评估
        if getattr(trainer, "objective", None) is None:
            # 执行评估并获取指标 metrics
            metrics = trainer.evaluate()
            # 计算训练器的 objective 属性
            trainer.objective = trainer.compute_objective(metrics)
            # 重新编写日志格式的指标
            format_metrics = rewrite_logs(metrics)
            # 如果指定的度量标准不在重新编写后的指标中，则发出警告
            if metric not in format_metrics:
                logger.warning(
                    f"Provided metric {metric} not found. This might result in unexpected sweeps charts. The available"
                    f" metrics are {format_metrics.keys()}"
                )
        
        # 初始化 best_score 为 False
        best_score = False
        # 如果 best_trial["run_id"] 不为 None
        if best_trial["run_id"] is not None:
            # 根据 direction 的设置，比较当前训练器的 objective 属性和最佳试验的 objective 属性
            if direction == "minimize":
                best_score = trainer.objective < best_trial["objective"]
            elif direction == "maximize":
                best_score = trainer.objective > best_trial["objective"]

        # 如果 best_score 为 True 或者 best_trial["run_id"] 为 None
        if best_score or best_trial["run_id"] is None:
            # 更新最佳试验的 run_id、objective 和 hyperparameters
            best_trial["run_id"] = run.id
            best_trial["objective"] = trainer.objective
            best_trial["hyperparameters"] = dict(config)

        # 返回训练器的 objective 属性作为函数 _objective 的结果
        return trainer.objective

    # 如果 sweep_id 不存在，则使用给定的 sweep_config 创建一个新的 wandb sweep，并指定项目和实体
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity) if not sweep_id else sweep_id
    # 输出当前的 wandb sweep id
    logger.info(f"wandb sweep id - {sweep_id}")
    # 使用 wandb agent 在指定的 sweep_id 上运行函数 _objective，并设置运行的次数为 n_trials
    wandb.agent(sweep_id, function=_objective, count=n_trials)

    # 返回包含最佳运行的 run_id、objective 和 hyperparameters 的 BestRun 对象
    return BestRun(best_trial["run_id"], best_trial["objective"], best_trial["hyperparameters"])
# 定义函数，返回所有可用的报告集成列表
def get_available_reporting_integrations():
    # 初始化空列表，用于存储可用的集成
    integrations = []
    # 检查 Azure ML 是否可用，并且 MLflow 不可用时，添加 "azure_ml" 到集成列表
    if is_azureml_available() and not is_mlflow_available():
        integrations.append("azure_ml")
    # 如果 Comet ML 可用，添加 "comet_ml" 到集成列表
    if is_comet_available():
        integrations.append("comet_ml")
    # 如果 DagsHub 可用，添加 "dagshub" 到集成列表
    if is_dagshub_available():
        integrations.append("dagshub")
    # 如果 DVC Live 可用，添加 "dvclive" 到集成列表
    if is_dvclive_available():
        integrations.append("dvclive")
    # 如果 MLflow 可用，添加 "mlflow" 到集成列表
    if is_mlflow_available():
        integrations.append("mlflow")
    # 如果 Neptune 可用，添加 "neptune" 到集成列表
    if is_neptune_available():
        integrations.append("neptune")
    # 如果 TensorBoard 可用，添加 "tensorboard" 到集成列表
    if is_tensorboard_available():
        integrations.append("tensorboard")
    # 如果 Weights & Biases 可用，添加 "wandb" 到集成列表
    if is_wandb_available():
        integrations.append("wandb")
    # 如果 CodeCarbon 可用，添加 "codecarbon" 到集成列表
    if is_codecarbon_available():
        integrations.append("codecarbon")
    # 如果 ClearML 可用，添加 "clearml" 到集成列表
    if is_clearml_available():
        integrations.append("clearml")
    # 返回所有已添加的集成列表
    return integrations


# 定义函数，重写输入字典的键名规则并返回新字典
def rewrite_logs(d):
    # 初始化空字典，用于存储重写后的键值对
    new_d = {}
    # 设置评估前缀和测试前缀
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    # 遍历输入字典的键值对
    for k, v in d.items():
        # 如果键以评估前缀开头，将键重写为 "eval/去除前缀后的键"，并将原值赋给新字典对应键
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        # 如果键以测试前缀开头，将键重写为 "test/去除前缀后的键"，并将原值赋给新字典对应键
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        # 否则，将键重写为 "train/原键"，并将原值赋给新字典对应键
        else:
            new_d["train/" + k] = v
    # 返回重写后的新字典
    return new_d


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        # 检查 TensorBoard 是否可用
        has_tensorboard = is_tensorboard_available()
        # 如果 TensorBoard 不可用，抛出运行时错误
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        # 如果 TensorBoard 可用
        if has_tensorboard:
            try:
                # 尝试导入 PyTorch 的 SummaryWriter
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401
                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    # 如果导入失败，尝试导入 tensorboardX 的 SummaryWriter
                    from tensorboardX import SummaryWriter
                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    # 如果都导入失败，设为 None
                    self._SummaryWriter = None
        else:
            # 如果 TensorBoard 不可用，设为 None
            self._SummaryWriter = None
        # 设置回调对象的写入器
        self.tb_writer = tb_writer

    # 初始化 TensorBoard 的 SummaryWriter
    def _init_summary_writer(self, args, log_dir=None):
        # 如果未提供日志目录，使用参数 args 的 logging_dir
        log_dir = log_dir or args.logging_dir
        # 如果 SummaryWriter 存在
        if self._SummaryWriter is not None:
            # 初始化回调对象的写入器
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)
    # 如果不是全局进程的第一个进程，则直接返回，不执行后续操作
    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        # 初始化日志目录为 None
        log_dir = None

        # 如果是超参数搜索状态，根据试验名称组合日志目录路径
        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        # 如果 tb_writer 为空，则初始化摘要写入器
        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        # 如果 tb_writer 不为空，则添加参数 args 的 JSON 字符串到 TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            # 如果 kwargs 中包含 "model"，则获取模型配置信息并添加到 TensorBoard
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

    # 如果不是全局进程的第一个进程，则直接返回，不执行后续操作
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        # 如果 tb_writer 为空，则初始化摘要写入器
        if self.tb_writer is None:
            self._init_summary_writer(args)

        # 如果 tb_writer 不为空，则重写日志并逐个处理
        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                # 如果值为整数或浮点数，则将其作为标量添加到 TensorBoard
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    # 否则记录警告信息，指出不正确的调用方式并丢弃该属性
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            # 刷新 TensorBoard 写入器
            self.tb_writer.flush()

    # 如果 tb_writer 存在，则关闭它，并将其置为 None
    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    # 初始化函数，检查是否安装了 wandb，如果未安装则抛出异常
    def __init__(self):
        # 检查是否安装了 wandb
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        # 如果 wandb 可用，则引入 wandb 模块
        if has_wandb:
            import wandb
            self._wandb = wandb
        # 标记初始化状态为 False
        self._initialized = False
        
        # 根据环境变量设置是否记录模型，同时给出警告信息
        if os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"}):
            DeprecationWarning(
                f"Setting `WANDB_LOG_MODEL` as {os.getenv('WANDB_LOG_MODEL')} is deprecated and will be removed in "
                "version 5 of transformers. Use one of `'end'` or `'checkpoint'` instead."
            )
            logger.info(f"Setting `WANDB_LOG_MODEL` from {os.getenv('WANDB_LOG_MODEL')} to `end` instead")
            # 将记录模型的设置从环境变量中读取并转换为小写字符串
            self._log_model = "end"
        else:
            # 如果环境变量中未设置，则默认为 false
            self._log_model = os.getenv("WANDB_LOG_MODEL", "false").lower()

    # 当训练开始时调用的函数，根据状态进行相应操作
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果未安装 wandb，则直接返回
        if self._wandb is None:
            return
        
        # 检查是否为超参数搜索，如果是，则结束 wandb 进程并重置初始化状态
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        
        # 如果未初始化，则进行设置
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
    # 在训练结束时触发的回调函数，用于上传模型和日志
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # 如果未初始化或者未配置 WandB，直接返回
        if self._wandb is None:
            return
        # 如果需要在结束或者检查点时记录模型，并且已经初始化，并且当前进程是主进程
        if self._log_model in ("end", "checkpoint") and self._initialized and state.is_world_process_zero:
            # 导入 Trainer 类用于模型保存
            from ..trainer import Trainer

            # 创建一个假的 Trainer 对象用于保存模型
            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            # 使用临时目录保存模型
            with tempfile.TemporaryDirectory() as temp_dir:
                # 将模型保存到临时目录
                fake_trainer.save_model(temp_dir)
                # 准备上传的元数据
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                # 记录上传日志
                logger.info("Logging model artifacts. ...")
                # 确定模型的名称
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                # 创建一个 WandB Artifact 对象，用于上传模型
                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                # 遍历临时目录下的所有文件
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        # 将每个文件添加到 Artifact 中
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                # 上传 Artifact 到 WandB
                self._wandb.run.log_artifact(artifact)

    # 在日志记录时触发的回调函数，用于记录单值标量和非标量日志
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 定义单值标量日志的关键字列表
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        # 如果未配置 WandB，直接返回
        if self._wandb is None:
            return
        # 如果未初始化，则进行初始化
        if not self._initialized:
            self.setup(args, state, model)
        # 如果当前进程是主进程
        if state.is_world_process_zero:
            # 遍历 logs 中的每个键值对
            for k, v in logs.items():
                # 如果键在单值标量列表中，则更新 WandB 的 summary
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            # 从 logs 中提取非标量日志，并进行重写处理
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            # 记录非标量日志和全局步数到 WandB
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})
    # 当保存操作触发时调用此方法，接收参数 args, state, control 和其他关键字参数 kwargs
    def on_save(self, args, state, control, **kwargs):
        # 检查日志模式是否为 "checkpoint"，且对象已初始化，并且当前进程是主进程
        if self._log_model == "checkpoint" and self._initialized and state.is_world_process_zero:
            # 创建一个包含非私有数值的摘要元数据字典
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }

            # 根据全局步数创建检查点目录名
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 构造完整的存储路径，放置检查点
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            # 记录日志，指示正在保存检查点工件
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            # 根据运行名和ID创建检查点名称
            checkpoint_name = (
                f"checkpoint-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"checkpoint-{self._wandb.run.name}"
            )
            # 创建一个 W&B Artifact 对象，类型为 "model"，并附带元数据
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            # 将检查点目录及其内容添加到工件中
            artifact.add_dir(artifact_path)
            # 使用全局步数作为别名，将工件记录到 W&B
            self._wandb.log_artifact(artifact, aliases=[f"checkpoint-{state.global_step}"])
# 定义一个名为 `CometCallback` 的类，继承自 `TrainerCallback`
class CometCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Comet ML](https://www.comet.ml/site/).
    """

    # 初始化方法，检查是否安装了 comet-ml 库，若未安装则抛出运行时错误
    def __init__(self):
        if not _has_comet:
            raise RuntimeError("CometCallback requires comet-ml to be installed. Run `pip install comet-ml`.")
        self._initialized = False  # 标记是否已初始化
        self._log_assets = False  # 标记是否记录训练资源

    # 设置方法，用于配置 Comet.ml 集成
    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
        - **COMET_MODE** (`str`, *optional*, defaults to `ONLINE`):
            Whether to create an online, offline experiment or disable Comet logging. Can be `OFFLINE`, `ONLINE`, or
            `DISABLED`.
        - **COMET_PROJECT_NAME** (`str`, *optional*):
            Comet project name for experiments.
        - **COMET_OFFLINE_DIRECTORY** (`str`, *optional*):
            Folder to use for saving offline experiments when `COMET_MODE` is `OFFLINE`.
        - **COMET_LOG_ASSETS** (`str`, *optional*, defaults to `TRUE`):
            Whether or not to log training assets (tf event logs, checkpoints, etc), to Comet. Can be `TRUE`, or
            `FALSE`.

        For a number of configurable items in the environment, see
        [here](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).
        """
        self._initialized = True  # 标记已初始化
        # 检查是否需要记录训练资源，根据环境变量 COMET_LOG_ASSETS 的设置决定
        log_assets = os.getenv("COMET_LOG_ASSETS", "FALSE").upper()
        if log_assets in {"TRUE", "1"}:
            self._log_assets = True
        # 如果是主进程（world_process_zero），根据环境变量 COMET_MODE 的设置创建相应的 Comet 实验
        if state.is_world_process_zero:
            comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
            experiment = None
            experiment_kwargs = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
            if comet_mode == "ONLINE":
                # 创建在线 Comet 实验，并记录相关信息
                experiment = comet_ml.Experiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml online logging enabled")
            elif comet_mode == "OFFLINE":
                # 创建离线 Comet 实验，并记录相关信息
                experiment_kwargs["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
                experiment = comet_ml.OfflineExperiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
            # 如果成功创建实验对象，则记录模型图和参数信息到 Comet
            if experiment is not None:
                experiment._set_model_graph(model, framework="transformers")
                experiment._log_parameters(args, prefix="args/", framework="transformers")
                if hasattr(model, "config"):
                    experiment._log_parameters(model.config, prefix="config/", framework="transformers")

    # 训练开始时的回调方法，如果未初始化则执行设置方法
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
    # 当日志事件触发时调用的方法，用于处理日志记录
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 如果对象尚未初始化，则执行初始化设置
        if not self._initialized:
            self.setup(args, state, model)
        # 如果当前进程是全局的第零个进程
        if state.is_world_process_zero:
            # 获取全局 Comet 实验对象
            experiment = comet_ml.config.get_global_experiment()
            # 如果实验对象不为空，则记录指标(metrics)
            if experiment is not None:
                experiment._log_metrics(logs, step=state.global_step, epoch=state.epoch, framework="transformers")

    # 当训练结束时调用的方法
    def on_train_end(self, args, state, control, **kwargs):
        # 如果对象已经初始化，并且当前进程是全局的第零个进程
        if self._initialized and state.is_world_process_zero:
            # 获取全局 Comet 实验对象
            experiment = comet_ml.config.get_global_experiment()
            # 如果实验对象不为空
            if experiment is not None:
                # 如果设置了记录资产(_log_assets)，则记录输出目录中的文件
                if self._log_assets is True:
                    logger.info("Logging checkpoints. This may take time.")
                    experiment.log_asset_folder(
                        args.output_dir, recursive=True, log_file_name=True, step=state.global_step
                    )
                # 结束 Comet 实验
                experiment.end()
class AzureMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [AzureML](https://pypi.org/project/azureml-sdk/).
    """

    def __init__(self, azureml_run=None):
        # 检查是否安装了 AzureML SDK，如果没有则抛出运行时错误
        if not is_azureml_available():
            raise RuntimeError("AzureMLCallback requires azureml to be installed. Run `pip install azureml-sdk`.")
        self.azureml_run = azureml_run

    def on_init_end(self, args, state, control, **kwargs):
        # 导入 AzureML 的 Run 类
        from azureml.core.run import Run

        # 如果未提供 azureml_run 并且是主进程，则获取当前运行的上下文
        if self.azureml_run is None and state.is_world_process_zero:
            self.azureml_run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 如果有提供 azureml_run 并且是主进程
        if self.azureml_run and state.is_world_process_zero:
            # 遍历 logs 字典，将其键值对作为日志项传递给 AzureML 的 run 对象
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.azureml_run.log(k, v, description=k)


class MLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """

    def __init__(self):
        # 检查是否安装了 MLflow，如果没有则抛出运行时错误
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        import mlflow

        # 设置 MLflow 相关的最大参数值长度和每批次参数标签的最大数
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果尚未初始化，则进行设置
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        # 如果尚未初始化，则进行设置
        if not self._initialized:
            self.setup(args, state, model)
        # 如果是主进程
        if state.is_world_process_zero:
            metrics = {}
            # 遍历 logs 字典，将数值类型的值作为指标(metrics)记录
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                else:
                    # 如果值不是数值类型，则记录警告日志并忽略该值
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )

            # 如果是异步日志，则异步记录指标(metrics)
            if self._async_log:
                self._ml_flow.log_metrics(metrics=metrics, step=state.global_step, synchronous=False)
            else:
                self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        # 如果已初始化并且是主进程，则根据设置自动结束 MLflow 的运行
        if self._initialized and state.is_world_process_zero:
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()
    # 当保存操作触发时执行的方法，接收参数 args, state, control 和 kwargs
    def on_save(self, args, state, control, **kwargs):
        # 检查对象是否已初始化，并且当前进程是世界中的主进程，同时日志记录已启用
        if self._initialized and state.is_world_process_zero and self._log_artifacts:
            # 构建检查点目录名称，使用全局步数来唯一标识
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 构建检查点的完整路径，基于指定的输出目录
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            # 记录信息日志，指示正在将检查点数据记录为 artifacts
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
            # 使用 MLflow 的 pyfunc 接口记录模型
            self._ml_flow.pyfunc.log_model(
                ckpt_dir,  # 记录模型时使用的名称
                artifacts={"model_path": artifact_path},  # 附加的 artifacts，指定模型路径
                python_model=self._ml_flow.pyfunc.PythonModel(),  # 使用的 Python 模型
            )

    # 析构函数，在对象被销毁时调用
    def __del__(self):
        # 如果设置了自动结束运行，并且 MLflow 的活跃运行状态是可调用的且不为空
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            # 结束当前的 MLflow 运行
            self._ml_flow.end_run()
class DagsHubCallback(MLflowCallback):
    """
    A [`TrainerCallback`] that logs to [DagsHub](https://dagshub.com/). Extends [`MLflowCallback`]
    """

    def __init__(self):
        super().__init__()
        # 检查是否安装了 DagsHub 相关库
        if not is_dagshub_available():
            raise ImportError("DagsHubCallback requires dagshub to be installed. Run `pip install dagshub`.")

        # 导入 DagsHub 的 Repo 类
        from dagshub.upload import Repo
        self.Repo = Repo

    def setup(self, *args, **kwargs):
        """
        Setup the DagsHub's Logging integration.

        Environment:
        - **HF_DAGSHUB_LOG_ARTIFACTS** (`str`, *optional*):
                Whether to save the data and model artifacts for the experiment. Default to `False`.
        """

        # 检查是否要记录数据和模型的 artifacts
        self.log_artifacts = os.getenv("HF_DAGSHUB_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        # 获取模型名称，如果未指定则默认为 "main"
        self.name = os.getenv("HF_DAGSHUB_MODEL_NAME") or "main"
        # 获取 MLflow 的远程跟踪 URI
        self.remote = os.getenv("MLFLOW_TRACKING_URI")
        # 根据远程跟踪 URI 创建 DagsHub Repo 对象
        self.repo = self.Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].split(".")[0],
            branch=os.getenv("BRANCH") or "main",
        )
        # 设置路径为 "artifacts"
        self.path = Path("artifacts")

        # 如果未设置远程跟踪 URI，则抛出运行时错误
        if self.remote is None:
            raise RuntimeError(
                "DagsHubCallback requires the `MLFLOW_TRACKING_URI` environment variable to be set. Did you run"
                " `dagshub.init()`?"
            )

        # 调用父类的 setup 方法
        super().setup(*args, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        # 如果要记录 artifacts
        if self.log_artifacts:
            # 如果存在 train_dataloader 属性，则保存数据集到 "dataset.pt" 文件
            if getattr(self, "train_dataloader", None):
                torch.save(self.train_dataloader.dataset, os.path.join(args.output_dir, "dataset.pt"))

            # 将输出目录下的内容添加到 DagsHub Repo 的指定目录下
            self.repo.directory(str(self.path)).add_dir(args.output_dir)


class NeptuneMissingConfiguration(Exception):
    def __init__(self):
        super().__init__(
            """
        ------ Unsupported ---- We were not able to create new runs. You provided a custom Neptune run to
        `NeptuneCallback` with the `run` argument. For the integration to work fully, provide your `api_token` and
        `project` by saving them as environment variables or passing them to the callback.
        """
        )


class NeptuneCallback(TrainerCallback):
    """TrainerCallback that sends the logs to [Neptune](https://app.neptune.ai).
    # NeptuneLogger 类定义，用于集成 Transformers 框架与 Neptune 平台
    class NeptuneLogger:
    
        # 集成版本键，用于 Neptune 运行日志中的源代码路径
        integration_version_key = "source_code/integrations/transformers"
        # 模型参数键，用于 Neptune 运行日志中的模型参数
        model_parameters_key = "model_parameters"
        # 实验名称键，用于 Neptune 运行日志中的实验名称
        trial_name_key = "trial"
        # 实验参数键，用于 Neptune 运行日志中的实验参数
        trial_params_key = "trial_params"
        # 训练器参数键，用于 Neptune 运行日志中的训练器参数
        trainer_parameters_key = "trainer_parameters"
        # 扁平化指标，用于 Neptune 运行日志中的扁平化指标记录
        flat_metrics = {"train/epoch"}
    
        # NeptuneLogger 类的初始化方法
        def __init__(
            self,
            *,
            api_token: Optional[str] = None,  # Neptune API token，可选参数，用于身份验证
            project: Optional[str] = None,  # Neptune 项目名称，可选参数，指定要记录的项目
            name: Optional[str] = None,  # 自定义运行名称，可选参数，指定 Neptune 运行的名称
            base_namespace: str = "finetuning",  # 基础命名空间，默认为 "finetuning"，用于 Neptune 日志的根命名空间
            run=None,  # Neptune 运行对象，可选参数，如果要继续记录到现有运行中
            log_parameters: bool = True,  # 是否记录训练器参数和模型参数的标志，可选参数，默认为 True
            log_checkpoints: Optional[str] = None,  # 检查点记录选项，可选参数，指定何时上传检查点文件
            **neptune_run_kwargs,  # 其他 Neptune 初始化函数的关键字参数，用于创建新的 Neptune 运行时
        ):
    ):
        # 检查 Neptune 是否可用，如果不可用则抛出 ValueError 异常
        if not is_neptune_available():
            raise ValueError(
                "NeptuneCallback requires the Neptune client library to be installed. "
                "To install the library, run `pip install neptune`."
            )

        try:
            # 尝试导入 Neptune 相关模块
            from neptune import Run
            from neptune.internal.utils import verify_type
        except ImportError:
            # 如果导入失败，则尝试从新路径导入 Neptune 相关模块
            from neptune.new.internal.utils import verify_type
            from neptune.new.metadata_containers.run import Run

        # 验证参数类型
        verify_type("api_token", api_token, (str, type(None)))
        verify_type("project", project, (str, type(None)))
        verify_type("name", name, (str, type(None)))
        verify_type("base_namespace", base_namespace, str)
        verify_type("run", run, (Run, type(None)))
        verify_type("log_parameters", log_parameters, bool)
        verify_type("log_checkpoints", log_checkpoints, (str, type(None)))

        # 设置内部变量
        self._base_namespace_path = base_namespace
        self._log_parameters = log_parameters
        self._log_checkpoints = log_checkpoints
        self._initial_run: Optional[Run] = run

        # 初始化变量
        self._run = None
        self._is_monitoring_run = False
        self._run_id = None
        self._force_reset_monitoring_run = False
        self._init_run_kwargs = {"api_token": api_token, "project": project, "name": name, **neptune_run_kwargs}

        self._volatile_checkpoints_dir = None
        self._should_upload_checkpoint = self._log_checkpoints is not None
        self._recent_checkpoint_path = None

        # 根据 log_checkpoints 的值设置目标检查点命名空间和是否清理最近上传的检查点
        if self._log_checkpoints in {"last", "best"}:
            self._target_checkpoints_namespace = f"checkpoints/{self._log_checkpoints}"
            self._should_clean_recently_uploaded_checkpoint = True
        else:
            self._target_checkpoints_namespace = "checkpoints"
            self._should_clean_recently_uploaded_checkpoint = False

    # 如果运行实例存在，则停止该运行实例
    def _stop_run_if_exists(self):
        if self._run:
            self._run.stop()
            del self._run
            self._run = None

    # 初始化 Neptune 运行实例
    def _initialize_run(self, **additional_neptune_kwargs):
        try:
            # 尝试从 neptune 包中导入 init_run 和异常处理类
            from neptune import init_run
            from neptune.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException
        except ImportError:
            # 如果导入失败，则尝试从新路径导入 init_run 和异常处理类
            from neptune.new import init_run
            from neptune.new.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException

        # 停止已存在的运行实例
        self._stop_run_if_exists()

        try:
            # 创建运行实例的参数集合
            run_params = additional_neptune_kwargs.copy()
            run_params.update(self._init_run_kwargs)
            # 使用参数初始化运行实例，并获取运行实例的 ID
            self._run = init_run(**run_params)
            self._run_id = self._run["sys/id"].fetch()
        except (NeptuneMissingProjectNameException, NeptuneMissingApiTokenException) as e:
            # 如果缺少项目名或 API token，则抛出 NeptuneMissingConfiguration 异常
            raise NeptuneMissingConfiguration() from e
    # 将初始运行设置为当前运行，并开启监控模式
    def _use_initial_run(self):
        self._run = self._initial_run
        self._is_monitoring_run = True
        self._run_id = self._run["sys/id"].fetch()
        self._initial_run = None

    # 确保存在带监控的运行环境
    def _ensure_run_with_monitoring(self):
        if self._initial_run is not None:
            # 如果存在初始运行，则使用初始运行
            self._use_initial_run()
        else:
            if not self._force_reset_monitoring_run and self._is_monitoring_run:
                return

            if self._run and not self._is_monitoring_run and not self._force_reset_monitoring_run:
                # 如果存在运行环境但未开启监控，则重新初始化运行并开启监控
                self._initialize_run(with_id=self._run_id)
                self._is_monitoring_run = True
            else:
                # 否则，初始化一个新的运行环境
                self._initialize_run()
                self._force_reset_monitoring_run = False

    # 确保至少存在一个不带监控的运行环境
    def _ensure_at_least_run_without_monitoring(self):
        if self._initial_run is not None:
            # 如果存在初始运行，则使用初始运行
            self._use_initial_run()
        else:
            if not self._run:
                # 如果没有运行环境，则初始化一个新的运行环境，不捕获 stdout、stderr、硬件指标和 traceback
                self._initialize_run(
                    with_id=self._run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False,
                    capture_traceback=False,
                )
                self._is_monitoring_run = False

    # 返回当前运行环境
    @property
    def run(self):
        if self._run is None:
            self._ensure_at_least_run_without_monitoring()
        return self._run

    # 返回运行环境的元数据命名空间
    @property
    def _metadata_namespace(self):
        return self.run[self._base_namespace_path]

    # 记录集成版本号到运行环境中
    def _log_integration_version(self):
        self.run[NeptuneCallback.integration_version_key] = version

    # 记录训练器参数到运行环境的元数据命名空间中
    def _log_trainer_parameters(self, args):
        self._metadata_namespace[NeptuneCallback.trainer_parameters_key] = args.to_sanitized_dict()

    # 记录模型参数到运行环境的元数据命名空间中
    def _log_model_parameters(self, model):
        from neptune.utils import stringify_unsupported

        if model and hasattr(model, "config") and model.config is not None:
            self._metadata_namespace[NeptuneCallback.model_parameters_key] = stringify_unsupported(
                model.config.to_dict()
            )

    # 记录超参数搜索参数到运行环境的元数据命名空间中
    def _log_hyper_param_search_parameters(self, state):
        if state and hasattr(state, "trial_name"):
            self._metadata_namespace[NeptuneCallback.trial_name_key] = state.trial_name

        if state and hasattr(state, "trial_params") and state.trial_params is not None:
            self._metadata_namespace[NeptuneCallback.trial_params_key] = state.trial_params
    # 将源目录和检查点路径合并成目标路径
    target_path = relative_path = os.path.join(source_directory, checkpoint)

    # 如果存在易失性检查点目录
    if self._volatile_checkpoints_dir is not None:
        # 构建一致性检查点路径
        consistent_checkpoint_path = os.path.join(self._volatile_checkpoints_dir, checkpoint)
        try:
            # 从相对路径中移除开头的 ../，并去掉开头的路径分隔符
            cpkt_path = relative_path.replace("..", "").lstrip(os.path.sep)
            copy_path = os.path.join(consistent_checkpoint_path, cpkt_path)
            # 复制整个目录树到一致性检查点路径
            shutil.copytree(relative_path, copy_path)
            # 更新目标路径为一致性检查点路径
            target_path = consistent_checkpoint_path
        except IOError as e:
            # 如果复制过程中出现 I/O 异常，则记录警告信息
            logger.warning(
                "NeptuneCallback was unable to made a copy of checkpoint due to I/O exception: '{}'. "
                "Could fail trying to upload.".format(e)
            )

    # 将目标路径中的文件上传到 Neptune 中
    self._metadata_namespace[self._target_checkpoints_namespace].upload_files(target_path)

    # 如果需要清理最近上传的检查点，并且最近的检查点路径不为 None，则删除它
    if self._should_clean_recently_uploaded_checkpoint and self._recent_checkpoint_path is not None:
        self._metadata_namespace[self._target_checkpoints_namespace].delete_files(self._recent_checkpoint_path)

    # 更新最近的检查点路径为相对路径
    self._recent_checkpoint_path = relative_path

def on_init_end(self, args, state, control, **kwargs):
    # 初始化易失性检查点目录为 None
    self._volatile_checkpoints_dir = None
    # 如果需要记录检查点，并且要求覆盖输出目录或设置了保存总数限制，则创建一个临时目录用于存储检查点
    if self._log_checkpoints and (args.overwrite_output_dir or args.save_total_limit is not None):
        self._volatile_checkpoints_dir = tempfile.TemporaryDirectory().name

    # 如果要求记录最佳检查点但未设置在训练结束时加载最佳模型，则引发 ValueError 异常
    if self._log_checkpoints == "best" and not args.load_best_model_at_end:
        raise ValueError("To save the best model checkpoint, the load_best_model_at_end argument must be enabled.")

def on_train_begin(self, args, state, control, model=None, **kwargs):
    # 如果不是全局进程的主进程，则直接返回
    if not state.is_world_process_zero:
        return

    # 确保在监控下运行
    self._ensure_run_with_monitoring()
    # 强制重置监控运行状态
    self._force_reset_monitoring_run = True

    # 记录集成版本信息
    self._log_integration_version()
    # 如果需要记录参数，则记录训练器参数和模型参数
    if self._log_parameters:
        self._log_trainer_parameters(args)
        self._log_model_parameters(model)

    # 如果是超参数搜索状态，则记录超参数搜索参数
    if state.is_hyper_param_search:
        self._log_hyper_param_search_parameters(state)

def on_train_end(self, args, state, control, **kwargs):
    # 如果存在运行，则停止该运行
    self._stop_run_if_exists()

def __del__(self):
    # 如果存在易失性检查点目录，则删除该目录及其内容，忽略所有错误
    if self._volatile_checkpoints_dir is not None:
        shutil.rmtree(self._volatile_checkpoints_dir, ignore_errors=True)

    # 停止 Neptune 运行，如果存在的话
    self._stop_run_if_exists()

def on_save(self, args, state, control, **kwargs):
    # 如果需要上传检查点，则记录模型检查点
    if self._should_upload_checkpoint:
        self._log_model_checkpoint(args.output_dir, f"checkpoint-{state.global_step}")
    # 定义一个方法，处理评估时的回调函数
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 如果设置了日志保存最佳模型
        if self._log_checkpoints == "best":
            # 获取用于最佳模型判定的指标名称
            best_metric_name = args.metric_for_best_model
            # 如果指标名称不以"eval_"开头，则添加前缀"eval_"
            if not best_metric_name.startswith("eval_"):
                best_metric_name = f"eval_{best_metric_name}"

            # 获取指定名称的指标值
            metric_value = metrics.get(best_metric_name)

            # 根据参数指定的条件判断函数选择比较操作符
            operator = np.greater if args.greater_is_better else np.less

            # 判断是否应上传检查点，判断标准是当前指标值是否优于之前保存的最佳指标值
            self._should_upload_checkpoint = state.best_metric is None or operator(metric_value, state.best_metric)

    # 类方法：获取训练器关联的 NeptuneCallback 实例的运行配置
    @classmethod
    def get_run(cls, trainer):
        # 遍历训练器回调处理程序中的回调函数
        for callback in trainer.callback_handler.callbacks:
            # 如果回调函数是 NeptuneCallback 的实例，则返回其运行配置
            if isinstance(callback, cls):
                return callback.run

        # 如果没有 NeptuneCallback 配置，抛出异常
        raise Exception("The trainer doesn't have a NeptuneCallback configured.")

    # 定义一个方法，处理记录日志时的回调函数
    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        # 如果不是全局进程的主进程，直接返回
        if not state.is_world_process_zero:
            return

        # 如果有日志内容
        if logs is not None:
            # 对每个重写后的日志项进行处理
            for name, value in rewrite_logs(logs).items():
                # 如果值是整数或浮点数
                if isinstance(value, (int, float)):
                    # 如果日志名称在 NeptuneCallback 的平坦指标中
                    if name in NeptuneCallback.flat_metrics:
                        # 将值记录到元数据命名空间中
                        self._metadata_namespace[name] = value
                    else:
                        # 否则，将值记录到元数据命名空间中并指定步骤为全局步骤数
                        self._metadata_namespace[name].log(value, step=state.global_step)
# 定义一个名为 CodeCarbonCallback 的类，继承自 TrainerCallback 类
class CodeCarbonCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that tracks the CO2 emission of training.
    """

    # 初始化方法
    def __init__(self):
        # 检查是否安装了 codecarbon 库，若未安装则引发运行时错误
        if not is_codecarbon_available():
            raise RuntimeError(
                "CodeCarbonCallback requires `codecarbon` to be installed. Run `pip install codecarbon`."
            )
        # 导入 codecarbon 库
        import codecarbon

        # 将 codecarbon 模块赋值给 self._codecarbon
        self._codecarbon = codecarbon
        # 初始化追踪器为 None
        self.tracker = None

    # 当初始化结束时触发的回调方法
    def on_init_end(self, args, state, control, **kwargs):
        # 如果追踪器为 None 并且是本地进程的第零号进程
        if self.tracker is None and state.is_local_process_zero:
            # 使用指定的输出目录创建 CO2 排放追踪器对象
            self.tracker = self._codecarbon.EmissionsTracker(output_dir=args.output_dir)

    # 当训练开始时触发的回调方法
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果追踪器存在并且是本地进程的第零号进程
        if self.tracker and state.is_local_process_zero:
            # 启动 CO2 排放追踪器
            self.tracker.start()

    # 当训练结束时触发的回调方法
    def on_train_end(self, args, state, control, **kwargs):
        # 如果追踪器存在并且是本地进程的第零号进程
        if self.tracker and state.is_local_process_zero:
            # 停止 CO2 排放追踪器
            self.tracker.stop()


# 定义一个名为 ClearMLCallback 的类，继承自 TrainerCallback 类
class ClearMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [ClearML](https://clear.ml/).

    Environment:
    - **CLEARML_PROJECT** (`str`, *optional*, defaults to `HuggingFace Transformers`):
        ClearML project name.
    - **CLEARML_TASK** (`str`, *optional*, defaults to `Trainer`):
        ClearML task name.
    - **CLEARML_LOG_MODEL** (`bool`, *optional*, defaults to `False`):
        Whether to log models as artifacts during training.
    """

    # 类级别的属性
    log_suffix = ""

    _hparams_section = "Transformers"
    _model_config_section = "Model Configuration"
    _ignore_hparams_overrides = "_ignore_hparams_ui_overrides_"
    _ignoge_model_config_overrides = "_ignore_model_config_ui_overrides_"
    _model_config_description = "The configuration of model number {}."
    _model_config_description_note = (
        "Note that, when cloning this task and running it remotely,"
        " the configuration might be applied to another model instead of this one."
        " To avoid this, initialize the task externally by calling `Task.init`"
        " before the `ClearMLCallback` is instantiated."
    )
    _train_run_counter = 0
    _model_connect_counter = 0
    _task_created_in_callback = False
    _should_close_on_train_end = None

    # 初始化方法
    def __init__(self):
        # 检查是否安装了 clearml 库，若未安装则引发运行时错误
        if is_clearml_available():
            import clearml

            # 导入 clearml 库
            self._clearml = clearml
        else:
            raise RuntimeError("ClearMLCallback requires 'clearml' to be installed. Run `pip install clearml`.")

        # 初始化标志为 False
        self._initialized = False
        # 初始化 ClearML 任务为 None
        self._clearml_task = None

        # 初始化日志模型为 False
        self._log_model = False
        # 初始化检查点保存列表为空列表
        self._checkpoints_saved = []
    # 当训练开始时调用的方法，初始化训练过程中需要用到的参数和模型
    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # 如果未初始化 ClearML，直接返回
        if self._clearml is None:
            return
        # 初始化一个空列表来存储保存的检查点文件名
        self._checkpoints_saved = []
        # 如果当前训练是超参数搜索，则标记为未初始化状态
        if state.is_hyper_param_search:
            self._initialized = False
        # 如果未初始化，调用 setup 方法来设置参数、模型和分词器等
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)
    
    # 当训练结束时调用的方法，用于清理和关闭 ClearML 相关的任务和计数器
    def on_train_end(self, args, state, control, **kwargs):
        # 如果应该在训练结束时关闭 ClearML 任务，则关闭当前任务
        if ClearMLCallback._should_close_on_train_end:
            self._clearml_task.close()
            # 重置训练运行计数器为零
            ClearMLCallback._train_run_counter = 0
    # 定义一个方法，用于处理日志信息，将其发送到 ClearML 平台
    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        # 如果 ClearML 客户端未初始化，则直接返回
        if self._clearml is None:
            return
        # 如果未初始化，则进行初始化设置
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)
        # 如果是全局进程的第一个进程（通常是主进程）
        if state.is_world_process_zero:
            # 定义评估数据的前缀和长度
            eval_prefix = "eval_"
            eval_prefix_len = len(eval_prefix)
            # 定义测试数据的前缀和长度
            test_prefix = "test_"
            test_prefix_len = len(test_prefix)
            # 定义单值标量的列表，这些值通常用于表示单个标量的日志信息
            single_value_scalars = [
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "total_flos",
                "epoch",
            ]
            # 遍历日志中的每个键值对
            for k, v in logs.items():
                # 如果值 v 是整数或浮点数
                if isinstance(v, (int, float)):
                    # 如果键 k 在单值标量列表中，则将其作为单值报告到 ClearML
                    if k in single_value_scalars:
                        self._clearml_task.get_logger().report_single_value(
                            name=k + ClearMLCallback.log_suffix, value=v
                        )
                    # 如果键 k 以评估数据前缀开头，则将其作为评估数据报告到 ClearML
                    elif k.startswith(eval_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title="eval" + ClearMLCallback.log_suffix,
                            series=k[eval_prefix_len:],
                            value=v,
                            iteration=state.global_step,
                        )
                    # 如果键 k 以测试数据前缀开头，则将其作为测试数据报告到 ClearML
                    elif k.startswith(test_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title="test" + ClearMLCallback.log_suffix,
                            series=k[test_prefix_len:],
                            value=v,
                            iteration=state.global_step,
                        )
                    # 否则，将其作为训练数据报告到 ClearML
                    else:
                        self._clearml_task.get_logger().report_scalar(
                            title="train" + ClearMLCallback.log_suffix,
                            series=k,
                            value=v,
                            iteration=state.global_step,
                        )
                else:
                    # 如果值 v 的类型不是整数或浮点数，则记录警告信息
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of ClearML logger's  report_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
    # 定义一个保存模型回调函数，根据特定条件执行保存操作
    def on_save(self, args, state, control, **kwargs):
        # 如果启用模型日志、存在 ClearML 任务并且当前进程为主进程
        if self._log_model and self._clearml_task and state.is_world_process_zero:
            # 根据全局步数创建检查点目录名
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 构建检查点在文件系统中的路径
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            # 定义保存的模型名，包括 ClearML 日志后缀
            name = ckpt_dir + ClearMLCallback.log_suffix
            # 输出日志，指示正在记录检查点信息
            logger.info(f"Logging checkpoint artifact `{name}`. This may take some time.")
            # 创建 ClearML 的 OutputModel 对象，关联到当前任务并设置名称
            output_model = self._clearml.OutputModel(task=self._clearml_task, name=name)
            output_model.connect(task=self._clearml_task, name=name)
            # 更新模型权重包，将指定路径的权重文件打包，指定迭代次数，并禁止自动删除文件
            output_model.update_weights_package(
                weights_path=artifact_path,
                target_filename=ckpt_dir,
                iteration=state.global_step,
                auto_delete_file=False,
            )
            # 将保存的模型对象添加到检查点列表中
            self._checkpoints_saved.append(output_model)
            # 当设置了保存总数限制并且当前保存的检查点数量超过限制时执行
            while args.save_total_limit and args.save_total_limit < len(self._checkpoints_saved):
                try:
                    # 尝试移除最早的检查点模型及其关联的权重文件
                    self._clearml.model.Model.remove(
                        self._checkpoints_saved[0],
                        delete_weights_file=True,
                        force=True,
                        raise_on_errors=True,
                    )
                except Exception as e:
                    # 记录警告，指示在超过保存限制后无法移除检查点的错误信息
                    logger.warning(
                        "Could not remove checkpoint `{}` after going over the `save_total_limit`. Error is: {}".format(
                            self._checkpoints_saved[0].name, e
                        )
                    )
                    # 中断循环，保持检查点列表不变
                    break
                # 移除成功后，更新检查点列表，去除最早的一个检查点对象
                self._checkpoints_saved = self._checkpoints_saved[1:]

    # 将训练参数复制为超参数，并将其传递给 ClearML 任务
    def _copy_training_args_as_hparams(self, training_args, prefix):
        # 将训练参数对象中的每个字段转换为字典，排除以 "_token" 结尾的字段
        as_dict = {
            field.name: getattr(training_args, field.name)
            for field in fields(training_args)
            if field.init and not field.name.endswith("_token")
        }
        # 扁平化字典，将所有键转换为字符串
        flat_dict = {str(k): v for k, v in self._clearml.utilities.proxy_object.flatten_dictionary(as_dict).items()}
        # 将扁平化后的字典作为超参数设置到 ClearML 任务中
        self._clearml_task._arguments.copy_from_dict(flat_dict, prefix=prefix)
# 定义一个名为 FlyteCallback 的类，继承自 TrainerCallback 类
class FlyteCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Flyte](https://flyte.org/).
    NOTE: This callback only works within a Flyte task.

    Args:
        save_log_history (`bool`, *optional*, defaults to `True`):
            When set to True, the training logs are saved as a Flyte Deck.

        sync_checkpoints (`bool`, *optional*, defaults to `True`):
            When set to True, checkpoints are synced with Flyte and can be used to resume training in the case of an
            interruption.

    Example:

    ```python
    # Note: This example skips over some setup steps for brevity.
    from flytekit import current_context, task


    @task
    def train_hf_transformer():
        cp = current_context().checkpoint
        trainer = Trainer(..., callbacks=[FlyteCallback()])
        output = trainer.train(resume_from_checkpoint=cp.restore())
    ```
    """

    # 初始化方法，接受两个可选参数：save_log_history 和 sync_checkpoints
    def __init__(self, save_log_history: bool = True, sync_checkpoints: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        
        # 检查 flytekit 是否可用，如果不可用则抛出 ImportError
        if not is_flytekit_available():
            raise ImportError("FlyteCallback requires flytekit to be installed. Run `pip install flytekit`.")
        
        # 检查是否安装了 flytekitplugins-deck-standard 和 pandas，如果未安装，则记录警告并将 save_log_history 设置为 False
        if not is_flyte_deck_standard_available() or not is_pandas_available():
            logger.warning(
                "Syncing log history requires both flytekitplugins-deck-standard and pandas to be installed. "
                "Run `pip install flytekitplugins-deck-standard pandas` to enable this feature."
            )
            save_log_history = False
        
        # 导入当前上下文的 checkpoint 对象
        from flytekit import current_context
        self.cp = current_context().checkpoint
        
        # 初始化实例变量
        self.save_log_history = save_log_history
        self.sync_checkpoints = sync_checkpoints

    # 在保存方法回调时执行的操作
    def on_save(self, args, state, control, **kwargs):
        # 如果 sync_checkpoints 为 True，并且当前状态的全局进程是零（即主进程）
        if self.sync_checkpoints and state.is_world_process_zero:
            # 构建检查点目录和存储路径
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)

            # 记录信息，将检查点同步到 Flyte。这可能需要一些时间。
            logger.info(f"Syncing checkpoint in {ckpt_dir} to Flyte. This may take time.")
            self.cp.save(artifact_path)

    # 在训练结束时执行的操作
    def on_train_end(self, args, state, control, **kwargs):
        # 如果 save_log_history 为 True
        if self.save_log_history:
            # 导入 pandas、Deck 类以及 TableRenderer
            import pandas as pd
            from flytekit import Deck
            from flytekitplugins.deck.renderer import TableRenderer

            # 创建日志历史的 DataFrame
            log_history_df = pd.DataFrame(state.log_history)
            
            # 创建一个名为 "Log History" 的 Flyte Deck，使用 TableRenderer 将 DataFrame 转换为 HTML 格式
            Deck("Log History", TableRenderer().to_html(log_history_df))


# DVCLiveCallback 类暂时省略注释
class DVCLiveCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [DVCLive](https://www.dvc.org/doc/dvclive).

    Use the environment variables below in `setup` to configure the integration. To customize this callback beyond
    those environment variables, see [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).
    """
    Args:
        live (`dvclive.Live`, *optional*, defaults to `None`):
            Optional Live instance. If None, a new instance will be created using **kwargs.
        log_model (Union[Literal["all"], bool], *optional*, defaults to `None`):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True`,
            the final checkpoint is logged at the end of training. If set to `"all"`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
    """
    
    # DVCLiveCallback 类的初始化方法，用于设置 DVCLive 相关的参数和实例
    def __init__(
        self,
        live: Optional[Any] = None,
        log_model: Optional[Union[Literal["all"], bool]] = None,
        **kwargs,
    ):
        # 检查 dvclive 是否可用，如果不可用则抛出运行时错误
        if not is_dvclive_available():
            raise RuntimeError("DVCLiveCallback requires dvclive to be installed. Run `pip install dvclive`.")
        # 导入 dvclive.Live
        from dvclive import Live

        # 初始化实例变量
        self._initialized = False
        self.live = None
        
        # 根据 live 参数的类型来设置 self.live 实例
        if isinstance(live, Live):
            self.live = live
        elif live is not None:
            raise RuntimeError(f"Found class {live.__class__} for live, expected dvclive.Live")

        # 设置日志模型的方式
        self._log_model = log_model
        if self._log_model is None:
            # 从环境变量 HF_DVCLIVE_LOG_MODEL 获取日志模型设置
            log_model_env = os.getenv("HF_DVCLIVE_LOG_MODEL", "FALSE")
            if log_model_env.upper() in ENV_VARS_TRUE_VALUES:
                self._log_model = True
            elif log_model_env.lower() == "all":
                self._log_model = "all"

    # 设置 DVCLiveCallback 的初始化状态，并在主进程中初始化 dvclive.Live 实例并记录参数
    def setup(self, args, state, model):
        """
        Setup the optional DVCLive integration. To customize this callback beyond the environment variables below, see
        [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

        Environment:
        - **HF_DVCLIVE_LOG_MODEL** (`str`, *optional*):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True` or
            *1*, the final checkpoint is logged at the end of training. If set to `all`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
        """
        # 导入 dvclive.Live
        from dvclive import Live

        # 设置初始化状态为 True
        self._initialized = True
        
        # 如果是主进程中的第一个进程，则初始化 dvclive.Live 实例并记录参数
        if state.is_world_process_zero:
            if not self.live:
                self.live = Live()
            self.live.log_params(args.to_dict())

    # 在训练开始时检查是否初始化，如果未初始化则调用 setup 方法初始化
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
    # 当日志事件发生时调用，处理日志相关操作
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 如果对象尚未初始化，则进行初始化设置
        if not self._initialized:
            self.setup(args, state, model)
        # 如果是全局进程中的主进程
        if state.is_world_process_zero:
            # 导入必要的库：Metric 类和标准化指标名称的工具函数
            from dvclive.plots import Metric
            from dvclive.utils import standardize_metric_name

            # 遍历日志中的键值对
            for key, value in logs.items():
                # 检查当前值是否可记录为 Metric
                if Metric.could_log(value):
                    # 使用标准化的名称记录指标到 DVCLive
                    self.live.log_metric(standardize_metric_name(key, "dvclive.huggingface"), value)
                else:
                    # 如果记录的值不符合 Metric 要求，发出警告并丢弃该属性
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{value}" of type {type(value)} for key "{key}" as a scalar. '
                        "This invocation of DVCLive's Live.log_metric() "
                        "is incorrect so we dropped this attribute."
                    )
            # 在 DVCLive 中记录下一个步骤
            self.live.next_step()

    # 当保存事件发生时调用，处理保存模型相关操作
    def on_save(self, args, state, control, **kwargs):
        # 如果设置为保存所有模型，并且对象已初始化且是全局进程中的主进程
        if self._log_model == "all" and self._initialized and state.is_world_process_zero:
            # 将输出目录作为 artifact 记录到 DVCLive 中
            self.live.log_artifact(args.output_dir)

    # 当训练结束事件发生时调用，处理训练结束相关操作
    def on_train_end(self, args, state, control, **kwargs):
        # 如果对象已初始化且是全局进程中的主进程
        if self._initialized and state.is_world_process_zero:
            # 导入 Transformers 库中的 Trainer 类
            from transformers.trainer import Trainer

            # 如果设置为保存模型
            if self._log_model is True:
                # 创建一个虚拟 Trainer 对象用于保存模型
                fake_trainer = Trainer(args=args, model=kwargs.get("model"), tokenizer=kwargs.get("tokenizer"))
                # 根据设置选择保存最佳模型还是最后模型
                name = "best" if args.load_best_model_at_end else "last"
                output_dir = os.path.join(args.output_dir, name)
                # 保存模型到指定目录
                fake_trainer.save_model(output_dir)
                # 将保存的模型目录作为 artifact 记录到 DVCLive 中
                self.live.log_artifact(output_dir, name=name, type="model", copy=True)
            # 在 DVCLive 中结束记录
            self.live.end()
# 定义一个映射，将集成名称映射到相应的回调类
INTEGRATION_TO_CALLBACK = {
    "azure_ml": AzureMLCallback,
    "comet_ml": CometCallback,
    "mlflow": MLflowCallback,
    "neptune": NeptuneCallback,
    "tensorboard": TensorBoardCallback,
    "wandb": WandbCallback,
    "codecarbon": CodeCarbonCallback,
    "clearml": ClearMLCallback,
    "dagshub": DagsHubCallback,
    "flyte": FlyteCallback,
    "dvclive": DVCLiveCallback,
}

# 根据给定的报告集成列表，返回对应的回调类列表
def get_reporting_integration_callbacks(report_to):
    # 遍历报告集成列表中的每个集成
    for integration in report_to:
        # 如果集成不在预定义的映射中，则引发 ValueError 异常
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )

    # 返回一个包含各个集成对应的回调类的列表
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
```