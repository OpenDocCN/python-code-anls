# `.\transformers\integrations\integration_utils.py`

```
# 版权声明和许可信息
"""
Integrations with other Python libraries.
"""
# 导入模块
import functools  # 用于创建高阶函数的工具
import importlib.metadata  # 用于检索安装的 Python 包的元数据信息
import importlib.util  # 提供有关模块的一般功能
import json  # 用于 JSON 数据的编码和解码
import numbers  # 提供数字抽象基类
import os  # 提供与操作系统交互的功能
import pickle  # 用于 Python 对象的序列化和反序列化
import shutil  # 提供高级文件操作功能
import sys  # 提供与 Python 解释器交互的变量和函数
import tempfile  # 提供用于临时文件和目录的功能
from dataclasses import asdict  # 从数据类实例创建一个字典，将字段名映射到它们的值
from pathlib import Path  # 提供了一个用于处理文件系统路径的类
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union  # 提供静态类型的支持

import numpy as np  # 数组处理工具

# 导入自定义的模块和函数
from .. import __version__ as version  # 导入当前包的版本信息
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging  # 导入一些实用函数和变量


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 如果 Torch 可用，导入 Torch 模块
if is_torch_available():
    import torch  # 导入 PyTorch 模块

# 检查是否存在 comet_ml，并且环境变量 COMET_MODE 不为 DISABLED
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    try:
        import comet_ml  # 导入 comet_ml 模块

        if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
            _has_comet = True
        else:
            if os.getenv("COMET_MODE", "").upper() != "DISABLED":
                logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
            _has_comet = False
    except (ImportError, ValueError):
        _has_comet = False

# 检查是否存在 neptune 或 neptune-client
_has_neptune = (
    importlib.util.find_spec("neptune") is not None or importlib.util.find_spec("neptune-client") is not None
)
# 如果是类型检查，并且存在 neptune，则记录 neptune 版本信息
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

# 导入 TrainerCallback 相关的模块
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
# 导入 TrainerUtils 相关的模块
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
# 导入 ParallelMode 枚举
from ..training_args import ParallelMode  # noqa: E402
# 导入 ENV_VARS_TRUE_VALUES 和 is_torch_tpu_available 函数
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402

# Integration functions:
# 判断是否可用 wandb 模块
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
``` 
    # 检查环境变量中是否设置了"WANDB_DISABLED"，如果设置且值为环境变量真值列表中的值之一，则进入条件判断
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        # 若"WANDB_DISABLED"环境变量被使用，给出警告，说明此用法已被弃用，并在下一个版本中将被移除
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        # 返回 False，表示禁用了 wandb
        return False
    # 使用 importlib.util.find_spec 函数来检查是否可以导入 wandb 模块，如果能导入，说明 wandb 未被禁用
    return importlib.util.find_spec("wandb") is not None
# 检查是否安装了 clearml 库
def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


# 检查是否 comet 可用
def is_comet_available():
    return _has_comet


# 检查是否安装了 tensorboard 或 tensorboardX 库
def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


# 检查是否安装了 optuna 库
def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


# 检查是否安装了 ray 库
def is_ray_available():
    return importlib.util.find_spec("ray") is not None


# 检查是否安装了 ray.tune 库
def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


# 检查是否安装了 sigopt 库
def is_sigopt_available():
    return importlib.util.find_spec("sigopt") is not None


# 检查是否安装了 azureml 库
def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


# 检查是否安装了 mlflow 库
def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


# 检查是否安装了 dagshub 和 mlflow 库
def is_dagshub_available():
    return None not in [importlib.util.find_spec("dagshub"), importlib.util.find_spec("mlflow")]


# 检查是否 neptune 可用
def is_neptune_available():
    return _has_neptune


# 检查是否安装了 codecarbon 库
def is_codecarbon_available():
    return importlib.util.find_spec("codecarbon") is not None


# 检查是否安装了 flytekit 库
def is_flytekit_available():
    return importlib.util.find_spec("flytekit") is not None


# 检查是否安装了 flytekitplugins.deck 库
def is_flyte_deck_standard_available():
    if not is_flytekit_available():
        return False
    return importlib.util.find_spec("flytekitplugins.deck") is not None


# 检查是否安装了 dvclive 库
def is_dvclive_available():
    return importlib.util.find_spec("dvclive") is not None


# 获取超参数
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

    if is_wandb_available():
        if isinstance(trial, dict):
            return trial

    raise RuntimeError(f"Unknown type for trial {trial.__class__}")


# 运行 optuna 超参数搜索
def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna
    # 检查当前进程是否为主进程（process_index为0）
    if trainer.args.process_index == 0:
    
        # 定义内部函数_objective，用于作为优化目标函数
        def _objective(trial, checkpoint_dir=None):
            # 初始化变量checkpoint为None
            checkpoint = None
            # 如果提供了checkpoint_dir，则尝试从中获取最新的checkpoint
            if checkpoint_dir:
                # 遍历checkpoint_dir下的所有子目录
                for subdir in os.listdir(checkpoint_dir):
                    # 如果子目录以PREFIX_CHECKPOINT_DIR开头，则将其作为checkpoint
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            # 重置trainer的objective为None
            trainer.objective = None
            # 如果world_size大于1，则需要进行分布式训练
            if trainer.args.world_size > 1:
                # 如果并行模式不是DISTRIBUTED，则抛出RuntimeError
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
                # 设置超参数搜索
                trainer._hp_search_setup(trial)
                # 将trainer的args对象序列化并广播给所有进程
                torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                # 开始训练，从checkpoint处恢复
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                # 开始训练，从checkpoint处恢复，并传入trial对象以供记录
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            # 如果训练循环中没有进行过评估
            if getattr(trainer, "objective", None) is None:
                # 进行一次评估
                metrics = trainer.evaluate()
                # 计算训练结果的目标值
                trainer.objective = trainer.compute_objective(metrics)
            # 返回训练结果的目标值
            return trainer.objective
    
        # 从kwargs中弹出timeout和n_jobs参数
        timeout = kwargs.pop("timeout", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        # 将direction转换为列表，如果direction已经是列表则不变
        directions = direction if isinstance(direction, list) else None
        # 如果directions存在则将direction置为None
        direction = None if directions is not None else direction
        # 创建一个Optuna Study对象
        study = optuna.create_study(direction=direction, directions=directions, **kwargs)
        # 使用Optuna进行超参数搜索优化
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        # 如果不是多目标优化，则返回最佳的运行结果
        if not study._is_multi_objective():
            best_trial = study.best_trial
            return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
        # 如果是多目标优化，则返回多个最佳的运行结果
        else:
            best_trials = study.best_trials
            return [BestRun(str(best.number), best.values, best.params) for best in best_trials]
    
    # 如果当前进程不是主进程，则执行以下代码
    else:
        # 对于指定的试验次数，进行模型训练并优化
        for i in range(n_trials):
            # 重置trainer的objective为None
            trainer.objective = None
            # 将trainer的args对象序列化，并广播给所有进程
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            # 从序列化的args对象中恢复参数
            args = pickle.loads(bytes(args_main_rank))
            # 将args中除了"local_rank"以外的键值对设置为trainer.args的属性
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            # 开始训练，从头开始
            trainer.train(resume_from_checkpoint=None)
            # 如果训练循环中没有进行过评估
            if getattr(trainer, "objective", None) is None:
                # 进行一次评估
                metrics = trainer.evaluate()
                # 计算训练结果的目标值
                trainer.objective = trainer.compute_objective(metrics)
        # 返回None
        return None
# 运行超参数搜索的函数，使用 Ray 来并行执行
def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    # 导入 Ray 库
    import ray
    # 导入 Ray 训练相关模块
    import ray.train

    # 定义目标函数，用于执行单个训练试验
    def _objective(trial: dict, local_trainer):
        try:
            # 尝试导入 notebook 相关进度条回调函数
            from transformers.utils.notebook import NotebookProgressCallback
            # 如果成功导入，则从本地训练器中移除笔记本进度回调函数并添加进度回调函数
            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            pass

        # 重置本地训练器的目标为 None
        local_trainer.objective = None

        # 获取检查点对象
        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            # 在试验恢复时，本地训练器的目标会被重置为 None
            # 如果 `local_trainer.train` 是一个空操作（训练已经达到目标的轮数/步数），
            # 这将触发不必要的额外检查点在训练结束时。
            # -> 在恢复时设置目标为一个虚拟值以解决此问题。
            local_trainer.objective = "objective"

            # 将检查点目录下的检查点路径作为参数，从该路径恢复训练
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = next(Path(checkpoint_dir).glob(f"{PREFIX_CHECKPOINT_DIR}*")).as_posix()
                local_trainer.train(resume_from_checkpoint=checkpoint_path, trial=trial)
        else:
            # 如果没有检查点，直接进行训练
            local_trainer.train(trial=trial)

        # 如果在训练循环期间没有进行评估
        if getattr(local_trainer, "objective", None) is None:
            # 计算评估指标
            metrics = local_trainer.evaluate()
            # 设置本地训练器的目标为计算得到的目标值
            local_trainer.objective = local_trainer.compute_objective(metrics)

            # 更新评估指标字典，包括目标值和标记训练完成的键值对
            metrics.update({"objective": local_trainer.objective, "done": True})

            # 使用临时目录创建临时检查点并报告指标
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                local_trainer._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                ray.train.report(metrics, checkpoint=checkpoint)

    # 如果不跳过内存指标跟踪，则将警告消息记录到日志中，因为内存跟踪器不可序列化
    if not trainer._memory_tracker.skip_memory_metrics:
        from ..trainer_utils import TrainerMemoryTracker

        logger.warning(
            "Memory tracking for your Trainer is currently "
            "enabled. Automatically disabling the memory tracker "
            "since the memory tracker is not serializable."
        )
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)

    # 在 Ray 超参数搜索期间，模型和 TensorBoard writer 不可序列化，因此需要移除它们
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None

    # 设置默认的 `resources_per_trial`
```  
    # 检查是否在参数中包含了"resources_per_trial"
    if "resources_per_trial" not in kwargs:
        # 默认每个 trial 使用 1 个 CPU 和 1 个 GPU（如果有的话）
        kwargs["resources_per_trial"] = {"cpu": 1}
        # 如果有 GPU，则设置每个 trial 使用 1 个 GPU
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        # 设置资源信息消息
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        # 输出日志信息
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # 确保每个训练器只使用为每个 trial 分配的 GPU
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # 设置默认的 `progress_reporter`
    if "progress_reporter" not in kwargs:
        # 导入 CLIReporter
        from ray.tune import CLIReporter
        # 设置默认的 `progress_reporter` 为 CLIReporter，并指定度量列为 ["objective"]
        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])

    if "scheduler" in kwargs:
        # 导入不同的调度器类
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # 检查是否需要中间报告的调度器是否启用了 `do_eval` 和 `eval_during_training`
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            # 如果未启用评估过程或评估策略为 IntervalStrategy.NO，则抛出异常
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    # 使用 ray.tune.with_parameters 将 _objective 函数与 local_trainer 绑定
    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    # 使用 functools.wraps 装饰器
    @functools.wraps(trainable)
    # 定义一个函数 dynamic_modules_import_trainable，用于动态导入模块并执行训练
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        # 检查是否可用 datasets 模块
        if is_datasets_available():
            # 导入 datasets.load 模块
            import datasets.load
            # 获取动态模块路径
            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), "__init__.py")
            # 根据路径加载动态模块
            spec = importlib.util.spec_from_file_location("datasets_modules", dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        # 调用传入的 trainable 函数，传递参数并执行训练
        return trainable(*args, **kwargs)

    # 如果 trainable 函数有特殊属性 "__mixins__"，则将 dynamic_modules_import_trainable 函数的 "__mixins__" 属性设置为相同值
    if hasattr(trainable, "__mixins__"):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__

    # 使用 ray.tune.run 函数执行训练任务
    analysis = ray.tune.run(
        dynamic_modules_import_trainable,
        # 使用 trainer.hp_space(None) 配置超参数空间
        config=trainer.hp_space(None),
        # 设置试验次数为 n_trials
        num_samples=n_trials,
        **kwargs,  # 传递额外的参数
    )
    # 获取最佳试验
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3], scope=trainer.args.ray_scope)
    # 创建 BestRun 对象，包括最佳试验的信息
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config, analysis)
    # 如果有 _tb_writer 回调函数，则将其添加到训练器中
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    # 返回最佳试验的结果
    return best_run
# 运行 Sigopt 的超参数搜索，返回最佳运行结果
def run_hp_search_sigopt(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    # 导入 Sigopt 库
    import sigopt

    # 如果没有指定超参数搜索的方向，则抛出错误
    else:
        # 循环执行超参数搜索的次数
        for i in range(n_trials):
            # 重置训练器的目标函数
            trainer.objective = None
            # 将训练器的参数转换为字节流并获取其排名
            args_main_rank = list(pickle.dumps(trainer.args))
            # 如果训练器的并行模式不是分布式，则抛出错误
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
            # 将参数字节流广播到所有进程
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            # 将字节流转换为参数对象
            args = pickle.loads(bytes(args_main_rank))
            # 设置训练器的参数
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            # 开始训练
            trainer.train(resume_from_checkpoint=None)
            # 如果训练循环期间没有进行评估
            if getattr(trainer, "objective", None) is None:
                # 进行评估
                metrics = trainer.evaluate()
                # 计算目标函数
                trainer.objective = trainer.compute_objective(metrics)
        # 返回空值
        return None


# 运行 Wandb 的超参数搜索，返回最佳运行结果
def run_hp_search_wandb(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    # 导入 Wandb 库
    from ..integrations import is_wandb_available

    # 如果 Wandb 不可用，则抛出 ImportError
    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")
    # 导入 Wandb 库
    import wandb

    # 如果训练器的回调中没有 WandbCallback，则添加之
    reporting_to_wandb = False
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            reporting_to_wandb = True
            break
    if not reporting_to_wandb:
        trainer.add_callback(WandbCallback())
    # 设置报告目标为 Wandb
    trainer.args.report_to = ["wandb"]
    # 初始化最佳试验信息
    best_trial = {"run_id": None, "objective": None, "hyperparameters": None}
    # 获取超参数搜索的参数
    sweep_id = kwargs.pop("sweep_id", None)
    project = kwargs.pop("project", None)
    name = kwargs.pop("name", None)
    entity = kwargs.pop("entity", None)
    metric = kwargs.pop("metric", "eval/loss")

    # 创建超参数搜索配置
    sweep_config = trainer.hp_space(None)
    # 设置搜索的目标方向
    sweep_config["metric"]["goal"] = direction
    # 设置搜索的目标指标
    sweep_config["metric"]["name"] = metric
    # 如果指定了名称，则设置超参数搜索的名称
    if name:
        sweep_config["name"] = name
    # 定义目标函数，用于训练和评估模型，并记录最佳结果的相关信息
    def _objective():
        # 如果已经有 WandB 的运行记录，则使用现有的运行，否则初始化一个新的运行
        run = wandb.run if wandb.run else wandb.init()
        # 将当前运行的名称作为试验名称存储在训练器的状态中
        trainer.state.trial_name = run.name
        # 更新配置，包括分配、指标等信息
        run.config.update({"assignments": {}, "metric": metric})
        config = wandb.config
    
        # 将训练器的目标设为 None，用于存储训练过程中的评估指标
        trainer.objective = None
    
        # 开始训练模型，将配置参数作为字典传递给训练器
        trainer.train(resume_from_checkpoint=None, trial=vars(config)["_items"])
        # 如果在训练循环中没有进行任何评估
        if getattr(trainer, "objective", None) is None:
            # 对模型进行评估
            metrics = trainer.evaluate()
            # 计算模型的评估指标
            trainer.objective = trainer.compute_objective(metrics)
            # 将评估指标重写为日志格式
            format_metrics = rewrite_logs(metrics)
            # 如果指定的指标不在重写后的指标中，则发出警告
            if metric not in format_metrics:
                logger.warning(
                    f"Provided metric {metric} not found. This might result in unexpected sweeps charts. The available"
                    f" metrics are {format_metrics.keys()}"
                )
        # 检查当前模型是否优于历史最佳模型
        best_score = False
        if best_trial["run_id"] is not None:
            if direction == "minimize":
                best_score = trainer.objective < best_trial["objective"]
            elif direction == "maximize":
                best_score = trainer.objective > best_trial["objective"]
    
        # 如果当前模型更好或者历史最佳模型不存在，则更新最佳模型信息
        if best_score or best_trial["run_id"] is None:
            best_trial["run_id"] = run.id
            best_trial["objective"] = trainer.objective
            best_trial["hyperparameters"] = dict(config)
    
        # 返回当前模型的评估指标
        return trainer.objective
    
    # 如果没有指定参数的话，使用 wandb 根据配置文件创建一个新的 Sweep
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity) if not sweep_id else sweep_id
    # 打印 WandB Sweep 的 ID
    logger.info(f"wandb sweep id - {sweep_id}")
    # 使用 WandB 的 agent 方法运行 Sweep 实验
    wandb.agent(sweep_id, function=_objective, count=n_trials)
    
    # 返回最佳运行的 ID、评估指标和超参数
    return BestRun(best_trial["run_id"], best_trial["objective"], best_trial["hyperparameters"])
# 获取可用的报告集成列表
def get_available_reporting_integrations():
    # 初始化一个空列表用于存储集成
    integrations = []
    # 检查 AzureML 是否可用且 MLflow 不可用时，添加 AzureML 到集成列表
    if is_azureml_available() and not is_mlflow_available():
        integrations.append("azure_ml")
    # 检查 Comet 是否可用，添加 Comet 到集成列表
    if is_comet_available():
        integrations.append("comet_ml")
    # 检查 Dagshub 是否可用，添加 Dagshub 到集成列表
    if is_dagshub_available():
        integrations.append("dagshub")
    # 检查 DVC Live 是否可用，添加 DVC Live 到集成列表
    if is_dvclive_available():
        integrations.append("dvclive")
    # 检查 MLflow 是否可用，添加 MLflow 到集成列表
    if is_mlflow_available():
        integrations.append("mlflow")
    # 检查 Neptune 是否可用，添加 Neptune 到集成列表
    if is_neptune_available():
        integrations.append("neptune")
    # 检查 TensorBoard 是否可用，添加 TensorBoard 到集成列表
    if is_tensorboard_available():
        integrations.append("tensorboard")
    # 检查 Weights & Biases 是否可用，添加 Weights & Biases 到集成列表
    if is_wandb_available():
        integrations.append("wandb")
    # 检查 CodeCarbon 是否可用，添加 CodeCarbon 到集成列表
    if is_codecarbon_available():
        integrations.append("codecarbon")
    # 检查 ClearML 是否可用，添加 ClearML 到集成列表
    if is_clearml_available():
        integrations.append("clearml")
    # 返回集成列表
    return integrations


# 重写日志键名
def rewrite_logs(d):
    # 初始化一个新的字典用于存储重写后的键名
    new_d = {}
    # 定义评估前缀和测试前缀
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    # 遍历原始字典的键值对
    for k, v in d.items():
        # 如果键名以评估前缀开头，将键名改为以"eval/"开头
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        # 如果键名以测试前缀开头，将键名改为以"test/"开头
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        # 其他情况下，将键名改为以"train/"开头
        else:
            new_d["train/" + k] = v
    # 返回重写后的字典
    return new_d


# TensorBoard 回调类
class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    # 初始化方法
    def __init__(self, tb_writer=None):
        # 检查是否有 TensorBoard 可用
        has_tensorboard = is_tensorboard_available()
        # 如果没有 TensorBoard，抛出运行时错误
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        # 如果有 TensorBoard
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401
                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        # 设置 TensorBoard 写入器
        self.tb_writer = tb_writer

    # 初始化摘要写入器方法
    def _init_summary_writer(self, args, log_dir=None):
        # 如果未设置日志目录，则使用参数中的日志目录
        log_dir = log_dir or args.logging_dir
        # 如果摘要写入器不为空
        if self._SummaryWriter is not None:
            # 实例化摘要写入器
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)
    # 当训练开始时触发的回调函数，根据参数、状态和控制信息执行相应操作
    def on_train_begin(self, args, state, control, **kwargs):
        # 如果不是世界进程的第一个进程，则直接返回
        if not state.is_world_process_zero:
            return

        log_dir = None

        # 如果是超参数搜索状态
        if state.is_hyper_param_search:
            trial_name = state.trial_name
            # 如果试验名称不为空，则设置日志目录为日志目录和试验名称的组合
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        # 如果 Tensorboard 写入器为空，则初始化摘要写入器
        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        # 如果 Tensorboard 写入器不为空
        if self.tb_writer is not None:
            # 将参数转换为 JSON 字符串并添加到 Tensorboard 中
            self.tb_writer.add_text("args", args.to_json_string())
            # 如果 kwargs 中包含 "model"
            if "model" in kwargs:
                model = kwargs["model"]
                # 如果模型具有配置信息
                if hasattr(model, "config") and model.config is not None:
                    # 将模型配置信息转换为 JSON 字符串并添加到 Tensorboard 中
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

    # 当记录日志时触发的回调函数，根据参数、状态和控制信息执行相应操作
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 如果不是世界进程的第一个进程，则直接返回
        if not state.is_world_process_zero:
            return

        # 如果 Tensorboard 写入器为空，则初始化摘要写入器
        if self.tb_writer is None:
            self._init_summary_writer(args)

        # 如果 Tensorboard 写入器不为空
        if self.tb_writer is not None:
            # 重写日志信息
            logs = rewrite_logs(logs)
            # 遍历日志字典
            for k, v in logs.items():
                # 如果值是整数或浮点数，则将其添加为标量到 Tensorboard 中
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    # 如果值不是整数或浮点数，则记录警告信息
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            # 刷新 Tensorboard 写入器
            self.tb_writer.flush()

    # 当训练结束时触发的回调函数，根据参数、状态和控制信息执行相应操作
    def on_train_end(self, args, state, control, **kwargs):
        # 如果 Tensorboard 写入器存在
        if self.tb_writer:
            # 关闭 Tensorboard 写入器并将其设置为 None
            self.tb_writer.close()
            self.tb_writer = None
class WandbCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        # 检查是否安装了 wandb 库
        has_wandb = is_wandb_available()
        # 如果没有安装 wandb，则抛出运行时错误
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        # 如果安装了 wandb，则导入 wandb 库
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # 检查是否需要记录模型
        if os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"}):
            # 发出弃用警告
            DeprecationWarning(
                f"Setting `WANDB_LOG_MODEL` as {os.getenv('WANDB_LOG_MODEL')} is deprecated and will be removed in "
                "version 5 of transformers. Use one of `'end'` or `'checkpoint'` instead."
            )
            logger.info(f"Setting `WANDB_LOG_MODEL` from {os.getenv('WANDB_LOG_MODEL')} to `end` instead")
            self._log_model = "end"
        else:
            self._log_model = os.getenv("WANDB_LOG_MODEL", "false").lower()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果 wandb 为空，则直接返回
        if self._wandb is None:
            return
        # 检查是否为超参数搜索
        hp_search = state.is_hyper_param_search
        # 如果是超参数搜索，则结束 wandb 进程，重置初始化状态，清空运行名称
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        # 如果未初始化，则进行设置
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
    # 在训练结束时执行的回调函数，用于保存模型和日志
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # 如果没有初始化 wandb，则直接返回
        if self._wandb is None:
            return
        # 如果需要在训练结束或检查点时记录模型，并且已经初始化，并且是主进程
        if self._log_model in ("end", "checkpoint") and self._initialized and state.is_world_process_zero:
            # 导入 Trainer 类
            from ..trainer import Trainer

            # 创建一个虚拟 Trainer 对象
            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            # 使用临时目录保存模型
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                # 根据条件选择要记录的元数据
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
                # 记录日志信息
                logger.info("Logging model artifacts. ...")
                # 根据运行名称创建模型名称
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                # 创建一个 wandb Artifact 对象
                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                # 遍历临时目录下的文件，并将其写入 Artifact
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                # 记录 Artifact
                self._wandb.run.log_artifact(artifact)

    # 在记录日志时执行的回调函数
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 如果没有初始化 wandb，则直接返回
        if self._wandb is None:
            return
        # 如果尚未初始化��则进行初始化
        if not self._initialized:
            self.setup(args, state, model)
        # 如果是主进程
        if state.is_world_process_zero:
            # 重写日志信息
            logs = rewrite_logs(logs)
            # 记录日志信息到 wandb
            self._wandb.log({**logs, "train/global_step": state.global_step})
    # 当保存模型时触发的方法，接受参数 args、state、control 和其他关键字参数 kwargs
    def on_save(self, args, state, control, **kwargs):
        # 如果日志模型为 "checkpoint"，并且已经初始化，并且状态是世界进程的第一个进程
        if self._log_model == "checkpoint" and self._initialized and state.is_world_process_zero:
            # 创建检查点元数据，包括不是以下划线开头的数值类型的所有 self._wandb.summary 中的键值对
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }

            # 创建检查点目录名，格式为 "checkpoint-全局步数"
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 构建检查点路径，即将检查点目录与输出目录连接
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            # 记录日志，指示正在将检查点工件记录在指定的目录中
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            # 根据情况确定检查点名称，如果运行名称为 None 或与输出目录相同，则使用运行 ID 作为名称，否则使用运行名称
            checkpoint_name = (
                f"checkpoint-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"checkpoint-{self._wandb.run.name}"
            )
            # 创建 Wandb 工件对象，类型为 "model"，并附带检查点元数据
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            # 向工件中添加指定目录下的所有文件
            artifact.add_dir(artifact_path)
            # 使用全局步数作为别名记录工件
            self._wandb.log_artifact(artifact, aliases=[f"checkpoint-{state.global_step}"])
# 定义一个 CometCallback 类，继承自 TrainerCallback 类
class CometCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Comet ML](https://www.comet.ml/site/).
    """

    # 初始化方法
    def __init__(self):
        # 检查是否安装了 comet-ml，如果没有则抛出运行时错误
        if not _has_comet:
            raise RuntimeError("CometCallback requires comet-ml to be installed. Run `pip install comet-ml`.")
        self._initialized = False
        self._log_assets = False

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
        self._initialized = True
        # 获取环境变量 COMET_LOG_ASSETS 的值，默认为 FALSE
        log_assets = os.getenv("COMET_LOG_ASSETS", "FALSE").upper()
        # 如果 log_assets 为 TRUE 或 1，则设置 _log_assets 为 True
        if log_assets in {"TRUE", "1"}:
            self._log_assets = True
        # 如果是主进程
        if state.is_world_process_zero:
            # 获取环境变量 COMET_MODE 的值，默认为 ONLINE
            comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
            experiment = None
            experiment_kwargs = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
            # 如果 comet_mode 为 ONLINE
            if comet_mode == "ONLINE":
                experiment = comet_ml.Experiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml online logging enabled")
            # 如果 comet_mode 为 OFFLINE
            elif comet_mode == "OFFLINE":
                experiment_kwargs["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
                experiment = comet_ml.OfflineExperiment(**experiment_kwargs)
                experiment.log_other("Created from", "transformers")
                logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
            # 如果实验对象不为空
            if experiment is not None:
                experiment._set_model_graph(model, framework="transformers")
                experiment._log_parameters(args, prefix="args/", framework="transformers")
                # 如果模型有 config 属性，则记录 config 参数
                if hasattr(model, "config"):
                    experiment._log_parameters(model.config, prefix="config/", framework="transformers")

    # 训练开始时的回调方法
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果未初始化，则调用 setup 方法进行初始化
        if not self._initialized:
            self.setup(args, state, model)
    # 当日志触发时执行的回调函数，用于记录训练过程中的日志信息
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 如果实例尚未初始化，则进行初始化操作
        if not self._initialized:
            self.setup(args, state, model)
        # 如果当前进程是全局的第一个进程
        if state.is_world_process_zero:
            # 获取全局的 Comet 实验对象
            experiment = comet_ml.config.get_global_experiment()
            # 如果实验对象存在
            if experiment is not None:
                # 向 Comet 实验对象记录指标信息
                experiment._log_metrics(logs, step=state.global_step, epoch=state.epoch, framework="transformers")

    # 当训练结束时执行的回调函数
    def on_train_end(self, args, state, control, **kwargs):
        # 如果实例已经初始化且当前进程是全局的第一个进程
        if self._initialized and state.is_world_process_zero:
            # 获取全局的 Comet 实验对象
            experiment = comet_ml.config.get_global_experiment()
            # 如果实验对象存在
            if experiment is not None:
                # 如果设置了日志资产的标志为真
                if self._log_assets is True:
                    # 输出日志信息
                    logger.info("Logging checkpoints. This may take time.")
                    # 将输出目录作为资产记录到 Comet 实验对象中，递归记录子目录，记录文件名，记录步骤信息
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
        # 如果未安装 AzureML 库，则引发运行时错误
        if not is_azureml_available():
            raise RuntimeError("AzureMLCallback requires azureml to be installed. Run `pip install azureml-sdk`.")
        # 初始化 AzureML 运行对象
        self.azureml_run = azureml_run

    def on_init_end(self, args, state, control, **kwargs):
        # 导入 AzureML 中的 Run 类
        from azureml.core.run import Run

        # 如果未指定 AzureML 运行对象并且是主进程，则获取当前运行上下文
        if self.azureml_run is None and state.is_world_process_zero:
            self.azureml_run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 如果存在 AzureML 运行对象并且是主进程
        if self.azureml_run and state.is_world_process_zero:
            # 遍历日志字典
            for k, v in logs.items():
                # 如果值为整数或浮点数
                if isinstance(v, (int, float)):
                    # 记录日志
                    self.azureml_run.log(k, v, description=k)


class MLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """

    def __init__(self):
        # 如果未安装 MLflow 库，则引发运行时错误
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        # 导入 MLflow 库
        import mlflow

        # 设置最大参数值长度和最大参数标签数
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        # 初始化标志
        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果尚未初始化，则进行初始化
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        # 如果尚未初始化，则进行初始化
        if not self._initialized:
            self.setup(args, state, model)
        # 如果是主进程
        if state.is_world_process_zero:
            metrics = {}
            # 遍历日志字典
            for k, v in logs.items():
                # 如果值为整数或浮点数
                if isinstance(v, (int, float)):
                    # 将其添加到度量值字典中
                    metrics[k] = v
                else:
                    # 如果值不是整数或浮点数，则记录警告
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )
            # 使用 MLflow 记录度量值
            self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        # 如果已初始化并且是主进程
        if self._initialized and state.is_world_process_zero:
            # 如果自动结束运行已启用且存在活动运行
            if self._auto_end_run and self._ml_flow.active_run():
                # 结束运行
                self._ml_flow.end_run()
    # 当保存模型时触发的方法，仅在初始化完成且是世界进程的第一个进程，并且允许记录模型时执行
    def on_save(self, args, state, control, **kwargs):
        # 检查是否已经初始化并且是世界进程的第一个进程，以及是否允许记录模型
        if self._initialized and state.is_world_process_zero and self._log_artifacts:
            # 创建检查点目录
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 拼接检查点目录路径
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            # 记录检查点文件的信息
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
            # 使用 MLflow 的 pyfunc 模块记录模型
            self._ml_flow.pyfunc.log_model(
                ckpt_dir,
                artifacts={"model_path": artifact_path},
                python_model=self._ml_flow.pyfunc.PythonModel(),
            )

    # 析构函数，在对象被销毁时调用
    def __del__(self):
        # 如果上一个运行没有正确终止，fluent API 将不允许在上一个运行被终止之前启动新的运行
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            # 结束当前运行
            self._ml_flow.end_run()
class DagsHubCallback(MLflowCallback):
    """
    一个 [`TrainerCallback`]，用于将日志记录到 [DagsHub](https://dagshub.com/)。继承自 [`MLflowCallback`]
    """

    def __init__(self):
        super().__init__()
        # 检查是否安装了 DagsHub 库，若未安装则引发 ImportError
        if not is_dagshub_available():
            raise ImportError("DagsHubCallback requires dagshub to be installed. Run `pip install dagshub`.")

        # 导入 DagsHub 库中的 Repo 类
        from dagshub.upload import Repo

        # 初始化 Repo 属性为导入的 Repo 类
        self.Repo = Repo

    def setup(self, *args, **kwargs):
        """
        设置 DagsHub 的日志集成。

        环境变量:
        - **HF_DAGSHUB_LOG_ARTIFACTS** (`str`, *可选*):
                是否保存实验的数据和模型结果。默认为 `False`。
        """

        # 根据环境变量设置是否保存实验的数据和模型结果
        self.log_artifacts = os.getenv("HF_DAGSHUB_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        # 获取模型名称，默认为 "main"
        self.name = os.getenv("HF_DAGSHUB_MODEL_NAME") or "main"
        # 获取 MLflow 追踪 URI
        self.remote = os.getenv("MLFLOW_TRACKING_URI")
        # 创建 Repo 对象
        self.repo = self.Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].split(".")[0],
            branch=os.getenv("BRANCH") or "main",
        )
        # 设置存储路径为 "artifacts"
        self.path = Path("artifacts")

        # 若未设置 MLflow 追踪 URI，引发 RuntimeError
        if self.remote is None:
            raise RuntimeError(
                "DagsHubCallback requires the `MLFLOW_TRACKING_URI` environment variable to be set. Did you run"
                " `dagshub.init()`?"
            )

        super().setup(*args, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        # 若需要保存实验的数据和模型结果
        if self.log_artifacts:
            # 若存在训练数据加载器，则保存数据集
            if getattr(self, "train_dataloader", None):
                torch.save(self.train_dataloader.dataset, os.path.join(args.output_dir, "dataset.pt"))

            # 将输出目录添加到 Repo 对象中
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
    """将日志发送到 [Neptune](https://app.neptune.ai) 的 TrainerCallback。
    """
    Args:
        api_token (`str`, *optional*): Neptune API token obtained upon registration.
            You can leave this argument out if you have saved your token to the `NEPTUNE_API_TOKEN` environment
            variable (strongly recommended). See full setup instructions in the
            [docs](https://docs.neptune.ai/setup/installation).
        project (`str`, *optional*): Name of an existing Neptune project, in the form "workspace-name/project-name".
            You can find and copy the name in Neptune from the project settings -> Properties. If None (default), the
            value of the `NEPTUNE_PROJECT` environment variable is used.
        name (`str`, *optional*): Custom name for the run.
        base_namespace (`str`, optional, defaults to "finetuning"): In the Neptune run, the root namespace
            that will contain all of the metadata logged by the callback.
        log_parameters (`bool`, *optional*, defaults to `True`):
            If True, logs all Trainer arguments and model parameters provided by the Trainer.
        log_checkpoints (`str`, *optional*): If "same", uploads checkpoints whenever they are saved by the Trainer.
            If "last", uploads only the most recently saved checkpoint. If "best", uploads the best checkpoint (among
            the ones saved by the Trainer). If `None`, does not upload checkpoints.
        run (`Run`, *optional*): Pass a Neptune run object if you want to continue logging to an existing run.
            Read more about resuming runs in the [docs](https://docs.neptune.ai/logging/to_existing_object).
        **neptune_run_kwargs (*optional*):
            Additional keyword arguments to be passed directly to the
            [`neptune.init_run()`](https://docs.neptune.ai/api/neptune#init_run) function when a new run is created.

    For instructions and examples, see the [Transformers integration
    guide](https://docs.neptune.ai/integrations/transformers) in the Neptune documentation.
    """

    # Key for the integration version in Neptune
    integration_version_key = "source_code/integrations/transformers"
    # Key for model parameters in Neptune
    model_parameters_key = "model_parameters"
    # Key for trial name in Neptune
    trial_name_key = "trial"
    # Key for trial parameters in Neptune
    trial_params_key = "trial_params"
    # Key for trainer parameters in Neptune
    trainer_parameters_key = "trainer_parameters"
    # Set of flat metrics to be logged in Neptune
    flat_metrics = {"train/epoch"}

    def __init__(
        self,
        *,
        api_token: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        base_namespace: str = "finetuning",
        run=None,
        log_parameters: bool = True,
        log_checkpoints: Optional[str] = None,
        **neptune_run_kwargs,
    ):
        # 检查 Neptune 是否可用，如果不可用则抛出数值错误
        if not is_neptune_available():
            raise ValueError(
                "NeptuneCallback requires the Neptune client library to be installed. "
                "To install the library, run `pip install neptune`."
            )

        try:
            from neptune import Run
            from neptune.internal.utils import verify_type
        except ImportError:
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

        # 初始化 NeptuneCallback 对象的属性
        self._base_namespace_path = base_namespace
        self._log_parameters = log_parameters
        self._log_checkpoints = log_checkpoints
        self._initial_run: Optional[Run] = run

        self._run = None
        self._is_monitoring_run = False
        self._run_id = None
        self._force_reset_monitoring_run = False
        self._init_run_kwargs = {"api_token": api_token, "project": project, "name": name, **neptune_run_kwargs}

        self._volatile_checkpoints_dir = None
        self._should_upload_checkpoint = self._log_checkpoints is not None
        self._recent_checkpoint_path = None

        # 根据 log_checkpoints 的值设置目标检查点的命名空间和是否清除最近上传的检查点
        if self._log_checkpoints in {"last", "best"}:
            self._target_checkpoints_namespace = f"checkpoints/{self._log_checkpoints}"
            self._should_clean_recently_uploaded_checkpoint = True
        else:
            self._target_checkpoints_namespace = "checkpoints"
            self._should_clean_recently_uploaded_checkpoint = False

    def _stop_run_if_exists(self):
        # 如果运行实例存在，则停止运行
        if self._run:
            self._run.stop()
            del self._run
            self._run = None

    def _initialize_run(self, **additional_neptune_kwargs):
        try:
            from neptune import init_run
            from neptune.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException
        except ImportError:
            from neptune.new import init_run
            from neptune.new.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException

        # 停止当前运行实例
        self._stop_run_if_exists()

        try:
            # 初始化运行实例，并获取运行实例的 ID
            self._run = init_run(**self._init_run_kwargs, **additional_neptune_kwargs)
            self._run_id = self._run["sys/id"].fetch()
        except (NeptuneMissingProjectNameException, NeptuneMissingApiTokenException) as e:
            raise NeptuneMissingConfiguration() from e

    def _use_initial_run(self):
        # 使用初始运行实例
        self._run = self._initial_run
        self._is_monitoring_run = True
        self._run_id = self._run["sys/id"].fetch()
        self._initial_run = None
    # 确保带监控运行环境
    def _ensure_run_with_monitoring(self):
        # 如果有初始运行，则使用初始运行
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            # 如果不强制重置监控运行并且当前为监控运行，则返回
            if not self._force_reset_monitoring_run and self._is_monitoring_run:
                return

            # 如果存在运行对象但不是监控运行并且不强制重置监控运行，则初始化监控运行
            if self._run and not self._is_monitoring_run and not self._force_reset_monitoring_run:
                self._initialize_run(with_id=self._run_id)
                self._is_monitoring_run = True
            # 否则初始化运行
            else:
                self._initialize_run()
                self._force_reset_monitoring_run = False

    # 确保至少运行一次但不带监控环境
    def _ensure_at_least_run_without_monitoring(self):
        # 如果有初始运行，则使用初始运行
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            # 如果没有运行对象，则初始化运行
            if not self._run:
                self._initialize_run(
                    with_id=self._run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False,
                    capture_traceback=False,
                )
                self._is_monitoring_run = False

    # 返回运行对象
    @property
    def run(self):
        # 如果运行对象为空，则确保至少运行一次但不带监控环境
        if self._run is None:
            self._ensure_at_least_run_without_monitoring()
        return self._run

    # 返回元数据命名空间
    @property
    def _metadata_namespace(self):
        return self.run[self._base_namespace_path]

    # 记录集成版本
    def _log_integration_version(self):
        self.run[NeptuneCallback.integration_version_key] = version

    # 记录训练器参数
    def _log_trainer_parameters(self, args):
        self._metadata_namespace[NeptuneCallback.trainer_parameters_key] = args.to_sanitized_dict()

    # 记录模型参数
    def _log_model_parameters(self, model):
        from neptune.utils import stringify_unsupported

        if model and hasattr(model, "config") and model.config is not None:
            self._metadata_namespace[NeptuneCallback.model_parameters_key] = stringify_unsupported(
                model.config.to_dict()
            )

    # 记录超参数搜索参数
    def _log_hyper_param_search_parameters(self, state):
        if state and hasattr(state, "trial_name"):
            self._metadata_namespace[NeptuneCallback.trial_name_key] = state.trial_name

        if state and hasattr(state, "trial_params") and state.trial_params is not None:
            self._metadata_namespace[NeptuneCallback.trial_params_key] = state.trial_params
    # 记录模型检查点，将检查点保存到指定目录
    def _log_model_checkpoint(self, source_directory: str, checkpoint: str):
        # 设置目标路径为源目录下的检查点路径
        target_path = relative_path = os.path.join(source_directory, checkpoint)

        # 如果存在易失性检查点目录
        if self._volatile_checkpoints_dir is not None:
            # 一致性检查点路径为易失性检查点目录下的检查点路径
            consistent_checkpoint_path = os.path.join(self._volatile_checkpoints_dir, checkpoint)
            try:
                # 移除相对路径中的前导 ../
                cpkt_path = relative_path.replace("..", "").lstrip(os.path.sep)
                # 复制相对路径下的内容到一致性检查点路径下
                copy_path = os.path.join(consistent_checkpoint_path, cpkt_path)
                shutil.copytree(relative_path, copy_path)
                # 设置目标路径为一致性检查点路径
                target_path = consistent_checkpoint_path
            except IOError as e:
                # 若复制失败则记录警告信息
                logger.warning(
                    "NeptuneCallback was unable to made a copy of checkpoint due to I/O exception: '{}'. "
                    "Could fail trying to upload.".format(e)
                )

        # 将目标路径下的文件上传到元数据命名空间
        self._metadata_namespace[self._target_checkpoints_namespace].upload_files(target_path)

        # 如果需要清理最近上传的检查点，并且最近检查点路径不为空
        if self._should_clean_recently_uploaded_checkpoint and self._recent_checkpoint_path is not None:
            # 删除最近上传的检查点
            self._metadata_namespace[self._target_checkpoints_namespace].delete_files(self._recent_checkpoint_path)

        # 更新最近检查点路径为相对路径
        self._recent_checkpoint_path = relative_path

    # 在初始化结束时执行的操作
    def on_init_end(self, args, state, control, **kwargs):
        # 若需要记录检查点，并且覆盖输出目录或设置了保存总数限制
        if self._log_checkpoints and (args.overwrite_output_dir or args.save_total_limit is not None):
            # 创建临时目录作为易失性检查点目录
            self._volatile_checkpoints_dir = tempfile.TemporaryDirectory().name

        # 若需要记录最佳检查点但未设置在结束时加载最佳模型，则抛出数值错误
        if self._log_checkpoints == "best" and not args.load_best_model_at_end:
            raise ValueError("To save the best model checkpoint, the load_best_model_at_end argument must be enabled.")

    # 在训练开始时执行的操作
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果不是世界进程的零号进程，则返回
        if not state.is_world_process_zero:
            return

        # 确保在监控模式下运行
        self._ensure_run_with_monitoring()
        # 强制重置监控运行
        self._force_reset_monitoring_run()

        # 记录集成版本信息
        self._log_integration_version()
        # 如果需要记录参数，则记录训练器参数和模型参数
        if self._log_parameters:
            self._log_trainer_parameters(args)
            self._log_model_parameters(model)

        # 如果是超参数搜索状态，则记录超参数搜索参数
        if state.is_hyper_param_search:
            self._log_hyper_param_search_parameters(state)

    # 在训练结束时执行的操作
    def on_train_end(self, args, state, control, **kwargs):
        # 如果运行存在，则停止运行
        self._stop_run_if_exists()

    # 对象被销毁时执行的操作
    def __del__(self):
        # 如果存在易失性检查点目录，则删除该目录及其内容
        if self._volatile_checkpoints_dir is not None:
            shutil.rmtree(self._volatile_checkpoints_dir, ignore_errors=True)

        # 如果运行存在，则停止运行
        self._stop_run_if_exists()

    # 在保存时执行的操作
    def on_save(self, args, state, control, **kwargs):
        # 如果应上传检查点
        if self._should_upload_checkpoint:
            # 记录模型检查点到输出目录下，以全局步骤号命名
            self._log_model_checkpoint(args.output_dir, f"checkpoint-{state.global_step}")
    # 在评估过程中，根据指定条件判断是否应该上传检查点
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 如果设置为只上传最佳检查点
        if self._log_checkpoints == "best":
            # 获取最佳模型度量名称
            best_metric_name = args.metric_for_best_model
            # 如果度量名称不以"eval_"开头，则添加"eval_"
            if not best_metric_name.startswith("eval_"):
                best_metric_name = f"eval_{best_metric_name}"

            # 获取度量值
            metric_value = metrics.get(best_metric_name)

            # 根据参数中的greater_is_better属性选择比较操作符
            operator = np.greater if args.greater_is_better else np.less

            # 判断是否应该上传检查点
            self._should_upload_checkpoint = state.best_metric is None or operator(metric_value, state.best_metric)

    # 获取训练运行实例
    @classmethod
    def get_run(cls, trainer):
        # 遍历训练回调处理程序中的回调列表
        for callback in trainer.callback_handler.callbacks:
            # 如果回调是 NeptuneCallback 类的实例，则返回其运行实例
            if isinstance(callback, cls):
                return callback.run

        # 如果没有 NeptuneCallback 配置，则抛出异常
        raise Exception("The trainer doesn't have a NeptuneCallback configured.")

    # 在记录日志时执行的操作
    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        # 如果不是全局进程的第一个进程，则直接返回
        if not state.is_world_process_zero:
            return

        # 如果日志不为空
        if logs is not None:
            # 遍历重写后的日志字典
            for name, value in rewrite_logs(logs).items():
                # 如果值是整数或浮点数
                if isinstance(value, (int, float)):
                    # 如果名称在 NeptuneCallback 的 flat_metrics 列表中
                    if name in NeptuneCallback.flat_metrics:
                        # 将值存储到元数据命名空间中
                        self._metadata_namespace[name] = value
                    else:
                        # 否则，记录值到 Neptune 平台，并指定步骤为全局步骤数
                        self._metadata_namespace[name].log(value, step=state.global_step)
class CodeCarbonCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that tracks the CO2 emission of training.
    """

    def __init__(self):
        # 检查是否安装了 codecarbon，若未安装则引发异常
        if not is_codecarbon_available():
            raise RuntimeError(
                "CodeCarbonCallback requires `codecarbon` to be installed. Run `pip install codecarbon`."
            )
        import codecarbon

        # 导入 codecarbon 模块
        self._codecarbon = codecarbon
        # 初始化追踪器变量
        self.tracker = None

    def on_init_end(self, args, state, control, **kwargs):
        # 如果追踪器尚未初始化且当前进程是主进程
        if self.tracker is None and state.is_local_process_zero:
            # 使用输出目录初始化追踪器，CodeCarbon 会自动处理环境变量配置
            self.tracker = self._codecarbon.EmissionsTracker(output_dir=args.output_dir)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 如果追踪器已初始化且当前进程是主进程
        if self.tracker and state.is_local_process_zero:
            # 开始追踪 CO2 排放
            self.tracker.start()

    def on_train_end(self, args, state, control, **kwargs):
        # 如果追踪器已初始化且当前进程是主进程
        if self.tracker and state.is_local_process_zero:
            # 停止追踪 CO2 排放
            self.tracker.stop()


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

    def __init__(self):
        # 检查是否安装了 clearml，若未安装则引发异常
        if is_clearml_available():
            import clearml

            self._clearml = clearml
        else:
            raise RuntimeError("ClearMLCallback requires 'clearml' to be installed. Run `pip install clearml`.")

        # 初始化变量
        self._initialized = False
        self._initialized_externally = False
        self._clearml_task = None

        # 从环境变量中获取是否记录模型日志的设置，默认为 False
        self._log_model = os.getenv("CLEARML_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})
    # 设置方法，用于初始化 ClearML 日志记录
    def setup(self, args, state, model, tokenizer, **kwargs):
        # 如果 ClearML 未被导入，则直接返回
        if self._clearml is None:
            return
        # 如果已经初始化过，则直接返回
        if self._initialized:
            return
        # 如果当前进程是第一个进程
        if state.is_world_process_zero:
            # 输出日志信息
            logger.info("Automatic ClearML logging enabled.")
            # 如果 ClearML 任务对象为空
            if self._clearml_task is None:
                # 这种情况可能发生在运行在管道中，任务已经在 Hugging Face 外部初始化的情况下
                if self._clearml.Task.current_task():
                    # 使用当前 ClearML 任务对象
                    self._clearml_task = self._clearml.Task.current_task()
                    self._initialized = True
                    self._initialized_externally = True
                    logger.info("External ClearML Task has been connected.")
                else:
                    # 初始化 ClearML 任务对象
                    self._clearml_task = self._clearml.Task.init(
                        project_name=os.getenv("CLEARML_PROJECT", "HuggingFace Transformers"),
                        task_name=os.getenv("CLEARML_TASK", "Trainer"),
                        auto_connect_frameworks={"tensorboard": False, "pytorch": False},
                        output_uri=True,
                    )
                    self._initialized = True
                    logger.info("ClearML Task has been initialized.")

            # 连接参数到 ClearML 任务
            self._clearml_task.connect(args, "Args")
            # 如果模型有配置信息，则连接模型配置到 ClearML 任务
            if hasattr(model, "config") and model.config is not None:
                self._clearml_task.connect(model.config, "Model Configuration")

    # 训练开始时的方法
    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # 如果 ClearML 未被导入，则直接返回
        if self._clearml is None:
            return
        # 如果是超参数搜索状态，则重置初始化状态
        if state.is_hyper_param_search:
            self._initialized = False
        # 如果未初始化，则调用 setup 方法进行初始化
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)

    # 训练结束时的方法
    def on_train_end(self, args, state, control, model=None, tokenizer=None, metrics=None, logs=None, **kwargs):
        # 如果 ClearML 未被导入，则直接返回
        if self._clearml is None:
            return
        # 如果 ClearML 任务对象存在且当前进程是第一个进程且非外部初始化
        if self._clearml_task and state.is_world_process_zero and not self._initialized_externally:
            # 在训练结束时关闭 ClearML 任务
            self._clearml_task.close()
    # 定义日志回调函数，用于处理训练过程中的日志信息
    def on_log(self, args, state, control, model=None, tokenizer=None, logs=None, **kwargs):
        # 如果未初始化 ClearML 任务，则直接返回
        if self._clearml is None:
            return
        # 如果未初始化，则进行初始化设置
        if not self._initialized:
            self.setup(args, state, model, tokenizer, **kwargs)
        # 如果是世界进程中的第一个进程
        if state.is_world_process_zero:
            # 定义评估和测试指标的前缀字符串
            eval_prefix = "eval_"
            eval_prefix_len = len(eval_prefix)
            test_prefix = "test_"
            test_prefix_len = len(test_prefix)
            # 定义只包含单个值的标量列表
            single_value_scalars = [
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "total_flos",
                "epoch",
            ]
            # 遍历日志字典
            for k, v in logs.items():
                # 如果值是整数或浮点数
                if isinstance(v, (int, float)):
                    # 如果键在单值标量列表中，则将其作为单值标量进行记录
                    if k in single_value_scalars:
                        self._clearml_task.get_logger().report_single_value(name=k, value=v)
                    # 如果键以评估前缀开头，则将其作为评估指标记录
                    elif k.startswith(eval_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title=k[eval_prefix_len:], series="eval", value=v, iteration=state.global_step
                        )
                    # 如果键以测试前缀开头，则将其作为测试指标记录
                    elif k.startswith(test_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title=k[test_prefix_len:], series="test", value=v, iteration=state.global_step
                        )
                    # 否则将其作为训练指标记录
                    else:
                        self._clearml_task.get_logger().report_scalar(
                            title=k, series="train", value=v, iteration=state.global_step
                        )
                # 如果值不是整数或浮点数，则记录警告信息
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of ClearML logger's  report_scalar() "
                        "is incorrect so we dropped this attribute."
                    )

    # 定义保存模型的回调函数
    def on_save(self, args, state, control, **kwargs):
        # 如果需要记录模型，并且已初始化 ClearML 任务，并且是世界进程中的第一个进程
        if self._log_model and self._clearml_task and state.is_world_process_zero:
            # 定义检查点文件夹名称
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 定义检查点的路径
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            # 记录检查点信息
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
            # 更新输出模型的路径，并设置不自动删除文件
            self._clearml_task.update_output_model(artifact_path, iteration=state.global_step, auto_delete_file=False)
class FlyteCallback(TrainerCallback):
    """A [`TrainerCallback`] that sends the logs to [Flyte](https://flyte.org/).
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

    def __init__(self, save_log_history: bool = True, sync_checkpoints: bool = True):
        # 继承父类构造方法
        super().__init__()
        # 检查是否安装了 Flyte 相关库
        if not is_flytekit_available():
            raise ImportError("FlyteCallback requires flytekit to be installed. Run `pip install flytekit`.")

        # 检查是否安装了 flytekitplugins-deck-standard 和 pandas 库
        if not is_flyte_deck_standard_available() or not is_pandas_available():
            logger.warning(
                "Syncing log history requires both flytekitplugins-deck-standard and pandas to be installed. "
                "Run `pip install flytekitplugins-deck-standard pandas` to enable this feature."
            )
            # 如果未安装相关库，则设置 save_log_history 为 False
            save_log_history = False

        # 导入 flytekit 中的 current_context 函数
        from flytekit import current_context

        # 获取当前任务的检查点
        self.cp = current_context().checkpoint
        # 保存日志历史记录的标志
        self.save_log_history = save_log_history
        # 是否同步检查点的标志
        self.sync_checkpoints = sync_checkpoints

    def on_save(self, args, state, control, **kwargs):
        # 如果开启了同步检查点并且是主进程
        if self.sync_checkpoints and state.is_world_process_zero:
            # 构建检查点目录名
            ckpt_dir = f"checkpoint-{state.global_step}"
            # 构建检查点的完整路径
            artifact_path = os.path.join(args.output_dir, ckpt_dir)

            # 输出同步检查点信息
            logger.info(f"Syncing checkpoint in {ckpt_dir} to Flyte. This may take time.")
            # 保存检查点
            self.cp.save(artifact_path)

    def on_train_end(self, args, state, control, **kwargs):
        # 如果开启了保存日志历史记录功能
        if self.save_log_history:
            # 导入 pandas 库
            import pandas as pd
            # 导入 flytekit 中的 Deck 类和 deck 渲染器
            from flytekit import Deck
            from flytekitplugins.deck.renderer import TableRenderer

            # 创建日志历史记录的 DataFrame
            log_history_df = pd.DataFrame(state.log_history)
            # 创建 Flyte Deck 并使用表格渲染器渲染日志历史记录
            Deck("Log History", TableRenderer().to_html(log_history_df))


class DVCLiveCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [DVCLive](https://www.dvc.org/doc/dvclive).

    Use the environment variables below in `setup` to configure the integration. To customize this callback beyond
    those environment variables, see [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).
    Args:
        live (`dvclive.Live`, *optional*, defaults to `None`):
            可选的 Live 实例。如果为 None，则将使用 **kwargs 创建一个新实例。
        log_model (Union[Literal["all"], bool], *optional*, defaults to `None`):
            是否使用 `dvclive.Live.log_artifact()` 来记录 [`Trainer`] 创建的检查点。如果设置为 `True`，
            则在训练结束时记录最终检查点。如果设置为 `"all"`，则在每个检查点记录整个 [`TrainingArguments`] 的 `output_dir`。
    """

    def __init__(
        self,
        live: Optional[Any] = None,
        log_model: Optional[Union[Literal["all"], bool]] = None,
        **kwargs,
    ):
        if not is_dvclive_available():
            raise RuntimeError("DVCLiveCallback 需要安装 dvclive。运行 `pip install dvclive`。")
        from dvclive import Live

        self._log_model = log_model  # 记录是否要记录模型
        self._initialized = False  # 初始化标志
        self.live = None
        if isinstance(live, Live):  # 如果 live 是 Live 的实例
            self.live = live
            self._initialized = True
        elif live is not None:
            raise RuntimeError(f"发现 {live.__class__} 类的 live，预期是 dvclive.Live")

    def setup(self, args, state, model):
        """
        设置可选的 DVCLive 集成。要自定义此回调，超出下面的环境变量之外，请参见
        [这里](https://dvc.org/doc/dvclive/ml-frameworks/huggingface)。

        环境变量:
        - **HF_DVCLIVE_LOG_MODEL** (`str`, *optional*):
            是否使用 `dvclive.Live.log_artifact()` 来记录 [`Trainer`] 创建的检查点。如果设置为 `True` 或
            *1*，则在训练结束时记录最终检查点。如果设置为 `all`，则在每个检查点记录整个
            [`TrainingArguments`] 的 `output_dir`。
        """
        from dvclive import Live

        self._initialized = True
        if self._log_model is not None:
            log_model_env = os.getenv("HF_DVCLIVE_LOG_MODEL")
            if log_model_env.upper() in ENV_VARS_TRUE_VALUES:
                self._log_model = True
            elif log_model_env.lower() == "all":
                self._log_model = "all"
        if state.is_world_process_zero:  # 如果是全局进程的第一个进程
            if not self.live:
                self.live = Live()  # 创建 Live 实例
            self.live.log_params(args.to_dict())  # 记录参数信息

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
    # 当日志事件发生时调用该方法，处理日志相关操作
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 如果未初始化，则进行初始化
        if not self._initialized:
            self.setup(args, state, model)
        # 如果是世界进程的第一个进程
        if state.is_world_process_zero:
            # 导入 Metric 类和 standardize_metric_name 方法
            from dvclive.plots import Metric
            from dvclive.utils import standardize_metric_name

            # 遍历日志字典中的键值对
            for key, value in logs.items():
                # 如果 Metric 类可以记录该值
                if Metric.could_log(value):
                    # 记录指标到 DVCLive
                    self.live.log_metric(standardize_metric_name(key, "dvclive.huggingface"), value)
                else:
                    # 记录警告信息
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{value}" of type {type(value)} for key "{key}" as a scalar. '
                        "This invocation of DVCLive's Live.log_metric() "
                        "is incorrect so we dropped this attribute."
                    )
            # 进行下一步操作
            self.live.next_step()

    # 当保存事件发生时调用该方法，处理保存模型相关操作
    def on_save(self, args, state, control, **kwargs):
        # 如果需要记录所有模型、已初始化且是世界进程的第一个进程
        if self._log_model == "all" and self._initialized and state.is_world_process_zero:
            # 记录模型到 DVCLive
            self.live.log_artifact(args.output_dir)

    # 当训练结束事件发生时调用该方法，处理训练结束相关操作
    def on_train_end(self, args, state, control, **kwargs):
        # 如果已初始化且是世界进程的第一个进程
        if self._initialized and state.is_world_process_zero:
            # 导入 Trainer 类
            from transformers.trainer import Trainer

            # 如果需要记录模型
            if self._log_model is True:
                # 创建一个虚拟 Trainer 对象
                fake_trainer = Trainer(args=args, model=kwargs.get("model"), tokenizer=kwargs.get("tokenizer"))
                # 根据参数确定模型名称
                name = "best" if args.load_best_model_at_end else "last"
                output_dir = os.path.join(args.output_dir, name)
                # 保存模型到指��目录
                fake_trainer.save_model(output_dir)
                # 记录模型到 DVCLive
                self.live.log_artifact(output_dir, name=name, type="model", copy=True)
            # 结束 DVCLive 记录
            self.live.end()
# 将不同的集成名称映射到相应的回调类
INTEGRATION_TO_CALLBACK = {
    "azure_ml": AzureMLCallback,  # Azure ML集成对应的回调类
    "comet_ml": CometCallback,    # Comet ML集成对应的回调类
    "mlflow": MLflowCallback,     # MLflow集成对应的回调类
    "neptune": NeptuneCallback,   # Neptune集成对应的回调类
    "tensorboard": TensorBoardCallback,  # TensorBoard集成对应的回调类
    "wandb": WandbCallback,       # Weights & Biases集成对应的回调类
    "codecarbon": CodeCarbonCallback,   # CodeCarbon集成对应的回调类
    "clearml": ClearMLCallback,   # ClearML集成对应的回调类
    "dagshub": DagsHubCallback,   # DagsHub集成对应的回调类
    "flyte": FlyteCallback,       # Flyte集成对应的回调类
    "dvclive": DVCLiveCallback,   # DVC Live集成对应的回调类
}

# 根据给定的报告集成，返回相应的回调类列表
def get_reporting_integration_callbacks(report_to):
    # 遍历每个报告集成
    for integration in report_to:
        # 如果集成不在预定义的映射中，则抛出值错误
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )

    # 返回所有报告集成对应的回调类列表
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
```