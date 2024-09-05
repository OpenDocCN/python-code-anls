# `.\yolov8\ultralytics\utils\callbacks\mlflow.py`

```py
# Ultralytics YOLO , AGPL-3.0 license
"""
MLflow Logging for Ultralytics YOLO.

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
    1. To set a project name:
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. To set a run name:
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. To start a local MLflow server:
        mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

    4. To kill all running MLflow server instances:
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

try:
    import os

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get("PYTEST_CURRENT_TEST", "")  # do not log pytest
    assert SETTINGS["mlflow"] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package is not directory
    from pathlib import Path

    PREFIX = colorstr("MLflow: ")

except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def on_pretrain_routine_end(trainer):
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Global:
        mlflow: The imported mlflow module to use for logging.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after the end of training.
    """
    global mlflow

    # 获取 MLflow 追踪的 URI，如果未设置，则默认为 RUNS_DIR 下的 'mlflow'
    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    # 设置 MLflow 追踪 URI
    mlflow.set_tracking_uri(uri)

    # 设置实验名称和运行名称
    # 如果环境变量中未设置 MLFLOW_EXPERIMENT_NAME，则默认使用 trainer.args.project 或者 '/Shared/YOLOv8'
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/YOLOv8"
    # 获取运行名称，优先从环境变量中获取，否则使用 trainer 的参数中的名称
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    
    # 设置 MLflow 实验名称
    mlflow.set_experiment(experiment_name)
    
    # 自动记录所有的参数和指标
    mlflow.autolog()
    
    try:
        # 获取当前活跃的 MLflow 运行，如果没有则启动一个新的运行，使用指定的运行名称
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        
        # 记录运行 ID 到日志中
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        
        # 如果指定的 URI 是一个目录，则记录一个查看 URI 的信息，包括本地访问地址
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        
        # 提示如何禁用 MLflow 记录
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        
        # 记录所有 trainer 参数到 MLflow 的参数日志中
        mlflow.log_params(dict(trainer.args))
    
    except Exception as e:
        # 如果出现异常，记录警告日志，提示初始化失败，并不跟踪这次运行
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n" f"{PREFIX}WARNING ⚠️ Not tracking this run")
# 在每个训练周期结束时将训练指标记录到 MLflow 中
def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    # 检查是否启用了 MLflow
    if mlflow:
        # 将训练学习率和标签损失项的指标进行处理和记录
        mlflow.log_metrics(
            metrics={
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )


# 在每个拟合（fit）周期结束时将训练指标记录到 MLflow 中
def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    # 检查是否启用了 MLflow
    if mlflow:
        # 将拟合周期的指标进行处理和记录
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)


# 在训练结束时记录模型工件到 MLflow
def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    # 如果没有启用 MLflow，则直接返回
    if not mlflow:
        return
    # 记录最佳模型和其他文件到 MLflow
    mlflow.log_artifact(str(trainer.best.parent))  # 记录最佳模型目录中的 weights 文件夹（包含 best.pt 和 last.pt）
    for f in trainer.save_dir.glob("*"):  # 记录保存目录中的所有其他文件
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            mlflow.log_artifact(str(f))
    # 检查是否需要保持 MLflow 运行活动状态
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
    else:
        mlflow.end_run()
        LOGGER.debug(f"{PREFIX}mlflow run ended")

    # 记录 MLflow 结果的输出信息
    LOGGER.info(
        f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
    )


# 如果启用了 MLflow，则配置相应的回调函数
callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}  # 如果未启用 MLflow，则回调为空字典
)
```