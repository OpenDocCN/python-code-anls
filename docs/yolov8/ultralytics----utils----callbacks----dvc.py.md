# `.\yolov8\ultralytics\utils\callbacks\dvc.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入必要的模块和变量
from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks

try:
    # 确保不在运行 pytest 时记录日志
    assert not TESTS_RUNNING
    # 确保集成设置已启用
    assert SETTINGS["dvc"] is True
    # 尝试导入 dvclive
    import dvclive

    # 检查 dvclive 版本是否符合要求
    assert checks.check_version("dvclive", "2.11.0", verbose=True)

    import os
    import re
    from pathlib import Path

    # DVCLive 日志实例
    live = None
    # 记录已处理的绘图
    _processed_plots = {}

    # `on_fit_epoch_end` 在最终验证时被调用（可能需要修复），目前是我们区分最佳模型的最终评估与最后一个 epoch 验证的方式
    _training_epoch = False

except (ImportError, AssertionError, TypeError):
    # 捕获异常，设定 dvclive 为 None
    dvclive = None


def _log_images(path, prefix=""):
    """使用 DVCLive 记录指定路径下的图像，可选添加前缀。"""
    if live:
        name = path.name

        # 根据批次分组图像，以便在用户界面中使用滑块浏览
        if m := re.search(r"_batch(\d+)", name):
            ni = m[1]
            new_stem = re.sub(r"_batch(\d+)", "_batch", path.stem)
            name = (Path(new_stem) / ni).with_suffix(path.suffix)

        live.log_image(os.path.join(prefix, name), path)


def _log_plots(plots, prefix=""):
    """记录训练进度的绘图，如果尚未处理过。"""
    for name, params in plots.items():
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp


def _log_confusion_matrix(validator):
    """使用 DVCLive 记录给定验证器的混淆矩阵。"""
    targets = []
    preds = []
    matrix = validator.confusion_matrix.matrix
    names = list(validator.names.values())
    if validator.confusion_matrix.task == "detect":
        names += ["background"]

    for ti, pred in enumerate(matrix.T.astype(int)):
        for pi, num in enumerate(pred):
            targets.extend([names[ti]] * num)
            preds.extend([names[pi]] * num)

    live.log_sklearn_plot("confusion_matrix", targets, preds, name="cf.json", normalized=True)


def on_pretrain_routine_start(trainer):
    """在预训练过程开始时初始化 DVCLive 记录器，用于记录训练元数据。"""
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info("DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable).")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ DVCLive installed but not initialized correctly, not logging this run. {e}")


def on_pretrain_routine_end(trainer):
    """在预训练过程结束时记录与训练进程相关的绘图。"""
    _log_plots(trainer.plots, "train")


def on_train_start(trainer):
    """如果 DVCLive 记录器处于活动状态，则记录训练参数。"""
    if live:
        live.log_params(trainer.args)


def on_train_epoch_start(trainer):
    # 这里留空，可能在后续实现具体功能
    # 设置全局变量 _training_epoch 在每个训练周期开始时为 True
    global _training_epoch
    # 将 _training_epoch 设置为 True，指示当前处于训练周期中
    _training_epoch = True
def on_fit_epoch_end(trainer):
    """Logs training metrics and model info, and advances to next step on the end of each fit epoch."""
    global _training_epoch
    if live and _training_epoch:
        # Collect all training metrics including loss, custom metrics, and learning rate
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        # Log each metric to DVCLive
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)

        # Log model information if it's the first epoch
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers
            # Log model-specific information to DVCLive
            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)

        # Log training plots
        _log_plots(trainer.plots, "train")
        # Log validation plots
        _log_plots(trainer.validator.plots, "val")

        # Advance to the next step in the training process
        live.next_step()
        _training_epoch = False


def on_train_end(trainer):
    """Logs the best metrics, plots, and confusion matrix at the end of training if DVCLive is active."""
    if live:
        # Log all final training metrics including loss, custom metrics, and learning rate
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        # Log each metric to DVCLive
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)

        # Log validation plots
        _log_plots(trainer.plots, "val")
        # Log validation plots from validator
        _log_plots(trainer.validator.plots, "val")

        # Log confusion matrix for validation data
        _log_confusion_matrix(trainer.validator)

        # If there exists a best model artifact, log it to DVCLive
        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type="model")

        # End the DVCLive logging session
        live.end()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if dvclive
    else {}
)
```