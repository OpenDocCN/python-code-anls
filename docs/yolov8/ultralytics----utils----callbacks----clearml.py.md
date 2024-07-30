# `.\yolov8\ultralytics\utils\callbacks\clearml.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 引入必要的日志器、设置和测试运行状态的标志
from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

# 尝试导入并验证 ClearML 相关的设置和环境
try:
    # 确保不在运行 pytest 时记录日志
    assert not TESTS_RUNNING
    # 确保 ClearML 整合已启用
    assert SETTINGS["clearml"] is True
    import clearml
    from clearml import Task

    # 确保 clearml 包已成功导入且有版本信息
    assert hasattr(clearml, "__version__")

except (ImportError, AssertionError):
    clearml = None


# 定义一个函数用于将文件路径列表中的图像作为调试样本记录到 ClearML 任务中
def _log_debug_samples(files, title="Debug Samples") -> None:
    """
    Log files (images) as debug samples in the ClearML task.

    Args:
        files (list): A list of file paths in PosixPath format.
        title (str): A title that groups together images with the same values.
    """
    import re

    # 如果当前存在 ClearML 任务，则依次处理文件
    if task := Task.current_task():
        for f in files:
            if f.exists():
                # 从文件名中提取批次号并转换为整数
                it = re.search(r"_batch(\d+)", f.name)
                iteration = int(it.groups()[0]) if it else 0
                # 将图像文件报告到 ClearML 任务日志
                task.get_logger().report_image(
                    title=title, series=f.name.replace(it.group(), ""), local_path=str(f), iteration=iteration
                )


# 定义一个函数用于将保存的图像文件作为绘图记录到 ClearML 的绘图部分
def _log_plot(title, plot_path) -> None:
    """
    Log an image as a plot in the plot section of ClearML.

    Args:
        title (str): The title of the plot.
        plot_path (str): The path to the saved image file.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    # 读取图像文件并创建绘图对象
    img = mpimg.imread(plot_path)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # 不显示刻度
    ax.imshow(img)

    # 报告 Matplotlib 绘制的图像到 ClearML 任务日志
    Task.current_task().get_logger().report_matplotlib_figure(
        title=title, series="", figure=fig, report_interactive=False
    )


# 定义一个函数，在预训练过程开始时初始化并连接/记录任务到 ClearML
def on_pretrain_routine_start(trainer):
    """Runs at start of pretraining routine; initializes and connects/ logs task to ClearML."""
    try:
        # 如果当前存在 ClearML 任务，则更新 PyTorch 和 Matplotlib 的绑定
        if task := Task.current_task():
            # 警告：确保禁用自动的 PyTorch 和 Matplotlib 绑定！
            # 我们正在手动在集成中记录这些绘图和模型文件
            from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
            from clearml.binding.matplotlib_bind import PatchedMatplotlib

            PatchPyTorchModelIO.update_current_task(None)
            PatchedMatplotlib.update_current_task(None)
        else:
            # 否则初始化一个新的 ClearML 任务
            task = Task.init(
                project_name=trainer.args.project or "YOLOv8",
                task_name=trainer.args.name,
                tags=["YOLOv8"],
                output_uri=True,
                reuse_last_task_id=False,
                auto_connect_frameworks={"pytorch": False, "matplotlib": False},
            )
            # 记录警告信息，提示用户如何在远程环境运行 YOLO
            LOGGER.warning(
                "ClearML Initialized a new task. If you want to run remotely, "
                "please add clearml-init and connect your arguments before initializing YOLO."
            )
        # 将训练器参数连接到 ClearML 任务
        task.connect(vars(trainer.args), name="General")
    # 捕获所有异常并将其存储在变量e中
    except Exception as e:
        # 使用WARNING级别的日志记录器LOGGER记录警告消息，指出ClearML未正确初始化，
        # 因此不能记录这次运行的日志。同时输出异常信息e。
        LOGGER.warning(f"WARNING ⚠️ ClearML installed but not initialized correctly, not logging this run. {e}")
def on_train_epoch_end(trainer):
    """Logs debug samples for the first epoch of YOLO training and report current training progress."""
    # 获取当前任务对象，如果存在
    if task := Task.current_task():
        # 如果当前是第一个 epoch，则记录调试样本
        if trainer.epoch == 1:
            _log_debug_samples(sorted(trainer.save_dir.glob("train_batch*.jpg")), "Mosaic")
        # 报告当前训练进度
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            task.get_logger().report_scalar("train", k, v, iteration=trainer.epoch)
        # 报告当前学习率
        for k, v in trainer.lr.items():
            task.get_logger().report_scalar("lr", k, v, iteration=trainer.epoch)


def on_fit_epoch_end(trainer):
    """Reports model information to logger at the end of an epoch."""
    # 获取当前任务对象，如果存在
    if task := Task.current_task():
        # 报告每个 epoch 的耗时
        task.get_logger().report_scalar(
            title="Epoch Time", series="Epoch Time", value=trainer.epoch_time, iteration=trainer.epoch
        )
        # 报告验证指标
        for k, v in trainer.metrics.items():
            task.get_logger().report_scalar("val", k, v, iteration=trainer.epoch)
        # 如果是第一个 epoch，报告模型信息给日志记录器
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for k, v in model_info_for_loggers(trainer).items():
                task.get_logger().report_single_value(k, v)


def on_val_end(validator):
    """Logs validation results including labels and predictions."""
    # 如果存在当前任务对象
    if Task.current_task():
        # 记录验证结果的标签和预测
        _log_debug_samples(sorted(validator.save_dir.glob("val*.jpg")), "Validation")


def on_train_end(trainer):
    """Logs final model and its name on training completion."""
    # 获取当前任务对象，如果存在
    if task := Task.current_task():
        # 记录最终结果，如混淆矩阵和精确率-召回率曲线
        files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        # 过滤存在的文件
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # filter
        for f in files:
            _log_plot(title=f.stem, plot_path=f)
        # 报告最终指标
        for k, v in trainer.validator.metrics.results_dict.items():
            task.get_logger().report_single_value(k, v)
        # 记录最终模型
        task.update_output_model(model_path=str(trainer.best), model_name=trainer.args.name, auto_delete_file=False)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)
```