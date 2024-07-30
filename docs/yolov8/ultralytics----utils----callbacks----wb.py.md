# `.\yolov8\ultralytics\utils\callbacks\wb.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入必要的模块和变量
from ultralytics.utils import SETTINGS, TESTS_RUNNING  # 从ultralytics.utils中导入SETTINGS和TESTS_RUNNING变量
from ultralytics.utils.torch_utils import model_info_for_loggers  # 从ultralytics.utils.torch_utils中导入model_info_for_loggers函数

try:
    assert not TESTS_RUNNING  # 确保不是在运行测试时记录日志，断言不应该是pytest
    assert SETTINGS["wandb"] is True  # 验证W&B集成是否启用

    # 尝试导入并验证wandb模块
    import wandb as wb
    assert hasattr(wb, "__version__")  # 确保wandb模块已经正确导入，而不是一个目录
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None  # 如果导入失败或者断言失败，则将wb设为None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (List): Values for the x-axis; expected to have length N.
        y (List): Corresponding values for the y-axis; also expected to have length N.
        classes (List): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot; defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis; defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis; defaults to 'Precision'.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import pandas  # 用于更快的导入ultralytics的作用域

    # 创建一个包含x、y和classes的DataFrame对象，并保留小数点后三位
    df = pandas.DataFrame({"class": classes, "y": y, "x": x}).round(3)
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    
    # 使用wandb.plot_table将数据表格化，并指定相关字段和字符串字段
    return wb.plot_table(
        "wandb/area-under-curve/v0", wb.Table(dataframe=df), fields=fields, string_fields=string_fields
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.
    """
    # 函数用于生成基于输入数据的度量曲线，并将其记录到wandb中
    pass  # 该函数当前没有实现任何功能，只是一个占位符
    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape CxN, where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C. Defaults to [].
        id (str, optional): Unique identifier for the logged data in wandb. Defaults to 'precision-recall'.
        title (str, optional): Title for the visualization plot. Defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis. Defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis. Defaults to 'Precision'.
        num_x (int, optional): Number of interpolated data points for visualization. Defaults to 100.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted. Defaults to True.

    Note:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    # Generate a new array of x values by linearly interpolating between the first and last x values
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    # Convert x_new to a list for logging purposes
    x_log = x_new.tolist()
    # Interpolate the mean values of y across the new x values and convert to list for logging
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    # Conditionally log either only the mean curve or all curves
    if only_mean:
        # Create a table with x and y data and log a line plot with WandB
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        # Prepare to log multiple curves with individual class names
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # Add new x values for the current class
            y_log.extend(np.interp(x_new, x, yi))  # Interpolate y values for the current class
            classes.extend([names[i]] * len(x_new))  # Append corresponding class names

        # Log a custom table visualization with WandB, without committing the log immediately
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)
# 定义一个函数，用于记录指定步骤中尚未记录的输入字典中的图表
def _log_plots(plots, step):
    # 使用浅拷贝以防止迭代过程中更改 plots 字典
    for name, params in plots.copy().items():
        # 获取图表的时间戳
        timestamp = params["timestamp"]
        # 如果未记录过这个图表（根据时间戳判断）
        if _processed_plots.get(name) != timestamp:
            # 记录图表到 wandb 的运行日志中，使用图表名称作为键，图像文件路径作为值
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            # 更新已处理的图表记录
            _processed_plots[name] = timestamp


# 当训练前例程开始时执行的回调函数，根据模块的存在初始化并启动项目
def on_pretrain_routine_start(trainer):
    # 如果 wb.run 不存在，则初始化一个 wandb 运行时
    wb.run or wb.init(project=trainer.args.project or "YOLOv8", name=trainer.args.name, config=vars(trainer.args))


# 每个训练周期结束时记录训练指标和模型信息的回调函数
def on_fit_epoch_end(trainer):
    # 记录训练指标到 wandb 运行日志中，使用当前周期数作为步骤
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    # 记录训练过程中的图表到 wandb 运行日志中
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    # 记录验证集的图表到 wandb 运行日志中
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    # 如果是第一个周期，记录模型信息到 wandb 运行日志中
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


# 每个训练周期结束时记录指标和保存图像的回调函数
def on_train_epoch_end(trainer):
    # 记录训练损失项到 wandb 运行日志中，使用当前周期数作为步骤
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    # 记录当前学习率到 wandb 运行日志中，使用当前周期数作为步骤
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    # 如果是第二个周期，记录训练过程中的图表到 wandb 运行日志中
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


# 训练结束时保存最佳模型作为 artifact 的回调函数
def on_train_end(trainer):
    # 记录验证集的图表到 wandb 运行日志中，使用当前周期数作为步骤
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    # 记录训练过程中的图表到 wandb 运行日志中，使用当前周期数作为步骤
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    # 创建一个类型为 "model" 的 artifact，用于保存最佳模型
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    # 如果存在最佳模型文件，则将其添加到 artifact 中
    if trainer.best.exists():
        art.add_file(trainer.best)
        # 记录 artifact 到 wandb 运行日志中，并指定别名为 "best"
        wb.run.log_artifact(art, aliases=["best"])
    # 遍历验证集的指标曲线并绘制到 wandb 运行日志中
    for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
        x, y, x_title, y_title = curve_values
        _plot_curve(
            x,
            y,
            names=list(trainer.validator.metrics.names.values()),
            id=f"curves/{curve_name}",
            title=curve_name,
            x_title=x_title,
            y_title=y_title,
        )
    # 结束 wandb 运行日志，必须调用以完成运行
    wb.run.finish()  # required or run continues on dashboard


# 定义回调函数集合，根据 wandb 是否可用来决定包含哪些回调函数
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb  # 如果 wb 可用，则包含上述四个回调函数
    else {}  # 否则为空字典
)
```