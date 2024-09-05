# `.\yolov8\ultralytics\utils\callbacks\neptune.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 从 ultralytics.utils 模块导入 LOGGER、SETTINGS 和 TESTS_RUNNING
from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

# 尝试检查测试是否运行，不记录 pytest 测试
try:
    assert not TESTS_RUNNING  
    # 确认 SETTINGS 中的 neptune 设置为 True，验证 Neptune 集成已启用
    assert SETTINGS["neptune"] is True  
    import neptune
    from neptune.types import File

    assert hasattr(neptune, "__version__")

    run = None  # NeptuneAI 实验记录器实例

except (ImportError, AssertionError):
    neptune = None


def _log_scalars(scalars, step=0):
    """Log scalars to the NeptuneAI experiment logger."""
    # 如果 run 不为 None，将标量写入 NeptuneAI 实验记录器
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)


def _log_images(imgs_dict, group=""):
    """Log scalars to the NeptuneAI experiment logger."""
    # 如果 run 不为 None，上传图像到 NeptuneAI 实验记录器
    if run:
        for k, v in imgs_dict.items():
            run[f"{group}/{k}"].upload(File(v))


def _log_plot(title, plot_path):
    """
    Log plots to the NeptuneAI experiment logger.

    Args:
        title (str): 图表的标题.
        plot_path (PosixPath | str): 图像文件的路径.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    # 读取图像文件
    img = mpimg.imread(plot_path)
    # 创建新的图表
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])  # 不显示刻度
    ax.imshow(img)
    # 上传图表到 NeptuneAI 实验记录器
    run[f"Plots/{title}"].upload(fig)


def on_pretrain_routine_start(trainer):
    """Callback function called before the training routine starts."""
    try:
        global run
        # 初始化 NeptuneAI 实验记录器
        run = neptune.init_run(project=trainer.args.project or "YOLOv8", name=trainer.args.name, tags=["YOLOv8"])
        # 记录超参数配置到 NeptuneAI 实验记录器
        run["Configuration/Hyperparameters"] = {k: "" if v is None else v for k, v in vars(trainer.args).items()}
    except Exception as e:
        # 若 NeptuneAI 安装但初始化不正确，记录警告信息
        LOGGER.warning(f"WARNING ⚠️ NeptuneAI installed but not initialized correctly, not logging this run. {e}")


def on_train_epoch_end(trainer):
    """Callback function called at end of each training epoch."""
    # 记录训练损失到 NeptuneAI 实验记录器
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    # 记录学习率到 NeptuneAI 实验记录器
    _log_scalars(trainer.lr, trainer.epoch + 1)
    # 如果是第一个 epoch，记录训练批次图像到 NeptuneAI 实验记录器中的"Mosaic"组
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob("train_batch*.jpg")}, "Mosaic")


def on_fit_epoch_end(trainer):
    """Callback function called at end of each fit (train+val) epoch."""
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        # 记录模型信息到 NeptuneAI 实验记录器
        run["Configuration/Model"] = model_info_for_loggers(trainer)
    # 记录指标到 NeptuneAI 实验记录器
    _log_scalars(trainer.metrics, trainer.epoch + 1)


def on_val_end(validator):
    """Callback function called at end of each validation."""
    if run:
        # 记录验证图像到 NeptuneAI 实验记录器中的"Validation"组
        _log_images({f.stem: str(f) for f in validator.save_dir.glob("val*.jpg")}, "Validation")


def on_train_end(trainer):
    """Callback function called at end of training."""
    # 如果 run 变量为真，则执行以下操作
    if run:
        # 定义要记录的文件列表，包括结果图像和混淆矩阵等
        files = [
            "results.png",  # 结果图像文件名
            "confusion_matrix.png",  # 混淆矩阵图像文件名
            "confusion_matrix_normalized.png",  # 归一化混淆矩阵图像文件名
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),  # F1、PR、P、R 曲线图像文件名
        ]
        # 使用列表推导式筛选出存在的文件路径
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]  # filter
        # 遍历筛选后的文件列表，记录每个文件的标题和路径
        for f in files:
            _log_plot(title=f.stem, plot_path=f)  # 记录图像，使用文件名的基本名称作为标题
        # 记录最终的模型权重文件
        run[f"weights/{trainer.args.name or trainer.args.task}/{trainer.best.name}"].upload(File(str(trainer.best)))
# 如果 neptune 变量为真，则定义一个包含多个回调函数的字典；否则定义一个空字典。
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if neptune
    else {}
)
```