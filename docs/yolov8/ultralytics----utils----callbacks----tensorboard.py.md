# `.\yolov8\ultralytics\utils\callbacks\tensorboard.py`

```py
# 导入上下文管理工具
import contextlib

# 从ultralytics.utils模块中导入必要的组件：LOGGER, SETTINGS, TESTS_RUNNING, colorstr
from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

try:
    # 尝试导入TensorBoard的SummaryWriter
    from torch.utils.tensorboard import SummaryWriter

    # 确保不处于测试运行中，避免记录pytest
    assert not TESTS_RUNNING
    # 确保SETTINGS中的tensorboard选项为True，验证集成已启用
    assert SETTINGS["tensorboard"] is True
    # 初始化TensorBoard的SummaryWriter实例为None
    WRITER = None
    # 定义输出前缀为TensorBoard:
    PREFIX = colorstr("TensorBoard: ")

    # 如果启用了TensorBoard，则需要以下导入
    import warnings
    from copy import deepcopy
    from ultralytics.utils.torch_utils import de_parallel, torch

except (ImportError, AssertionError, TypeError, AttributeError):
    # 处理导入错误、断言错误、类型错误和属性错误异常
    # TypeError用于处理Windows中的'Descriptors cannot not be created directly.' protobuf错误
    # AttributeError: 如果未安装'tensorflow'，则模块'tensorflow'没有'io'属性
    SummaryWriter = None


def _log_scalars(scalars, step=0):
    """将标量值记录到TensorBoard中。"""
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)


def _log_tensorboard_graph(trainer):
    """将模型图记录到TensorBoard中。"""

    # 输入图像尺寸
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())  # 获取模型的第一个参数以确定设备和类型
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # 输入图像（必须是零，而不是空）

    with warnings.catch_warnings():
        # 忽略特定警告以减少干扰
        warnings.simplefilter("ignore", category=UserWarning)  # 抑制jit追踪警告
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)  # 抑制jit追踪警告

        # 首先尝试简单方法（例如YOLO）
        with contextlib.suppress(Exception):
            trainer.model.eval()  # 将模型置于评估模式，避免BatchNorm统计量的更改
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])
            LOGGER.info(f"{PREFIX}模型图可视化已添加 ✅")
            return

        # 退回到TorchScript导出步骤（例如RTDETR）
        try:
            model = deepcopy(de_parallel(trainer.model))
            model.eval()
            model = model.fuse(verbose=False)
            for m in model.modules():
                if hasattr(m, "export"):  # 检测是否为RTDETRDecoder等，需使用Detect基类
                    m.export = True
                    m.format = "torchscript"
            model(im)  # 进行一次干跑
            WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
            LOGGER.info(f"{PREFIX}模型图可视化已添加 ✅")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}警告 ⚠️ TensorBoard模型图可视化失败 {e}")


def on_pretrain_routine_start(trainer):
    """使用SummaryWriter初始化TensorBoard日志记录。"""
    # 检查是否存在 SummaryWriter 类
    if SummaryWriter:
        # 尝试创建全局变量 WRITER，并初始化 SummaryWriter 实例
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            # 记录日志，指示如何启动 TensorBoard 并查看日志
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        # 捕获可能发生的异常
        except Exception as e:
            # 记录警告日志，指示 TensorBoard 初始化失败，当前运行未记录日志
            LOGGER.warning(f"{PREFIX}WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}")
# 在训练开始时调用的回调函数，用于记录 TensorBoard 图。
def on_train_start(trainer):
    # 如果存在 SummaryWriter 对象，则记录 TensorBoard 图
    if WRITER:
        _log_tensorboard_graph(trainer)


# 在每个训练周期结束时记录标量统计信息的回调函数。
def on_train_epoch_end(trainer):
    # 记录训练损失相关项的标量统计信息，使用指定的前缀 "train"
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    # 记录学习率的标量统计信息
    _log_scalars(trainer.lr, trainer.epoch + 1)


# 在每个训练周期结束时记录周期度量指标的回调函数。
def on_fit_epoch_end(trainer):
    # 记录训练器的度量指标的标量统计信息
    _log_scalars(trainer.metrics, trainer.epoch + 1)


# 根据条件创建回调函数字典，可能包括各个训练阶段的回调函数。
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 在预训练过程开始时调用的回调函数
        "on_train_start": on_train_start,  # 在训练开始时调用的回调函数
        "on_fit_epoch_end": on_fit_epoch_end,  # 在每个训练周期结束时调用的回调函数
        "on_train_epoch_end": on_train_epoch_end,  # 在每个训练周期结束时调用的回调函数
    }
    if SummaryWriter  # 如果存在 SummaryWriter 对象，则添加相应的回调函数到 callbacks 字典中
    else {}  # 如果不存在 SummaryWriter 对象，则 callbacks 字典为空
)
```