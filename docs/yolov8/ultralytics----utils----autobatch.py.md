# `.\yolov8\ultralytics\utils\autobatch.py`

```py
# 导入深度复制函数
from copy import deepcopy
# 导入必要的库
import numpy as np
import torch

# 导入自定义模块
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, profile

# 定义检查训练批大小的函数
def check_train_batch_size(model, imgsz=640, amp=True, batch=-1):
    """
    使用 autobatch() 函数计算最佳 YOLO 训练批大小。

    Args:
        model (torch.nn.Module): 要检查批大小的 YOLO 模型。
        imgsz (int): 用于训练的图像尺寸。
        amp (bool): 如果为 True，使用自动混合精度 (AMP) 进行训练。

    Returns:
        (int): 使用 autobatch() 函数计算的最佳批大小。
    """

    with autocast(enabled=amp):
        return autobatch(deepcopy(model).train(), imgsz, fraction=batch if 0.0 < batch < 1.0 else 0.6)


# 定义自动估算最佳批大小的函数
def autobatch(model, imgsz=640, fraction=0.60, batch_size=DEFAULT_CFG.batch):
    """
    自动估算最佳 YOLO 批大小，以利用可用 CUDA 内存的一部分。

    Args:
        model (torch.nn.module): 要计算批大小的 YOLO 模型。
        imgsz (int, optional): YOLO 模型输入的图像大小。默认为 640。
        fraction (float, optional): 要使用的可用 CUDA 内存的分数。默认为 0.60。
        batch_size (int, optional): 如果检测到错误，则使用的默认批大小。默认为 16。

    Returns:
        (int): 最佳批大小。
    """

    # 检查设备
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}正在计算 imgsz={imgsz} 时，{fraction * 100}% CUDA 内存利用率下的最佳批大小。")
    device = next(model.parameters()).device  # 获取模型设备
    if device.type in {"cpu", "mps"}:
        LOGGER.info(f"{prefix} ⚠️ 仅适用于 CUDA 设备，使用默认批大小 {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ 需要设置 torch.backends.cudnn.benchmark=False，使用默认批大小 {batch_size}")
        return batch_size

    # 检查 CUDA 内存
    gb = 1 << 30  # 字节转换为 GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # 设备属性
    t = properties.total_memory / gb  # 总 GiB
    r = torch.cuda.memory_reserved(device) / gb  # 已预留 GiB
    a = torch.cuda.memory_allocated(device) / gb  # 已分配 GiB
    f = t - (r + a)  # 可用 GiB
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G 总共, {r:.2f}G 已预留, {a:.2f}G 已分配, {f:.2f}G 可用")

    # 分析不同批大小
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        # 创建一个空的 PyTorch 张量列表，每个张量的大小为 (b, 3, imgsz, imgsz)，其中 b 从 batch_sizes 中取值
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        # 使用 profile 函数对 img、model 进行分析，n=3，使用指定设备 device
        results = profile(img, model, n=3, device=device)

        # 对结果进行拟合
        y = [x[2] for x in results if x]  # 获取结果中的内存使用情况 [2]
        # 对 batch_sizes 中的数据和 y 进行一次度为 1 的多项式拟合
        p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # 第一次多项式拟合
        # 计算出最佳的 batch size，通过拟合得到的 y 截距 (optimal batch size)
        b = int((f * fraction - p[1]) / p[0])

        # 如果结果中有 None，表示某些尺寸的计算失败
        if None in results:  # 一些尺寸计算失败
            i = results.index(None)  # 第一个失败的索引
            # 如果 b 大于等于失败尺寸的 batch size，选择前一个安全点的 batch size
            if b >= batch_sizes[i]:  # y 截距超过失败点
                b = batch_sizes[max(i - 1, 0)]  # 选择前一个安全点

        # 如果 b 小于 1 或大于 1024，超出安全范围，使用默认的 batch_size
        if b < 1 or b > 1024:  # b 超出安全范围
            b = batch_size
            LOGGER.info(f"{prefix}WARNING ⚠️ CUDA anomaly detected, using default batch-size {batch_size}.")

        # 计算实际预测的分数 fraction
        fraction = (np.polyval(p, b) + r + a) / t  # 实际预测的分数
        # 记录使用的 batch size 以及相关的内存信息
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        # 返回计算得到的 batch size
        return b
    except Exception as e:
        # 捕获异常情况，记录警告信息，使用默认的 batch_size
        LOGGER.warning(f"{prefix}WARNING ⚠️ error detected: {e},  using default batch-size {batch_size}.")
        # 返回默认的 batch_size
        return batch_size
```