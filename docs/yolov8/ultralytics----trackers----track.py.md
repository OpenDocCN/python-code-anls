# `.\yolov8\ultralytics\trackers\track.py`

```py
# 导入必要的模块和库
from functools import partial
from pathlib import Path
import torch

# 导入自定义工具函数和类
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

# 导入自定义的追踪器类
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# 将追踪器类型映射到相应的追踪器类的字典
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    初始化对象追踪器，用于预测过程中的目标追踪。

    Args:
        predictor (object): 需要初始化追踪器的预测器对象。
        persist (bool, optional): 是否在追踪器已存在时持久化。默认为 False。

    Raises:
        AssertionError: 如果追踪器类型不是 'bytetrack' 或 'botsort'。
    """
    # 如果预测器对象已经有 'trackers' 属性且 persist=True，则直接返回
    if hasattr(predictor, "trackers") and persist:
        return

    # 检查并加载追踪器配置信息
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    # 如果配置中的追踪器类型不是支持的 'bytetrack' 或 'botsort'，抛出异常
    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    # 根据数据集的批量大小初始化追踪器对象列表
    for _ in range(predictor.dataset.bs):
        # 根据配置中的追踪器类型创建对应的追踪器对象
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        # 如果数据集模式不是 'stream'，只需要一个追踪器对象
        if predictor.dataset.mode != "stream":
            break
    # 将创建的追踪器对象列表赋值给预测器对象的 'trackers' 属性
    predictor.trackers = trackers
    # 初始化一个列表用于存储每个批次视频路径，用于在新视频时重置追踪器
    predictor.vid_path = [None] * predictor.dataset.bs


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    对检测到的框进行后处理，并更新对象追踪信息。

    Args:
        predictor (object): 包含预测结果的预测器对象。
        persist (bool, optional): 是否在追踪器已存在时持久化。默认为 False。
    """
    # 获取批次中的路径和图像数据
    path, im0s = predictor.batch[:2]

    # 是否为旋转矩形目标检测任务
    is_obb = predictor.args.task == "obb"
    # 是否为数据流模式
    is_stream = predictor.dataset.mode == "stream"

    # 遍历每张图像
    for i in range(len(im0s)):
        # 获取当前图像对应的追踪器对象
        tracker = predictor.trackers[i if is_stream else 0]
        # 获取当前图像对应的视频路径
        vid_path = predictor.save_dir / Path(path[i]).name

        # 如果不需要持久化且当前视频路径与上一次不同，则重置追踪器
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        # 获取检测到的目标框信息
        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()

        # 如果没有检测到目标，则继续处理下一张图像
        if len(det) == 0:
            continue

        # 更新追踪器并获取更新后的轨迹信息
        tracks = tracker.update(det, im0s[i])

        # 如果没有有效的轨迹信息，则继续处理下一张图像
        if len(tracks) == 0:
            continue

        # 根据轨迹信息对预测结果进行重新排序
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        # 更新预测结果对象中的目标框或旋转矩形信息
        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """
    空函数，用于注册模型的追踪器。

    Args:
        model (object): 需要注册追踪器的模型对象。
        persist (bool): 是否持久化追踪器。
    """
    pass
    # 为模型注册回调函数，用于在预测期间进行对象跟踪

    # 在预测开始时添加回调函数，使用 partial 函数将 persist 参数绑定到 on_predict_start 函数
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))

    # 在预测后处理结束时添加回调函数，使用 partial 函数将 persist 参数绑定到 on_predict_postprocess_end 函数
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
```