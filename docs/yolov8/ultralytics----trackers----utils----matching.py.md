# `.\yolov8\ultralytics\trackers\utils\matching.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np  # 导入 NumPy 库，用于数值计算
import scipy  # 导入 SciPy 库，用于科学计算
from scipy.spatial.distance import cdist  # 从 SciPy 库的 spatial.distance 模块导入 cdist 函数

from ultralytics.utils.metrics import batch_probiou, bbox_ioa  # 从 ultralytics.utils.metrics 模块导入 batch_probiou 和 bbox_ioa 函数

try:
    import lap  # 尝试导入 lap 模块，用于 linear_assignment

    assert lap.__version__  # 验证 lap 模块不是一个目录
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lapx>=0.5.2")  # 检查 lapx 版本是否符合要求，推荐从 https://github.com/rathaROG/lapx 更新到 lap 包
    import lap  # 导入 lap 模块

def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    """
    Perform linear assignment using scipy or lap.lapjv.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool, optional): Whether to use lap.lapjv. Defaults to True.

    Returns:
        Tuple with:
            - matched indices
            - unmatched indices from 'a'
            - unmatched indices from 'b'
    """

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]  # 构建匹配对列表，包含匹配的索引对
        unmatched_a = np.where(x < 0)[0]  # 找出未匹配的 'a' 索引
        unmatched_b = np.where(y < 0)[0]  # 找出未匹配的 'b' 索引
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # 使用 SciPy 中的 linear_sum_assignment 函数进行线性分配
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])  # 根据阈值筛选出有效匹配
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))  # 如果没有有效匹配，则所有 'a' 都为未匹配
            unmatched_b = list(np.arange(cost_matrix.shape[1]))  # 如果没有有效匹配，则所有 'b' 都为未匹配
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))  # 找出未匹配的 'a' 索引
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))  # 找出未匹配的 'b' 索引

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU.
    """

    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks  # 将 atracks 赋值给 atlbrs
        btlbrs = btracks  # 将 btracks 赋值给 btlbrs
    # 如果条件不满足，执行以下操作：
    else:
        # 生成包含所有 A 跟踪目标边界框坐标的列表，若目标存在角度信息，则使用 xywha 格式，否则使用 xyxy 格式
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        # 生成包含所有 B 跟踪目标边界框坐标的列表，若目标存在角度信息，则使用 xywha 格式，否则使用 xyxy 格式
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    # 创建一个形状为 (len(atlbrs), len(btlbrs)) 的全零数组，用于存储计算的 IoU 值
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    
    # 如果 atlbrs 和 btlbrs 列表都不为空
    if len(atlbrs) and len(btlbrs):
        # 如果列表中第一个元素是长度为 5 的数组，则调用 batch_probiou 计算 IoU
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),  # 转换为连续内存的 numpy 数组
                np.ascontiguousarray(btlbrs, dtype=np.float32),  # 转换为连续内存的 numpy 数组
            ).numpy()  # 将结果转换为 numpy 数组
        else:
            # 否则，调用 bbox_ioa 计算 IoU，设定参数 iou=True
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),  # 转换为连续内存的 numpy 数组
                np.ascontiguousarray(btlbrs, dtype=np.float32),  # 转换为连续内存的 numpy 数组
                iou=True,  # 设置计算 IoU
            )
    
    # 返回 1 减去计算得到的 IoU 矩阵，即得到的 cost matrix
    return 1 - ious
# 计算跟踪目标与检测目标之间基于嵌入向量的距离矩阵

def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks.
            跟踪目标的列表，每个元素是一个STrack对象。
        detections (list[BaseTrack]): List of detections.
            检测目标的列表，每个元素是一个BaseTrack对象。
        metric (str, optional): Metric for distance computation. Defaults to 'cosine'.
            距离计算所用的度量标准，默认为余弦相似度。

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings.
            基于嵌入向量计算的成本矩阵。
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    # 提取检测目标的特征向量
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)

    # 提取跟踪目标的平滑特征向量
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)

    # 计算距离矩阵，使用指定的度量标准（metric）
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features

    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """
    Fuses cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
            包含分配成本值的矩阵。
        detections (list[BaseTrack]): List of detections with scores.
            带有分数的检测目标列表。

    Returns:
        (np.ndarray): Fused similarity matrix.
            融合后的相似性矩阵。
    """

    if cost_matrix.size == 0:
        return cost_matrix

    # 根据成本矩阵计算IoU相似度
    iou_sim = 1 - cost_matrix

    # 提取检测目标的分数
    det_scores = np.array([det.score for det in detections])

    # 将检测目标的分数扩展为与成本矩阵相同的形状
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)

    # 将IoU相似度与检测目标分数相乘，进行融合
    fuse_sim = iou_sim * det_scores

    # 返回融合后的相似性矩阵，即融合成本
    return 1 - fuse_sim
```