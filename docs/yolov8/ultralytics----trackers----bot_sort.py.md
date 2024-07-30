# `.\yolov8\ultralytics\trackers\bot_sort.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import deque  # 导入 deque 数据结构，用于存储特征向量历史
import numpy as np  # 导入 NumPy 库，用于数值计算

from .basetrack import TrackState  # 导入 TrackState 类，用于跟踪状态
from .byte_tracker import BYTETracker, STrack  # 导入 BYTETracker 和 STrack 类
from .utils import matching  # 导入 matching 函数，用于匹配操作
from .utils.gmc import GMC  # 导入 GMC 类
from .utils.kalman_filter import KalmanFilterXYWH  # 导入 KalmanFilterXYWH 类


class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8, adding object tracking features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.
        predict(): Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Usage:
        bo_track = BOTrack(tlwh, score, cls, feat)
        bo_track.predict()
        bo_track.update(new_track, frame_id)
    """

    shared_kalman = KalmanFilterXYWH()  # 创建一个共享的 KalmanFilterXYWH 对象

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """Initialize YOLOv8 object with temporal parameters, such as feature history, alpha and current features."""
        super().__init__(tlwh, score, cls)  # 调用父类 STrack 的初始化方法

        self.smooth_feat = None  # 初始化平滑后的特征向量
        self.curr_feat = None  # 初始化当前的特征向量
        if feat is not None:
            self.update_features(feat)  # 若提供了特征向量 feat，则更新特征向量

        self.features = deque([], maxlen=feat_history)  # 创建一个空的 deque，用于存储特征向量历史，最大长度为 feat_history
        self.alpha = 0.9  # 初始化指数移动平均的平滑因子

    def update_features(self, feat):
        """Update features vector and smooth it using exponential moving average."""
        feat /= np.linalg.norm(feat)  # 对特征向量进行归一化处理
        self.curr_feat = feat  # 更新当前特征向量

        if self.smooth_feat is None:
            self.smooth_feat = feat  # 若平滑后的特征向量还未初始化，则直接赋值为当前特征向量
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat  # 否则进行指数移动平均平滑处理

        self.features.append(feat)  # 将当前特征向量添加到 deque 中
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  # 对平滑后的特征向量再次进行归一化处理
    def predict(self):
        """Predicts the mean and covariance using Kalman filter."""
        # 复制当前的均值状态
        mean_state = self.mean.copy()
        # 如果跟踪状态不是已跟踪，则将速度置为零
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        # 使用卡尔曼滤波器进行预测，更新均值和协方差
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        # 如果新的轨迹具有当前特征，则更新特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        # 调用父类的重新激活方法，传递新的轨迹、帧ID和是否需要新ID的信息
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Update the YOLOv8 instance with new track and frame ID."""
        # 如果新的轨迹具有当前特征，则更新特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        # 调用父类的更新方法，传递新的轨迹和帧ID
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        # 如果均值为空，则返回私有属性 `_tlwh` 的副本
        if self.mean is None:
            return self._tlwh.copy()
        # 否则，从均值中获取当前位置的副本，并计算其左上角坐标和大小
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance of multiple object tracks using shared Kalman filter."""
        # 如果输入的轨迹数小于等于0，则直接返回
        if len(stracks) <= 0:
            return
        # 获取所有轨迹的均值和协方差，并转换为数组
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        # 对于每一个轨迹，如果其状态不是已跟踪，则将速度置为零
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        # 使用共享卡尔曼滤波器进行多目标预测，更新均值和协方差
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        # 将更新后的均值和协方差分配回每一个轨迹
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
        # 调用静态方法 tlwh_to_xywh 进行坐标转换
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        # 复制传入的 tlwh 数组
        ret = np.asarray(tlwh).copy()
        # 计算中心点坐标并更新 ret 数组
        ret[:2] += ret[2:] / 2
        return ret
# BOTSORT 类是 BYTETracker 类的扩展版本，专为使用 ReID 和 GMC 算法进行对象跟踪的 YOLOv8 设计。

class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (object): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (object): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.
        init_track(dets, scores, cls, img): Initialize track with detections, scores, and classes.
        get_dists(tracks, detections): Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.

    Usage:
        bot_sort = BOTSORT(args, frame_rate)
        bot_sort.init_track(dets, scores, cls, img)
        bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize YOLOv8 object with ReID module and GMC algorithm."""
        # 调用父类的初始化方法
        super().__init__(args, frame_rate)
        # 设置空间接近度阈值和外观相似度阈值
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            # 如果启用了 ReID，但尚未支持 BoT-SORT(reid)
            self.encoder = None
        # 初始化 GMC 算法实例
        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for object tracking."""
        # 返回用于对象跟踪的 KalmanFilterXYWH 实例
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize track with detections, scores, and classes."""
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            # 如果启用了 ReID 并且有编码器对象，则进行特征推断
            features_keep = self.encoder.inference(img, dets)
            # 返回包含位置、分数、类别和特征的 BOTrack 实例列表
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]
        else:
            # 返回包含位置、分数和类别的 BOTrack 实例列表
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]
    # 获取跟踪目标与检测目标之间的距离，使用IoU和（可选的）ReID嵌入特征。
    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)  # 计算使用IoU的距离
        dists_mask = dists > self.proximity_thresh  # 创建距离阈值掩码

        # TODO: mot20
        # 如果不是使用mot20数据集，则进行融合评分处理
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)

        # 如果启用了ReID并且有编码器
        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0  # 计算嵌入距离
            emb_dists[emb_dists > self.appearance_thresh] = 1.0  # 应用外观阈值
            emb_dists[dists_mask] = 1.0  # 应用距离掩码
            dists = np.minimum(dists, emb_dists)  # 取最小距离

        return dists  # 返回距离矩阵

    # 使用YOLOv8模型进行多对象预测和跟踪
    def multi_predict(self, tracks):
        BOTrack.multi_predict(tracks)

    # 重置跟踪器状态
    def reset(self):
        super().reset()  # 调用父类的重置方法
        self.gmc.reset_params()  # 重置特定的参数
```