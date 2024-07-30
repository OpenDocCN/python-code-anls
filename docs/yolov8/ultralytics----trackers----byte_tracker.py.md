# `.\yolov8\ultralytics\trackers\byte_tracker.py`

```py
# 导入 NumPy 库，用于处理数值计算
import numpy as np

# 从上级目录的 utils 模块中导入 LOGGER 对象
from ..utils import LOGGER
# 从 utils.ops 模块中导入 xywh2ltwh 函数，用于坐标转换
from ..utils.ops import xywh2ltwh
# 从当前目录下的 basetrack 模块中导入 BaseTrack 类和 TrackState 枚举
from .basetrack import BaseTrack, TrackState
# 从当前目录下的 utils 模块中导入 matching 函数
from .utils import matching
# 从当前目录下的 utils.kalman_filter 模块中导入 KalmanFilterXYAH 类
from .utils.kalman_filter import KalmanFilterXYAH

# 定义 STrack 类，继承自 BaseTrack 类
class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah(tlwh): Convert tlwh bounding box to xyah format.
    """

    # 共享的 Kalman 过滤器实例，用于所有 STrack 实例的预测
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """Initialize new STrack instance."""
        super().__init__()
        # 将 xywh 转换为 tlwh 格式并存储在私有属性 _tlwh 中
        # xywh 是包含位置和尺寸信息的数组，score 是置信度，cls 是类别标签
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        # 初始化 Kalman 过滤器、状态估计向量和协方差矩阵
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        # 标识是否已激活的布尔值
        self.is_activated = False

        # 分数（置信度）、轨迹长度、类别标签和对象标识符
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]  # 最后一个元素是对象的唯一标识符
        self.angle = xywh[4] if len(xywh) == 6 else None  # 如果有角度信息则存储角度

    def predict(self):
        """Predicts mean and covariance using Kalman filter."""
        # 复制当前均值状态
        mean_state = self.mean.copy()
        # 如果跟踪状态不是已跟踪状态，则将速度信息置零
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        # 使用 Kalman 过滤器预测下一个状态的均值和协方差
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for given stracks."""
        # 如果输入的stracks为空，则直接返回，不进行处理
        if len(stracks) <= 0:
            return
        
        # 从每个STrack对象中提取mean属性的副本，形成numpy数组multi_mean
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        # 从每个STrack对象中提取covariance属性，形成numpy数组multi_covariance
        multi_covariance = np.asarray([st.covariance for st in stracks])
        
        # 遍历每个STrack对象，如果其状态不是Tracked，则将其mean属性的第8个元素设为0
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        
        # 调用共享的Kalman滤波器进行多对象预测，更新multi_mean和multi_covariance
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        
        # 将预测得到的mean和covariance更新回各个STrack对象中
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix."""
        # 如果输入的stracks不为空
        if len(stracks) > 0:
            # 从每个STrack对象中提取mean属性的副本，形成numpy数组multi_mean
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            # 从每个STrack对象中提取covariance属性，形成numpy数组multi_covariance
            multi_covariance = np.asarray([st.covariance for st in stracks])

            # 从H矩阵中提取旋转部分R，并构造一个扩展到8x8维度的R矩阵R8x8
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            # 提取H矩阵中的平移部分t
            t = H[:2, 2]

            # 遍历每个STrack对象，根据H矩阵更新其mean和covariance属性
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)  # 通过R8x8变换更新mean
                mean[:2] += t  # 加上平移t
                cov = R8x8.dot(cov).dot(R8x8.transpose())  # 更新covariance

                # 将更新后的mean和covariance重新赋值回各个STrack对象中
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        # 初始化新的跟踪对象，设置其kalman_filter、track_id、mean和covariance等属性
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track with a new detection."""
        # 使用新的检测数据更新已丢失的跟踪对象的mean和covariance属性
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
    # 更新匹配跟踪的状态。
    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        # 更新帧 ID
        self.frame_id = frame_id
        # 增加跟踪段长度
        self.tracklet_len += 1

        # 获取新跟踪的位置信息
        new_tlwh = new_track.tlwh
        # 使用卡尔曼滤波器更新跟踪状态的均值和协方差
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        # 设置跟踪状态为已跟踪
        self.state = TrackState.Tracked
        # 激活跟踪器
        self.is_activated = True

        # 更新跟踪器的分数
        self.score = new_track.score
        # 更新跟踪器的类别
        self.cls = new_track.cls
        # 更新跟踪器的角度
        self.angle = new_track.angle
        # 更新跟踪器的索引
        self.idx = new_track.idx

    # 将边界框的 top-left-width-height 格式转换为 x-y-aspect-height 格式。
    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    # 返回当前位置的边界框格式 (top left x, top left y, width, height)。
    @property
    def tlwh(self):
        """Get current position in bounding box format (top left x, top left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    # 将边界框转换为格式 (min x, min y, max x, max y)，即 (top left, bottom right)。
    @property
    def xyxy(self):
        """Convert bounding box to format (min x, min y, max x, max y), i.e., (top left, bottom right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    # 将边界框转换为格式 (center x, center y, aspect ratio, height)，其中 aspect ratio 是宽度 / 高度。
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format (center x, center y, aspect ratio, height), where the aspect ratio is width /
        height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    # 返回当前位置的边界框格式 (center x, center y, width, height)。
    @property
    def xywh(self):
        """Get current position in bounding box format (center x, center y, width, height)."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    # 返回当前位置的边界框格式 (center x, center y, width, height, angle)。
    @property
    def xywha(self):
        """Get current position in bounding box format (center x, center y, width, height, angle)."""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    # 返回当前跟踪结果。
    @property
    def result(self):
        """Get current tracking results."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    # 返回 BYTETracker 对象的字符串表示，包括开始和结束帧以及跟踪 ID。
    def __repr__(self):
        """Return a string representation of the BYTETracker object with start and end frames and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"
class BYTETracker:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    The class is responsible for initializing, updating, and managing the tracks for detected objects in a video
    sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for
    predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (object): Kalman Filter object.

    Methods:
        update(results, img=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IoU.
    """

    def __init__(self, args, frame_rate=30):
        """Initialize a YOLOv8 object to track objects with given arguments and frame rate."""
        # 初始化已跟踪的轨迹列表
        self.tracked_stracks = []  # type: list[STrack]
        # 初始化丢失的轨迹列表
        self.lost_stracks = []  # type: list[STrack]
        # 初始化移除的轨迹列表
        self.removed_stracks = []  # type: list[STrack]

        # 当前帧的 ID
        self.frame_id = 0
        # 命令行参数
        self.args = args
        # 计算最大丢失帧数，用于确定轨迹丢失
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        # 获取 Kalman 过滤器对象
        self.kalman_filter = self.get_kalmanfilter()
        # 重置 STrack 的 ID 计数器
        self.reset_id()

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes."""
        # 返回一个 Kalman 过滤器对象，用于跟踪边界框
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize object tracking with detections and scores using STrack algorithm."""
        # 使用 STrack 算法，根据检测结果和分数初始化对象跟踪
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU and fuses scores."""
        # 使用 IoU 计算轨迹和检测之间的距离，并融合分数
        dists = matching.iou_distance(tracks, detections)
        # 如果不是 MOT20 数据集，执行分数融合
        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        return dists
    def multi_predict(self, tracks):
        """
        Returns the predicted tracks using the YOLOv8 network.
        """
        # 调用 STrack 类的 multi_predict 方法来进行多目标跟踪预测
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """
        Resets the ID counter of STrack.
        """
        # 调用 STrack 类的 reset_id 静态方法，重置 STrack 的 ID 计数器
        STrack.reset_id()

    def reset(self):
        """
        Reset tracker.
        """
        # 初始化跟踪器，重置所有跟踪状态
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()  # 调用重置 ID 计数器的方法

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """
        Combine two lists of stracks into a single one.
        """
        # 使用字典记录已经存在的 track_id
        exists = {}
        res = []
        # 将 tlista 中的每个元素添加到 res 中，并记录其 track_id 到 exists 字典中
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        # 遍历 tlistb，如果其 track_id 不存在于 exists 字典中，则添加到 res 中
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """
        DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        """
        # 使用集合推导式创建 tlista 中每个元素的 track_id 集合
        track_ids_b = {t.track_id for t in tlistb}
        # 返回 tlista 中所有 track_id 不在 track_ids_b 集合中的元素列表
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """
        Remove duplicate stracks with non-maximum IoU distance.
        """
        # 计算 stracksa 和 stracksb 之间的 IoU 距离矩阵
        pdist = matching.iou_distance(stracksa, stracksb)
        # 找到 IoU 距离小于 0.15 的索引对
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        # 根据跟踪器的时间长度，判断哪些是重复的跟踪器
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        # 返回去除重复跟踪器后的结果列表
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
```