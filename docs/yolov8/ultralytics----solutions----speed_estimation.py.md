# `.\yolov8\ultralytics\solutions\speed_estimation.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict
from time import time

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


class SpeedEstimator:
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

    def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, region_thickness=5, spdl_dist_thresh=10):
        """
        Initializes the SpeedEstimator with the given parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list, optional): List of region points for speed estimation. Defaults to [(20, 400), (1260, 400)].
            view_img (bool, optional): Whether to display the image with annotations. Defaults to False.
            line_thickness (int, optional): Thickness of the lines for drawing boxes and tracks. Defaults to 2.
            region_thickness (int, optional): Thickness of the region lines. Defaults to 5.
            spdl_dist_thresh (int, optional): Distance threshold for speed calculation. Defaults to 10.
        """
        # Visual & image information
        self.im0 = None  # 初始化原始图像为 None
        self.annotator = None  # 初始化标注器为 None
        self.view_img = view_img  # 设置是否显示图像的标志

        # Region information
        self.reg_pts = reg_pts if reg_pts is not None else [(20, 400), (1260, 400)]  # 设置用于速度估计的区域点，默认为 [(20, 400), (1260, 400)]
        self.region_thickness = region_thickness  # 设置区域线的粗细

        # Tracking information
        self.clss = None  # 初始化类别信息为 None
        self.names = names  # 设置类别名称字典
        self.boxes = None  # 初始化边界框信息为 None
        self.trk_ids = None  # 初始化跟踪 ID 信息为 None
        self.line_thickness = line_thickness  # 设置绘制框和轨迹线的粗细
        self.trk_history = defaultdict(list)  # 初始化跟踪历史为默认字典列表

        # Speed estimation information
        self.current_time = 0  # 初始化当前时间为 0
        self.dist_data = {}  # 初始化距离数据字典为空字典
        self.trk_idslist = []  # 初始化跟踪 ID 列表为空列表
        self.spdl_dist_thresh = spdl_dist_thresh  # 设置速度计算的距离阈值
        self.trk_previous_times = {}  # 初始化上一个时间点的跟踪时间信息为空字典
        self.trk_previous_points = {}  # 初始化上一个时间点的跟踪点信息为空字典

        # Check if the environment supports imshow
        self.env_check = check_imshow(warn=True)  # 检查环境是否支持 imshow 函数并设置警告为 True

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided tracking data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()  # 提取边界框信息并转换为 CPU 格式
        self.clss = tracks[0].boxes.cls.cpu().tolist()  # 提取类别信息并转换为列表格式
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()  # 提取跟踪 ID 并转换为整数列表格式
    def store_track_info(self, track_id, box):
        """
        存储跟踪数据。

        Args:
            track_id (int): 对象的跟踪ID。
            box (list): 对象边界框数据。

        Returns:
            (list): 给定track_id的更新跟踪历史记录。
        """
        # 获取当前跟踪ID对应的历史跟踪数据
        track = self.trk_history[track_id]
        
        # 计算边界框中心点坐标
        bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
        
        # 将计算得到的中心点坐标添加到跟踪历史中
        track.append(bbox_center)

        # 如果跟踪历史长度超过30，移除最早的数据
        if len(track) > 30:
            track.pop(0)

        # 将跟踪历史转换为numpy数组，并更新self.trk_pts
        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        
        # 返回更新后的跟踪历史
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        绘制跟踪路径和边界框。

        Args:
            track_id (int): 对象的跟踪ID。
            box (list): 对象边界框数据。
            cls (str): 对象类别名称。
            track (list): 用于绘制跟踪路径的跟踪历史。
        """
        # 根据跟踪ID是否在速度数据中确定显示的速度标签
        speed_label = f"{int(self.dist_data[track_id])} km/h" if track_id in self.dist_data else self.names[int(cls)]
        
        # 根据跟踪ID是否在速度数据中确定绘制边界框的颜色
        bbox_color = colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)

        # 在图像上绘制边界框和速度标签
        self.annotator.box_label(box, speed_label, bbox_color)
        
        # 在图像上绘制跟踪路径
        cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1)
        
        # 在图像上绘制跟踪路径的最后一个点
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        计算对象的速度。

        Args:
            trk_id (int): 对象的跟踪ID。
            track (list): 用于绘制跟踪路径的跟踪历史。
        """
        # 如果对象最后一个位置不在指定的区域内，则返回
        if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
            return
        
        # 根据对象最后一个位置的y坐标是否在指定距离范围内确定运动方向
        if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
            direction = "known"
        elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
            direction = "known"
        else:
            direction = "unknown"

        # 如果前一次跟踪时间不为0，并且运动方向已知且跟踪ID不在列表中
        if self.trk_previous_times.get(trk_id) != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            # 将跟踪ID添加到列表中
            self.trk_idslist.append(trk_id)

            # 计算跟踪点的时间差和位置差，从而计算速度
            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                speed = dist_difference / time_difference
                self.dist_data[trk_id] = speed

        # 更新跟踪ID的前一次跟踪时间和位置
        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]
    def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Estimates the speed of objects based on tracking data.

        Args:
            im0 (ndarray): Image.
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple, optional): Color to use when drawing regions. Defaults to (255, 0, 0).

        Returns:
            (ndarray): The image with annotated boxes and tracks.
        """
        # 将传入的图像赋给对象属性
        self.im0 = im0
        # 检查第一个轨迹是否具有有效的标识符，如果没有，显示图像并返回原始图像
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                # 在视图模式开启且环境检查通过时，显示当前帧图像
                self.display_frames()
            return im0

        # 提取轨迹信息
        self.extract_tracks(tracks)
        # 创建一个注解器对象，并设置线宽度
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)
        # 绘制区域，使用给定的颜色和线条粗细
        self.annotator.draw_region(reg_pts=self.reg_pts, color=region_color, thickness=self.region_thickness)

        # 遍历每个框、轨迹ID和类别，并处理其信息
        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            # 存储轨迹信息，并返回当前轨迹
            track = self.store_track_info(trk_id, box)

            # 如果当前轨迹ID不在之前时间的记录中，将其初始化为0
            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            # 绘制框和轨迹，并将其绘制到图像上
            self.plot_box_and_track(trk_id, box, cls, track)
            # 计算当前轨迹的速度
            self.calculate_speed(trk_id, track)

        # 如果视图模式开启且环境检查通过，显示当前帧图像
        if self.view_img and self.env_check:
            self.display_frames()

        # 返回带有注释框和轨迹的图像
        return im0

    def display_frames(self):
        """Displays the current frame."""
        # 显示当前帧图像，窗口标题为 "Ultralytics Speed Estimation"
        cv2.imshow("Ultralytics Speed Estimation", self.im0)
        # 检测键盘输入是否是 'q'，如果是则退出显示
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
if __name__ == "__main__":
    # 如果这个脚本被直接执行而不是被导入为模块，则执行以下代码块
    names = {0: "person", 1: "car"}  # 示例类别名称，用于初始化速度估计器对象
    speed_estimator = SpeedEstimator(names)
```