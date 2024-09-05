# `.\yolov8\ultralytics\solutions\distance_calculation.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入数学库
import math

# 导入 OpenCV 库
import cv2

# 导入自定义模块
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

# 距离计算类，用于实时视频流中基于对象轨迹计算距离
class DistanceCalculation:
    """A class to calculate distance between two objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        pixels_per_meter=10,
        view_img=False,
        line_thickness=2,
        line_color=(255, 255, 0),
        centroid_color=(255, 0, 255),
    ):
        """
        Initializes the DistanceCalculation class with the given parameters.

        Args:
            names (dict): Dictionary of classes names.
            pixels_per_meter (int, optional): Conversion factor from pixels to meters. Defaults to 10.
            view_img (bool, optional): Flag to indicate if the video stream should be displayed. Defaults to False.
            line_thickness (int, optional): Thickness of the lines drawn on the image. Defaults to 2.
            line_color (tuple, optional): Color of the lines drawn on the image (BGR format). Defaults to (255, 255, 0).
            centroid_color (tuple, optional): Color of the centroids drawn (BGR format). Defaults to (255, 0, 255).
        """
        # 图像和注解器相关信息初始化
        self.im0 = None  # 初始图像置空
        self.annotator = None  # 注解器置空
        self.view_img = view_img  # 是否显示视频流
        self.line_color = line_color  # 线条颜色
        self.centroid_color = centroid_color  # 质心颜色

        # 预测和跟踪信息初始化
        self.clss = None  # 类别信息置空
        self.names = names  # 类别名称字典
        self.boxes = None  # 边界框信息置空
        self.line_thickness = line_thickness  # 线条粗细
        self.trk_ids = None  # 跟踪 ID 信息置空

        # 距离计算信息初始化
        self.centroids = []  # 质心列表
        self.pixel_per_meter = pixels_per_meter  # 像素与米的转换因子

        # 鼠标事件信息初始化
        self.left_mouse_count = 0  # 左键点击次数
        self.selected_boxes = {}  # 选中的边界框字典

        # 检查环境是否支持 imshow 函数
        self.env_check = check_imshow(warn=True)
    # 处理鼠标事件以选择实时视频流中的区域

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        Handles mouse events to select regions in a real-time video stream.

        Args:
            event (int): Type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): X-coordinate of the mouse pointer.
            y (int): Y-coordinate of the mouse pointer.
            flags (int): Flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY, etc.).
            param (dict): Additional parameters passed to the function.
        """
        # 如果是左键单击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            # 增加左键点击计数
            self.left_mouse_count += 1
            # 如果左键点击次数小于等于2
            if self.left_mouse_count <= 2:
                # 遍历每个盒子和其对应的跟踪 ID
                for box, track_id in zip(self.boxes, self.trk_ids):
                    # 如果鼠标点击在当前盒子的范围内，并且该跟踪 ID 不在已选择的盒子中
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        # 将该跟踪 ID 和盒子加入已选择的盒子字典中
                        self.selected_boxes[track_id] = box

        # 如果是右键单击事件
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 清空已选择的盒子字典
            self.selected_boxes = {}
            # 重置左键点击计数为 0
            self.left_mouse_count = 0

    # 从提供的数据中提取跟踪结果
    def extract_tracks(self, tracks):
        """
        Extracts tracking results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        # 获取第一个轨迹的盒子坐标并转换为 CPU 上的数组
        self.boxes = tracks[0].boxes.xyxy.cpu()
        # 获取第一个轨迹的类别并转换为 CPU 上的列表
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        # 获取第一个轨迹的 ID 并转换为 CPU 上的列表
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    # 静态方法：计算边界框的质心
    @staticmethod
    def calculate_centroid(box):
        """
        Calculates the centroid of a bounding box.

        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            (tuple): Centroid coordinates (x, y).
        """
        # 计算边界框的中心点坐标
        return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

    # 计算两个质心之间的距离
    def calculate_distance(self, centroid1, centroid2):
        """
        Calculates the distance between two centroids.

        Args:
            centroid1 (tuple): Coordinates of the first centroid (x, y).
            centroid2 (tuple): Coordinates of the second centroid (x, y).

        Returns:
            (tuple): Distance in meters and millimeters.
        """
        # 计算像素距离
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        # 将像素距离转换为米
        distance_m = pixel_distance / self.pixel_per_meter
        # 将米转换为毫米
        distance_mm = distance_m * 1000
        # 返回距离的米和毫米表示
        return distance_m, distance_mm
    def start_process(self, im0, tracks):
        """
        Processes the video frame and calculates the distance between two bounding boxes.

        Args:
            im0 (ndarray): The image frame.
            tracks (list): List of tracks obtained from the object tracking process.

        Returns:
            (ndarray): The processed image frame.
        """
        # 将传入的图像帧赋给对象的成员变量
        self.im0 = im0

        # 检查第一个跟踪目标的边界框是否有标识号
        if tracks[0].boxes.id is None:
            # 如果没有标识号，根据需要显示图像帧，并返回未处理的图像帧
            if self.view_img:
                self.display_frames()
            return im0

        # 提取跟踪目标的信息
        self.extract_tracks(tracks)

        # 创建一个图像注释器对象
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)

        # 对每个边界框进行标注
        for box, cls, track_id in zip(self.boxes, self.clss, self.trk_ids):
            # 标注边界框及其类别
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            # 如果已选择了两个边界框，则更新选定的边界框信息
            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        # 如果已选择了两个边界框，则计算它们的质心
        if len(self.selected_boxes) == 2:
            self.centroids = [self.calculate_centroid(self.selected_boxes[trk_id]) for trk_id in self.selected_boxes]

            # 计算并绘制两个边界框之间的距离及线条
            distance_m, distance_mm = self.calculate_distance(self.centroids[0], self.centroids[1])
            self.annotator.plot_distance_and_line(
                distance_m, distance_mm, self.centroids, self.line_color, self.centroid_color
            )

        # 清空质心列表
        self.centroids = []

        # 如果需要显示图像并且环境检查通过，则显示图像帧
        if self.view_img and self.env_check:
            self.display_frames()

        # 返回处理后的图像帧
        return im0

    def display_frames(self):
        """Displays the current frame with annotations."""
        # 创建一个窗口并显示图像帧及其相关注释
        cv2.namedWindow("Ultralytics Distance Estimation")
        cv2.setMouseCallback("Ultralytics Distance Estimation", self.mouse_event_for_distance)
        cv2.imshow("Ultralytics Distance Estimation", self.im0)

        # 等待用户按键操作，如果按下 'q' 键则退出函数
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
if __name__ == "__main__":
    # 当该脚本作为主程序运行时执行以下代码块

    names = {0: "person", 1: "car"}  # 示例类别名称的字典，键为索引，值为类别名称

    # 创建 DistanceCalculation 的实例，传入类别名称的字典作为参数
    distance_calculation = DistanceCalculation(names)
```