# `.\yolov8\ultralytics\solutions\heatmap.py`

```py
# 导入必要的库和模块
from collections import defaultdict  # 导入collections模块中的defaultdict类
import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库

# 导入自定义的工具函数和类
from ultralytics.utils.checks import check_imshow, check_requirements  # 导入检查函数
from ultralytics.utils.plotting import Annotator  # 导入标注类

# 检查并确保所需的第三方库安装正确
check_requirements("shapely>=2.0.0")

# 导入用于空间几何计算的shapely库中的特定类和函数
from shapely.geometry import LineString, Point, Polygon

class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""
    
    def __init__(
        self,
        names,
        imw=0,
        imh=0,
        colormap=cv2.COLORMAP_JET,
        heatmap_alpha=0.5,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        count_reg_pts=None,
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        count_reg_color=(255, 0, 255),
        region_thickness=5,
        line_dist_thresh=15,
        line_thickness=2,
        decay_factor=0.99,
        shape="circle",
    ):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        # Visual information
        self.annotator = None  # 初始化注释器为None
        self.view_img = view_img  # 设置是否显示图像
        self.shape = shape  # 设置热图形状

        self.initialized = False  # 标记对象是否已初始化
        self.names = names  # 类别名称列表

        # Image information
        self.imw = imw  # 图像宽度
        self.imh = imh  # 图像高度
        self.im0 = None  # 初始化图像对象为None
        self.tf = line_thickness  # 线条粗细
        self.view_in_counts = view_in_counts  # 是否显示计数内部
        self.view_out_counts = view_out_counts  # 是否显示计数外部

        # Heatmap colormap and heatmap np array
        self.colormap = colormap  # 热图颜色映射
        self.heatmap = None  # 初始化热图数组为None
        self.heatmap_alpha = heatmap_alpha  # 热图透明度

        # Predict/track information
        self.boxes = []  # 目标框列表
        self.track_ids = []  # 跟踪目标的ID列表
        self.clss = []  # 目标类别列表
        self.track_history = defaultdict(list)  # 跟踪历史记录

        # Region & Line Information
        self.counting_region = None  # 计数区域对象
        self.line_dist_thresh = line_dist_thresh  # 线段距离阈值
        self.region_thickness = region_thickness  # 区域厚度
        self.region_color = count_reg_color  # 区域颜色

        # Object Counting Information
        self.in_counts = 0  # 进入计数
        self.out_counts = 0  # 离开计数
        self.count_ids = []  # 计数的目标ID列表
        self.class_wise_count = {}  # 按类别统计计数
        self.count_txt_color = count_txt_color  # 计数文本颜色
        self.count_bg_color = count_bg_color  # 计数背景颜色
        self.cls_txtdisplay_gap = 50  # 类别文本显示间隔

        # Decay factor
        self.decay_factor = decay_factor  # 衰减因子

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)  # 检查环境是否支持imshow函数

        # Region and line selection
        self.count_reg_pts = count_reg_pts  # 计数区域的点集
        print(self.count_reg_pts)  # 打印计数区域的点集
        if self.count_reg_pts is not None:
            if len(self.count_reg_pts) == 2:
                print("Line Counter Initiated.")  # 打印线条计数器初始化信息
                self.counting_region = LineString(self.count_reg_pts)  # 使用两点创建线计数器区域
            elif len(self.count_reg_pts) >= 3:
                print("Polygon Counter Initiated.")  # 打印多边形计数器初始化信息
                self.counting_region = Polygon(self.count_reg_pts)  # 使用多于三个点创建多边形计数器区域
            else:
                print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
                print("Using Line Counter Now")
                self.counting_region = LineString(self.count_reg_pts)  # 使用线计数器作为默认选择

        # Shape of heatmap, if not selected
        if self.shape not in {"circle", "rect"}:
            print("Unknown shape value provided, 'circle' & 'rect' supported")
            print("Using Circular shape now")
            self.shape = "circle"  # 如果未选择热图形状，则默认选择圆形

    def extract_results(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        if tracks[0].boxes.id is not None:
            self.boxes = tracks[0].boxes.xyxy.cpu()  # 提取目标框坐标并转换为CPU格式
            self.clss = tracks[0].boxes.cls.tolist()  # 提取目标类别并转换为列表格式
            self.track_ids = tracks[0].boxes.id.int().tolist()  # 提取目标ID并转换为整数列表格式
    def display_frames(self):
        """Display frames method."""
        # 使用OpenCV显示图像窗口，标题为"Ultralytics Heatmap"，显示self.im0图像
        cv2.imshow("Ultralytics Heatmap", self.im0)
        
        # 等待用户按键输入，等待时间为1毫秒，并检查是否按下键盘上的q键
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # 如果检测到按下q键，返回退出方法
            return
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行
    classes_names = {0: "person", 1: "car"}  # 示例类别名称字典，映射类别编号到类别名称
    # 创建一个 Heatmap 对象，传入类别名称字典作为参数
    heatmap = Heatmap(classes_names)
```