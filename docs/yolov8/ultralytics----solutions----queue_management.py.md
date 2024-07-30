# `.\yolov8\ultralytics\solutions\queue_management.py`

```py
# 引入 Python 内置的 collections 模块中的 defaultdict 类
from collections import defaultdict

# 引入 OpenCV 库
import cv2

# 引入自定义的检查函数
from ultralytics.utils.checks import check_imshow, check_requirements

# 引入自定义的绘图相关模块
from ultralytics.utils.plotting import Annotator, colors

# 检查运行环境是否满足要求，要求 shapely 版本 >= 2.0.0
check_requirements("shapely>=2.0.0")

# 引入 shapely 库中的几何对象 Point 和 Polygon
from shapely.geometry import Point, Polygon


class QueueManager:
    """A class to manage the queue in a real-time video stream based on object tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        region_color=(255, 0, 255),
        view_queue_counts=True,
        draw_tracks=False,
        count_txt_color=(255, 255, 255),
        track_color=None,
        region_thickness=5,
        fontsize=0.7,
        """
        Initializes the QueueManager with specified parameters for tracking and counting objects.

        Args:
            names (dict): A dictionary mapping class IDs to class names.
            reg_pts (list of tuples, optional): Points defining the counting region polygon. Defaults to a predefined
                rectangle.
            line_thickness (int, optional): Thickness of the annotation lines. Defaults to 2.
            track_thickness (int, optional): Thickness of the track lines. Defaults to 2.
            view_img (bool, optional): Whether to display the image frames. Defaults to False.
            region_color (tuple, optional): Color of the counting region lines (BGR). Defaults to (255, 0, 255).
            view_queue_counts (bool, optional): Whether to display the queue counts. Defaults to True.
            draw_tracks (bool, optional): Whether to draw tracks of the objects. Defaults to False.
            count_txt_color (tuple, optional): Color of the count text (BGR). Defaults to (255, 255, 255).
            track_color (tuple, optional): Color of the tracks. If None, different colors will be used for different
                tracks. Defaults to None.
            region_thickness (int, optional): Thickness of the counting region lines. Defaults to 5.
            fontsize (float, optional): Font size for the text annotations. Defaults to 0.7.
        """

        # Mouse events state
        self.is_drawing = False  # 初始设定为不绘制状态
        self.selected_point = None  # 初始选择点为空

        # Region & Line Information
        self.reg_pts = reg_pts if reg_pts is not None else [(20, 60), (20, 680), (1120, 680), (1120, 60)]  # 设置计数区域多边形顶点
        self.counting_region = (
            Polygon(self.reg_pts) if len(self.reg_pts) >= 3 else Polygon([(20, 60), (20, 680), (1120, 680), (1120, 60)])
        )  # 根据顶点创建计数区域多边形对象
        self.region_color = region_color  # 设置计数区域线的颜色
        self.region_thickness = region_thickness  # 设置计数区域线的粗细

        # Image and annotation Information
        self.im0 = None  # 初始化图像为空
        self.tf = line_thickness  # 设置注解线的粗细
        self.view_img = view_img  # 是否显示图像帧
        self.view_queue_counts = view_queue_counts  # 是否显示队列计数
        self.fontsize = fontsize  # 设置文本注释的字体大小

        self.names = names  # 类别名称字典
        self.annotator = None  # 注释器对象
        self.window_name = "Ultralytics YOLOv8 Queue Manager"  # 窗口名称

        # Object counting Information
        self.counts = 0  # 对象计数初始为0
        self.count_txt_color = count_txt_color  # 设置计数文本的颜色

        # Tracks info
        self.track_history = defaultdict(list)  # 使用默认字典存储轨迹历史
        self.track_thickness = track_thickness  # 设置轨迹线的粗细
        self.draw_tracks = draw_tracks  # 是否绘制对象的轨迹
        self.track_color = track_color  # 设置轨迹线的颜色，如果为None则使用不同颜色区分不同轨迹

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)  # 检查环境是否支持imshow函数
    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for queue management in a video stream."""

        # 初始化注释器并绘制队列区域
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # 检查是否有跟踪目标的盒子信息
        if tracks[0].boxes.id is not None:
            # 提取跟踪目标的盒子坐标并转换为CPU可处理的格式
            boxes = tracks[0].boxes.xyxy.cpu()
            # 提取类别信息并转换为列表
            clss = tracks[0].boxes.cls.cpu().tolist()
            # 提取跟踪目标的ID并转换为整数格式的列表
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # 遍历每个跟踪目标
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # 在图像上绘制边界框和标签
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                # 更新跟踪历史
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # 如果启用了绘制轨迹功能，则绘制轨迹
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color or colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                # 获取前一个位置信息
                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                # 检查物体是否在计数区域内
                if len(self.reg_pts) >= 3:
                    is_inside = self.counting_region.contains(Point(track_line[-1]))
                    if prev_position is not None and is_inside:
                        self.counts += 1

        # 显示队列计数
        label = f"Queue Counts : {str(self.counts)}"
        if label is not None:
            self.annotator.queue_counts_display(
                label,
                points=self.reg_pts,
                region_color=self.region_color,
                txt_color=self.count_txt_color,
            )

        # 显示完成后重置计数
        self.counts = 0
        self.display_frames()

    def display_frames(self):
        """Displays the current frame with annotations."""
        if self.env_check and self.view_img:
            # 绘制区域边界
            self.annotator.draw_region(reg_pts=self.reg_pts, thickness=self.region_thickness, color=self.region_color)
            # 创建窗口并显示图像
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, self.im0)
            # 在按下 'q' 键时关闭窗口
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    # 存储当前帧到对象的实例变量中
    self.im0 = im0  # Store the current frame
    
    # 调用对象的方法，从跟踪列表中提取并处理跟踪信息
    self.extract_and_process_tracks(tracks)  # Extract and process tracks

    # 如果视图图像标志为真，则显示当前帧
    if self.view_img:
        self.display_frames()  # Display the frame if enabled
    
    # 返回存储的当前帧
    return self.im0
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下代码块

    classes_names = {0: "person", 1: "car"}  # 示例类别名称字典，将整数类别映射到字符串
    queue_manager = QueueManager(classes_names)
    # 创建一个队列管理器对象，使用给定的类别名称字典初始化
```