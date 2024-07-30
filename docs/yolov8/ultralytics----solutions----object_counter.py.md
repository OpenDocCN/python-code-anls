# `.\yolov8\ultralytics\solutions\object_counter.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入必要的库
from collections import defaultdict
import cv2
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

# 检查并确保安装了必需的第三方库
check_requirements("shapely>=2.0.0")

# 导入 shapely 库中的几何图形类
from shapely.geometry import LineString, Point, Polygon

class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        count_reg_color=(255, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        # 初始化对象计数器的各种参数
        # names: 物体类别的名称列表
        # reg_pts: 计数区域的定义点列表
        # count_reg_color: 计数区域的颜色
        # count_txt_color: 计数文本的颜色
        # count_bg_color: 计数文本的背景颜色
        # line_thickness: 绘制线条的粗细
        # track_thickness: 绘制轨迹的粗细
        # view_img: 是否显示图像
        # view_in_counts: 是否显示进入计数区域的物体计数
        # view_out_counts: 是否显示离开计数区域的物体计数
        # draw_tracks: 是否绘制物体轨迹
        # track_color: 轨迹颜色
        # region_thickness: 计数区域的线条粗细
        # line_dist_thresh: 线段连接的最大距离阈值
        # cls_txtdisplay_gap: 不同类别文本显示的间隔

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 处理鼠标左键按下事件，检查是否点击到计数区域的定义点
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            # 处理鼠标移动事件，如果正在绘制且选中了点，则更新计数区域的定义点
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            # 处理鼠标左键松开事件，停止绘制计数区域
            self.is_drawing = False
            self.selected_point = None

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            # 如果环境检查通过，创建窗口并显示图像
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:  # 如果用户绘制了计数区域，则添加鼠标事件处理
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            # 检测按键事件，如果按下 'q' 键则关闭窗口
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    # 开始对象计数的主要函数，用于启动对象计数过程。
    # 将当前帧从视频流存储到 self.im0 中
    self.im0 = im0  # store image

    # 对从对象跟踪过程获取的轨迹进行提取和处理
    self.extract_and_process_tracks(tracks)  # draw region even if no objects

    # 如果 self.view_img 为 True，则显示帧
    if self.view_img:
        self.display_frames()

    # 返回处理后的帧 self.im0
    return self.im0
# 如果当前模块被直接运行（而不是被导入到其他模块中），则执行以下代码块
if __name__ == "__main__":
    # 定义一个示例的类名字典，用于对象计数器
    classes_names = {0: "person", 1: "car"}  # example class names
    # 创建一个对象计数器实例，传入类名字典作为参数
    ObjectCounter(classes_names)
```