# `.\yolov8\examples\YOLOv8-Region-Counter\yolov8_region_counter.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse  # 导入命令行参数解析模块
from collections import defaultdict  # 导入默认字典模块
from pathlib import Path  # 导入处理文件路径的模块

import cv2  # 导入 OpenCV 模块
import numpy as np  # 导入 NumPy 数学计算库
from shapely.geometry import Polygon  # 从 Shapely 几何库中导入多边形对象
from shapely.geometry.point import Point  # 从 Shapely 几何库中导入点对象

from ultralytics import YOLO  # 导入 Ultralytics YOLO 模块
from ultralytics.utils.files import increment_path  # 导入路径增量函数
from ultralytics.utils.plotting import Annotator, colors  # 导入标注和颜色模块

track_history = defaultdict(list)  # 初始化一个默认字典，用于跟踪历史记录

current_region = None  # 初始化当前选定的区域为空

counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",  # 区域名称
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # 多边形顶点坐标
        "counts": 0,  # 区域计数初始值
        "dragging": False,  # 拖动状态标志
        "region_color": (255, 42, 4),  # 区域颜色，BGR 值
        "text_color": (255, 255, 255),  # 文字颜色
    },
    {
        "name": "YOLOv8 Rectangle Region",  # 区域名称
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # 多边形顶点坐标
        "counts": 0,  # 区域计数初始值
        "dragging": False,  # 拖动状态标志
        "region_color": (37, 255, 225),  # 区域颜色，BGR 值
        "text_color": (0, 0, 0),  # 文字颜色
    },
]


def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for region manipulation.

    Parameters:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters passed to the callback (not used in this function).

    Global Variables:
        current_region (dict): A dictionary representing the current selected region.

    Mouse Events:
        - LBUTTONDOWN: Initiates dragging for the region containing the clicked point.
        - MOUSEMOVE: Moves the selected region if dragging is active.
        - LBUTTONUP: Ends dragging for the selected region.

    Notes:
        - This function is intended to be used as a callback for OpenCV mouse events.
        - Requires the existence of the 'counting_regions' list and the 'Polygon' class.

    Example:
        >>> cv2.setMouseCallback(window_name, mouse_callback)
    """
    global current_region  # 引用全局变量 current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:  # 如果是鼠标左键按下事件
        for region in counting_regions:  # 遍历计数区域列表
            if region["polygon"].contains(Point((x, y))):  # 如果鼠标点击点在某个区域内
                current_region = region  # 将当前选中区域设置为该区域
                current_region["dragging"] = True  # 开始拖动该区域
                current_region["offset_x"] = x  # 记录拖动起始的 x 坐标
                current_region["offset_y"] = y  # 记录拖动起始的 y 坐标

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:  # 如果是鼠标移动事件
        if current_region is not None and current_region["dragging"]:  # 如果当前有选定区域且正在拖动
            dx = x - current_region["offset_x"]  # 计算 x 方向上的移动距离
            dy = y - current_region["offset_y"]  # 计算 y 方向上的移动距离
            current_region["polygon"] = Polygon(  # 更新区域的多边形顶点坐标
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x  # 更新拖动后的 x 坐标
            current_region["offset_y"] = y  # 更新拖动后的 y 坐标
    # 如果鼠标左键抬起事件被触发
    elif event == cv2.EVENT_LBUTTONUP:
        # 如果当前区域不为空且正在拖拽状态
        if current_region is not None and current_region["dragging"]:
            # 将当前区域的拖拽状态设置为 False，表示停止拖拽
            current_region["dragging"] = False
# 定义一个函数 `run`，用于运行基于 YOLOv8 和 ByteTrack 的视频区域计数。
def run(
    weights="yolov8n.pt",  # 模型权重文件路径，默认为 "yolov8n.pt"
    source=None,           # 视频文件路径，必须提供
    device="cpu",          # 处理设备选择，默认为 CPU
    view_img=False,        # 是否显示结果，默认为 False
    save_img=False,        # 是否保存结果，默认为 False
    exist_ok=False,        # 是否覆盖现有文件，默认为 False
    classes=None,          # 要检测和跟踪的类别列表，默认为 None
    line_thickness=2,      # 边界框厚度，默认为 2
    track_thickness=2,     # 跟踪线厚度，默认为 2
    region_thickness=2,    # 区域厚度，默认为 2
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """

    # 初始化视频帧计数器
    vid_frame_count = 0

    # 检查视频源路径是否存在
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # 设置 YOLO 模型并加载权重
    model = YOLO(f"{weights}")

    # 根据设备选择加载模型到 CPU 或 CUDA
    model.to("cuda") if device == "0" else model.to("cpu")

    # 提取模型中的类别名称列表
    names = model.model.names

    # 设置视频捕捉对象
    videocapture = cv2.VideoCapture(source)

    # 获取视频帧宽度和高度
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))

    # 获取视频帧率和视频编码格式
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # 设置保存结果的目录
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)

    # 创建保存目录（如果不存在）
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建视频写入对象，指定输出视频文件路径、帧率和帧尺寸
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    # 释放视频帧计数器
    del vid_frame_count

    # 释放视频写入对象资源
    video_writer.release()

    # 释放视频捕捉对象资源
    videocapture.release()

    # 关闭所有的 OpenCV 窗口
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数，用于解析权重文件路径，默认为 "yolov8n.pt"
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")

    # 添加命令行参数，用于指定 CUDA 设备编号，支持多个设备，例如 "0,1,2,3" 或 "cpu"
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")

    # 添加命令行参数，用于指定视频文件路径，必须提供
    parser.add_argument("--source", type=str, required=True, help="video file path")

    # 添加命令行参数，用于显示结果的标志，默认为 False
    parser.add_argument("--view-img", action="store_true", help="show results")

    # 添加命令行参数，用于保存结果的标志，默认为 False
    parser.add_argument("--save-img", action="store_true", help="save results")

    # 添加命令行参数，用于覆盖现有文件的标志，默认为 False
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")

    # 添加命令行参数，用于指定要检测和跟踪的类别列表，支持多个类别编号
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")

    # 添加命令行参数，用于指定边界框厚度，默认为 2
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")

    # 添加命令行参数，用于指定跟踪线厚度，默认为 2
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")

    # 添加命令行参数，用于指定区域厚度，默认为 4
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    # 解析命令行参数并返回解析结果
    return parser.parse_args()


def main(opt):
    # 在这里实现主函数的逻辑，根据命令行参数 `opt` 进行相应操作
    """Main function."""
    # 执行程序的主函数，通过解析命令行参数获取参数字典并传递给 run 函数
    run(**vars(opt))
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中），则执行以下代码块
if __name__ == "__main__":
    # 调用 parse_opt 函数，返回命令行参数的解析结果，存储在 opt 变量中
    opt = parse_opt()
    # 调用 main 函数，并将 opt 作为参数传递给它，执行主程序的逻辑
    main(opt)
```