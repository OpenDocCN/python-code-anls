# `.\yolov8\ultralytics\solutions\parking_management.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import json  # 导入处理 JSON 格式数据的模块

import cv2  # 导入 OpenCV 库，用于图像处理
import numpy as np  # 导入 NumPy 库，用于处理数值数据

from ultralytics.utils.checks import check_imshow, check_requirements  # 导入检查函数，用于检查必要的依赖项
from ultralytics.utils.plotting import Annotator  # 导入绘图类，用于标注图像

class ParkingPtsSelection:
    """Class for selecting and managing parking zone points on images using a Tkinter-based UI."""

    def __init__(self):
        """Initializes the UI for selecting parking zone points in a tkinter window."""
        check_requirements("tkinter")  # 检查是否安装了 tkinter 库，必要时抛出异常

        import tkinter as tk  # 导入 tkinter 库，用于构建图形用户界面

        self.tk = tk  # 赋值 tkinter 模块给实例变量 self.tk
        self.master = tk.Tk()  # 创建主窗口实例
        self.master.title("Ultralytics Parking Zones Points Selector")  # 设置窗口标题

        # Disable window resizing
        self.master.resizable(False, False)  # 禁止窗口大小调整

        # Setup canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")  # 在主窗口中创建画布用于显示图像

        # Setup buttons
        button_frame = self.tk.Frame(self.master)  # 创建按钮框架
        button_frame.pack(side=self.tk.TOP)  # 放置在顶部

        self.tk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0)
        # 创建上传图像的按钮，点击后调用 upload_image 方法，放置在第一行第一列
        self.tk.Button(button_frame, text="Remove Last BBox", command=self.remove_last_bounding_box).grid(
            row=0, column=1
        )
        # 创建移除最后一个边界框的按钮，点击后调用 remove_last_bounding_box 方法，放置在第一行第二列
        self.tk.Button(button_frame, text="Save", command=self.save_to_json).grid(row=0, column=2)
        # 创建保存按钮，点击后调用 save_to_json 方法，放置在第一行第三列

        # Initialize properties
        self.image_path = None  # 初始化图像路径为空
        self.image = None  # 初始化图像对象为空
        self.canvas_image = None  # 初始化画布图像对象为空
        self.bounding_boxes = []  # 初始化边界框列表为空
        self.current_box = []  # 初始化当前边界框为空
        self.img_width = 0  # 初始化图像宽度为 0
        self.img_height = 0  # 初始化图像高度为 0

        # Constants
        self.canvas_max_width = 1280  # 设置画布最大宽度为 1280
        self.canvas_max_height = 720  # 设置画布最大高度为 720

        self.master.mainloop()  # 进入主事件循环，等待用户交互
    def upload_image(self):
        """Upload an image and resize it to fit canvas."""
        # 导入文件对话框模块
        from tkinter import filedialog
        # 导入PIL图像处理库及其图像展示模块ImageTk，因为ImageTk需要tkinter库

        from PIL import Image, ImageTk  

        # 请求用户选择图片文件路径，限定文件类型为png、jpg、jpeg
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not self.image_path:
            return  # 如果未选择文件，则结束函数

        # 打开选择的图片文件
        self.image = Image.open(self.image_path)
        self.img_width, self.img_height = self.image.size

        # 计算图片的宽高比并调整图片大小以适应画布
        aspect_ratio = self.img_width / self.img_height
        if aspect_ratio > 1:
            # 横向图片
            canvas_width = min(self.canvas_max_width, self.img_width)
            canvas_height = int(canvas_width / aspect_ratio)
        else:
            # 纵向图片
            canvas_height = min(self.canvas_max_height, self.img_height)
            canvas_width = int(canvas_height * aspect_ratio)

        # 如果画布已经初始化，则销毁之前的画布对象
        if self.canvas:
            self.canvas.destroy()

        # 创建新的画布对象，并设置其大小及背景色
        self.canvas = self.tk.Canvas(self.master, bg="white", width=canvas_width, height=canvas_height)

        # 调整图片大小，并转换为ImageTk.PhotoImage格式以在画布上展示
        resized_image = self.image.resize((canvas_width, canvas_height), Image.LANCZOS)
        self.canvas_image = ImageTk.PhotoImage(resized_image)

        # 在画布上创建图片对象
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

        # 将画布放置在窗口底部
        self.canvas.pack(side=self.tk.BOTTOM)

        # 绑定画布的鼠标左键点击事件到特定处理函数
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # 重置边界框和当前边界框数据
        self.bounding_boxes = []
        self.current_box = []

    def on_canvas_click(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        # 在画布上处理鼠标左键点击事件，用于创建边界框的顶点
        self.current_box.append((event.x, event.y))
        x0, y0 = event.x - 3, event.y - 3
        x1, y1 = event.x + 3, event.y + 3

        # 在画布上绘制红色的小圆点以标记边界框顶点
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")

        if len(self.current_box) == 4:
            # 如果当前边界框的顶点数为4，则将其添加到边界框列表中，并绘制边界框
            self.bounding_boxes.append(self.current_box)
            self.draw_bounding_box(self.current_box)
            self.current_box = []

    def draw_bounding_box(self, box):
        """
        Draw bounding box on canvas.

        Args:
            box (list): Bounding box data
        """
        # 在画布上绘制边界框
        for i in range(4):
            x1, y1 = box[i]
            x2, y2 = box[(i + 1) % 4]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
    # 从画布中移除最后一个绘制的边界框
    def remove_last_bounding_box(self):
        """Remove the last drawn bounding box from canvas."""
        from tkinter import messagebox  # 为了多环境兼容性而导入消息框

        # 如果存在边界框
        if self.bounding_boxes:
            self.bounding_boxes.pop()  # 移除最后一个边界框
            self.canvas.delete("all")  # 清空画布
            self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)  # 重新绘制图像

            # 重新绘制所有边界框
            for box in self.bounding_boxes:
                self.draw_bounding_box(box)

            messagebox.showinfo("Success", "Last bounding box removed.")  # 显示成功消息
        else:
            messagebox.showwarning("Warning", "No bounding boxes to remove.")  # 显示警告消息：没有边界框可移除

    # 将按图像到画布大小比例重新缩放的边界框保存到 'bounding_boxes.json'
    def save_to_json(self):
        """Saves rescaled bounding boxes to 'bounding_boxes.json' based on image-to-canvas size ratio."""
        from tkinter import messagebox  # 为了多环境兼容性而导入消息框

        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        width_scaling_factor = self.img_width / canvas_width
        height_scaling_factor = self.img_height / canvas_height
        bounding_boxes_data = []

        # 遍历所有边界框
        for box in self.bounding_boxes:
            rescaled_box = []
            for x, y in box:
                rescaled_x = int(x * width_scaling_factor)
                rescaled_y = int(y * height_scaling_factor)
                rescaled_box.append((rescaled_x, rescaled_y))
            bounding_boxes_data.append({"points": rescaled_box})

        # 将数据以缩进格式写入到 'bounding_boxes.json'
        with open("bounding_boxes.json", "w") as f:
            json.dump(bounding_boxes_data, f, indent=4)

        messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")  # 显示成功消息
class ParkingManagement:
    """Manages parking occupancy and availability using YOLOv8 for real-time monitoring and visualization."""

    def __init__(
        self,
        model_path,
        txt_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        occupied_region_color=(0, 255, 0),
        available_region_color=(0, 0, 255),
        margin=10,
    ):
        """
        Initializes the parking management system with a YOLOv8 model and visualization settings.

        Args:
            model_path (str): Path to the YOLOv8 model.
            txt_color (tuple): RGB color tuple for text.
            bg_color (tuple): RGB color tuple for background.
            occupied_region_color (tuple): RGB color tuple for occupied regions.
            available_region_color (tuple): RGB color tuple for available regions.
            margin (int): Margin for text display.
        """
        # Model path and initialization
        self.model_path = model_path
        self.model = self.load_model()  # 载入YOLOv8模型

        # Labels dictionary
        self.labels_dict = {"Occupancy": 0, "Available": 0}  # 初始化标签字典

        # Visualization details
        self.margin = margin  # 文字显示的边距
        self.bg_color = bg_color  # 背景颜色设置
        self.txt_color = txt_color  # 文字颜色设置
        self.occupied_region_color = occupied_region_color  # 占用区域的颜色设置
        self.available_region_color = available_region_color  # 空闲区域的颜色设置

        self.window_name = "Ultralytics YOLOv8 Parking Management System"  # 窗口名称
        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)  # 检查环境是否支持imshow函数

    def load_model(self):
        """Load the Ultralytics YOLO model for inference and analytics."""
        from ultralytics import YOLO

        return YOLO(self.model_path)  # 使用路径加载Ultralytics YOLO模型

    @staticmethod
    def parking_regions_extraction(json_file):
        """
        Extract parking regions from json file.

        Args:
            json_file (str): file that have all parking slot points
        """
        with open(json_file, "r") as f:
            return json.load(f)  # 从JSON文件中提取停车区域信息
    def process_data(self, json_data, im0, boxes, clss):
        """
        Process the model data for parking lot management.

        Args:
            json_data (str): json data for parking lot management
            im0 (ndarray): inference image
            boxes (list): bounding boxes data
            clss (list): bounding boxes classes list

        Returns:
            filled_slots (int): total slots that are filled in parking lot
            empty_slots (int): total slots that are available in parking lot
        """
        # 创建一个Annotator对象，用于在图像上标注信息
        annotator = Annotator(im0)
        
        # 初始化空车位数为json_data的长度，已占用车位数为0
        empty_slots, filled_slots = len(json_data), 0
        
        # 遍历json_data中的每个区域
        for region in json_data:
            # 将区域的点坐标转换为numpy数组形式
            points_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            # 初始化区域占用状态为False
            region_occupied = False

            # 遍历所有检测到的边界框及其类别
            for box, cls in zip(boxes, clss):
                # 计算边界框中心点的坐标
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                # 获取类别名称对应的文本信息
                text = f"{self.model.names[int(cls)]}"

                # 在图像上显示对象标签信息
                annotator.display_objects_labels(
                    im0, text, self.txt_color, self.bg_color, x_center, y_center, self.margin
                )
                
                # 计算当前中心点到区域边界的距离
                dist = cv2.pointPolygonTest(points_array, (x_center, y_center), False)
                
                # 如果距离大于等于0，表示中心点在区域内，标记该区域已被占用
                if dist >= 0:
                    region_occupied = True
                    break

            # 根据区域占用状态确定绘制区域的颜色
            color = self.occupied_region_color if region_occupied else self.available_region_color
            # 在图像上绘制多边形边界
            cv2.polylines(im0, [points_array], isClosed=True, color=color, thickness=2)
            
            # 如果区域被占用，更新已占用车位数和空车位数
            if region_occupied:
                filled_slots += 1
                empty_slots -= 1

        # 将已占用和空余车位数存入标签字典
        self.labels_dict["Occupancy"] = filled_slots
        self.labels_dict["Available"] = empty_slots
        
        # 在图像上显示分析结果
        annotator.display_analytics(im0, self.labels_dict, self.txt_color, self.bg_color, self.margin)

    def display_frames(self, im0):
        """
        Display frame.

        Args:
            im0 (ndarray): inference image
        """
        # 如果开启了环境检测模式，创建并显示图像窗口
        if self.env_check:
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, im0)
            
            # 检测键盘输入，如果按下 'q' 键，关闭窗口
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
```