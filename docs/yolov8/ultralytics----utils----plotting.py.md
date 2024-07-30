# `.\yolov8\ultralytics\utils\plotting.py`

```py
# 导入需要的库
import contextlib  # 上下文管理模块，用于创建上下文管理器
import math  # 数学函数模块，提供数学函数的实现
import warnings  # 警告模块，用于处理警告信息
from pathlib import Path  # 路径操作模块，用于处理文件和目录路径
from typing import Callable, Dict, List, Optional, Union  # 类型提示模块，用于类型注解

import cv2  # OpenCV图像处理库
import matplotlib.pyplot as plt  # 绘图库matplotlib的pyplot模块
import numpy as np  # 数值计算库numpy
import torch  # 深度学习框架PyTorch
from PIL import Image, ImageDraw, ImageFont  # Python Imaging Library，用于图像处理

from PIL import __version__ as pil_version  # PIL版本信息

from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings, threaded  # 导入自定义工具函数和变量
from ultralytics.utils.checks import check_font, check_version, is_ascii  # 导入自定义检查函数
from ultralytics.utils.files import increment_path  # 导入路径处理函数

# 颜色类，包含Ultralytics默认色彩方案和转换函数
class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        # 初始化颜色调色板，将16进制颜色代码转换为RGB元组
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        # 预定义特定颜色调色板，用于特定应用场景
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        # 返回调色板中第i个颜色的RGB值，支持BGR格式
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        # 将16进制颜色代码转换为RGB元组（PIL默认顺序）
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # 创建颜色对象实例，用于绘图颜色选择

class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.
    """
    # 定义类属性，用于图像注释
    Attributes:
        # im是要注释的图像，可以是PIL图像(Image.Image)或者numpy数组
        im (Image.Image or numpy array): The image to annotate.
        # pil标志指示是否使用PIL库进行注释，而不是cv2
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        # font用于文本注释的字体，可以是ImageFont.truetype或ImageFont.load_default
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        # lw是用于绘制注释的线条宽度
        lw (float): Line width for drawing.
        # skeleton是关键点的骨架结构的列表，其中每个元素是一个列表，表示连接的两个关键点的索引
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        # limb_color是绘制骨架连接的颜色调色板，以RGB整数列表形式表示
        limb_color (List[int]): Color palette for limbs.
        # kpt_color是绘制关键点的颜色调色板，以RGB整数列表形式表示
        kpt_color (List[int]): Color palette for keypoints.
    """
    # 初始化 Annotator 类，接受图像 im、线宽 line_width、字体大小 font_size、字体名称 font、是否使用 PIL 的标志 pil、示例 example
    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        # 检查示例是否包含非 ASCII 字符，用于确定是否使用 PIL
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        # 检查输入的图像是否为 PIL Image 对象
        input_is_pil = isinstance(im, Image.Image)
        # 根据条件判断是否使用 PIL
        self.pil = pil or non_ascii or input_is_pil
        # 计算线宽，默认为图像尺寸或形状的一半乘以 0.003，取整后至少为 2
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        
        if self.pil:  # 如果使用 PIL
            # 如果输入的是 PIL Image，则直接使用；否则将其转换为 PIL Image
            self.im = im if input_is_pil else Image.fromarray(im)
            # 创建一个用于绘制的 ImageDraw 对象
            self.draw = ImageDraw.Draw(self.im)
            try:
                # 根据示例中是否包含非 ASCII 字符，选择适当的字体文件（Unicode 或 Latin）
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                # 计算字体大小，默认为图像尺寸的一半乘以 0.035，取整后至少为 12
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                # 加载选择的字体文件并设置字体大小
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                # 如果加载字体文件出错，则使用默认字体
                self.font = ImageFont.load_default()
            # 如果 PIL 版本高于等于 9.2.0，则修复 getsize 方法的用法为 getbbox 方法的结果中的宽度和高度
            if check_version(pil_version, "9.2.0"):
                self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # 如果使用 cv2
            # 断言输入的图像数据是连续的，否则提出警告
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            # 如果图像数据不可写，则创建其副本
            self.im = im if im.flags.writeable else im.copy()
            # 计算字体粗细，默认为线宽减 1，至少为 1
            self.tf = max(self.lw - 1, 1)  # font thickness
            # 计算字体缩放比例，默认为线宽的三分之一
            self.sf = self.lw / 3  # font scale
        
        # 姿态关键点的连接关系
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
        ]

        # 姿态关键点连接线的颜色
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        # 姿态关键点的颜色
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        
        # 深色调色板，用于姿态显示
        self.dark_colors = {
            (235, 219, 11), (243, 243, 243), (183, 223, 0), (221, 111, 255),
            (0, 237, 204), (68, 243, 0), (255, 255, 0), (179, 255, 1),
            (11, 255, 162),
        }
        # 浅色调色板，用于姿态显示
        self.light_colors = {
            (255, 42, 4), (79, 68, 255), (255, 0, 189), (255, 180, 0),
            (186, 0, 221), (0, 192, 38), (255, 36, 125), (104, 0, 123),
            (108, 27, 255), (47, 109, 252), (104, 31, 17),
        }
    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Assign text color based on background color."""
        # 检查给定的背景颜色是否为暗色
        if color in self.dark_colors:
            # 如果是暗色，则返回预定义的深色文本颜色
            return 104, 31, 17
        elif color in self.light_colors:
            # 如果是亮色，则返回白色作为文本颜色
            return 255, 255, 255
        else:
            # 如果背景颜色既不是暗色也不是亮色，则返回默认的文本颜色
            return txt_color

    def circle_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (R, G, B).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """

        # 如果标签超过3个字符，打印警告信息，并仅使用前三个字符作为圆形标注的文本
        if len(label) > 3:
            print(
                f"Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!"
            )
            label = label[:3]

        # 计算框的中心点坐标
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # 获取文本的大小
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        # 计算需要的半径，以适应文本和边距
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # 在图像上绘制圆形标注
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # 计算文本位置
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # 绘制文本
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            # 获取文本颜色，根据背景颜色自动选择
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )
    def text_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=5):
        """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (R, G, B).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """

        # Calculate the center of the bounding box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the size of the text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw the background rectangle
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,  # 目标图像，在其上绘制
            label,  # 要绘制的文本
            (text_x, text_y),  # 文本的起始坐标（左下角位置）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
            self.sf - 0.1,  # 字体比例因子
            self.get_txt_color(color, txt_color),  # 文本颜色
            self.tf,  # 文本线宽
            lineType=cv2.LINE_AA,  # 线型
        )
    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """
        Plot masks on image.

        Args:
            masks (tensor): Predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
            retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
        """

        # 如果使用 PIL，先转换为 numpy 数组
        if self.pil:
            self.im = np.asarray(self.im).copy()

        # 如果没有预测到任何 mask，则直接将原始图像拷贝到 self.im
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255

        # 如果图像和 masks 不在同一个设备上，则将 im_gpu 移动到 masks 所在的设备上
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)

        # 将 colors 转换为 torch.tensor，并归一化到 [0, 1] 的范围
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)

        # 扩展维度以便进行广播操作，将 colors 变为 shape(n,1,1,3)
        colors = colors[:, None, None]  # shape(n,1,1,3)

        # 增加一个维度到 masks 上，使其变为 shape(n,h,w,1)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)

        # 将 masks 与颜色相乘，乘以 alpha 控制透明度，得到彩色的 masks，shape(n,h,w,3)
        masks_color = masks * (colors * alpha)

        # 计算反向透明度 masks，用于混合原始图像和 masks_color，shape(n,h,w,1)
        inv_alpha_masks = (1 - masks * alpha).cumprod(0)

        # 计算最大通道值，用于融合图像和 masks_color，shape(n,h,w,3)
        mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

        # 翻转图像的通道顺序，从 RGB 转为 BGR
        im_gpu = im_gpu.flip(dims=[0])

        # 调整张量的维度顺序，从 (3,h,w) 转为 (h,w,3)
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()

        # 使用 inv_alpha_masks[-1] 和 mcs 进行图像的混合
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs

        # 将混合后的图像乘以 255，并转为 numpy 数组
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()

        # 根据 retina_masks 参数选择是否缩放图像
        self.im[:] = im_mask_np if retina_masks else ops.scale_image(im_mask_np, self.im.shape)

        # 如果使用 PIL，将处理后的 numpy 数组转回 PIL 格式，并更新 draw
        if self.pil:
            self.fromarray(self.im)
    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True, conf_thres=0.25):
        """
        Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note:
            `kpt_line=True` currently only supports human pose plotting.
        """

        if self.pil:
            # If working with PIL image, convert to numpy array for processing
            self.im = np.asarray(self.im).copy()  # Convert PIL image to numpy array
        
        # Get the number of keypoints and dimensions from the input tensor
        nkpt, ndim = kpts.shape
        # Check if the keypoints represent a human pose (17 keypoints with 2 or 3 dimensions)
        is_pose = nkpt == 17 and ndim in {2, 3}
        # Adjust kpt_line based on whether it's a valid human pose and the argument value
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting

        # Loop through each keypoint and plot a circle on the image
        for i, k in enumerate(kpts):
            # Determine color for the keypoint based on whether it's a pose or not
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            # Check if the keypoint coordinates are within image boundaries
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                # If confidence score is provided (3 dimensions), skip keypoints below threshold
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                # Draw a circle on the image at the keypoint location
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        # If kpt_line is True, draw lines connecting keypoints (for human pose)
        if kpt_line:
            ndim = kpts.shape[-1]
            # Iterate over predefined skeleton connections and draw lines between keypoints
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                # If confidence scores are provided, skip lines for keypoints below threshold
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                # Check if keypoints' positions are within image boundaries before drawing lines
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                # Draw a line connecting two keypoints on the image
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        if self.pil:
            # Convert numpy array (image) back to PIL image format and update self.im
            self.fromarray(self.im)  # Convert numpy array back to PIL image

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)
    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """Adds text to an image using PIL or cv2."""
        # 如果锚点是"bottom"，从字体底部开始计算y坐标
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # 获取文本的宽度和高度
            xy[1] += 1 - h
        if self.pil:
            # 如果需要使用方框样式
            if box_style:
                w, h = self.font.getsize(text)  # 获取文本的宽度和高度
                # 在图像上绘制一个矩形框作为背景，并使用txt_color填充
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # 将txt_color作为背景颜色，将文本以白色填充前景绘制
                txt_color = (255, 255, 255)
            # 如果文本中包含换行符
            if "\n" in text:
                lines = text.split("\n")  # 拆分成多行文本
                _, h = self.font.getsize(text)  # 获取单行文本的高度
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)  # 绘制每一行文本
                    xy[1] += h  # 更新y坐标以绘制下一行文本
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)  # 绘制单行文本
        else:
            # 如果需要使用方框样式
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # 获取文本的宽度和高度
                h += 3  # 增加一些像素以填充文本
                outside = xy[1] >= h  # 判断标签是否适合在框外
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # 填充矩形框
                # 将txt_color作为背景颜色，将文本以白色填充前景绘制
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)  # 使用cv2绘制文本

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)  # 将numpy数组或PIL图像赋值给self.im
        self.draw = ImageDraw.Draw(self.im)  # 使用PIL的ImageDraw创建绘图对象

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)  # 将PIL图像转换为numpy数组并返回

    def show(self, title=None):
        """Show the annotated image."""
        Image.fromarray(np.asarray(self.im)[..., ::-1]).show(title)  # 将numpy数组转换为RGB模式的PIL图像并显示

    def save(self, filename="image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))  # 将numpy数组保存为图像文件

    def get_bbox_dimension(self, bbox=None):
        """
        Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            angle (degree): Degree value of angle between three points
        """
        x_min, y_min, x_max, y_max = bbox  # 解构包围框坐标
        width = x_max - x_min  # 计算包围框宽度
        height = y_max - y_min  # 计算包围框高度
        return width, height, width * height  # 返回宽度、高度和面积的元组
    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """

        # 使用 cv2.polylines 方法在图像上绘制多边形线段，reg_pts 是多边形的顶点坐标
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """

        # 将轨迹点连接成连续的线段，并绘制到图像上
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)

        # 在轨迹的最后一个点处画一个实心圆圈，表示物体的当前位置
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): queue counts label
            points (tuple): region points for center point calculation to display text
            region_color (RGB): queue region color
            txt_color (RGB): text display color
        """

        # 计算区域中心点的坐标
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        # 计算显示文本的大小和位置
        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)

        # 在图像上绘制一个填充的矩形框作为背景
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # 在指定位置绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )
    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): inference image
            text (str): object/class name
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            x_center (float): x position center point for bounding box
            y_center (float): y position center point for bounding box
            margin (int): gap between text and rectangle for better display
        """

        # Calculate the size of the text to be displayed
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        # Calculate the x and y coordinates for placing the text centered at (x_center, y_center)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        # Calculate the coordinates of the rectangle surrounding the text
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw a filled rectangle with specified background color around the text
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        # Draw the text on the image at (text_x, text_y)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        Display the overall statistics for parking lots.

        Args:
            im0 (ndarray): inference image
            text (dict): labels dictionary
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            margin (int): gap between text and rectangle for better display
        """

        # Calculate horizontal and vertical gaps based on image dimensions
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0  # Initialize offset for vertical placement of text

        # Iterate through each label and value pair in the provided dictionary
        for label, value in text.items():
            txt = f"{label}: {value}"  # Format the label and value into a string
            # Calculate the size of the text to be displayed
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            # Ensure minimum size for text dimensions to avoid errors
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            # Calculate the x and y coordinates for placing the text on the image
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            # Calculate the coordinates of the rectangle surrounding the text
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            # Draw a filled rectangle with specified background color around the text
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            # Draw the text on the image at (text_x, text_y)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            # Update the vertical offset for placing the next text block
            text_y_offset = rect_y2

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle between three points.

        Args:
            a (float) : The coordinates of pose point a
            b (float): The coordinates of pose point b
            c (float): The coordinates of pose point c

        Returns:
            angle (degree): Degree value of the angle between the points
        """

        # Convert input points to numpy arrays for calculations
        a, b, c = np.array(a), np.array(b), np.array(c)

        # Calculate the angle using arctangent and convert from radians to degrees
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        # Normalize angle to be within [0, 180] degrees
        if angle > 180.0:
            angle = 360 - angle

        return angle

    def draw_specific_points(self, keypoints, indices=None, shape=(640, 640), radius=2, conf_thres=0.25):
        """
        Draw specific keypoints on an image.

        Args:
            keypoints (list): List of keypoints to be plotted
            indices (list): Indices of keypoints to be plotted
            shape (tuple): Size of the image (width, height)
            radius (int): Radius of the keypoints
            conf_thres (float): Confidence threshold for keypoints
        """

        # If indices are not provided, default to drawing keypoints 2, 5, and 7
        if indices is None:
            indices = [2, 5, 7]

        # Iterate through keypoints and draw circles for specific indices
        for i, k in enumerate(keypoints):
            if i in indices:
                x_coord, y_coord = k[0], k[1]

                # Check if the keypoints are within image bounds
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        # Skip drawing if confidence is below the threshold
                        if conf < conf_thres:
                            continue

                    # Draw a circle on the image at the keypoint coordinates
                    cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        # Return the image with keypoints drawn
        return self.im

    def plot_angle_and_count_and_stage(
        self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)
    ):
        """
        Plot angle, count, and stage information on the image.

        Args:
            angle_text (str): Text to display for the angle
            count_text (str): Text to display for the count
            stage_text (str): Text to display for the stage
            center_kpt (tuple): Center keypoint coordinates
            color (tuple): Color of the plotted elements
            txt_color (tuple): Color of the text
        """

        # Implementation details are missing in the provided snippet.
        # The function definition is incomplete.
        pass

    def seg_bbox(self, mask, mask_color=(255, 0, 255), label=None, txt_color=(255, 255, 255)):
        """
        Draw a segmented object with a bounding box on the image.

        Args:
            mask (list): List of mask data points for the segmented object
            mask_color (RGB): Color for the mask
            label (str): Text label for the detection
            txt_color (RGB): Text color
        """

        # Draw the polygonal lines around the mask region
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)

        # Calculate text size for label and draw a rectangle around the label
        text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)
        cv2.rectangle(
            self.im,
            (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
            (int(mask[0][0]) + text_size[0] // 2 + 10, int(mask[0][1] + 10)),
            mask_color,
            -1,
        )

        # Draw the label text on the image
        if label:
            cv2.putText(
                self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1])), 0, self.sf, txt_color, self.tf
            )
    def plot_distance_and_line(self, distance_m, distance_mm, centroids, line_color, centroid_color):
        """
        Plot the distance and line on frame.

        Args:
            distance_m (float): Distance between two bbox centroids in meters.
            distance_mm (float): Distance between two bbox centroids in millimeters.
            centroids (list): Bounding box centroids data.
            line_color (RGB): Distance line color.
            centroid_color (RGB): Bounding box centroid color.
        """

        # 计算 "Distance M" 文本的宽度和高度
        (text_width_m, text_height_m), _ = cv2.getTextSize(f"Distance M: {distance_m:.2f}m", 0, self.sf, self.tf)
        # 绘制包围 "Distance M" 文本的矩形框
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 10, 25 + text_height_m + 20), line_color, -1)
        # 在图像中绘制 "Distance M" 文本
        cv2.putText(
            self.im,
            f"Distance M: {distance_m:.2f}m",
            (20, 50),
            0,
            self.sf,
            centroid_color,
            self.tf,
            cv2.LINE_AA,
        )

        # 计算 "Distance MM" 文本的宽度和高度
        (text_width_mm, text_height_mm), _ = cv2.getTextSize(f"Distance MM: {distance_mm:.2f}mm", 0, self.sf, self.tf)
        # 绘制包围 "Distance MM" 文本的矩形框
        cv2.rectangle(self.im, (15, 75), (15 + text_width_mm + 10, 75 + text_height_mm + 20), line_color, -1)
        # 在图像中绘制 "Distance MM" 文本
        cv2.putText(
            self.im,
            f"Distance MM: {distance_mm:.2f}mm",
            (20, 100),
            0,
            self.sf,
            centroid_color,
            self.tf,
            cv2.LINE_AA,
        )

        # 在图像中绘制两个中心点之间的直线
        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        # 在图像中绘制第一个中心点
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        # 在图像中绘制第二个中心点
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
        """

        # 计算 bounding box 的中心点坐标
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # 在图像中绘制 visioneye 点的中心点
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        # 在图像中绘制 bounding box 的中心点
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        # 在图像中绘制 visioneye 点与 bounding box 中心点之间的连线
        cv2.line(self.im, center_point, center_bbox, color, self.tf)
@TryExcept()  # 使用 TryExcept 装饰器，处理已知问题 https://github.com/ultralytics/yolov5/issues/5395
@plt_settings()  # 使用 plt_settings 函数进行绘图设置
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics."""
    import pandas  # 导入 pandas 库，用于数据处理
    import seaborn  # 导入 seaborn 库，用于统计图表绘制

    # 过滤掉 matplotlib>=3.7.2 的警告和 Seaborn 的 use_inf 和 is_categorical 的 FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 绘制数据集标签
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # 计算类别数量
    boxes = boxes[:1000000]  # 限制最多处理 100 万个框
    x = pandas.DataFrame(boxes, columns=["x", "y", "width", "height"])  # 创建包含框坐标的 DataFrame

    # 绘制 Seaborn 相关性图
    seaborn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)  # 保存相关性图
    plt.close()

    # 绘制 Matplotlib 标签
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)  # 绘制类别直方图
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])  # 设置直方图颜色
    ax[0].set_ylabel("instances")  # 设置 y 轴标签
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)  # 设置 x 轴标签
    else:
        ax[0].set_xlabel("classes")  # 设置 x 轴标签为类别

    seaborn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)  # 绘制 x、y 分布图
    seaborn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)  # 绘制宽度、高度分布图

    # 绘制矩形框
    boxes[:, 0:2] = 0.5  # 将框坐标调整为中心点
    boxes = ops.xywh2xyxy(boxes) * 1000  # 转换为绝对坐标并放大
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)  # 创建空白图像
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # 绘制矩形框
    ax[1].imshow(img)  # 显示图像
    ax[1].axis("off")  # 关闭坐标轴显示

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)  # 隐藏图表边框

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)  # 保存最终标签图像
    plt.close()  # 关闭绘图窗口
    if on_plot:
        on_plot(fname)  # 如果指定了回调函数，则调用回调函数
    # 根据传入的边界框信息 xyxy，裁剪输入图像 im，并返回裁剪后的图像。
    def save_one_box(xyxy, im, file='im.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
        """
        Args:
            xyxy (torch.Tensor or list): 表示边界框的张量或列表，格式为 xyxy。
            im (numpy.ndarray): 输入图像。
            file (Path, optional): 裁剪后的图像保存路径。默认为 'im.jpg'。
            gain (float, optional): 边界框尺寸增益因子。默认为 1.02。
            pad (int, optional): 边界框宽度和高度增加的像素数。默认为 10。
            square (bool, optional): 如果为 True，则将边界框转换为正方形。默认为 False。
            BGR (bool, optional): 如果为 True，则保存图像为 BGR 格式；否则保存为 RGB 格式。默认为 False。
            save (bool, optional): 如果为 True，则保存裁剪后的图像到磁盘。默认为 True。
    
        Returns:
            (numpy.ndarray): 裁剪后的图像。
    
        Example:
            ```python
            from ultralytics.utils.plotting import save_one_box
    
            xyxy = [50, 50, 150, 150]
            im = cv2.imread('image.jpg')
            cropped_im = save_one_box(xyxy, im, file='cropped.jpg', square=True)
            ```py
        """
    
        if not isinstance(xyxy, torch.Tensor):  # 如果 xyxy 不是 torch.Tensor 类型，可能是列表
            xyxy = torch.stack(xyxy)  # 转换为 torch.Tensor
    
        b = ops.xyxy2xywh(xyxy.view(-1, 4))  # 将 xyxy 格式的边界框转换为 xywh 格式
        if square:
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 尝试将矩形边界框转换为正方形
    
        b[:, 2:] = b[:, 2:] * gain + pad  # 计算边界框宽高乘以增益因子后加上 pad 像素
        xyxy = ops.xywh2xyxy(b).long()  # 将 xywh 格式的边界框转换回 xyxy 格式，并转换为整型坐标
        xyxy = ops.clip_boxes(xyxy, im.shape)  # 将边界框坐标限制在图像范围内
        crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]  # 根据边界框坐标裁剪图像
    
        if save:
            file.parent.mkdir(parents=True, exist_ok=True)  # 创建保存图像的文件夹
            f = str(increment_path(file).with_suffix(".jpg"))  # 生成带有递增数字的文件名，并设置为 jpg 后缀
            # cv2.imwrite(f, crop)  # 保存为 BGR 格式图像（存在色度抽样问题）
            Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # 保存为 RGB 格式图像
    
        return crop  # 返回裁剪后的图像
# 使用装饰器标记该函数为可多线程执行的函数
@threaded
# 定义函数用于绘制带有标签、边界框、掩码和关键点的图像网格
def plot_images(
    # 图像数据，可以是 torch.Tensor 或 np.ndarray 类型，形状为 (batch_size, channels, height, width)
    images: Union[torch.Tensor, np.ndarray],
    # 每个检测的批次索引，形状为 (num_detections,)
    batch_idx: Union[torch.Tensor, np.ndarray],
    # 每个检测的类别标签，形状为 (num_detections,)
    cls: Union[torch.Tensor, np.ndarray],
    # 每个检测的边界框，形状为 (num_detections, 4) 或 (num_detections, 5)（用于旋转边界框）
    bboxes: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.float32),
    # 每个检测的置信度分数，形状为 (num_detections,)
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    # 实例分割掩码，形状为 (num_detections, height, width) 或 (1, height, width)
    masks: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.uint8),
    # 每个检测的关键点，形状为 (num_detections, 51)
    kpts: Union[torch.Tensor, np.ndarray] = np.zeros((0, 51), dtype=np.float32),
    # 图像文件路径列表，与批次中每个图像对应
    paths: Optional[List[str]] = None,
    # 输出图像网格的文件名
    fname: str = "images.jpg",
    # 类别索引到类别名称的映射字典
    names: Optional[Dict[int, str]] = None,
    # 绘图完成后的回调函数，可选
    on_plot: Optional[Callable] = None,
    # 输出图像网格的最大尺寸
    max_size: int = 1920,
    # 图像网格中最大子图数目
    max_subplots: int = 16,
    # 是否保存绘制的图像网格到文件
    save: bool = True,
    # 显示检测结果所需的置信度阈值
    conf_thres: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """

    # 如果 images 是 torch.Tensor 类型，则转换为 numpy 数组类型
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    # 如果 cls 是 torch.Tensor 类型，则转换为 numpy 数组类型
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    # 如果 bboxes 是 torch.Tensor 类型，则转换为 numpy 数组类型
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    # 如果 masks 是 torch.Tensor 类型，则转换为 numpy 数组类型，并将类型转换为 int
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    # 如果 kpts 是 torch.Tensor 类型，则转换为 numpy 数组类型
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    # 如果 batch_idx 是 torch.Tensor 类型，则转换为 numpy 数组类型
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    # 获取图像的批次大小、通道数、高度和宽度
    bs, _, h, w = images.shape  # batch size, _, height, width
    # 限制要绘制的图像数量，最多为 max_subplots
    bs = min(bs, max_subplots)
    # 计算图像网格中子图的行数和列数（向上取整）
    ns = np.ceil(bs**0.5)
    
    # 如果图像的最大像素值小于等于1，则将其转换为 0-255 范围的值（去除标准化）
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)
    # 构建图像拼接
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # 初始化一个白色背景的图像数组

    # 遍历每个图像块，将其放置在合适的位置
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 计算当前块的起始位置
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)  # 将图像块放置到拼接图像上

    # 可选的调整大小操作
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)  # 计算调整后的高度
        w = math.ceil(scale * w)  # 计算调整后的宽度
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))  # 调整拼接后的图像大小

    # 添加注释
    fs = int((h + w) * ns * 0.01)  # 计算字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)  # 创建一个注释器对象
    if not save:
        return np.asarray(annotator.im)  # 如果不需要保存，返回注释后的图像数组
    annotator.im.save(fname)  # 否则保存注释后的图像
    if on_plot:
        on_plot(fname)  # 如果有指定的绘图函数，调用它并传入保存的文件名
@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results('path/to/results.csv', segment=True)
        ```py
    """
    import pandas as pd  # 导入 pandas 库，用于处理 CSV 文件
    from scipy.ndimage import gaussian_filter1d  # 导入 scipy 库中的高斯滤波函数

    # 确定保存图片的目录
    save_dir = Path(file).parent if file else Path(dir)

    # 根据不同的数据类型和设置，选择合适的子图布局和指数索引
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)  # 分类数据的布局
        index = [1, 4, 2, 3]  # 对应子图的索引顺序
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)  # 分割数据的布局
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]  # 对应子图的索引顺序
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)  # 姿态估计数据的布局
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]  # 对应子图的索引顺序
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)  # 默认数据的布局
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]  # 对应子图的索引顺序

    ax = ax.ravel()  # 将子图数组展平，便于迭代处理

    files = list(save_dir.glob("results*.csv"))  # 查找保存结果的 CSV 文件列表
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."  # 断言确保找到了结果文件，否则报错

    for f in files:
        try:
            data = pd.read_csv(f)  # 读取 CSV 文件中的数据
            s = [x.strip() for x in data.columns]  # 清理列名，去除空格
            x = data.values[:, 0]  # 获取 X 轴数据，通常是第一列数据

            # 遍历子图索引，绘制每个子图的数据曲线和平滑曲线
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")  # 获取 Y 轴数据，并转换为浮点数类型
                # y[y == 0] = np.nan  # 不显示值为零的点，可选功能

                # 绘制实际结果曲线
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)
                # 绘制平滑后的曲线
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)
                ax[i].set_title(s[j], fontsize=12)  # 设置子图标题

                # 如果是指定的子图索引，共享训练和验证损失的 Y 轴
                # if j in {8, 9, 10}:
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")  # 捕获并记录绘图过程中的异常信息

    ax[1].legend()  # 在第二个子图上添加图例
    # 指定文件名为 save_dir 下的 "results.png"
    fname = save_dir / "results.png"
    # 将当前图形保存为 PNG 文件，设置 DPI 为 200
    fig.savefig(fname, dpi=200)
    # 关闭当前图形，释放资源
    plt.close()
    # 如果定义了 on_plot 回调函数，则调用该函数，传递保存的文件名作为参数
    if on_plot:
        on_plot(fname)
def plot_tune_results(csv_file="tune_results.csv"):
    """
    Plot the evolution results stored in an 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.

    Examples:
        >>> plot_tune_results('path/to/tune_results.csv')
    """

    import pandas as pd  # 导入 pandas 库，用于处理数据
    from scipy.ndimage import gaussian_filter1d  # 导入 scipy 库中的高斯滤波函数

    def _save_one_file(file):
        """Save one matplotlib plot to 'file'."""
        plt.savefig(file, dpi=200)  # 保存当前 matplotlib 图形为指定文件，设置分辨率为200dpi
        plt.close()  # 关闭当前 matplotlib 图形
        LOGGER.info(f"Saved {file}")  # 记录日志信息，显示保存成功的文件名

    # Scatter plots for each hyperparameter
    csv_file = Path(csv_file)  # 将传入的 CSV 文件路径转换为 Path 对象
    data = pd.read_csv(csv_file)  # 使用 pandas 读取 CSV 文件中的数据
    num_metrics_columns = 1  # 指定要跳过的列数（这里是第一列的列数）
    keys = [x.strip() for x in data.columns][num_metrics_columns:]  # 获取 CSV 文件中的列名，并去除首尾空白字符
    x = data.values  # 获取 CSV 文件中的所有数据值
    fitness = x[:, 0]  # 从数据中提取 fitness（适应度）列数据
    j = np.argmax(fitness)  # 找到 fitness 列中最大值的索引
    n = math.ceil(len(keys) ** 0.5)  # 计算绘图的行数和列数，向上取整以确保足够的子图空间
    plt.figure(figsize=(10, 10), tight_layout=True)  # 创建一个 10x10 英寸大小的图形，并启用紧凑布局
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]  # 获取当前列（除 fitness 外的其他列）的数据
        mu = v[j]  # 获取当前列中 fitness 最大值对应的数据点
        plt.subplot(n, n, i + 1)  # 在 n x n 的子图中，选择第 i+1 个子图
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")  # 调用 plt_color_scatter 函数绘制散点图
        plt.plot(mu, fitness.max(), "k+", markersize=15)  # 在散点图上绘制 fitness 最大值对应的点
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # 设置子图标题，显示参数名和对应的最佳单个结果
        plt.tick_params(axis="both", labelsize=8)  # 设置坐标轴标签的大小为 8
        if i % n != 0:
            plt.yticks([])  # 如果不是每行的第一个子图，则不显示 y 轴刻度

    _save_one_file(csv_file.with_name("tune_scatter_plots.png"))  # 调用保存函数，将绘制好的图形保存为 PNG 文件

    # Fitness vs iteration
    # 生成 x 轴的数值范围，从1到fitness列表长度加1
    x = range(1, len(fitness) + 1)
    # 创建一个图形对象，设置图形大小为10x6，启用紧凑布局
    plt.figure(figsize=(10, 6), tight_layout=True)
    # 绘制 fitness 列表的数据点，使用圆形标记，折线样式为无，设置标签为"fitness"
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")
    # 绘制 fitness 列表数据点的高斯平滑曲线，设置折线样式为冒号，设置标签为"smoothed"，设置线宽为2，说明是平滑线
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)  # smoothing line
    # 设置图形的标题为"Fitness vs Iteration"
    plt.title("Fitness vs Iteration")
    # 设置 x 轴标签为"Iteration"
    plt.xlabel("Iteration")
    # 设置 y 轴标签为"Fitness"
    plt.ylabel("Fitness")
    # 启用网格线
    plt.grid(True)
    # 显示图例
    plt.legend()
    # 调用保存图形的函数，保存文件名为csv_file的名称加上"tune_fitness.png"作为后缀
    _save_one_file(csv_file.with_name("tune_fitness.png"))
def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    # 检查模块类型是否属于需要可视化的类型，如果不属于则直接返回
    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:
            return

    # 检查输入特征是否为Tensor类型
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # 获取特征张量的形状信息：batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # 构建保存文件路径和名称

            # 按照通道数拆分特征图块
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # 选择批次索引为0的数据，并按通道拆分
            n = min(n, channels)  # 确定要绘制的特征图块数量，不超过通道数
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 创建绘图布局，8行 n/8 列
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # 显示特征图块，去除单维度条目
                ax[i].axis("off")  # 关闭坐标轴显示

            LOGGER.info(f"Saving {f}... ({n}/{channels})")  # 记录保存文件信息
            plt.savefig(f, dpi=300, bbox_inches="tight")  # 保存绘制结果为PNG文件，300dpi，紧凑边界
            plt.close()  # 关闭绘图窗口
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # 保存特征数据为.npy文件
```