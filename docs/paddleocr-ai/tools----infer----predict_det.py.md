# `.\PaddleOCR\tools\infer\predict_det.py`

```py
# 版权声明
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入 OpenCV 库
import cv2
# 导入 NumPy 库
import numpy as np
# 导入时间模块
import time
# 再次导入 sys 模块
import sys

# 导入自定义的 utility 模块
import tools.infer.utility as utility
# 从 ppocr.utils.logging 模块中导入 get_logger 函数
from ppocr.utils.logging import get_logger
# 从 ppocr.utils.utility 模块中导入 get_image_file_list 和 check_and_read 函数
from ppocr.utils.utility import get_image_file_list, check_and_read
# 从 ppocr.data 模块中导入 create_operators 和 transform 函数
from ppocr.data import create_operators, transform
# 从 ppocr.postprocess 模块中导入 build_post_process 函数
from ppocr.postprocess import build_post_process
# 导入 json 模块
import json
# 获取日志记录器
logger = get_logger()

# 定义 TextDetector 类
class TextDetector(object):
    # 定义 order_points_clockwise 方法，用于按顺时针顺序排列点
    def order_points_clockwise(self, pts):
        # 创建一个全零矩阵，用于存储排序后的点
        rect = np.zeros((4, 2), dtype="float32")
        # 计算点的和
        s = pts.sum(axis=1)
        # 找到和最小的点作为矩形的第一个点
        rect[0] = pts[np.argmin(s)]
        # 找到和最大的点作为矩形的第三个点
        rect[2] = pts[np.argmax(s)]
        # 删除最小和最大的两个点，剩下的两个点中，横坐标之差最小的点作为第二个点，差值最大的点作为第四个点
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    # 定义 clip_det_res 方法，用于裁剪检测结果的点坐标
    def clip_det_res(self, points, img_height, img_width):
        # 遍历每个点的坐标
        for pno in range(points.shape[0]):
            # 将点的横坐标限制在 [0, img_width-1] 范围内
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            # 将点的纵坐标限制在 [0, img_height-1] 范围内
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    # 过滤文本检测结果中的标签框，根据图像形状进行裁剪
    def filter_tag_det_res(self, dt_boxes, image_shape):
        # 获取图像的高度和宽度
        img_height, img_width = image_shape[0:2]
        # 创建一个新的空列表，用于存储处理后的标签框
        dt_boxes_new = []
        # 遍历每个标签框
        for box in dt_boxes:
            # 如果标签框是列表类型，则转换为 NumPy 数组
            if type(box) is list:
                box = np.array(box)
            # 对标签框的顶点按顺时针排序
            box = self.order_points_clockwise(box)
            # 裁剪标签框，确保不超出图像范围
            box = self.clip_det_res(box, img_height, img_width)
            # 计算标签框的宽度和高度
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            # 如果标签框的宽度或高度小于等于3，则跳过该标签框
            if rect_width <= 3 or rect_height <= 3:
                continue
            # 将处理后的标签框添加到新列表中
            dt_boxes_new.append(box)
        # 将新列表转换为 NumPy 数组并返回
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    # 仅裁剪文本检测结果中的标签框，不进行其他处理
    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        # 获取图像的高度和宽度
        img_height, img_width = image_shape[0:2]
        # 创建一个新的空列表，用于存储处理后的标签框
        dt_boxes_new = []
        # 遍历每个标签框
        for box in dt_boxes:
            # 如果标签框是列表类型，则转换为 NumPy 数组
            if type(box) is list:
                box = np.array(box)
            # 裁剪标签框，确保不超出图像范围
            box = self.clip_det_res(box, img_height, img_width)
            # 将处理后的标签框添加到新列表中
            dt_boxes_new.append(box)
        # 将新列表转换为 NumPy 数组并返回
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 解析命令行参数
    args = utility.parse_args()
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建文本检测器对象
    text_detector = TextDetector(args)
    # 初始化总时间
    total_time = 0
    # 获取绘制图像保存目录
    draw_img_save_dir = args.draw_img_save_dir
    # 如果目录不存在，则创建目录
    os.makedirs(draw_img_save_dir, exist_ok=True)

    # 如果需要预热
    if args.warmup:
        # 创建一个随机图像
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        # 进行两次文本检测
        for i in range(2):
            res = text_detector(img)

    # 初始化保存结果列表
    save_results = []
    # 打开保存检测结果的文件
    with open(os.path.join(draw_img_save_dir, "det_results.txt"), 'w') as f:
        # 将结果写入文件
        f.writelines(save_results)
        # 关闭文件
        f.close()
    
    # 如果需要进行基准测试
    if args.benchmark:
        # 输出性能日志
        text_detector.autolog.report()
```