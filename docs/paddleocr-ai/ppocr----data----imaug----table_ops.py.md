# `.\PaddleOCR\ppocr\data\imaug\table_ops.py`

```
"""
# 版权声明
# 2020年版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 导入系统模块
import sys
# 导入兼容 Python 2 和 Python 3 的模块
import six
# 导入 OpenCV 模块
import cv2
# 导入 NumPy 模块
import numpy as np

# 定义 GenTableMask 类
class GenTableMask(object):
    """ 生成表格掩码 """

    # 初始化方法
    def __init__(self, shrink_h_max, shrink_w_max, mask_type=0, **kwargs):
        # 设置最大高度缩小值
        self.shrink_h_max = 5
        # 设置最大宽度缩小值
        self.shrink_w_max = 5
        # 设置掩码类型，默认为 0
        self.mask_type = mask_type
    # 对二值化图像进行水平投影，返回投影直方图和字符区域的边界框列表
    def projection(self, erosion, h, w, spilt_threshold=0):
        # 创建与二值化图像相同大小的全1矩阵作为投影图
        projection_map = np.ones_like(erosion)
        # 初始化一个长度为h的全0数组，用于记录每一行的像素点数量
        project_val_array = [0 for _ in range(0, h)]

        # 遍历每一行每一列的像素值，统计每一行的像素点数量
        for j in range(0, h):
            for i in range(0, w):
                if erosion[j, i] == 255:
                    project_val_array[j] += 1
        
        # 初始化变量用于记录字符区域的起始和结束索引，以及是否在字符区域内
        start_idx = 0  # 记录进入字符区的索引
        end_idx = 0  # 记录进入空白区域的索引
        in_text = False  # 是否遍历到了字符区内
        box_list = []

        # 根据投影直方图数组获取字符区域的边界框列表
        for i in range(len(project_val_array)):
            if in_text == False and project_val_array[i] > spilt_threshold:  # 进入字符区了
                in_text = True
                start_idx = i
            elif project_val_array[i] <= spilt_threshold and in_text == True:  # 进入空白区了
                end_idx = i
                in_text = False
                if end_idx - start_idx <= 2:
                    continue
                box_list.append((start_idx, end_idx + 1))

        # 处理最后一个字符区域
        if in_text:
            box_list.append((start_idx, h - 1))
        
        # 根据投影直方图数组绘制投影直方图
        for j in range(0, h):
            for i in range(0, project_val_array[j]):
                projection_map[j, i] = 0
        
        # 返回字符区域的边界框列表和投影直方图
        return box_list, projection_map

    # 缩小边界框的大小
    def shrink_bbox(self, bbox):
        left, top, right, bottom = bbox
        # 计算缩小的高度和宽度
        sh_h = min(max(int((bottom - top) * 0.1), 1), self.shrink_h_max)
        sh_w = min(max(int((right - left) * 0.1), 1), self.shrink_w_max)
        # 根据缩小的高度和宽度计算新的边界框坐标
        left_new = left + sh_w
        right_new = right - sh_w
        top_new = top + sh_h
        bottom_new = bottom - sh_h
        # 处理边界情况，确保边界框合法
        if left_new >= right_new:
            left_new = left
            right_new = right
        if top_new >= bottom_new:
            top_new = top
            bottom_new = bottom
        # 返回新的边界框坐标
        return [left_new, top_new, right_new, bottom_new]
    # 定义一个类的调用方法，接受数据作为参数
    def __call__(self, data):
        # 从数据中获取图像数据
        img = data['image']
        # 从数据中获取细胞数据
        cells = data['cells']
        # 获取图像的高度和宽度
        height, width = img.shape[0:2]
        # 根据不同的mask类型创建空白的mask图像
        if self.mask_type == 1:
            mask_img = np.zeros((height, width), dtype=np.float32)
        else:
            mask_img = np.zeros((height, width, 3), dtype=np.float32)
        # 获取细胞的数量
        cell_num = len(cells)
        # 遍历每个细胞
        for cno in range(cell_num):
            # 如果细胞中包含bbox信息
            if "bbox" in cells[cno]:
                # 获取bbox信息
                bbox = cells[cno]['bbox']
                left, top, right, bottom = bbox
                # 从原图像中裁剪出bbox对应的区域
                box_img = img[top:bottom, left:right, :].copy()
                # 对裁剪出的区域进行投影变换
                split_bbox_list = self.projection_cx(box_img)
                # 调整投影变换后的bbox坐标
                for sno in range(len(split_bbox_list)):
                    split_bbox_list[sno][0] += left
                    split_bbox_list[sno][1] += top
                    split_bbox_list[sno][2] += left
                    split_bbox_list[sno][3] += top

                # 遍历调整后的bbox列表
                for sno in range(len(split_bbox_list)):
                    left, top, right, bottom = split_bbox_list[sno]
                    # 缩小bbox区域
                    left, top, right, bottom = self.shrink_bbox(
                        [left, top, right, bottom])
                    # 根据mask类型填充mask图像
                    if self.mask_type == 1:
                        mask_img[top:bottom, left:right] = 1.0
                        data['mask_img'] = mask_img
                    else:
                        mask_img[top:bottom, left:right, :] = (255, 255, 255)
                        data['image'] = mask_img
        # 返回处理后的数据
        return data
# 定义一个类 ResizeTableImage，用于调整表格图像的大小
class ResizeTableImage(object):
    # 初始化方法，设置最大长度、是否调整边界框、是否推断模式等参数
    def __init__(self, max_len, resize_bboxes=False, infer_mode=False,
                 **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len
        self.resize_bboxes = resize_bboxes
        self.infer_mode = infer_mode

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 获取图像的高度和宽度
        height, width = img.shape[0:2]
        # 计算调整比例
        ratio = self.max_len / (max(height, width) * 1.0)
        # 计算调整后的高度和宽度
        resize_h = int(height * ratio)
        resize_w = int(width * ratio)
        # 调整图像大小
        resize_img = cv2.resize(img, (resize_w, resize_h))
        # 如果需要调整边界框且不是推断模式
        if self.resize_bboxes and not self.infer_mode:
            # 调整边界框坐标
            data['bboxes'] = data['bboxes'] * ratio
        # 更新图像数据
        data['image'] = resize_img
        data['src_img'] = img
        # 更新数据形状信息
        data['shape'] = np.array([height, width, ratio, ratio])
        data['max_len'] = self.max_len
        return data

# 定义一个类 PaddingTableImage，用于填充表格图像
class PaddingTableImage(object):
    # 初始化方法，设置填充大小等参数
    def __init__(self, size, **kwargs):
        super(PaddingTableImage, self).__init__()
        self.size = size

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 获取填充高度和宽度
        pad_h, pad_w = self.size
        # 创建填充后的图像
        padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        # 获取原图像的高度和宽度
        height, width = img.shape[0:2]
        # 将原图像复制到填充图像中
        padding_img[0:height, 0:width, :] = img.copy()
        # 更新图像数据
        data['image'] = padding_img
        # 更新数据形状信息
        shape = data['shape'].tolist()
        shape.extend([pad_h, pad_w])
        data['shape'] = np.array(shape)
        return data
```