# `.\PaddleOCR\ppocr\data\imaug\random_crop_data.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引入未来的绝对导入、除法、打印函数、unicode 字符串
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 引入所需的库
import numpy as np
import cv2
import random

# 判断多边形是否在矩形内部
def is_poly_in_rect(poly, x, y, w, h):
    poly = np.array(poly)
    # 判断多边形的 x 坐标范围是否在矩形内
    if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
        return False
    # 判断多边形的 y 坐标范围是否在矩形内
    if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
        return False
    return True

# 判断多边形是否在矩形外部
def is_poly_outside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    # 判断多边形的 x 坐标范围是否在矩形外
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    # 判断多边形的 y 坐标范围是否在矩形外
    if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
        return True
    return False

# 将轴向数组分割成区域
def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions

# 随机选择区域
def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax

# 区域智能随机选择
def region_wise_random_select(regions, max_size):
    # 从 regions 列表中随机选择两个索引
    selected_index = list(np.random.choice(len(regions), 2))
    # 初始化一个空列表用于存储选定的数值
    selected_values = []
    # 遍历选定的索引
    for index in selected_index:
        # 获取对应索引的 axis
        axis = regions[index]
        # 从 axis 中随机选择一个数值
        xx = int(np.random.choice(axis, size=1))
        # 将选定的数值添加到列表中
        selected_values.append(xx)
    # 计算选定数值列表中的最小值
    xmin = min(selected_values)
    # 计算选定数值列表中的最大值
    xmax = max(selected_values)
    # 返回最小值和最大值
    return xmin, xmax
# 根据给定的图像和文本多边形裁剪区域
def crop_area(im, text_polys, min_crop_side_ratio, max_tries):
    # 获取图像的高度、宽度和通道数
    h, w, _ = im.shape
    # 创建一个长度为 h 的零数组和一个长度为 w 的零数组
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    # 遍历文本多边形的点集
    for points in text_polys:
        # 将点集四舍五入到整数，并转换为整型
        points = np.round(points, decimals=0).astype(np.int32)
        # 获取 x 轴最小值和最大值
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        # 在 w_array 中标记 x 轴范围
        w_array[minx:maxx] = 1
        # 获取 y 轴最小值和最大值
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])
        # 在 h_array 中标记 y 轴范围
        h_array[miny:maxy] = 1
    # 确保裁剪区域不跨越文本
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h

    # 将 h_axis 和 w_axis 分割成多个区域
    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            # 在 w 轴区域中随机选择 xmin 和 xmax
            xmin, xmax = region_wise_random_select(w_regions, w)
        else:
            # 在 w_axis 中随机选择 xmin 和 xmax
            xmin, xmax = random_select(w_axis, w)
        if len(h_regions) > 1:
            # 在 h 轴区域中随机选择 ymin 和 ymax
            ymin, ymax = region_wise_random_select(h_regions, h)
        else:
            # 在 h_axis 中随机选择 ymin 和 ymax
            ymin, ymax = random_select(h_axis, h)

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # 裁剪区域太小，继续尝试
            continue
        num_poly_in_rect = 0
        for poly in text_polys:
            if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                        ymax - ymin):
                num_poly_in_rect += 1
                break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h

# 定义一个类用于随机裁剪数据
class EastRandomCropData(object):
    def __init__(self,
                 size=(640, 640),
                 max_tries=10,
                 min_crop_side_ratio=0.1,
                 keep_ratio=True,
                 **kwargs):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio
    # 定义一个类的调用方法，接受一个数据字典作为参数
    def __call__(self, data):
        # 从数据字典中获取图像数据
        img = data['image']
        # 从数据字典中获取文本框的多边形坐标
        text_polys = data['polys']
        # 从数据字典中获取忽略标签
        ignore_tags = data['ignore_tags']
        # 从数据字典中获取文本内容
        texts = data['texts']
        # 从所有文本框中筛选出不被忽略的文本框
        all_care_polys = [
            text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
        ]
        # 计算裁剪区域的坐标和尺寸
        crop_x, crop_y, crop_w, crop_h = crop_area(
            img, all_care_polys, self.min_crop_side_ratio, self.max_tries)
        # 计算裁剪后的图片缩放比例
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        # 如果保持比例，填充裁剪后的图片
        if self.keep_ratio:
            padimg = np.zeros((self.size[1], self.size[0], img.shape[2]),
                              img.dtype)
            padimg[:h, :w] = cv2.resize(
                img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            # 调整裁剪后的图片大小
            img = cv2.resize(
                img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))
        # 裁剪文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            # 根据裁剪区域和缩放比例调整文本框坐标
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            # 如果文本框在裁剪后的图片内部，则保留
            if not is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        # 更新数据字典中的图像、文本框、忽略标签和文本内容
        data['image'] = img
        data['polys'] = np.array(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        # 返回更新后的数据字典
        return data
class RandomCropImgMask(object):
    # 初始化函数，设置裁剪尺寸、主要关键字、裁剪关键字、概率等参数
    def __init__(self, size, main_key, crop_keys, p=3 / 8, **kwargs):
        self.size = size
        self.main_key = main_key
        self.crop_keys = crop_keys
        self.p = p

    # 调用函数，对数据进行裁剪处理
    def __call__(self, data):
        # 获取图像数据
        image = data['image']

        # 获取图像的高度和宽度
        h, w = image.shape[0:2]
        # 获取裁剪尺寸
        th, tw = self.size
        # 如果图像尺寸与裁剪尺寸相同，则直接返回数据
        if w == tw and h == th:
            return data

        # 获取掩模数据
        mask = data[self.main_key]
        # 如果掩模中存在文本区域且随机数大于概率值
        if np.max(mask) > 0 and random.random() > self.p:
            # 确保裁剪文本区域
            tl = np.min(np.where(mask > 0), axis=1) - (th, tw)
            tl[tl < 0] = 0
            br = np.max(np.where(mask > 0), axis=1) - (th, tw)
            br[br < 0] = 0

            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, h - th) if h - th > 0 else 0
            j = random.randint(0, w - tw) if w - tw > 0 else 0

        # 对数据进行裁剪处理
        for k in data:
            if k in self.crop_keys:
                if len(data[k].shape) == 3:
                    if np.argmin(data[k].shape) == 0:
                        img = data[k][:, i:i + th, j:j + tw]
                        if img.shape[1] != img.shape[2]:
                            a = 1
                    elif np.argmin(data[k].shape) == 2:
                        img = data[k][i:i + th, j:j + tw, :]
                        if img.shape[1] != img.shape[0]:
                            a = 1
                    else:
                        img = data[k]
                else:
                    img = data[k][i:i + th, j:j + tw]
                    if img.shape[0] != img.shape[1]:
                        a = 1
                data[k] = img
        return data
```