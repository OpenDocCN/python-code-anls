# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\modules\random_crop_data.py`

```
# 导入 random 模块
import random

# 导入 OpenCV 模块
import cv2

# 导入 NumPy 模块
import numpy as np

# 定义一个类，实现类似于 https://github.com/argman/EAST 的随机裁剪算法
class EastRandomCropData():
    # 初始化方法，设置参数
    def __init__(self,
                 size=(640, 640),  # 裁剪后的大小
                 max_tries=50,  # 最大尝试次数
                 min_crop_side_ratio=0.1,  # 最小裁剪边长比例
                 require_original_image=False,  # 是否需要原始图像
                 keep_ratio=True):  # 是否保持宽高比
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image
        self.keep_ratio = keep_ratio
    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags'}
        :return: 返回处理后的数据字典
        """
        # 获取输入数据中的图片、文本框坐标、忽略标签和文本内容
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        # 从所有非忽略文本框中获取坐标信息
        all_care_polys = [
            text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
        ]
        # 计算裁剪区域的坐标
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        # 计算缩放比例
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        # 根据是否保持比例进行填充
        if self.keep_ratio:
            if len(im.shape) == 3:
                padimg = np.zeros((self.size[1], self.size[0], im.shape[2]),
                                  im.dtype)
            else:
                padimg = np.zeros((self.size[1], self.size[0]), im.dtype)
            padimg[:h, :w] = cv2.resize(
                im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                             tuple(self.size))
        # 裁剪文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        # 更新数据字典中的图片、文本框坐标、忽略标签和文本内容
        data['img'] = img
        data['text_polys'] = np.float32(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data
    # 检查多边形是否完全在矩形内部
    def is_poly_in_rect(self, poly, x, y, w, h):
        # 将多边形转换为 NumPy 数组
        poly = np.array(poly)
        # 检查多边形的 x 坐标范围是否在矩形内
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        # 检查多边形的 y 坐标范围是否在矩形内
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    # 检查多边形是否完全在矩形外部
    def is_poly_outside_rect(self, poly, x, y, w, h):
        # 将多边形转换为 NumPy 数组
        poly = np.array(poly)
        # 检查多边形的 x 坐标范围是否完全在矩形外
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        # 检查多边形的 y 坐标范围是否完全在矩形外
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    # 将轴向量分割成不连续的区域
    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    # 在轴向量中随机选择两个值
    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    # 在不连续区域中按区域随机选择值
    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax
    # 根据给定的文本多边形裁剪图像区域
    def crop_area(self, im, text_polys):
        # 获取图像的高度和宽度
        h, w = im.shape[:2]
        # 创建一个与图像高度相同的零数组
        h_array = np.zeros(h, dtype=np.int32)
        # 创建一个与图像宽度相同的零数组
        w_array = np.zeros(w, dtype=np.int32)
        # 遍历文本多边形的每个点
        for points in text_polys:
            # 将点坐标四舍五入并转换为整数
            points = np.round(points, decimals=0).astype(np.int32)
            # 获取点的最小和最大 x 坐标
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            # 将 x 坐标范围内的数组值设为1
            w_array[minx:maxx] = 1
            # 获取点的最小和最大 y 坐标
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            # 将 y 坐标范围内的数组值设为1
            h_array[miny:maxy] = 1
        # 确保裁剪区域不跨越文本
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        # 如果没有可用的裁剪区域，则返回整个图像的尺寸
        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        # 将高度和宽度分割成多个区域
        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        # 尝试多次随机选择裁剪区域
        for i in range(self.max_tries):
            # 如果存在多个宽度区域，则在这些区域中随机选择
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            # 如果存在多个高度区域，则在这些区域中随机选择
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            # 如果裁剪区域太小，则继续尝试
            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                continue
            num_poly_in_rect = 0
            # 检查裁剪区域内是否有文本多边形
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                                 ymax - ymin):
                    num_poly_in_rect += 1
                    break

            # 如果裁剪区域内有文本多边形，则返回该区域的坐标和尺寸
            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        # 如果无法找到合适的裁剪区域，则返回整个图像的尺寸
        return 0, 0, w, h
# 定义一个类 PSERandomCrop，用于实现随机裁剪操作
class PSERandomCrop():
    # 初始化方法，接收裁剪尺寸参数
    def __init__(self, size):
        self.size = size

    # 调用方法，接收数据并进行裁剪操作
    def __call__(self, data):
        # 获取数据中的图像列表
        imgs = data['imgs']

        # 获取第一张图像的高度和宽度
        h, w = imgs[0].shape[0:2]
        # 获取裁剪尺寸
        th, tw = self.size
        # 如果图像尺寸与裁剪尺寸相同，则直接返回原图像
        if w == tw and h == th:
            return imgs

        # 如果图像中存在文本实例，并且按照概率进行裁剪，使用threshold_label_map控制
        if np.max(imgs[2]) > 0 and random.random() > 3 / 8:
            # 获取文本实例的左上角点
            tl = np.min(np.where(imgs[2] > 0), axis=1) - self.size
            tl[tl < 0] = 0
            # 获取文本实例的右下角点
            br = np.max(np.where(imgs[2] > 0), axis=1) - self.size
            br[br < 0] = 0
            # 确保选到右下角点时，有足够的距离进行裁剪
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            # 循环50000次，随机选择裁剪区域
            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])
                # 确保裁剪区域包含文本
                if imgs[1][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            # 随机选择裁剪区域
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # 对所有图像进行裁剪操作
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        # 更新数据中的图像列表
        data['imgs'] = imgs
        return data
```