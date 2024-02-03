# `.\PaddleOCR\ppocr\data\imaug\fce_aug.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制
"""
这段代码参考自:
https://github.com/open-mmlab/mmocr/blob/main/mmocr/datasets/pipelines/transforms.py
"""
# 导入所需的库
import numpy as np
from PIL import Image, ImageDraw
import cv2
from shapely.geometry import Polygon
import math
from ppocr.utils.poly_nms import poly_intersection

# 定义一个类 RandomScaling
class RandomScaling:
    def __init__(self, size=800, scale=(3. / 4, 5. / 2), **kwargs):
        """随机缩放图像，保持长宽比不变。

        Args:
            size (int) : 缩放前的基本大小。
            scale (tuple(float)) : 缩放范围。
        """
        # 断言确保 size 是整数类型
        assert isinstance(size, int)
        # 断言确保 scale 是浮点数或元组类型
        assert isinstance(scale, float) or isinstance(scale, tuple)
        # 初始化 size 和 scale 属性
        self.size = size
        self.scale = scale if isinstance(scale, tuple) \
            else (1 - scale, 1 + scale)
    # 定义一个类的调用方法，接受一个数据字典作为参数
    def __call__(self, data):
        # 从数据字典中获取图像数据和文本多边形数据
        image = data['image']
        text_polys = data['polys']
        # 获取图像的高度、宽度和通道数
        h, w, _ = image.shape

        # 从指定范围内随机选择一个长宽比
        aspect_ratio = np.random.uniform(min(self.scale), max(self.scale))
        # 计算缩放比例，使得图像的最大边缩放到指定大小，并根据长宽比进行调整
        scales = self.size * 1.0 / max(h, w) * aspect_ratio
        scales = np.array([scales, scales])
        # 计算缩放后的图像尺寸
        out_size = (int(h * scales[1]), int(w * scales[0]))
        # 使用双线性插值对图像进行缩放
        image = cv2.resize(image, out_size[::-1])

        # 更新数据字典中的图像数据
        data['image'] = image
        # 根据缩放比例对文本多边形的 x 坐标进行缩放
        text_polys[:, :, 0::2] = text_polys[:, :, 0::2] * scales[1]
        # 根据缩放比例对文本多边形的 y 坐标进行缩放
        text_polys[:, :, 1::2] = text_polys[:, :, 1::2] * scales[0]
        # 更新数据字典中的文本多边形数据
        data['polys'] = text_polys

        # 返回更新后的数据字典
        return data
# 定义 RandomCropFlip 类，用于随机裁剪和翻转图像的一个 patch
class RandomCropFlip:
    # 初始化方法，设置裁剪和翻转的参数
    def __init__(self,
                 pad_ratio=0.1,
                 crop_ratio=0.5,
                 iter_num=1,
                 min_area_ratio=0.2,
                 **kwargs):
        """Random crop and flip a patch of the image.

        Args:
            crop_ratio (float): The ratio of cropping.
            iter_num (int): Number of operations.
            min_area_ratio (float): Minimal area ratio between cropped patch
                and original image.
        """
        # 断言确保参数的类型正确
        assert isinstance(crop_ratio, float)
        assert isinstance(iter_num, int)
        assert isinstance(min_area_ratio, float)

        # 设置各个参数的值
        self.pad_ratio = pad_ratio
        self.epsilon = 1e-2
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    # 定义 __call__ 方法，用于对输入的 results 进行随机裁剪和翻转
    def __call__(self, results):
        # 循环执行指定次数的随机裁剪和翻转操作
        for i in range(self.iter_num):
            results = self.random_crop_flip(results)

        # 返回处理后的 results
        return results
    def generate_crop_target(self, image, all_polys, pad_h, pad_w):
        """Generate crop target and make sure not to crop the polygon
        instances.

        Args:
            image (ndarray): The image waited to be crop.
            all_polys (list[list[ndarray]]): All polygons including ground
                truth polygons and ground truth ignored polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.
        Returns:
            h_axis (ndarray): Vertical cropping range.
            w_axis (ndarray): Horizontal cropping range.
        """
        # 获取图像的高度和宽度
        h, w, _ = image.shape
        # 创建一个全零数组，用于记录垂直方向的裁剪范围
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        # 创建一个全零数组，用于记录水平方向的裁剪范围
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        # 存储所有文本多边形的顶点坐标
        text_polys = []
        # 遍历所有多边形
        for polygon in all_polys:
            # 获取最小外接矩形
            rect = cv2.minAreaRect(polygon.astype(np.int32).reshape(-1, 2))
            # 获取矩形的四个顶点
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        # 将多边形转换为 NumPy 数组
        polys = np.array(text_polys, dtype=np.int32)
        # 遍历所有多边形
        for poly in polys:
            # 对多边形进行四舍五入取整
            poly = np.round(poly, decimals=0).astype(np.int32)
            # 获取多边形在水平方向上的最小和最大值
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            # 在水平方向上标记裁剪范围
            w_array[minx + pad_w:maxx + pad_w] = 1
            # 获取多边形在垂直方向上的最小和最大值
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            # 在垂直方向上标记裁剪范围
            h_array[miny + pad_h:maxy + pad_h] = 1

        # 获取垂直方向上未被标记的裁剪范围
        h_axis = np.where(h_array == 0)[0]
        # 获取水平方向上未被标记的裁剪范围
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis
class RandomCropPolyInstances:
    """随机裁剪图像，并确保至少包含一个完整实例。"""

    def __init__(self, crop_ratio=5.0 / 8.0, min_side_ratio=0.4, **kwargs):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.min_side_ratio = min_side_ratio

    def sample_valid_start_end(self, valid_array, min_len, max_start, min_end):
        """从有效数组中随机选择起始和结束位置，确保长度大于最小长度，并在指定范围内选择起始和结束位置。"""

        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start = min(len(start_array) - min_len, max_start)
        start_array[max_start:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end = max(start + min_len, min_end)
        end_array[:min_end] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end
    def sample_crop_box(self, img_size, results):
        """Generate crop box and make sure not to crop the polygon instances.

        Args:
            img_size (tuple(int)): The image size (h, w).
            results (dict): The results dict.
        """

        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        key_masks = results['polys']

        # 创建一个长度为图像宽度的数组，用于标记有效的 x 坐标
        x_valid_array = np.ones(w, dtype=np.int32)
        # 创建一个长度为图像高度的数组，用于标记有效的 y 坐标
        y_valid_array = np.ones(h, dtype=np.int32)

        # 从关键 mask 中随机选择一个作为裁剪的目标
        selected_mask = key_masks[np.random.randint(0, len(key_masks))]
        selected_mask = selected_mask.reshape((-1, 2)).astype(np.int32)
        # 计算裁剪框的左上角和右下角坐标
        max_x_start = max(np.min(selected_mask[:, 0]) - 2, 0)
        min_x_end = min(np.max(selected_mask[:, 0]) + 3, w - 1)
        max_y_start = max(np.min(selected_mask[:, 1]) - 2, 0)
        min_y_end = min(np.max(selected_mask[:, 1]) + 3, h - 1)

        # 遍历所有关键 mask，更新 x 和 y 方向上的有效坐标数组
        for mask in key_masks:
            mask = mask.reshape((-1, 2)).astype(np.int32)
            clip_x = np.clip(mask[:, 0], 0, w - 1)
            clip_y = np.clip(mask[:, 1], 0, h - 1)
            min_x, max_x = np.min(clip_x), np.max(clip_x)
            min_y, max_y = np.min(clip_y), np.max(clip_y)

            x_valid_array[min_x - 2:max_x + 3] = 0
            y_valid_array[min_y - 2:max_y + 3] = 0

        # 计算裁剪框的最小宽度和高度
        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        # 根据有效的 x 和 y 坐标数组，以及最小宽度和高度，生成裁剪框的起始和结束坐标
        x1, x2 = self.sample_valid_start_end(x_valid_array, min_w, max_x_start,
                                             min_x_end)
        y1, y2 = self.sample_valid_start_end(y_valid_array, min_h, max_y_start,
                                             min_y_end)

        return np.array([x1, y1, x2, y2])

    def crop_img(self, img, bbox):
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        # 根据给定的裁剪框坐标，对图像进行裁剪并返回
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # 定义一个类方法，用于对输入的结果进行处理
    def __call__(self, results):
        # 从结果中获取图像数据、多边形坐标和忽略标签
        image = results['image']
        polygons = results['polys']
        ignore_tags = results['ignore_tags']
        # 如果多边形数量小于1，则直接返回结果
        if len(polygons) < 1:
            return results

        # 根据概率进行裁剪
        if np.random.random_sample() < self.crop_ratio:
            # 获取裁剪框
            crop_box = self.sample_crop_box(image.shape, results)
            # 对图像进行裁剪
            img = self.crop_img(image, crop_box)
            results['image'] = img
            # 裁剪并过滤掩模
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            # 调整多边形坐标
            polygons[:, :, 0::2] = polygons[:, :, 0::2] - x1
            polygons[:, :, 1::2] = polygons[:, :, 1::2] - y1

            valid_masks_list = []
            valid_tags_list = []
            # 遍历多边形列表，筛选有效的多边形和对应的忽略标签
            for ind, polygon in enumerate(polygons):
                if (polygon[:, ::2] > -4).all() and (
                        polygon[:, ::2] < w + 4).all() and (
                            polygon[:, 1::2] > -4).all() and (
                                polygon[:, 1::2] < h + 4).all():
                    polygon[:, ::2] = np.clip(polygon[:, ::2], 0, w)
                    polygon[:, 1::2] = np.clip(polygon[:, 1::2], 0, h)
                    valid_masks_list.append(polygon)
                    valid_tags_list.append(ignore_tags[ind])

            results['polys'] = np.array(valid_masks_list)
            results['ignore_tags'] = valid_tags_list

        # 返回处理后的结果
        return results

    # 定义一个类方法，用于返回类的字符串表示
    def __repr__(self):
        # 获取类名作为字符串表示
        repr_str = self.__class__.__name__
        return repr_str
class RandomRotatePolyInstances:
    def __init__(self,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        # 初始化随机旋转类的实例
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        # 旋转多边形的顶点
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[:, ::2], points[:, 1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[:, ::2], points[:, 1::2] = _x, _y
        return points

    def cal_canvas_size(self, ori_size, degree):
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size
    # 从 0 到 max_angle 之间生成一个随机角度
    def sample_angle(self, max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    # 旋转图像并将其放置在指定的画布大小中
    def rotate_img(self, img, angle, canvas_size):
        # 获取图像的高度和宽度
        h, w = img.shape[:2]
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # 调整旋转矩阵以适应画布大小
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        # 如果需要用固定颜色填充边界
        if self.pad_with_fixed_color:
            # 使用最近邻插值方法旋转图像
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            # 创建一个与图像相同大小的全零矩阵作为蒙版
            mask = np.zeros_like(img)
            # 随机选择一个区域并对其进行裁剪和缩放
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            img_cut = cv2.resize(img_cut, (canvas_size[1], canvas_size[0]))

            # 使用蒙版旋转空白图像
            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            # 使用固定颜色填充边界旋转图像
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            # 将裁剪的图像叠加到旋转后的图像上
            target_img = target_img + img_cut * mask

        return target_img
    # 定义一个方法，用于对输入的结果进行旋转操作
    def __call__(self, results):
        # 如果生成的随机数小于旋转比例，则执行旋转操作
        if np.random.random_sample() < self.rotate_ratio:
            # 获取结果中的图像和多边形信息
            image = results['image']
            polygons = results['polys']
            # 获取图像的高度和宽度
            h, w = image.shape[:2]

            # 随机生成旋转角度
            angle = self.sample_angle(self.max_angle)
            # 计算旋转后的画布大小
            canvas_size = self.cal_canvas_size((h, w), angle)
            # 计算中心点的偏移量
            center_shift = (int((canvas_size[1] - w) / 2), int(
                (canvas_size[0] - h) / 2))
            # 对图像进行旋转操作
            image = self.rotate_img(image, angle, canvas_size)
            results['image'] = image
            # 旋转多边形
            rotated_masks = []
            for mask in polygons:
                rotated_mask = self.rotate((w / 2, h / 2), mask, angle,
                                           center_shift)
                rotated_masks.append(rotated_mask)
            results['polys'] = np.array(rotated_masks)

        # 返回处理后的结果
        return results

    # 定义一个方法，用于返回对象的字符串表示
    def __repr__(self):
        # 获取对象的类名作为字符串表示
        repr_str = self.__class__.__name__
        return repr_str
class SquareResizePad:
    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        # 断言目标大小为整数
        assert isinstance(target_size, int)
        # 断言填充比例为浮点数
        assert isinstance(pad_ratio, float)
        # 断言是否使用固定颜色填充为布尔值
        assert isinstance(pad_with_fixed_color, bool)
        # 断言填充颜色为元组
        assert isinstance(pad_value, tuple)

        # 初始化类的属性
        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        # 获取图像的高度、宽度和通道数
        h, w, _ = img.shape
        # 如果保持比例
        if keep_ratio:
            # 计算调整后的高度和宽度
            t_h = self.target_size if h >= w else int(h * self.target_size / w)
            t_w = self.target_size if h <= w else int(w * self.target_size / h)
        else:
            # 不保持比例时，高度和宽度都为目标大小
            t_h = t_w = self.target_size
        # 调整图像大小
        img = cv2.resize(img, (t_w, t_h))
        # 返回调整后的图像和调整后的高度、宽度
        return img, (t_h, t_w)
    # 对输入的图像进行填充，使其变成正方形
    def square_pad(self, img):
        # 获取图像的高度和宽度
        h, w = img.shape[:2]
        # 如果高度等于宽度，则无需填充，直接返回原图像和偏移量(0, 0)
        if h == w:
            return img, (0, 0)
        # 计算填充后的大小为高度和宽度的最大值
        pad_size = max(h, w)
        # 如果选择使用固定颜色填充
        if self.pad_with_fixed_color:
            # 创建一个全为指定值的三通道图像
            expand_img = np.ones((pad_size, pad_size, 3), dtype=np.uint8)
            expand_img[:] = self.pad_value
        else:
            # 随机选择裁剪区域并进行缩放填充
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            expand_img = cv2.resize(img_cut, (pad_size, pad_size))
        # 根据高度和宽度的大小关系确定图像的放置位置
        if h > w:
            y0, x0 = 0, (h - w) // 2
        else:
            y0, x0 = (w - h) // 2, 0
        # 将原图像放置在填充后的图像中
        expand_img[y0:y0 + h, x0:x0 + w] = img
        offset = (x0, y0)

        return expand_img, offset

    # 对点集进行填充，根据偏移量进行调整
    def square_pad_mask(self, points, offset):
        x0, y0 = offset
        pad_points = points.copy()
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    # 数据增强函数，对图像和多边形进行填充处理
    def __call__(self, results):
        image = results['image']
        polygons = results['polys']
        h, w = image.shape[:2]

        # 根据概率选择是否进行填充
        if np.random.random_sample() < self.pad_ratio:
            # 对图像进行缩放处理
            image, out_size = self.resize_img(image, keep_ratio=True)
            # 对图像进行填充处理
            image, offset = self.square_pad(image)
        else:
            # 对图像进行缩放处理
            image, out_size = self.resize_img(image, keep_ratio=False)
            offset = (0, 0)
        results['image'] = image
        try:
            # 根据缩放比例和偏移量对多边形进行调整
            polygons[:, :, 0::2] = polygons[:, :, 0::2] * out_size[
                1] / w + offset[0]
            polygons[:, :, 1::2] = polygons[:, :, 1::2] * out_size[
                0] / h + offset[1]
        except:
            pass
        results['polys'] = polygons

        return results

    # 返回类的字符串表示
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
```