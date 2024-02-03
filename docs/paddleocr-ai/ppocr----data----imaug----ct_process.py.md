# `.\PaddleOCR\ppocr\data\imaug\ct_process.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。

# 导入所需的库
import os
import cv2
import random
import pyclipper
import paddle

import numpy as np
from ppocr.utils.utility import check_install

import scipy.io as scio

from PIL import Image
import paddle.vision.transforms as transforms

# 定义 RandomScale 类
class RandomScale():
    def __init__(self, short_size=640, **kwargs):
        self.short_size = short_size

    # 将图像按比例缩放并保持对齐
    def scale_aligned(self, img, scale):
        oh, ow = img.shape[0:2]
        h = int(oh * scale + 0.5)
        w = int(ow * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        factor_h = h / oh
        factor_w = w / ow
        return img, factor_h, factor_w

    # 随机缩放图像
    def __call__(self, data):
        img = data['image']

        h, w = img.shape[0:2]
        random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
        scale = (np.random.choice(random_scale) * self.short_size) / min(h, w)
        img, factor_h, factor_w = self.scale_aligned(img, scale)

        data['scale_factor'] = (factor_w, factor_h)
        data['image'] = img
        return data

# 定义 MakeShrink 类
class MakeShrink():
    def __init__(self, kernel_scale=0.7, **kwargs):
        self.kernel_scale = kernel_scale

    # 计算两点之间的距离
    def dist(self, a, b):
        return np.linalg.norm((a - b), ord=2, axis=0)
    # 计算多边形的周长
    def perimeter(self, bbox):
        # 初始化周长为0
        peri = 0.0
        # 遍历多边形的每个顶点
        for i in range(bbox.shape[0]):
            # 计算当前顶点到下一个顶点的距离，并累加到周长上
            peri += self.dist(bbox[i], bbox[(i + 1) % bbox.shape[0])
        # 返回计算得到的周长
        return peri

    # 缩小多边形的边界框
    def shrink(self, bboxes, rate, max_shr=20):
        # 检查并导入所需的模块
        check_install('Polygon', 'Polygon3')
        import Polygon as plg
        # 计算缩小率的平方
        rate = rate * rate
        # 初始化存储缩小后边界框的列表
        shrinked_bboxes = []
        # 遍历每个边界框
        for bbox in bboxes:
            # 计算边界框的面积
            area = plg.Polygon(bbox).area()
            # 计算边界框的周长
            peri = self.perimeter(bbox)

            try:
                # 创建 PyclipperOffset 对象
                pco = pyclipper.PyclipperOffset()
                # 添加边界框的路径到 PyclipperOffset 对象中
                pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                # 计算缩小的偏移量
                offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
                # 执行缩小操作
                shrinked_bbox = pco.Execute(-offset)
                # 如果缩小后的边界框为空，则保持原边界框
                if len(shrinked_bbox) == 0:
                    shrinked_bboxes.append(bbox)
                    continue

                # 将缩小后的边界框转换为 NumPy 数组
                shrinked_bbox = np.array(shrinked_bbox[0])
                # 如果缩小后的边界框顶点数量小于等于2，则保持原边界框
                if shrinked_bbox.shape[0] <= 2:
                    shrinked_bboxes.append(bbox)
                    continue

                # 将缩小后的边界框添加到结果列表中
                shrinked_bboxes.append(shrinked_bbox)
            except Exception as e:
                # 出现异常时，保持原边界框
                shrinked_bboxes.append(bbox)

        # 返回所有缩小后的边界框
        return shrinked_bboxes
class GroupRandomHorizontalFlip():
    # 初始化函数，设置水平翻转的概率
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    # 调用函数，对数据进行水平翻转操作
    def __call__(self, data):
        imgs = data['image']

        # 根据设定的概率进行水平翻转操作
        if random.random() < self.p:
            for i in range(len(imgs)):
                # 对每张图片进行水平翻转并复制
                imgs[i] = np.flip(imgs[i], axis=1).copy()
        data['image'] = imgs
        return data


class GroupRandomRotate():
    # 初始化函数
    def __init__(self, **kwargs):
        pass

    # 调用函数，对数据进行随机旋转操作
    def __call__(self, data):
        imgs = data['image']

        # 设置最大旋转角度
        max_angle = 10
        # 随机生成旋转角度
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            # 获取旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            # 进行仿射变换
            img_rotation = cv2.warpAffine(
                img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
            imgs[i] = img_rotation

        data['image'] = imgs
        return data


class GroupRandomCropPadding():
    # 初始化函数，设置目标尺寸
    def __init__(self, target_size=(640, 640), **kwargs):
        self.target_size = target_size
    # 定义一个类的调用方法，接受一个数据字典作为参数
    def __call__(self, data):
        # 从数据字典中获取图像数据
        imgs = data['image']

        # 获取第一张图像的高度和宽度
        h, w = imgs[0].shape[0:2]
        # 获取目标尺寸
        t_w, t_h = self.target_size
        # 获取填充尺寸
        p_w, p_h = self.target_size
        # 如果图像的宽度和高度与目标尺寸相同，则直接返回数据字典
        if w == t_w and h == t_h:
            return data

        # 如果目标高度大于图像高度，则将目标高度设置为图像高度
        t_h = t_h if t_h < h else h
        # 如果目标宽度大于图像宽度，则将目标宽度设置为图像宽度
        t_w = t_w if t_w < w else w

        # 如果随机数大于3.0/8.0且第二张图像中的最大值大于0
        if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
            # 确保裁剪文本区域
            tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            br[br < 0] = 0
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, h - t_h) if h - t_h > 0 else 0
            j = random.randint(0, w - t_w) if w - t_w > 0 else 0

        # 初始化新图像列表
        n_imgs = []
        # 遍历所有图像
        for idx in range(len(imgs)):
            # 如果图像是三通道的
            if len(imgs[idx].shape) == 3:
                # 获取通道数
                s3_length = int(imgs[idx].shape[-1])
                # 裁剪图像并进行填充
                img = imgs[idx][i:i + t_h, j:j + t_w, :]
                img_p = cv2.copyMakeBorder(
                    img,
                    0,
                    p_h - t_h,
                    0,
                    p_w - t_w,
                    borderType=cv2.BORDER_CONSTANT,
                    value=tuple(0 for i in range(s3_length)))
            else:
                # 裁剪灰度图像并进行填充
                img = imgs[idx][i:i + t_h, j:j + t_w]
                img_p = cv2.copyMakeBorder(
                    img,
                    0,
                    p_h - t_h,
                    0,
                    p_w - t_w,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, ))
            # 将处理后的图像添加到新图像列表中
            n_imgs.append(img_p)

        # 更新数据字典中的图像数据为处理后的新图像列表
        data['image'] = n_imgs
        # 返回更新后的数据字典
        return data
# 定义一个类 MakeCentripetalShift
class MakeCentripetalShift():
    # 初始化方法，接受关键字参数
    def __init__(self, **kwargs):
        # pass 表示什么也不做，保持方法结构完整
        pass

    # 计算 Jaccard 距离的方法
    def jaccard(self, As, Bs):
        # 获取数组 As 的行数，表示小的数组
        A = As.shape[0]  # small
        # 获取数组 Bs 的行数，表示大的数组
        B = Bs.shape[0]  # large

        # 计算两个数组之间的欧氏距离
        dis = np.sqrt(
            np.sum((As[:, np.newaxis, :].repeat(
                B, axis=1) - Bs[np.newaxis, :, :].repeat(
                    A, axis=0))**2,
                   axis=-1))

        # 找到每个小数组元素对应到大数组中距离最小的索引
        ind = np.argmin(dis, axis=-1)

        # 返回索引结果
        return ind
    # 定义一个类的调用方法，接受数据作为参数
    def __call__(self, data):
        # 从数据中获取图像相关的内容
        imgs = data['image']

        # 将图像、实例标注、训练掩码、内核实例标注、内核标注、内核内部标注、距离训练掩码分别赋值给对应变量
        img, gt_instance, training_mask, gt_kernel_instance, gt_kernel, gt_kernel_inner, training_mask_distance = \
                        imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5], imgs[6]

        # 计算实例标注中的最大值
        max_instance = np.max(gt_instance)  # num bbox

        # 创建一个全零数组，用于存储距离信息
        gt_distance = np.zeros((2, *img.shape[0:2]), dtype=np.float32)
        
        # 遍历实例标注中的每个实例
        for i in range(1, max_instance + 1):
            # 找到内核内部标注中对应实例的位置
            ind = (gt_kernel_inner == i)

            # 如果找不到对应实例，则将训练掩码和距离训练掩码置零，并继续下一个实例
            if np.sum(ind) == 0:
                training_mask[gt_instance == i] = 0
                training_mask_distance[gt_instance == i] = 0
                continue

            # 找到内核内部标注和实例标注中对应实例的位置
            kpoints = np.array(np.where(ind)).transpose(
                (1, 0))[:, ::-1].astype('float32')

            ind = (gt_instance == i) * (gt_kernel_instance == 0)
            if np.sum(ind) == 0:
                continue
            pixels = np.where(ind)

            points = np.array(pixels).transpose(
                (1, 0))[:, ::-1].astype('float32')

            # 计算偏移量
            bbox_ind = self.jaccard(points, kpoints)

            offset_gt = kpoints[bbox_ind] - points

            # 将偏移量信息存储到距离数组中
            gt_distance[:, pixels[0], pixels[1]] = offset_gt.T * 0.1

        # 将图像转换为 PIL 图像对象，并转换为 RGB 模式
        img = Image.fromarray(img)
        img = img.convert('RGB')

        # 更新数据中的图像和其他相关信息
        data["image"] = img
        data["gt_kernel"] = gt_kernel.astype("int64")
        data["training_mask"] = training_mask.astype("int64")
        data["gt_instance"] = gt_instance.astype("int64")
        data["gt_kernel_instance"] = gt_kernel_instance.astype("int64")
        data["training_mask_distance"] = training_mask_distance.astype("int64")
        data["gt_distance"] = gt_distance.astype("float32")

        # 返回更新后的数据
        return data
# 定义一个名为 ScaleAlignedShort 的类
class ScaleAlignedShort():
    # 初始化方法，设置默认的 short_size 参数
    def __init__(self, short_size=640, **kwargs):
        self.short_size = short_size

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取数据中的图像
        img = data['image']

        # 记录原始图像的形状
        org_img_shape = img.shape

        # 获取图像的高度和宽度
        h, w = img.shape[0:2]
        # 计算缩放比例
        scale = self.short_size * 1.0 / min(h, w)
        # 根据缩放比例调整图像的高度和宽度
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        # 如果调整后的高度不能被32整除，则向上取整到最接近的32的倍数
        if h % 32 != 0:
            h = h + (32 - h % 32)
        # 如果调整后的宽度不能被32整除，则向上取整到最接近的32的倍数
        if w % 32 != 0:
            w = w + (32 - w % 32)
        # 调整图像的大小
        img = cv2.resize(img, dsize=(w, h))

        # 记录调整后的图像形状
        new_img_shape = img.shape
        # 将原始图像形状和调整后的图像形状合并为一个数组
        img_shape = np.array(org_img_shape + new_img_shape)

        # 将调整后的图像形状存储到数据中
        data['shape'] = img_shape
        # 将调整后的图像存储到数据中
        data['image'] = img

        # 返回处理后的数据
        return data
```