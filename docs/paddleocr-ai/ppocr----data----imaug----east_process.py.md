# `.\PaddleOCR\ppocr\data\imaug\east_process.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的具体语言
"""
# 引用自 https://github.com/songdejia/EAST/blob/master/data_utils.py
import math
import cv2
import numpy as np
import json
import sys
import os

# 定义类 EASTProcessTrain
class EASTProcessTrain(object):
    # 初始化方法
    def __init__(self,
                 image_shape=[512, 512],  # 图像形状，默认为 [512, 512]
                 background_ratio=0.125,  # 背景比例，默认为 0.125
                 min_crop_side_ratio=0.1,  # 最小裁剪边比例，默认为 0.1
                 min_text_size=10,  # 最小文本大小，默认为 10
                 **kwargs):  # 其他参数
        self.input_size = image_shape[1]  # 输入大小为图像形状的第二个值
        self.random_scale = np.array([0.5, 1, 2.0, 3.0])  # 随机缩放比例数组
        self.background_ratio = background_ratio  # 背景比例
        self.min_crop_side_ratio = min_crop_side_ratio  # 最小裁剪边比例
        self.min_text_size = min_text_size  # 最小文本大小
    # 对输入图像进行预处理，包括缩放、归一化等操作
    def preprocess(self, im):
        # 获取输入图像的大小
        input_size = self.input_size
        # 获取图像的形状信息
        im_shape = im.shape
        # 获取图像最小的尺寸
        im_size_min = np.min(im_shape[0:2])
        # 获取图像最大的尺寸
        im_size_max = np.max(im_shape[0:2])
        # 计算缩放比例
        im_scale = float(input_size) / float(im_size_max)
        # 根据缩放比例对图像进行缩放
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale)
        # 图像均值
        img_mean = [0.485, 0.456, 0.406]
        # 图像标准差
        img_std = [0.229, 0.224, 0.225]
        # 将图像通道顺序从 BGR 转换为 RGB，并转换为浮点型
        # im = im[:, :, ::-1].astype(np.float32)
        # 图像像素值归一化到 [0, 1] 区间
        im = im / 255
        # 对图像进行标准化处理
        im -= img_mean
        im /= img_std
        # 获取处理后图像的新尺寸
        new_h, new_w, _ = im.shape
        # 创建一个指定尺寸的全零数组
        im_padded = np.zeros((input_size, input_size, 3), dtype=np.float32)
        # 将处理后的图像放入全零数组中
        im_padded[:new_h, :new_w, :] = im
        # 调整图像维度顺序
        im_padded = im_padded.transpose((2, 0, 1))
        # 在第一维度上增加一个维度
        im_padded = im_padded[np.newaxis, :]
        # 返回预处理后的图像和缩放比例
        return im_padded, im_scale
    def rotate_im_poly(self, im, text_polys):
        """
        旋转图像90/180/270度
        """
        # 获取图像的宽度和高度
        im_w, im_h = im.shape[1], im.shape[0]
        # 复制输入图像
        dst_im = im.copy()
        # 存储旋转后的多边形
        dst_polys = []
        # 生成随机角度
        rand_degree_ratio = np.random.rand()
        rand_degree_cnt = 1
        # 根据随机角度确定旋转次数
        if 0.333 < rand_degree_ratio < 0.666:
            rand_degree_cnt = 2
        elif rand_degree_ratio > 0.666:
            rand_degree_cnt = 3
        # 根据旋转次数进行旋转
        for i in range(rand_degree_cnt):
            dst_im = np.rot90(dst_im)
        # 计算旋转角度
        rot_degree = -90 * rand_degree_cnt
        rot_angle = rot_degree * math.pi / 180.0
        # 获取多边形的数量
        n_poly = text_polys.shape[0]
        # 计算中心点坐标
        cx, cy = 0.5 * im_w, 0.5 * im_h
        ncx, ncy = 0.5 * dst_im.shape[1], 0.5 * dst_im.shape[0]
        # 遍历每个多边形
        for i in range(n_poly):
            wordBB = text_polys[i]
            poly = []
            # 计算旋转后的多边形顶点坐标
            for j in range(4):
                sx, sy = wordBB[j][0], wordBB[j][1]
                dx = math.cos(rot_angle) * (sx - cx) - math.sin(rot_angle) * (sy - cy) + ncx
                dy = math.sin(rot_angle) * (sx - cx) + math.cos(rot_angle) * (sy - cy) + ncy
                poly.append([dx, dy])
            dst_polys.append(poly)
        # 转换为numpy数组并返回
        dst_polys = np.array(dst_polys, dtype=np.float32)
        return dst_im, dst_polys

    def polygon_area(self, poly):
        """
        计算多边形的面积
        :param poly: 多边形的顶点坐标
        :return: 多边形的面积
        """
        # 计算多边形每条边的面积并求和
        edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
                (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
                (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
                (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])]
        # 返回多边形的面积
        return np.sum(edge) / 2.
    def check_and_validate_polys(self, polys, tags, img_height, img_width):
        """
        检查并验证多边形，确保文本多边形方向一致，并过滤一些无效多边形
        :param polys: 多边形数组
        :param tags: 标签数组
        :param img_height: 图像高度
        :param img_width: 图像宽度
        :return: 经过验证的多边形数组和标签数组
        """
        h, w = img_height, img_width
        如果多边形数组为空，则直接返回
        if polys.shape[0] == 0:
            return polys
        将多边形的 x 坐标限制在 [0, w-1] 范围内
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        将多边形的 y 坐标限制在 [0, h-1] 范围内
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []
        遍历多边形数组和标签数组
        for poly, tag in zip(polys, tags):
            计算多边形的面积
            p_area = self.polygon_area(poly)
            如果多边形面积小于1，则认为是无效多边形，跳过
            if abs(p_area) < 1:
                continue
            如果多边形面积大于0，表示多边形方向错误
            if p_area > 0:
                如果标签为 False，则将标签设为 True，表示忽略反向的情况
                if not tag:
                    tag = True  #反向的情况应该被忽略
                将多边形的顶点顺序反转
                poly = poly[(0, 3, 2, 1), :]
            将验证后的多边形和标签添加到对应的数组中
            validated_polys.append(poly)
            validated_tags.append(tag)
        返回经过验证的多边形数组和标签数组
        return np.array(validated_polys), np.array(validated_tags)

    def draw_img_polys(self, img, polys):
        如果图像的维度为4，则压缩维度
        if len(img.shape) == 4:
            img = np.squeeze(img, axis=0)
        如果图像的通道数为3，则转置通道
        if img.shape[0] == 3:
            img = img.transpose((1, 2, 0))
            对图像进行通道均值归一化
            img[:, :, 2] += 123.68
            img[:, :, 1] += 116.78
            img[:, :, 0] += 103.94
        将图像保存为临时文件
        cv2.imwrite("tmp.jpg", img)
        重新读取保存的图像
        img = cv2.imread("tmp.jpg")
        遍历多边形数组，绘制多边形
        for box in polys:
            将多边形顶点转换为整数类型，并重塑形状
            box = box.astype(np.int32).reshape((-1, 1, 2))
            绘制多边形
            cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
        导入随机模块，生成随机数
        ino = random.randint(0, 100)
        将绘制好的图像保存为带随机数后缀的文件
        cv2.imwrite("tmp_%d.jpg" % ino, img)
        返回
        return
    # 裁剪包含背景信息的图像和文本区域
    def crop_background_infor(self, im, text_polys, text_tags):
        # 裁剪图像和文本区域，包括背景信息
        im, text_polys, text_tags = self.crop_area(
            im, text_polys, text_tags, crop_background=True)

        # 如果文本区域数量大于0，则返回None
        if len(text_polys) > 0:
            return None
        # 填充和调整图像大小
        input_size = self.input_size
        im, ratio = self.preprocess(im)
        score_map = np.zeros((input_size, input_size), dtype=np.float32)
        geo_map = np.zeros((input_size, input_size, 9), dtype=np.float32)
        training_mask = np.ones((input_size, input_size), dtype=np.float32)
        return im, score_map, geo_map, training_mask

    # 裁剪包含前景信息的图像和文本区域
    def crop_foreground_infor(self, im, text_polys, text_tags):
        # 裁剪图像和文本区域，不包括背景信息
        im, text_polys, text_tags = self.crop_area(
            im, text_polys, text_tags, crop_background=False)

        # 如果文本区域的行数为0，则返回None
        if text_polys.shape[0] == 0:
            return None
        # 继续处理所有忽略的情况
        if np.sum((text_tags * 1.0)) >= text_tags.size:
            return None
        # 填充和调整图像大小
        input_size = self.input_size
        im, ratio = self.preprocess(im)
        text_polys[:, :, 0] *= ratio
        text_polys[:, :, 1] *= ratio
        _, _, new_h, new_w = im.shape
        # 生成四边形得分图、几何图和训练掩码
        score_map, geo_map, training_mask = self.generate_quad(
            (new_h, new_w), text_polys, text_tags)
        return im, score_map, geo_map, training_mask
    # 定义一个类的调用方法，接受一个数据字典作为参数
    def __call__(self, data):
        # 从数据字典中获取图像数据
        im = data['image']
        # 从数据字典中获取文本多边形坐标
        text_polys = data['polys']
        # 从数据字典中获取文本标签
        text_tags = data['ignore_tags']
        # 如果图像为空，则返回空
        if im is None:
            return None
        # 如果文本多边形的行数为0，则返回空
        if text_polys.shape[0] == 0:
            return None

        # 添加旋转情况
        if np.random.rand() < 0.5:
            # 对图像和文本多边形进行旋转
            im, text_polys = self.rotate_im_poly(im, text_polys)
        # 获取图像的高度、宽度和通道数
        h, w, _ = im.shape
        # 检查和验证文本多边形的有效性
        text_polys, text_tags = self.check_and_validate_polys(text_polys,
                                                              text_tags, h, w)
        # 如果文本多边形的行数为0，则返回空
        if text_polys.shape[0] == 0:
            return None

        # 随机缩放图像
        rd_scale = np.random.choice(self.random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale
        # 如果随机数小于背景比例，则裁剪背景信息
        if np.random.rand() < self.background_ratio:
            outs = self.crop_background_infor(im, text_polys, text_tags)
        else:
            outs = self.crop_foreground_infor(im, text_polys, text_tags)

        # 如果结果为空，则返回空
        if outs is None:
            return None
        # 从结果中获取图像、得分图、几何图和训练掩码
        im, score_map, geo_map, training_mask = outs
        # 将得分图转换为浮点数类型，并进行维度变换
        score_map = score_map[np.newaxis, ::4, ::4].astype(np.float32)
        geo_map = np.swapaxes(geo_map, 1, 2)
        geo_map = np.swapaxes(geo_map, 1, 0)
        geo_map = geo_map[:, ::4, ::4].astype(np.float32)
        training_mask = training_mask[np.newaxis, ::4, ::4]
        training_mask = training_mask.astype(np.float32)

        # 更新数据字典中的图像、得分图、几何图和训练掩码
        data['image'] = im[0]
        data['score_map'] = score_map
        data['geo_map'] = geo_map
        data['training_mask'] = training_mask
        # 返回更新后的数据字典
        return data
```