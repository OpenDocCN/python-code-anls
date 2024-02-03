# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\data_loader\modules\iaa_augment.py`

```
# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 18:06
# @Author  : zhoujun
# 导入所需的库
import numpy as np
import imgaug
import imgaug.augmenters as iaa

# 定义一个AugmenterBuilder类
class AugmenterBuilder(object):
    def __init__(self):
        pass

    # 根据参数构建数据增强序列
    def build(self, args, root=True):
        # 如果参数为空或长度为0，则返回None
        if args is None or len(args) == 0:
            return None
        # 如果参数是列表
        elif isinstance(args, list):
            # 如果是根节点
            if root:
                # 递归构建数据增强序列
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(
                    iaa,
                    args[0])(* [self.to_tuple_if_list(a) for a in args[1:]])
        # 如果参数是字典
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{
                k: self.to_tuple_if_list(v)
                for k, v in args['args'].items()
            })
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    # 如果对象是列表，则转换为元组
    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj

# 定义一个IaaAugment类
class IaaAugment():
    def __init__(self, augmenter_args):
        self.augmenter_args = augmenter_args
        # 根据参数构建数据增强器
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    # 对数据进行增强处理
    def __call__(self, data):
        image = data['img']
        shape = image.shape

        if self.augmenter:
            # 将增强器转换为确定性的
            aug = self.augmenter.to_deterministic()
            data['img'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    # 对标注进行可能的增强处理
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['text_polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['text_polys'] = np.array(line_polys)
        return data
    # 定义一个方法，用于根据给定的augmentation对象aug，图像形状img_shape和多边形poly来可能增强多边形
    def may_augment_poly(self, aug, img_shape, poly):
        # 将多边形的每个点转换为Keypoint对象并存储在列表中
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        # 使用aug对象对KeypointsOnImage对象进行增强，然后获取增强后的关键点列表
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        # 将增强后的关键点列表转换为多边形的坐标列表
        poly = [(p.x, p.y) for p in keypoints]
        # 返回增强后的多边形坐标列表
        return poly
```