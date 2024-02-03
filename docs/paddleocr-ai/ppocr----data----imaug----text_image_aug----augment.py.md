# `.\PaddleOCR\ppocr\data\imaug\text_image_aug\augment.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/augment.py

import numpy as np
# 导入 WarpMLS 模块
from .warp_mls import WarpMLS

# 定义函数 tia_distort，接受参数 src 和 segment，默认值为 4
def tia_distort(src, segment=4):
    # 获取输入图像的高度和宽度
    img_h, img_w = src.shape[:2]

    # 将图像宽度分成 segment 段
    cut = img_w // segment
    # 阈值为 cut 的三分之一
    thresh = cut // 3

    # 初始化源点和目标点列表
    src_pts = list()
    dst_pts = list()

    # 添加图像四个角的点到源点列表
    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    # 随机生成目标点列表的四个点
    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append(
        [np.random.randint(thresh), img_h - np.random.randint(thresh)])

    # 阈值的一半
    half_thresh = thresh * 0.5

    # 对于每个切割段，生成源点和目标点
    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            np.random.randint(thresh) - half_thresh
        ])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            img_h + np.random.randint(thresh) - half_thresh
        ])

    # 使用 WarpMLS 类对图像进行变换
    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    # 返回变量 dst 的值
    return dst
# 对输入的图像进行拉伸变换，将图像分割成指定段数，默认为4段
def tia_stretch(src, segment=4):
    # 获取输入图像的高度和宽度
    img_h, img_w = src.shape[:2]

    # 计算每段的宽度
    cut = img_w // segment
    # 设置阈值为宽度的4/5
    thresh = cut * 4 // 5

    # 初始化源点和目标点列表
    src_pts = list()
    dst_pts = list()

    # 添加图像的四个角点到源点列表
    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    # 添加图像的四个角点到目标点列表
    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    # 计算阈值的一半
    half_thresh = thresh * 0.5

    # 对每一段进行随机移动
    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    # 使用 WarpMLS 类进行坐标变换
    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    # 生成变换后的图像
    dst = trans.generate()

    return dst


# 对输入的图像进行透视变换
def tia_perspective(src):
    # 获取输入图像的高度和宽度
    img_h, img_w = src.shape[:2]

    # 设置阈值为高度的一半
    thresh = img_h // 2

    # 初始化源点和目标点列表
    src_pts = list()
    dst_pts = list()

    # 添加图像的四个角点到源点列表
    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    # 随机生成目标点的y坐标
    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    # 使用 WarpMLS 类进行坐标变换
    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    # 生成变换后的图像
    dst = trans.generate()

    return dst
```