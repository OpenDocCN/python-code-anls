# `.\PaddleOCR\ppocr\utils\e2e_utils\visual.py`

```py
# 导入所需的库
import numpy as np
import cv2
import time

# 将图像调整为网络所需的最大步长的倍数
def resize_image(im, max_side_len=512):
    """
    resize image to a size multiple of max_stride which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    # 获取图像的高度、宽度和通道数
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # 根据高度和宽度的比例计算调整比例
    if resize_h > resize_w:
        ratio = float(max_side_len) / resize_h
    else:
        ratio = float(max_side_len) / resize_w

    # 根据比例调整高度和宽度
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    # 将调整后的高度和宽度调整为最大步长的倍数
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    # 调整图像大小
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def resize_image_min(im, max_side_len=512):
    """
    resize image to a size multiple of max_stride which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    # 获取图像的高度、宽度和通道数
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # 根据高度和宽度的比例计算调整比例
    if resize_h < resize_w:
        ratio = float(max_side_len) / resize_h
    else:
        ratio = float(max_side_len) / resize_w

    # 根据比例调整高度和宽度
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    max_stride = 128
    # 将调整后的高度和宽度调整为最大步长的倍数
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    # 将 resize_w 调整为最接近且不小于原始值 resize_w 的 max_stride 的倍数
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    # 调整图像大小为 resize_w x resize_h
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    # 计算高度缩放比例
    ratio_h = resize_h / float(h)
    # 计算宽度缩放比例
    ratio_w = resize_w / float(w)
    # 返回调整大小后的图像和高度、宽度缩放比例
    return im, (ratio_h, ratio_w)
# 调整图像大小，使其最长边不超过指定长度
def resize_image_for_totaltext(im, max_side_len=512):
    # 获取图像的高度、宽度和通道数
    h, w, _ = im.shape

    # 初始化调整后的宽度和高度
    resize_w = w
    resize_h = h
    ratio = 1.25
    # 如果高度乘以比例大于最大边长，则重新计算比例
    if h * ratio > max_side_len:
        ratio = float(max_side_len) / resize_h

    # 根据比例调整高度和宽度
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    # 设置最大步长
    max_stride = 128
    # 根据最大步长调整高度和宽度
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    # 调整图像大小
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    # 计算高度和宽度的比例
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)


# 将垂直点对转换为顺时针的多边形点
def point_pair2poly(point_pair_list):
    # 计算每个点对的长度
    pair_length_list = []
    for point_pair in point_pair_list:
        pair_length = np.linalg.norm(point_pair[0] - point_pair[1])
        pair_length_list.append(pair_length)
    pair_length_list = np.array(pair_length_list)
    pair_info = (pair_length_list.max(), pair_length_list.min(),
                 pair_length_list.mean())

    # 计算点的数量
    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    # 将点对转换为点列表
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2), pair_info


# 沿着宽度收缩四边形
def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    # 生成沿着宽度收缩后的四边形
    ratio_pair = np.array(
        [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])


# 沿着宽度扩展多边形
def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    # 获取多边形的点数
    point_num = poly.shape[0]
    left_quad = np.array(
        [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    # 计算左侧四边形的缩小比例
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                 (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    # 根据计算出的比例，沿着宽度缩小左侧四边形
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    # 创建右侧四边形，包括四个点
    right_quad = np.array(
        [
            poly[point_num // 2 - 2], poly[point_num // 2 - 1],
            poly[point_num // 2], poly[point_num // 2 + 1]
        ],
        dtype=np.float32)
    # 计算右侧四边形的扩展比例
    right_ratio = 1.0 + \
                  shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                  (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    # 根据计算出的比例，沿着宽度扩展右侧四边形
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    # 更新多边形的第一个点和最后一个点为左侧四边形的第一个点和最后一个点
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    # 更新多边形的中间两个点为右侧四边形的第二个点和第三个点
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    # 返回更新后的多边形
    return poly
# 计算向量 x 的 L2 范数
def norm2(x, axis=None):
    # 如果指定了轴，则计算沿指定轴的 L2 范数
    if axis:
        return np.sqrt(np.sum(x**2, axis=axis)
    # 否则计算整个向量的 L2 范数
    return np.sqrt(np.sum(x**2))

# 计算两个向量的余弦相似度
def cos(p1, p2):
    # 计算两个向量的点积，并除以它们的 L2 范数的乘积
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))
```