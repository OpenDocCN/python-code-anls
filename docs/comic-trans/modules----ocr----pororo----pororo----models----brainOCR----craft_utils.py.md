# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\craft_utils.py`

```py
# 导入数学库
import math

# 导入OpenCV和NumPy库
import cv2
import numpy as np

# 根据逆变换矩阵Minv，对点pt进行逆变换，返回变换后的坐标
def warp_coord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])

# 核心函数：获取检测框（detection boxes）
def get_det_boxes_core(textmap, linkmap, text_threshold, link_threshold,
                       low_text):
    # 准备数据
    # 复制链接图像和文本图像
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    # 获取文本图像的高度和宽度
    img_h, img_w = textmap.shape

    # 标签方法
    # 根据低文本阈值将文本图像二值化为0或1
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    # 根据链接阈值将链接图像二值化为0或1
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    # 合并文本和链接图像的分数，范围限制在0到1之间
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # 使用连通组件分析获取文本区域的数量、标签、统计信息和质心
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4)

    # 初始化检测框和映射器
    det = []
    mapper = []

    # 遍历每个文本区域标签
    for k in range(1, nLabels):
        # 尺寸过滤
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # 阈值过滤
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # 创建分割图
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        # 去除链接区域
        segmap[np.logical_and(link_score == 1,
                              text_score == 0)] = 0
        # 计算区域的边界框
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # 边界检查
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        # 获取结构元素
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1 + niter, 1 + niter),
        )
        # 对分割图进行膨胀操作
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # 创建最小外接矩形
        np_contours = (np.roll(np.array(np.where(segmap != 0)), 1,
                               axis=0).transpose().reshape(-1, 2))
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # 对齐菱形区域
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # 将框按顺时针顺序排列
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        # 将框添加到检测框列表中
        det.append(box)
        # 将标签添加到映射器中
        mapper.append(k)

    # 返回检测框、标签和映射器
    return det, labels, mapper
# 定义函数 `get_poly_core`，用于生成多边形区域的核心逻辑
def get_poly_core(boxes, labels, mapper, linkmap):
    # 配置参数
    num_cp = 5  # 控制多边形点数的参数
    max_len_ratio = 0.7  # 控制多边形最大长度比例的参数
    expand_ratio = 1.45  # 控制多边形扩展比例的参数
    max_r = 2.0  # 多边形最大半径
    step_r = 0.2  # 多边形半径的步进值

    # 初始化多边形列表
    polys = []
    return polys


# 定义函数 `get_det_boxes`，获取检测到的文本框及其多边形区域
def get_det_boxes(
    textmap,
    linkmap,
    text_threshold,
    link_threshold,
    low_text,
    poly=False,
):
    # 调用核心函数获取检测框及相关信息
    boxes, labels, mapper = get_det_boxes_core(
        textmap,
        linkmap,
        text_threshold,
        link_threshold,
        low_text,
    )

    # 如果需要生成多边形区域
    if poly:
        polys = get_poly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)  # 否则初始化多边形列表为 None

    return boxes, polys


# 定义函数 `adjust_result_coordinates`，调整结果的坐标信息
def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    # 如果传入的多边形列表不为空
    if len(polys) > 0:
        polys = np.array(polys)  # 转换为 NumPy 数组
        # 遍历多边形列表中的每个多边形
        for k in range(len(polys)):
            if polys[k] is not None:  # 如果多边形不为 None
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)  # 根据比例因子调整多边形的坐标

    return polys  # 返回调整后的多边形列表
```