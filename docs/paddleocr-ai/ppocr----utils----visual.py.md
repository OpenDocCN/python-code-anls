# `.\PaddleOCR\ppocr\utils\visual.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 绘制识别结果到图像上
def draw_ser_results(image,
                     ocr_results,
                     font_path="doc/fonts/simfang.ttf",
                     font_size=14):
    # 设置随机种子
    np.random.seed(2021)
    # 生成随机颜色
    color = (np.random.permutation(range(255)),
             np.random.permutation(range(255)),
             np.random.permutation(range(255)))
    # 构建颜色映射表
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx])
        for idx in range(1, 255)
    }
    # 判断输入的图像类型，转换为 PIL.Image 对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')
    # 复制输入图像
    img_new = image.copy()
    # 创建绘图对象
    draw = ImageDraw.Draw(img_new)

    # 加载字体文件
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    # 遍历识别结果
    for ocr_info in ocr_results:
        # 如果预测 ID 不在颜色映射表中，则跳过
        if ocr_info["pred_id"] not in color_map:
            continue
        # 获取颜色
        color = color_map[ocr_info["pred_id"]]
        # 构建要绘制的文本
        text = "{}: {}".format(ocr_info["pred"], ocr_info["transcription"])

        if "bbox" in ocr_info:
            # 使用 OCR 引擎绘制
            bbox = ocr_info["bbox"]
        else:
            # 使用 OCR 地面真值绘制
            bbox = trans_poly_to_bbox(ocr_info["points"])
        # 绘制带有文本的框
        draw_box_txt(bbox, text, draw, font, font_size, color)
    # 使用Image.blend()函数将两个图像进行混合，第一个参数为底图，第二个参数为要混合的图像，第三个参数为混合程度
    img_new = Image.blend(image, img_new, 0.7)
    # 将混合后的图像转换为NumPy数组并返回
    return np.array(img_new)
# 绘制带有文本框的文本，包括文本框、文本内容和文本样式
def draw_box_txt(bbox, text, draw, font, font_size, color):

    # 绘制OCR结果的轮廓
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # 绘制OCR结果的文本内容
    left, top, right, bottom = font.getbbox(text)
    tw, th = right - left, bottom - top
    start_y = max(0, bbox[0][1] - th)
    draw.rectangle(
        [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + th)],
        fill=(0, 0, 255))
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)


# 将多边形转换为边界框
def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


# 绘制识别结果
def draw_re_results(image,
                    result,
                    font_path="doc/fonts/simfang.ttf",
                    font_size=18):
    np.random.seed(0)
    # 如果输入是numpy数组，则转换为Image对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # 如果输入是文件路径，则打开并转换为RGB格式的Image对象
    elif isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image).convert('RGB')
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    # 使用指定字体路径和大小创建字体对象
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    color_head = (0, 0, 255)
    color_tail = (255, 0, 0)
    color_line = (0, 255, 0)
    # 遍历结果中的每个 OCR 信息头和尾
    for ocr_info_head, ocr_info_tail in result:
        # 在图像上绘制 OCR 信息头的边界框和文本
        draw_box_txt(ocr_info_head["bbox"], ocr_info_head["transcription"],
                     draw, font, font_size, color_head)
        # 在图像上绘制 OCR 信息尾的边界框和文本
        draw_box_txt(ocr_info_tail["bbox"], ocr_info_tail["transcription"],
                     draw, font, font_size, color_tail)

        # 计算 OCR 信息头中心点的坐标
        center_head = (
            (ocr_info_head['bbox'][0] + ocr_info_head['bbox'][2]) // 2,
            (ocr_info_head['bbox'][1] + ocr_info_head['bbox'][3]) // 2)
        # 计算 OCR 信息尾中心点的坐标
        center_tail = (
            (ocr_info_tail['bbox'][0] + ocr_info_tail['bbox'][2]) // 2,
            (ocr_info_tail['bbox'][1] + ocr_info_tail['bbox'][3]) // 2)

        # 在图像上绘制连接 OCR 信息头和尾中心点的直线
        draw.line([center_head, center_tail], fill=color_line, width=5)

    # 将新图像与原图像混合
    img_new = Image.blend(image, img_new, 0.5)
    # 将混合后的图像转换为 NumPy 数组并返回
    return np.array(img_new)
# 根据给定的图像路径和矩形框坐标绘制矩形框
def draw_rectangle(img_path, boxes):
    # 将矩形框坐标转换为 NumPy 数组
    boxes = np.array(boxes)
    # 读取图像文件
    img = cv2.imread(img_path)
    # 复制一份图像用于显示
    img_show = img.copy()
    # 遍历每个矩形框的坐标
    for box in boxes.astype(int):
        # 获取矩形框的四个顶点坐标
        x1, y1, x2, y2 = box
        # 在显示图像上绘制矩形框
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # 返回绘制了矩形框的图像
    return img_show
```