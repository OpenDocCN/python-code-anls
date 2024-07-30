# `.\comic-translate\modules\ocr\pororo\utils\image_util.py`

```py
import cv2
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt


def plt_imshow(title='image', img=None, figsize=(8, 5)):
    # 创建一个指定大小的 Matplotlib 图像窗口
    plt.figure(figsize=figsize)

    # 如果输入的是字符串，则读取图像文件
    if type(img) is str:
        img = cv2.imread(img)

    # 如果输入的是列表，则显示多张图像
    if type(img) == list:
        # 如果标题也是列表，则使用对应的标题
        if type(title) == list:
            titles = title
        else:
            # 如果标题不是列表，则用相同标题复制多份
            titles = []
            for i in range(len(img)):
                titles.append(title)

        # 遍历图像列表，将每张图像显示在单独的子图中
        for i in range(len(img)):
            # 如果图像是灰度图，则转换为RGB格式
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            # 显示图像
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        # 显示整个图像窗口
        plt.show()
    else:
        # 如果图像不是列表，则显示单张图像
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 显示图像
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    # 如果输入的图像是 ndarray 类型，则转换为 PIL 图像格式
    if type(image) == np.ndarray:
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_converted)

    # 根据不同的操作系统选择字体
    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font = 'malgun.ttf'

    # 使用指定字体和大小创建 ImageFont 对象
    image_font = ImageFont.truetype(font, font_size)

    # 使用默认字体创建 ImageFont 对象（注：此行可能有误，应该使用上面的 image_font 变量）
    font = ImageFont.load_default()

    # 创建 ImageDraw 对象，用于在图像上绘制文本
    draw = ImageDraw.Draw(image)

    # 在指定位置绘制文本
    draw.text((x, y), text, font=image_font, fill=color)

    # 将 PIL 图像格式转换为 numpy 数组
    numpy_image = np.array(image)

    # 将 RGB 格式的图像转换为 BGR 格式（OpenCV 的默认格式）
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # 返回处理后的图像
    return opencv_image
```