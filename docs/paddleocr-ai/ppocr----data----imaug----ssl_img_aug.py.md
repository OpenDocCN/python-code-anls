# `.\PaddleOCR\ppocr\data\imaug\ssl_img_aug.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 导入所需的库
import math
import cv2
import numpy as np
import random
from PIL import Image

# 从当前目录下的 rec_img_aug 模块中导入 resize_norm_img 函数
from .rec_img_aug import resize_norm_img

# 定义 SSLRotateResize 类
class SSLRotateResize(object):
    # 初始化函数，接受参数 image_shape, padding, select_all, mode 和 kwargs
    def __init__(self,
                 image_shape,
                 padding=False,
                 select_all=True,
                 mode="train",
                 **kwargs):
        # 初始化对象的属性
        self.image_shape = image_shape
        self.padding = padding
        self.select_all = select_all
        self.mode = mode
    # 定义一个类的方法，用于对输入的数据进行处理
    def __call__(self, data):
        # 从输入数据中获取图像数据
        img = data["image"]

        # 将图像顺时针旋转90度，并存储在数据中
        data["image_r90"] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # 将上一步旋转后的图像再次顺时针旋转180度，并存储在数据中
        data["image_r180"] = cv2.rotate(data["image_r90"], cv2.ROTATE_90_CLOCKWISE)
        # 将上一步旋转后的图像再次顺时针旋转270度，并存储在数据中
        data["image_r270"] = cv2.rotate(data["image_r180"], cv2.ROTATE_90_CLOCKWISE)

        # 初始化一个空列表用于存储处理后的图像数据
        images = []
        # 遍历指定的键值，依次处理对应的图像数据
        for key in ["image", "image_r90", "image_r180", "image_r270"]:
            # 调用resize_norm_img函数对图像进行处理，并将处理后的图像数据添加到列表中
            images.append(
                resize_norm_img(
                    data.pop(key),
                    image_shape=self.image_shape,
                    padding=self.padding)[0])
        # 将处理后的图像数据堆叠在一起，形成一个新的数组，并存储在数据中
        data["image"] = np.stack(images, axis=0)
        # 创建一个包含0到3的数组，并存储在数据中作为标签
        data["label"] = np.array(list(range(4)))
        
        # 如果不选择全部数据
        if not self.select_all:
            # 仅保留0和180度旋转后的图像数据
            data["image"] = data["image"][0::2]
            # 标签只保留0和1
            data["label"] = data["label"][0:2]
        
        # 如果处理模式为测试模式
        if self.mode == "test":
            # 仅保留第一个图像数据
            data["image"] = data["image"][0]
            # 仅保留第一个标签
            data["label"] = data["label"][0]
        
        # 返回处理后的数据
        return data
```