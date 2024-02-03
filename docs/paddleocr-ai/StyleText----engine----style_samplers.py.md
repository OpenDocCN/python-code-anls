# `.\PaddleOCR\StyleText\engine\style_samplers.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 有关特定语言的权限和限制，请参阅许可证
import numpy as np
import random
import cv2

# 定义数据集采样器类
class DatasetSampler(object):
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 设置图像路径
        self.image_home = config["StyleSampler"]["image_home"]
        # 获取标签文件路径
        label_file = config["StyleSampler"]["label_file"]
        # 是否包含标签信息
        self.dataset_with_label = config["StyleSampler"]["with_label"]
        # 图像高度
        self.height = config["Global"]["image_height"]
        # 初始化索引
        self.index = 0
        # 读取标签文件内容
        with open(label_file, "r") as f:
            label_raw = f.read()
            # 将路径和标签信息以换行符分割为列表
            self.path_label_list = label_raw.split("\n")[:-1]
        # 断言路径和标签列表长度大于0
        assert len(self.path_label_list) > 0
        # 随机打乱路径和标签列表顺序
        random.shuffle(self.path_label_list)
    # 定义一个方法用于获取样本数据
    def sample(self):
        # 如果当前索引超出路径标签列表的长度，则重新打乱列表并重置索引
        if self.index >= len(self.path_label_list):
            random.shuffle(self.path_label_list)
            self.index = 0
        # 如果数据集包含标签信息
        if self.dataset_with_label:
            # 获取当前索引对应的路径和标签
            path_label = self.path_label_list[self.index]
            rel_image_path, label = path_label.split('\t')
        else:
            # 如果数据集不包含标签信息，则只获取路径
            rel_image_path = self.path_label_list[self.index]
            label = None
        # 拼接完整的图像路径
        img_path = "{}/{}".format(self.image_home, rel_image_path)
        # 读取图像数据
        image = cv2.imread(img_path)
        # 获取原始图像高度
        origin_height = image.shape[0]
        # 计算缩放比例
        ratio = self.height / origin_height
        # 根据比例计算新的宽度和高度
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)
        # 缩放图像
        image = cv2.resize(image, (width, height))

        # 更新索引
        self.index += 1
        # 如果存在标签信息，则返回图像和标签
        if label:
            return {"image": image, "label": label}
        # 如果不存在标签信息，则只返回图像
        else:
            return {"image": image}
# 复制图像以填充指定宽度
def duplicate_image(image, width):
    # 获取图像的宽度
    image_width = image.shape[1]
    # 计算需要复制的次数
    dup_num = width // image_width + 1
    # 沿着指定轴复制图像
    image = np.tile(image, reps=[1, dup_num, 1])
    # 裁剪图像到指定宽度
    cropped_image = image[:, :width, :]
    # 返回裁剪后的图像
    return cropped_image
```