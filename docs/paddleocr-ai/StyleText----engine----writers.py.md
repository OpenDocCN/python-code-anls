# `.\PaddleOCR\StyleText\engine\writers.py`

```
# 版权声明和许可信息
# 该代码版权归 PaddlePaddle 作者所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入所需的库
import os
import cv2
import glob

# 从自定义的 logging 模块中导入 get_logger 函数
from utils.logging import get_logger

# 定义 SimpleWriter 类
class SimpleWriter(object):
    # 初始化函数，接受配置和标签作为参数
    def __init__(self, config, tag):
        # 获取日志记录器
        self.logger = get_logger()
        # 设置输出目录
        self.output_dir = config["Global"]["output_dir"]
        # 计数器初始化为 0
        self.counter = 0
        # 标签字典初始化为空
        self.label_dict = {}
        # 保存标签
        self.tag = tag
        # 标签文件索引初始化为 0
        self.label_file_index = 0

    # 保存图像函数，接受图像和文本输入标签作为参数
    def save_image(self, image, text_input_label):
        # 图像保存路径
        image_home = os.path.join(self.output_dir, "images", self.tag)
        # 如果路径不存在，则创建路径
        if not os.path.exists(image_home):
            os.makedirs(image_home)

        # 图像文件路径
        image_path = os.path.join(image_home, "{}.png".format(self.counter))
        # 保存图像
        cv2.imwrite(image_path, image)
        self.logger.info("generate image: {}".format(image_path))

        # 图像名称
        image_name = os.path.join(self.tag, "{}.png".format(self.counter))
        # 将图像名称和文本输入标签存入标签字典
        self.label_dict[image_name] = text_input_label

        # 计数器加一
        self.counter += 1
        # 每保存 100 张图像就保存一次标签
        if not self.counter % 100:
            self.save_label()
    # 保存标签信息到文件
    def save_label(self):
        # 初始化标签内容为空字符串
        label_raw = ""
        # 拼接标签文件夹路径
        label_home = os.path.join(self.output_dir, "label")
        # 如果标签文件夹不存在，则创建
        if not os.path.exists(label_home):
            os.mkdir(label_home)
        # 遍历标签字典中的图片路径和标签信息，拼接到标签内容中
        for image_path in self.label_dict:
            label = self.label_dict[image_path]
            label_raw += "{}\t{}\n".format(image_path, label)
        # 拼接标签文件路径
        label_file_path = os.path.join(label_home,
                                       "{}_label.txt".format(self.tag))
        # 将标签内容写入文件
        with open(label_file_path, "w") as f:
            f.write(label_raw)
        # 标签文件索引自增
        self.label_file_index += 1

    # 合并所有标签文件
    def merge_label(self):
        # 初始化标签内容为空字符串
        label_raw = ""
        # 匹配所有标签文件路径
        label_file_regex = os.path.join(self.output_dir, "label",
                                        "*_label.txt")
        # 获取所有匹配的标签文件路径列表
        label_file_list = glob.glob(label_file_regex)
        # 遍历所有标签文件，读取内容并拼接到标签内容中
        for label_file_i in label_file_list:
            with open(label_file_i, "r") as f:
                label_raw += f.read()
        # 拼接合并后的标签文件路径
        label_file_path = os.path.join(self.output_dir, "label.txt")
        # 将合并后的标签内容写入文件
        with open(label_file_path, "w") as f:
            f.write(label_raw)
```