# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\_dataset.py`

```py
import os  # 导入标准库 os，用于处理操作系统相关功能
from natsort import natsorted  # 导入 natsorted 函数，用于自然排序
from PIL import Image  # 导入 PIL 库中的 Image 模块，用于图像处理
from torch.utils.data import Dataset  # 导入 PyTorch 中的 Dataset 类，用于自定义数据集


class RawDataset(Dataset):
    """
    自定义数据集类 RawDataset，继承自 Dataset 类
    """

    def __init__(self, root, imgW, imgH):
        """
        初始化方法，用于初始化数据集对象
        
        参数:
        - root: 数据集根目录路径
        - imgW: 图像的宽度
        - imgH: 图像的高度
        """
        self.imgW = imgW  # 设置图像宽度
        self.imgH = imgH  # 设置图像高度
        self.image_path_list = []  # 初始化图像路径列表为空列表
        for dirpath, _, filenames in os.walk(root):
            # 遍历 root 目录及其子目录下的所有文件和文件夹
            for name in filenames:
                _, ext = os.path.splitext(name)  # 获取文件名和扩展名
                ext = ext.lower()  # 将扩展名转换为小写
                if ext in (".jpg", ".jpeg", ".png"):  # 如果扩展名是常见图像格式之一
                    self.image_path_list.append(os.path.join(dirpath, name))  # 将图像路径添加到列表中

        self.image_path_list = natsorted(self.image_path_list)  # 对图像路径列表进行自然排序
        self.nSamples = len(self.image_path_list)  # 记录数据集中样本的数量

    def __len__(self):
        """
        返回数据集中样本的数量
        
        返回:
        - 数据集中样本的数量
        """
        return self.nSamples

    def __getitem__(self, index):
        """
        获取数据集中指定索引处的样本
        
        参数:
        - index: 样本的索引
        
        返回:
        - img: PIL 图像对象，表示索引处的图像
        - self.image_path_list[index]: 索引处的图像文件路径
        """
        try:
            img = Image.open(self.image_path_list[index]).convert("L")
            # 尝试打开并转换为灰度图像

        except IOError:
            print(f"Corrupted image for {index}")
            img = Image.new("L", (self.imgW, self.imgH))
            # 处理图像文件损坏的情况，创建一个空白的灰度图像

        return img, self.image_path_list[index]  # 返回图像对象和图像文件路径
```