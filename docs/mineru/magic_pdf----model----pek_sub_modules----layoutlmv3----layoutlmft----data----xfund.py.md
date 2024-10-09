# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\data\xfund.py`

```
# 导入操作系统相关模块
import os
# 导入 JSON 处理模块
import json

# 导入 PyTorch 和相关模块
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

# 从本地模块导入图像处理工具
from .image_utils import Compose, RandomResizedCropAndInterpolationWithTwoPic

# 定义标签到 ID 的映射字典
XFund_label2ids = {
    "O":0,
    'B-HEADER':1,
    'I-HEADER':2,
    'B-QUESTION':3,
    'I-QUESTION':4,
    'B-ANSWER':5,
    'I-ANSWER':6,
}

# 定义 xfund_dataset 类，继承自 Dataset
class xfund_dataset(Dataset):
    # 归一化边界框
    def box_norm(self, box, width, height):
        # 定义裁剪函数，限制数值在指定范围内
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        # 解包边界框坐标
        x0, y0, x1, y1 = box
        # 将边界框坐标归一化到 [0, 1000] 范围
        x0 = clip(0, int((x0 / width) * 1000), 1000)
        y0 = clip(0, int((y0 / height) * 1000), 1000)
        x1 = clip(0, int((x1 / width) * 1000), 1000)
        y1 = clip(0, int((y1 / height) * 1000), 1000)
        # 确保右下角坐标大于左上角
        assert x1 >= x0
        assert y1 >= y0
        # 返回归一化后的边界框
        return [x0, y0, x1, y1]

    # 获取段落 ID
    def get_segment_ids(self, bboxs):
        segment_ids = []  # 初始化段落 ID 列表
        for i in range(len(bboxs)):  # 遍历每个边界框
            if i == 0:  # 第一个边界框的段落 ID 为 0
                segment_ids.append(0)
            else:
                # 如果当前边界框与前一个相同，保持 ID 不变
                if bboxs[i - 1] == bboxs[i]:
                    segment_ids.append(segment_ids[-1])
                else:
                    # 否则，段落 ID 增加 1
                    segment_ids.append(segment_ids[-1] + 1)
        return segment_ids  # 返回段落 ID 列表

    # 获取位置 ID
    def get_position_ids(self, segment_ids):
        position_ids = []  # 初始化位置 ID 列表
        for i in range(len(segment_ids)):  # 遍历段落 ID
            if i == 0:  # 第一个位置 ID 为 2
                position_ids.append(2)
            else:
                # 如果当前段落 ID 与前一个相同，位置 ID 增加 1
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    # 否则，重置位置 ID 为 2
                    position_ids.append(2)
        return position_ids  # 返回位置 ID 列表

    # 加载数据
    def load_data(
            self,
            data_file,
    # 初始化方法
    def __init__(
            self,
            args,
            tokenizer,
            mode
    ):
        # 保存参数
        self.args = args
        # 保存模式
        self.mode = mode
        # 当前语言设置
        self.cur_la = args.language
        # 保存分词器
        self.tokenizer = tokenizer
        # 标签到 ID 映射
        self.label2ids = XFund_label2ids

        # 定义通用转换操作
        self.common_transform = Compose([
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, interpolation=args.train_interpolation,
            ),
        ])

        # 定义补丁转换操作
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(
                mean=torch.tensor((0.5, 0.5, 0.5)),  # 正规化均值
                std=torch.tensor((0.5, 0.5, 0.5)))  # 正规化标准差
        ])

        # 加载数据文件并解析为 JSON
        data_file = json.load(
            open(os.path.join(args.data_dir, "{}.{}.json".format(self.cur_la, 'train' if mode == 'train' else 'val')),
                 'r'))

        # 通过加载数据文件构建特征
        self.feature = self.load_data(data_file)

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.feature['input_ids'])  # 返回输入 ID 的数量
    # 定义获取特定索引项的特殊方法
        def __getitem__(self, index):
            # 从特征字典中获取指定索引的输入 ID
            input_ids = self.feature["input_ids"][index]
    
            # 生成注意力掩码，这里暂时注释掉的代码是从特征字典获取的
            # attention_mask = self.feature["attention_mask"][index]
            # 生成与输入 ID 等长的注意力掩码，所有值为 1
            attention_mask = [1] * len(input_ids)
            # 从特征字典中获取指定索引的标签
            labels = self.feature["labels"][index]
            # 从特征字典中获取指定索引的边界框信息
            bbox = self.feature["bbox"][index]
            # 从特征字典中获取指定索引的段 ID
            segment_ids = self.feature['segment_ids'][index]
            # 从特征字典中获取指定索引的位置 ID
            position_ids = self.feature['position_ids'][index]
    
            # 根据图像路径加载图像
            img = pil_loader(self.feature['image_path'][index])
            # 对图像进行常规变换，未使用增强
            for_patches, _ = self.common_transform(img, augmentation=False)
            # 对变换后的图像进行补丁变换
            patch = self.patch_transform(for_patches)
    
            # 断言确保输入 ID、注意力掩码、标签、边界框、段 ID 长度一致
            assert len(input_ids) == len(attention_mask) == len(labels) == len(bbox) == len(segment_ids)
    
            # 将所有提取的特征整理成字典格式
            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "bbox": bbox,
                "segment_ids": segment_ids,
                "position_ids": position_ids,
                "images": patch,
            }
            # 返回整理好的特征字典
            return res
# 定义一个函数，接收文件路径并返回图像对象
def pil_loader(path: str) -> Image.Image:
    # 以二进制模式打开指定路径的文件，避免资源警告
    with open(path, 'rb') as f:
        # 使用 Pillow 库打开文件对象，加载图像
        img = Image.open(f)
        # 将图像转换为 RGB 格式并返回
        return img.convert('RGB')
```