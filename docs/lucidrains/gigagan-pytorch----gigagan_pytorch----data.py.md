# `.\lucidrains\gigagan-pytorch\gigagan_pytorch\data.py`

```
# 导入必要的库
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms as T

from beartype.door import is_bearable
from beartype.typing import Tuple

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 将图像转换为指定格式的函数
def convert_image_to_fn(img_type, image):
    if image.mode == img_type:
        return image

    return image.convert(img_type)

# 自定义数据集拼接函数
# 使数据集可以返回字符串并将其拼接成 List[str]
def collate_tensors_or_str(data):
    is_one_data = not isinstance(data[0], tuple)

    if is_one_data:
        data = torch.stack(data)
        return (data,)

    outputs = []
    for datum in zip(*data):
        if is_bearable(datum, Tuple[str, ...]):
            output = list(datum)
        else:
            output = torch.stack(datum)

        outputs.append(output)

    return tuple(outputs)

# 数据集类

# 图像数据集类
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # 获取文件夹中指定扩展名的所有文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 断言确保文件路径数量大于0
        assert len(self.paths) > 0, 'your folder contains no images'
        # 断言确保文件路径数量大于100
        assert len(self.paths) > 100, 'you need at least 100 images, 10k for research paper, millions for miraculous results (try Laion-5B)'

        # 创建转换函数
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 图像转换操作序列
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    # 获取数据加载器
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, shuffle = True, drop_last = True, **kwargs)

    # 返回数据集长度
    def __len__(self):
        return len(self.paths)

    # 获取数据���中的数据
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# 文本图像数据集类
class TextImageDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    # 获取数据加载器
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

# 模拟文本图像数据集类
class MockTextImageDataset(TextImageDataset):
    def __init__(
        self,
        image_size,
        length = int(1e5),
        channels = 3
    ):
        self.image_size = image_size
        self.channels = channels
        self.length = length

    # 获取数据加载器
    def get_dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, collate_fn = collate_tensors_or_str, **kwargs)

    # 返回数据集长度
    def __len__(self):
        return self.length

    # 获取数据集中的数据
    def __getitem__(self, index):
        mock_image = torch.randn(self.channels, self.image_size, self.image_size)
        return mock_image, 'mock text'
```