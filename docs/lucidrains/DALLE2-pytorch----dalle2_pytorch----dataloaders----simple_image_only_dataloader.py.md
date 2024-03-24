# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\dataloaders\simple_image_only_dataloader.py`

```py
# 导入所需的库
from pathlib import Path
import torch
from torch.utils import data
from torchvision import transforms, utils
from PIL import Image

# 定义一个循环生成器函数，用于无限循环遍历数据集
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 定义数据集类
class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # 获取指定文件夹下所有指定扩展名的文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 定义数据预处理的操作
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    # 返回数据集的长度
    def __len__(self):
        return len(self.paths)

    # 根据索引获取数据
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# 获取图像数据的数据加载器
def get_images_dataloader(
    folder,
    *,
    batch_size,
    image_size,
    shuffle = True,
    cycle_dl = True,
    pin_memory = True
):
    # 创建数据集对象
    ds = Dataset(folder, image_size)
    # 创建数据加载器对象
    dl = data.DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

    # 如果需要循环遍历数据加载器，则将数据加载器设置为循环生成器
    if cycle_dl:
        dl = cycle(dl)
    return dl
```