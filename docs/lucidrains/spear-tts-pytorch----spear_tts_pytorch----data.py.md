# `.\lucidrains\spear-tts-pytorch\spear_tts_pytorch\data.py`

```py
# 导入必要的模块
from pathlib import Path
import torch
from torch.utils.data import Dataset
from beartype import beartype

# 模拟数据集类
class MockDataset(Dataset):
    # 初始化方法，接受数据集长度参数
    def __init__(self, length: int):
        self.length = length

    # 返回数据集长度
    def __len__(self):
        return self.length

    # 获取数据集中指定索引的数据
    def __getitem__(self, ind):
        return torch.randn(1024)

# 生成音频文本数据集类
class GeneratedAudioTextDataset(Dataset):
    # 初始化方法，接受文件夹路径和分隔符ID参数
    @beartype
    def __init__(
        self,
        folder: str,
        delimiter_id: int = -1
    ):
        # 将文件夹路径转换为Path对象
        self.folder = Path(folder)
        # 断言文件夹存在且是一个目录
        assert self.folder.exists() and self.folder.is_dir()
        # 获取文件夹中所有以'.pt'结尾的文件路径列表
        self.paths = list(self.folder.glob('*.pt'))
        # 设置分隔符ID
        self.delimiter_id = delimiter_id

    # 返回数据集的长度
    def __len__(self):
        return len(self.paths)

    # 获取数据集中指定索引的数据
    def __getitem__(self, ind):
        # 获取指定索引的文件路径
        path = self.paths[ind]
        # 加载文件中的数据为张量
        tensor = torch.load(str(path))

        # 创建一个布尔张量，标记分隔符ID的位置
        delimiter_mask = tensor == self.delimiter_id
        # 断言至少存在一个分隔符，否则抛出异常
        assert delimiter_mask.any(), f'delimeter (<audio> <delimeter> <text>) not found'

        # 找到第一个分隔符的位置
        ind = (delimiter_mask.cumsum(dim=-1) == 0).sum().item()

        # 返回分隔符之前的部分和分隔符之后的部分作为数据
        return tensor[:ind], tensor[(ind + 1):]
```