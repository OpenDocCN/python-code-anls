# `.\lucidrains\voicebox-pytorch\voicebox_pytorch\data.py`

```
# 导入必要的模块
from pathlib import Path
from functools import wraps

# 从 einops 模块中导入 rearrange 函数
from einops import rearrange

# 从 beartype 模块中导入 beartype 函数和 is_bearable 函数，以及 Optional、Tuple 和 Union 类型
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union

# 导入 torch 模块
import torch
# 从 torch.nn.utils.rnn 模块中导入 pad_sequence 函数
from torch.nn.utils.rnn import pad_sequence
# 从 torch.utils.data 模块中导入 Dataset 和 DataLoader 类
from torch.utils.data import Dataset, DataLoader

# 导入 torchaudio 模块
import torchaudio

# utilities

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 将值转换为元组的函数
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# dataset functions

# 定义 AudioDataset 类，继承自 Dataset 类
class AudioDataset(Dataset):
    # 初始化函数
    @beartype
    def __init__(
        self,
        folder,
        audio_extension = ".flac"
    ):
        super().__init__()
        # 将文件夹路径转换为 Path 对象
        path = Path(folder)
        # 断言文件夹存在
        assert path.exists(), 'folder does not exist'

        self.audio_extension = audio_extension

        # 获取文件夹下所有指定扩展名的文件列表
        files = list(path.glob(f'**/*{audio_extension}'))
        # 断言找到了文件
        assert len(files) > 0, 'no files found'

        self.files = files

    # 返回数据集的长度
    def __len__(self):
        return len(self.files)

    # 获取指定索引处的数据
    def __getitem__(self, idx):
        file = self.files[idx]

        # 加载音频文件
        wave, _ = torchaudio.load(file)
        # 重新排列音频数据的维度
        wave = rearrange(wave, '1 ... -> ...')

        return wave

# dataloader functions

# 定义装饰器函数，用于处理单个或多个张量的数据
def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

# 裁剪数据到最短长度的函数
@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

# 填充数据到最长长度的函数
@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

# 获取 DataLoader 对象的函数
def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
```