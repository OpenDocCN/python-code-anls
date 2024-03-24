# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\data.py`

```
# 导入必要的模块
from pathlib import Path
from functools import partial, wraps

# 导入 beartype 模块及相关类型
from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable

# 导入 torchaudio 模块及相关函数
import torchaudio
from torchaudio.functional import resample

# 导入 torch 模块及相关函数
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# 导入自定义工具函数
from audiolm_pytorch.utils import curtail_to_multiple

# 导入 einops 模块中的函数
from einops import rearrange, reduce

# 定义一些辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 将值转换为元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 判断列表中的元素是否唯一
def is_unique(arr):
    return len(set(arr)) == len(arr

# 定义数据集类
class SoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        target_sample_hz: Union[int, Tuple[int, ...]],  # 目标采样率必须指定，或者是一个包含多个目标采样率的元组
        exts = ['flac', 'wav', 'mp3', 'webm'],
        max_length: Optional[int] = None,               # 如果有多个目标采样率，最大长度将应用于最高的采样率
        seq_len_multiple_of: Optional[Union[int, Tuple[Optional[int], ...]]] = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), f'folder "{str(path)}" does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.max_length = max_length
        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        # 如果有多个目标采样率，首先将其重采样为最高的采样率，然后应用最大长度，最后再重采样为其他采样率

        self.max_target_sample_hz = max(self.target_sample_hz)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # 如果音频有多个通道，转换为单声道
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        # 首先将数据重采样为最大目标频率

        data = resample(data, sample_hz, self.max_target_sample_hz)
        sample_hz = self.max_target_sample_hz

        # 根据最大长度截断或填充音频

        max_length = self.max_length
        audio_length = data.size(1)

        if exists(max_length):
            if audio_length > max_length:
                max_start = audio_length - max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + max_length]
            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

        data = rearrange(data, '1 ... -> ...')

        # 如果目标采样率不是元组中的 None，则重采样

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        data_tuple = tuple(resample(d, sample_hz, target_sample_hz) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # 逐个处理不同频率下的数据以符合多个长度的要求

        for data, seq_len_multiple_of in zip(data_tuple, self.seq_len_multiple_of):
            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # 从列表转换为元组

        output = tuple(output)

        # 如果只有一个目标重采样频率，则返回一个音频

        if num_outputs == 1:
            return output[0]

        return output

# 数据加载函数

# 定义一个装饰器函数，用于处理单个或多个张量的填充
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

# 对最短的数据进行填充
@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

# 对最长的数据进行填充
@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

# 获取数据加载器
def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
```