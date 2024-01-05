# `d:/src/tocomm/Bert-VITS2\monotonic_align\__init__.py`

```
from numpy import zeros, int32, float32
from torch import from_numpy

from .core import maximum_path_jit


def maximum_path(neg_cent, mask):
    # 获取输入张量的设备类型
    device = neg_cent.device
    # 获取输入张量的数据类型
    dtype = neg_cent.dtype
    # 将输入张量转换为numpy数组，并转换为float32类型
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    # 创建一个形状与neg_cent相同的全零数组，数据类型为int32
    path = zeros(neg_cent.shape, dtype=int32)

    # 计算mask张量每行的和，并转换为numpy数组，数据类型为int32
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    # 计算mask张量每列的和，并转换为numpy数组，数据类型为int32
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    # 调用maximum_path_jit函数，计算最大路径，并将结果保存在path数组中
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    # 将path数组转换为torch张量，并将其移动到指定设备上，数据类型为dtype
    return from_numpy(path).to(device=device, dtype=dtype)
```

注释解释了每个语句的作用，包括获取设备类型和数据类型、转换张量为numpy数组、创建全零数组、计算张量的和、调用maximum_path_jit函数计算最大路径、将结果转换为torch张量并移动到指定设备上。
```