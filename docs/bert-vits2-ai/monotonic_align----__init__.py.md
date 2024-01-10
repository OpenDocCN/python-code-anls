# `Bert-VITS2\monotonic_align\__init__.py`

```
# 从 numpy 库中导入 zeros, int32, float32 方法
# 从 torch 库中导入 from_numpy 方法
from numpy import zeros, int32, float32
from torch import from_numpy

# 从当前目录下的 core 模块中导入 maximum_path_jit 方法
from .core import maximum_path_jit

# 定义 maximum_path 函数，接受 neg_cent 和 mask 两个参数
def maximum_path(neg_cent, mask):
    # 获取 neg_cent 的设备信息
    device = neg_cent.device
    # 获取 neg_cent 的数据类型信息
    dtype = neg_cent.dtype
    # 将 neg_cent 转换为 numpy 数组，再转换为 float32 类型
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    # 创建一个与 neg_cent 形状相同的、数据类型为 int32 的全零数组
    path = zeros(neg_cent.shape, dtype=int32)

    # 计算 mask 沿第一维度的和，再将结果转换为 numpy 数组，再转换为 int32 类型
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    # 计算 mask 沿第二维度的和，再将结果转换为 numpy 数组，再转换为 int32 类型
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    # 调用 maximum_path_jit 方法，传入 path, neg_cent, t_t_max, t_s_max 四个参数
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    # 将 path 转换为 torch tensor，并指定设备和数据类型，然后返回
    return from_numpy(path).to(device=device, dtype=dtype)
```