# `so-vits-svc\vdecoder\nsf_hifigan\utils.py`

```
# 导入所需的模块
import glob
import os

import matplotlib
import matplotlib.pylab as plt
import torch
from torch.nn.utils import weight_norm

# 设置 matplotlib 使用的后端为 "Agg"
matplotlib.use("Agg")

# 定义绘制频谱图的函数
def plot_spectrogram(spectrogram):
    # 创建一个图形和一个轴对象，设置图形大小为 (10, 2)
    fig, ax = plt.subplots(figsize=(10, 2))
    # 在轴上绘制频谱图，并设置参数
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    # 在轴上添加颜色条
    plt.colorbar(im, ax=ax)

    # 绘制图形
    fig.canvas.draw()
    # 关闭图形
    plt.close()

    # 返回绘制的图形对象
    return fig

# 初始化模型权重的函数
def init_weights(m, mean=0.0, std=0.01):
    # 获取模型类名
    classname = m.__class__.__name__
    # 如果模型类名中包含 "Conv"，则对权重进行正态分布初始化
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# 对模型应用权重归一化的函数
def apply_weight_norm(m):
    # 获取模型类名
    classname = m.__class__.__name__
    # 如果模型类名中包含 "Conv"，则对权重应用归一化
    if classname.find("Conv") != -1:
        weight_norm(m)

# 计算填充大小的函数
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# 加载检查点的函数
def load_checkpoint(filepath, device):
    # 断言文件路径存在
    assert os.path.isfile(filepath)
    # 打印加载的文件路径
    print("Loading '{}'".format(filepath))
    # 使用指定设备加载检查点文件
    checkpoint_dict = torch.load(filepath, map_location=device)
    # 打印加载完成信息
    print("Complete.")
    # 返回加载的检查点字典
    return checkpoint_dict

# 保存检查点的函数
def save_checkpoint(filepath, obj):
    # 打印保存检查点的文件路径
    print("Saving checkpoint to {}".format(filepath))
    # 保存检查点对象到指定文件路径
    torch.save(obj, filepath)
    # 打印保存完成信息
    print("Complete.")

# 删除旧检查点的函数
def del_old_checkpoints(cp_dir, prefix, n_models=2):
    # 构建匹配模式
    pattern = os.path.join(cp_dir, prefix + '????????')
    # 获取匹配模式下的检查点路径列表
    cp_list = glob.glob(pattern)
    # 对检查点路径列表进行排序
    cp_list = sorted(cp_list)
    # 如果检查点数量超过指定数量
    if len(cp_list) > n_models:
        # 遍历并删除旧的检查点文件
        for cp in cp_list[:-n_models]:
            # 清空文件内容
            open(cp, 'w').close()
            # 删除文件
            os.unlink(cp)

# 扫描检查点的函数
def scan_checkpoint(cp_dir, prefix):
    # 构建匹配模式
    pattern = os.path.join(cp_dir, prefix + '????????')
    # 获取匹配模式下的检查点路径列表
    cp_list = glob.glob(pattern)
    # 如果没有找到检查点文件，则返回 None
    if len(cp_list) == 0:
        return None
    # 返回最新的检查点文件路径
    return sorted(cp_list)[-1]
```