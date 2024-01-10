# `so-vits-svc\vdecoder\hifigan\utils.py`

```
# 导入glob和os模块
import glob
import os

# 导入matplotlib.pylab模块并重命名为plt
# import matplotlib.use("Agg")
import matplotlib.pylab as plt
# 导入torch模块
import torch
# 从torch.nn.utils模块中导入weight_norm函数
from torch.nn.utils import weight_norm

# 定义函数plot_spectrogram，用于绘制频谱图
def plot_spectrogram(spectrogram):
    # 创建一个新的图形和子图
    fig, ax = plt.subplots(figsize=(10, 2))
    # 在子图上绘制频谱图
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    # 在子图上添加颜色条
    plt.colorbar(im, ax=ax)

    # 绘制图形
    fig.canvas.draw()
    # 关闭图形
    plt.close()

    # 返回绘制的图形
    return fig

# 定义函数init_weights，用于初始化模型权重
def init_weights(m, mean=0.0, std=0.01):
    # 获取模型类名
    classname = m.__class__.__name__
    # 如果模型类名中包含"Conv"，则对权重进行正态分布初始化
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# 定义函数apply_weight_norm，用于对模型应用权重归一化
def apply_weight_norm(m):
    # 获取模型类名
    classname = m.__class__.__name__
    # 如果模型类名中包含"Conv"，则对模型应用权重归一化
    if classname.find("Conv") != -1:
        weight_norm(m)

# 定义函数get_padding，用于计算填充大小
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# 定义函数load_checkpoint，用于加载检查点文件
def load_checkpoint(filepath, device):
    # 断言检查文件是否存在
    assert os.path.isfile(filepath)
    # 打印加载文件的信息
    print("Loading '{}'".format(filepath))
    # 使用torch.load加载检查点文件，并指定设备
    checkpoint_dict = torch.load(filepath, map_location=device)
    # 打印加载完成信息
    print("Complete.")
    # 返回加载的检查点数据
    return checkpoint_dict

# 定义函数save_checkpoint，用于保存检查点文件
def save_checkpoint(filepath, obj):
    # 打印保存检查点文件的信息
    print("Saving checkpoint to {}".format(filepath))
    # 使用torch.save保存检查点文件
    torch.save(obj, filepath)
    # 打印保存完成信息
    print("Complete.")

# 定义函数del_old_checkpoints，用于删除旧的检查点文件
def del_old_checkpoints(cp_dir, prefix, n_models=2):
    # 构建检查点文件路径的匹配模式
    pattern = os.path.join(cp_dir, prefix + '????????')
    # 获取匹配模式下的检查点文件列表
    cp_list = glob.glob(pattern) # get checkpoint paths
    # 对检查点文件列表进行排序
    cp_list = sorted(cp_list)# sort by iter
    # 如果检查点文件数量超过指定数量
    if len(cp_list) > n_models: # if more than n_models models are found
        # 遍历并删除旧的检查点文件
        for cp in cp_list[:-n_models]:# delete the oldest models other than lastest n_models
            # 清空文件内容
            open(cp, 'w').close()# empty file contents
            # 删除文件（在Colab中移动到回收站）
            os.unlink(cp)# delete file (move to trash when using Colab)

# 定义函数scan_checkpoint，用于扫描最新的检查点文件
def scan_checkpoint(cp_dir, prefix):
    # 构建检查点文件路径的匹配模式
    pattern = os.path.join(cp_dir, prefix + '????????')
    # 获取匹配模式下的检查点文件列表
    cp_list = glob.glob(pattern)
    # 如果没有找到检查点文件，则返回None
    if len(cp_list) == 0:
        return None
    # 对检查点文件列表进行排序，并返回最新的检查点文件路径
    return sorted(cp_list)[-1]
```