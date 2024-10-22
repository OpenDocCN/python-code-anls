# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\warplayer.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 设置计算设备为 CUDA（如果可用），否则为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化一个空字典，用于存储反向扭曲的网格
backwarp_tenGrid = {}

# 定义一个函数，用于根据输入和光流进行扭曲
def warp(tenInput, tenFlow):
    # 将光流的设备和尺寸转换为字符串，作为字典的键
    k = (str(tenFlow.device), str(tenFlow.size()))
    # 如果该键不在字典中
    if k not in backwarp_tenGrid:
        # 创建水平网格，从 -1 到 1 线性分布，尺寸与光流的宽度相同
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)  # 生成水平坐标
            .view(1, 1, 1, tenFlow.shape[3])  # 重塑为适合的形状
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)  # 扩展到输入的批次大小
        )
        # 创建垂直网格，从 -1 到 1 线性分布，尺寸与光流的高度相同
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)  # 生成垂直坐标
            .view(1, 1, tenFlow.shape[2], 1)  # 重塑为适合的形状
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])  # 扩展到输入的批次大小
        )
        # 将水平和垂直网格合并并存储到字典中
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    # 将光流进行归一化处理，以适应输入尺寸
    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),  # 归一化宽度
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),  # 归一化高度
        ],
        1,  # 在通道维度上连接
    )

    # 将网格与光流相加，并调整维度顺序以适应 grid_sample 函数
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    # 使用 grid_sample 函数进行图像扭曲，返回扭曲后的图像
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True
    )
```