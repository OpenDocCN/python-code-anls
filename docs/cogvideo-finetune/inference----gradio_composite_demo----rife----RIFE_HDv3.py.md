# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\RIFE_HDv3.py`

```py
# 导入 PyTorch 及其子模块和其他必要的库
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from .warplayer import warp  # 导入自定义模块 warp
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行模块
from .IFNet_HDv3 import *  # 导入 IFNet_HDv3 模块中的所有内容
import torch.nn.functional as F  # 导入功能性激活函数
from .loss import *  # 导入自定义损失函数模块

# 检查是否有可用的 GPU，并选择相应的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    # 初始化模型，设置本地进程的排名（默认值为 -1）
    def __init__(self, local_rank=-1):
        # 实例化 IFNet 网络
        self.flownet = IFNet()
        # 将模型移动到指定的设备
        self.device()
        # 使用 AdamW 优化器，设置学习率和权重衰减
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        # 实例化 EPE 损失对象
        self.epe = EPE()
        # self.vgg = VGGPerceptualLoss().to(device)  # （注释掉的）实例化 VGG 感知损失对象并移动到设备
        # 实例化 SOBEL 边缘检测对象
        self.sobel = SOBEL()
        # 如果 local_rank 不为 -1，则使用分布式数据并行
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    # 设置模型为训练模式
    def train(self):
        self.flownet.train()

    # 设置模型为评估模式
    def eval(self):
        self.flownet.eval()

    # 将模型移动到指定的设备
    def device(self):
        self.flownet.to(device)

    # 从指定路径加载模型参数
    def load_model(self, path, rank=0):
        # 内部函数用于转换参数名称
        def convert(param):
            # 如果 rank 为 -1，移除参数名称中的 "module." 前缀
            if rank == -1:
                return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}
            else:
                return param

        # 如果 rank 小于等于 0，则加载模型
        if rank <= 0:
            # 如果有可用的 GPU，加载到 GPU
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path))))
            # 否则加载到 CPU
            else:
                self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path), map_location="cpu")))

    # 保存模型参数到指定路径
    def save_model(self, path, rank=0):
        # 如果 rank 为 0，保存模型状态字典
        if rank == 0:
            torch.save(self.flownet.state_dict(), "{}/flownet.pkl".format(path))

    # 进行推理，合并两幅图像并返回结果
    def inference(self, img0, img1, scale=1.0):
        # 将 img0 和 img1 在通道维度上进行拼接
        imgs = torch.cat((img0, img1), 1)
        # 根据输入缩放比例生成缩放列表
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        # 调用 flownet 进行推理，获取光流、掩膜和合并结果
        flow, mask, merged = self.flownet(imgs, scale_list)
        # 返回合并结果的第三个输出
        return merged[2]

    # 更新模型参数，计算损失并进行优化
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        # 更新优化器中的学习率
        for param_group in self.optimG.param_groups:
            param_group["lr"] = learning_rate
        # 从输入图像中分离出 img0 和 img1
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        # 根据训练状态设置模型模式
        if training:
            self.train()
        else:
            self.eval()
        # 定义缩放比例
        scale = [4, 2, 1]
        # 调用 flownet 进行前向计算
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        # 计算 L1 损失
        loss_l1 = (merged[2] - gt).abs().mean()
        # 计算平滑损失
        loss_smooth = self.sobel(flow[2], flow[2] * 0).mean()
        # loss_vgg = self.vgg(merged[2], gt)  # （注释掉的）计算 VGG 感知损失
        # 如果处于训练状态，进行反向传播和优化步骤
        if training:
            self.optimG.zero_grad()  # 清空梯度
            loss_G = loss_cons + loss_smooth * 0.1  # 计算总损失
            loss_G.backward()  # 反向传播
            self.optimG.step()  # 更新参数
        else:
            flow_teacher = flow[2]  # 获取教师模型的光流
        # 返回合并结果和各项损失
        return merged[2], {
            "mask": mask,
            "flow": flow[2][:, :2],
            "loss_l1": loss_l1,
            "loss_cons": loss_cons,
            "loss_smooth": loss_smooth,
        }
```