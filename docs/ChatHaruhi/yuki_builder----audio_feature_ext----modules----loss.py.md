# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\modules\loss.py`

```py
import math  # 导入数学库，用于数学计算

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数库，提供神经网络常用的功能

class AdditiveAngularMargin(nn.Module):
    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        """实现加性角度边界（Additive Angular Margin，AAM），参考以下论文:
           'Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'
           (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): 边界因子，默认为0.0.
            scale (float, optional): 缩放因子，默认为1.0.
            easy_margin (bool, optional): 是否启用简单边界标志，默认为False.
        """
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin  # 设置边界因子
        self.scale = scale  # 设置缩放因子
        self.easy_margin = easy_margin  # 设置是否启用简单边界标志

        # 计算预先计算的角度边界相关值
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        cosine = outputs.float()  # 将输出转换为浮点数
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # 计算余弦的对应正弦值
        phi = cosine * self.cos_m - sine * self.sin_m  # 计算加性角度边界函数
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)  # 根据简单边界标志进行调整
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # 根据角度边界进行调整
        outputs = (targets * phi) + ((1.0 - targets) * cosine)  # 计算最终的输出
        return self.scale * outputs  # 返回经过缩放的输出结果


class AAMLoss(nn.Module):
    def __init__(self, margin=0.2, scale=30, easy_margin=False):
        super(AAMLoss, self).__init__()
        self.loss_fn = AdditiveAngularMargin(margin=margin, scale=scale, easy_margin=easy_margin)  # 初始化加性角度边界损失函数
        self.criterion = torch.nn.KLDivLoss(reduction="sum")  # 使用KL散度损失作为标准

    def forward(self, outputs, targets):
        targets = F.one_hot(targets, outputs.shape[1]).float()  # 将目标标签转换为独热编码格式
        predictions = self.loss_fn(outputs, targets)  # 计算加性角度边界损失
        predictions = F.log_softmax(predictions, dim=1)  # 对预测结果进行log_softmax处理
        loss = self.criterion(predictions, targets) / targets.sum()  # 计算最终的损失值
        return loss  # 返回损失值作为前向传播的输出
```