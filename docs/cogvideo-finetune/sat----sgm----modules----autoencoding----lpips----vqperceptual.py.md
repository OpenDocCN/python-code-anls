# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\vqperceptual.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F


# 定义一个函数，计算对抗网络中判别器的铰链损失
def hinge_d_loss(logits_real, logits_fake):
    # 计算真实样本的损失，使用 ReLU 激活函数
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    # 计算假样本的损失，使用 ReLU 激活函数
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    # 计算总的判别器损失，取真实和假样本损失的平均
    d_loss = 0.5 * (loss_real + loss_fake)
    # 返回判别器的铰链损失
    return d_loss


# 定义一个函数，计算判别器的经典损失
def vanilla_d_loss(logits_real, logits_fake):
    # 计算判别器损失，使用 Softplus 函数处理真实和假样本
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    # 返回判别器的经典损失
    return d_loss
```