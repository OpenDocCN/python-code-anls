# `.\cogview3-finetune\sat\sgm\modules\autoencoding\lpips\vqperceptual.py`

```py
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的功能性模块
import torch.nn.functional as F


# 定义对抗训练中的判别器损失函数（hinge 损失）
def hinge_d_loss(logits_real, logits_fake):
    # 计算真实样本的损失，使用 ReLU 激活函数
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    # 计算假样本的损失，使用 ReLU 激活函数
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    # 计算总的判别器损失，取真实和假样本损失的平均值
    d_loss = 0.5 * (loss_real + loss_fake)
    # 返回判别器损失
    return d_loss


# 定义对抗训练中的另一种判别器损失函数（vanilla 损失）
def vanilla_d_loss(logits_real, logits_fake):
    # 计算总的判别器损失，使用 softplus 函数
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))  # 计算真实样本的 softplus 损失
        + torch.mean(torch.nn.functional.softplus(logits_fake))   # 计算假样本的 softplus 损失
    )
    # 返回判别器损失
    return d_loss
```