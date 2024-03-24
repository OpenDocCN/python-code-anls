# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\weighted_objective_gaussian_diffusion.py`

```
# 导入 torch 库
import torch
# 从 inspect 库中导入 isfunction 函数
from inspect import isfunction
# 从 torch 库中导入 nn 和 einsum 模块
from torch import nn, einsum
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 从 denoising_diffusion_pytorch.denoising_diffusion_pytorch 模块中导入 GaussianDiffusion 类

# 辅助函数

# 判断变量是否存在的函数
def exists(x):
    return x is not None

# 默认值函数，如果值存在则返回该值，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 在我的改进中
# 模型学习预测噪声和 x0，并根据时间步长学习加权和

# WeightedObjectiveGaussianDiffusion 类继承自 GaussianDiffusion 类
class WeightedObjectiveGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        model,
        *args,
        pred_noise_loss_weight = 0.1,
        pred_x_start_loss_weight = 0.1,
        **kwargs
    ):
        # 调用父类的构造函数
        super().__init__(model, *args, **kwargs)
        channels = model.channels
        # 断言模型输出维度必须是通道数的两倍加上 2
        assert model.out_dim == (channels * 2 + 2), 'dimension out (out_dim) of unet must be twice the number of channels + 2 (for the softmax weighted sum) - for channels of 3, this should be (3 * 2) + 2 = 8'
        assert not model.self_condition, 'not supported yet'
        assert not self.is_ddim_sampling, 'ddim sampling cannot be used'

        self.split_dims = (channels, channels, 2)
        self.pred_noise_loss_weight = pred_noise_loss_weight
        self.pred_x_start_loss_weight = pred_x_start_loss_weight

    # 计算均值和方差
    def p_mean_variance(self, *, x, t, clip_denoised, model_output = None):
        # 调用模型得到输出
        model_output = self.model(x, t)

        # 将模型输出拆分为预测的噪声、预测的 x_start 和权重
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim = 1)
        normalized_weights = weights.softmax(dim = 1)

        # 从预测的噪声中预测 x_start
        x_start_from_noise = self.predict_start_from_noise(x, t = t, noise = pred_noise)
        
        x_starts = torch.stack((x_start_from_noise, pred_x_start), dim = 1)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', normalized_weights, x_starts)

        if clip_denoised:
            weighted_x_start.clamp_(-1., 1.)

        # 计算模型的均值、方差和对数方差
        model_mean, model_variance, model_log_variance = self.q_posterior(weighted_x_start, x, t)

        return model_mean, model_variance, model_log_variance

    # 计算损失
    def p_losses(self, x_start, t, noise = None, clip_denoised = False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

        model_output = self.model(x_t, t)
        pred_noise, pred_x_start, weights = model_output.split(self.split_dims, dim = 1)

        # 计算预测噪声和 x_start 的损失，并乘以初始化时给定的损失权重
        noise_loss = F.mse_loss(noise, pred_noise) * self.pred_noise_loss_weight
        x_start_loss = F.mse_loss(x_start, pred_x_start) * self.pred_x_start_loss_weight

        # 从预测的噪声中计算 x_start，然后对 x_start 预测和模型预测的权重进行加权和
        x_start_from_pred_noise = self.predict_start_from_noise(x_t, t, pred_noise)
        x_start_from_pred_noise = x_start_from_pred_noise.clamp(-2., 2.)
        weighted_x_start = einsum('b j h w, b j c h w -> b c h w', weights.softmax(dim = 1), torch.stack((x_start_from_pred_noise, pred_x_start), dim = 1))

        # 主要损失为 x_start 与加权 x_start 的均方误差
        weighted_x_start_loss = F.mse_loss(x_start, weighted_x_start)
        return weighted_x_start_loss + x_start_loss + noise_loss
```