# `stable-diffusion-webui\modules\models\diffusion\ddpm_edit.py`

```
"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

# File modified by authors of InstructPix2Pix from original (https://github.com/CompVis/stable-diffusion).
# See more details in LICENSE.

# 导入所需的库
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

# 导入自定义的工具函数和模块
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

# 尝试导入VQModelInterface类，如果失败则定义一个空的类
try:
    from ldm.models.autoencoder import VQModelInterface
except Exception:
    class VQModelInterface:
        pass

# 定义条件键的映射关系
__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

# 定义一个函数，用于覆盖模型的train方法，使得训练/评估模式不再改变
def disabled_train(self, mode=True):
    return self

# 在指定设备上生成指定范围内的均匀分布随机数
def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

# 定义一个类，继承自LightningModule，实现了classic DDPM with Gaussian diffusion, in image space
class DDPM(pl.LightningModule):
    # 上下文管理器
    @contextmanager
    # 定义一个函数，用于在指定上下文中使用指数移动平均（EMA）权重
    def ema_scope(self, context=None):
        # 如果使用EMA
        if self.use_ema:
            # 将模型参数存储到EMA模型中
            self.model_ema.store(self.model.parameters())
            # 将EMA模型的权重复制到原始模型中
            self.model_ema.copy_to(self.model)
            # 如果有指定上下文，则打印切换到EMA权重的消息
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            # 返回一个空值
            yield None
        finally:
            # 如果使用EMA
            if self.use_ema:
                # 恢复原始模型的权重
                self.model_ema.restore(self.model.parameters())
                # 如果有指定上下文，则打印恢复训练权重的消息
                if context is not None:
                    print(f"{context}: Restored training weights")
    # 从检查点文件中初始化模型参数，可以选择忽略某些键值对
    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        # 如果忽略键值对为空，则初始化为空列表
        ignore_keys = ignore_keys or []

        # 加载检查点文件中的参数到内存中
        sd = torch.load(path, map_location="cpu")
        # 如果检查点文件中有 "state_dict" 键，则将其赋值给 sd
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        # 获取所有键值对的键
        keys = list(sd.keys())

        # 我们的模型在第一层中添加了额外的通道以便对输入图像进行条件化
        # 对于第一层，复制现有通道权重并将新通道权重初始化为零
        input_keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model_ema.diffusion_modelinput_blocks00weight",
        ]

        # 获取当前模型的参数字典
        self_sd = self.state_dict()
        # 遍历输入键列表
        for input_key in input_keys:
            # 如果输入键不在 sd 或者不在 self_sd 中，则跳过
            if input_key not in sd or input_key not in self_sd:
                continue

            # 获取当前模型参数中的输入权重
            input_weight = self_sd[input_key]

            # 如果输入权重的大小与 sd 中的输入权重大小不一致
            if input_weight.size() != sd[input_key].size():
                # 打印提示信息
                print(f"Manual init: {input_key}")
                # 将输入权重初始化为零
                input_weight.zero_()
                # 将 sd 中的输入权重的前四个通道复制到输入权重中
                input_weight[:, :4, :, :].copy_(sd[input_key])
                # 将当前键添加到忽略键列表中
                ignore_keys.append(input_key)

        # 遍历所有键值对的键
        for k in keys:
            # 遍历忽略键列表
            for ik in ignore_keys:
                # 如果键以忽略键开头
                if k.startswith(ik):
                    # 打印删除键的提示信息
                    print(f"Deleting key {k} from state_dict.")
                    # 从 sd 中删除该键
                    del sd[k]
        # 加载模型参数字典到当前模型中，如果只加载模型则加载到 model 中
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        # 打印恢复模型参数的信息，包括缺失和意外的键数量
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        # 如果有缺失的键，则打印缺失的键
        if missing:
            print(f"Missing Keys: {missing}")
        # 如果有意外的键，则打印意外的键
        if unexpected:
            print(f"Unexpected Keys: {unexpected}")
    # 计算给定时间步 t 的条件分布 q(x_t | x_0)
    def q_mean_variance(self, x_start, t):
        # 计算均值，使用噪声输入 x_start 乘以累积平方根 alpha 的乘积
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        # 计算方差，使用 1 减去累积 alpha 的乘积
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        # 计算对数方差，使用累积对数 (1 - alpha) 的乘积
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        # 返回均值、方差和对数方差
        return mean, variance, log_variance

    # 根据噪声和时间步 t 预测起始值
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                # 使用累积平方根 alpha 的倒数乘以 x_t
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                # 使用累积平方根 (1 - alpha) 的倒数乘以噪声
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 计算后验分布 q(x_start, x_t | t)
    def q_posterior(self, x_start, x_t, t):
        # 计算后验均值，使用两个系数乘以 x_start 和 x_t
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 计算后验方差
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        # 计算修剪后的后验对数方差
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        # 返回后验均值、后验方差和修剪后的后验对数方差
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 计算模型输出的均值和后验方差
    def p_mean_variance(self, x, t, clip_denoised: bool):
        # 获取模型输出
        model_out = self.model(x, t)
        # 根据参数化方式选择重构值
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        # 如果需要对重构值进行修剪
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # 计算模型均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        # 返回模型均值、后验方差和后验对数方差
        return model_mean, posterior_variance, posterior_log_variance
    # 使用 torch.no_grad() 装饰器，表示在该函数中不需要计算梯度
    @torch.no_grad()
    # 生成样本，根据输入 x 和 t 生成对应的样本，可以选择是否裁剪去噪声和重复噪声
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        # 获取输入 x 的形状和设备信息
        b, *_, device = *x.shape, x.device
        # 调用 self.p_mean_variance 方法，获取模型均值、方差和对应的设备信息
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        # 生成与输入 x 相同形状的噪声
        noise = noise_like(x.shape, device, repeat_noise)
        # 当 t == 0 时，不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 返回生成的样本，根据模型均值、方差和噪声计算得到
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 使用 torch.no_grad() 装饰器，表示在该函数中不需要计算梯度
    @torch.no_grad()
    # 循环生成样本，根据输入形状生成对应数量的样本，可以选择是否返回中间结果
    def p_sample_loop(self, shape, return_intermediates=False):
        # 获取设备信息
        device = self.betas.device
        # 获取 batch 大小
        b = shape[0]
        # 生成符合正态分布的随机数作为初始图像
        img = torch.randn(shape, device=device)
        # 初始化中间结果列表
        intermediates = [img]
        # 循环生成样本，根据当前时间步数调用 p_sample 方法生成样本
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            # 每隔一定时间步数记录中间结果
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        # 如果需要返回中间结果，则返回最终生成的图像和中间结果列表
        if return_intermediates:
            return img, intermediates
        # 否则只返回最终生成的图像
        return img

    # 使用 torch.no_grad() 装饰器，表示在该函数中不需要计算梯度
    @torch.no_grad()
    # 生成样本，可以指定批量大小和是否返回中间结果
    def sample(self, batch_size=16, return_intermediates=False):
        # 获取图像大小和通道数
        image_size = self.image_size
        channels = self.channels
        # 调用 p_sample_loop 方法生成样本
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    # 生成 q 分布的样本，根据初始图像 x_start 和时间步数 t 生成对应的样本
    def q_sample(self, x_start, t, noise=None):
        # 如果没有提供噪声，则生成与 x_start 相同形状的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 根据 alpha 和 beta 参数，计算 q 分布的样本
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    # 计算损失函数，根据预测值和目标值计算损失
    def get_loss(self, pred, target, mean=True):
        # 如果损失类型是 L1 损失
        if self.loss_type == 'l1':
            # 计算绝对值误差
            loss = (target - pred).abs()
            # 如果需要计算均值
            if mean:
                # 计算平均值
                loss = loss.mean()
        # 如果损失类型是 L2 损失
        elif self.loss_type == 'l2':
            # 如果需要计算均值
            if mean:
                # 计算均方误差损失
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                # 计算均方误差损失，不进行降维
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            # 抛出未实现的错误
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # 计算损失函数并返回损失值和损失字典
    def p_losses(self, x_start, t, noise=None):
        # 如果没有提供噪声，则生成一个与 x_start 相同形状的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 在 x_start 上添加噪声，得到 x_noisy
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 使用模型预测 x_noisy 的输出
        model_out = self.model(x_noisy, t)

        # 初始化损失字典
        loss_dict = {}
        # 如果参数化方式是 "eps"
        if self.parameterization == "eps":
            # 目标值为噪声
            target = noise
        # 如果参数化方式是 "x0"
        elif self.parameterization == "x0":
            # 目标值为 x_start
            target = x_start
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # 计算损失值，不进行均值计算，然后在指定维度上计算均值
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        # 根据训练状态确定日志前缀
        log_prefix = 'train' if self.training else 'val'

        # 更新损失字典，记录简单损失的均值
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        # 计算简单损失
        loss_simple = loss.mean() * self.l_simple_weight

        # 计算 VL-Bound 损失
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        # 更新损失字典，记录 VL-Bound 损失
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        # 计算总损失
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        # 更新损失字典，记录总损失
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    # 前向传播函数，根据输入 x 和时间步 t 计算损失
    def forward(self, x, *args, **kwargs):
        # 生成随机时间步 t
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # 调用 p_losses 函数计算损失
        return self.p_losses(x, t, *args, **kwargs)

    # 获取输入数据中的指定键值对应的值
    def get_input(self, batch, k):
        return batch[k]
    # 定义一个共享步骤函数，接收一个批次数据，获取输入数据并计算损失，返回损失和损失字典
    def shared_step(self, batch):
        # 获取输入数据
        x = self.get_input(batch, self.first_stage_key)
        # 调用模型进行前向传播计算损失
        loss, loss_dict = self(x)
        # 返回损失和损失字典
        return loss, loss_dict

    # 训练步骤函数，接收一个批次数据和批次索引，调用共享步骤函数计算损失
    def training_step(self, batch, batch_idx):
        # 调用共享步骤函数计算损失
        loss, loss_dict = self.shared_step(batch)

        # 记录损失字典到日志，显示在进度条中，同时记录到日志文件中，每步骤和每轮次都记录
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        # 记录全局步骤数到日志，显示在进度条中，同时记录到日志文件中，每步骤记录，每轮次不记录
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # 如果使用学习率调度器，记录学习率到日志，显示在进度条中，同时记录到日志文件中，每步骤记录，每轮次不记录
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # 返回损失
        return loss

    # 验证步骤函数，接收一个批次数据和批次索引，调用共享步骤函数计算损失，记录损失字典到日志
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # 调用共享步骤函数计算损失字典（不使用指数移动平均）
        _, loss_dict_no_ema = self.shared_step(batch)
        # 在指数移动平均范围内计算损失字典
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            # 为指数移动平均的损失字典添加后缀"_ema"
            loss_dict_ema = {f"{key}_ema": loss_dict_ema[key] for key in loss_dict_ema}
        # 记录不使用指数移动平均的损失字典到日志，不显示在进度条中，记录到日志文件中，每步骤不记录，每轮次记录
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # 记录使用指数移动平均的损失字典到日志，不显示在进度条中，记录到日志文件中，每步骤不记录，每轮次记录
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    # 训练批次结束时的回调函数，如果使用指数移动平均，更新模型的指数移动平均参数
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # 从样本列表中获取行数据的辅助函数
    def _get_rows_from_list(self, samples):
        # 计算每行图片数量
        n_imgs_per_row = len(samples)
        # 重排样本数据，使得每行图片排列在一起
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        # 重排样本数据，将每行图片数据合并到一起
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        # 将合并后的图片数据生成网格
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        # 返回生成的图片网格
        return denoise_grid

    # 无需梯度的装饰器
    @torch.no_grad()
    # 记录图像数据和相关信息，返回一个日志字典
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        # 初始化日志字典
        log = {}
        # 获取输入数据
        x = self.get_input(batch, self.first_stage_key)
        # 确定要展示的图像数量
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # 将输入数据移动到指定设备上
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # 获取扩散行
        diffusion_row = []
        x_start = x[:n_row]

        # 遍历时间步长
        for t in range(self.num_timesteps):
            # 每隔一定时间或者在最后一个时间步长时记录数据
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                # 重复时间步长信息以匹配数据维度
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                # 生成噪声数据
                noise = torch.randn_like(x_start)
                # 生成噪声数据的采样结果
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # 如果需要采样
        if sample:
            # 获取去噪行
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        # 如果需要返回特定键的数据
        if return_keys:
            # 如果返回的键和日志字典中的键有交集
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    # 配置优化器
    def configure_optimizers(self):
        # 获取学习率
        lr = self.learning_rate
        # 获取模型参数
        params = list(self.model.parameters())
        # 如果需要学习对数方差
        if self.learn_logvar:
            params = params + [self.logvar]
        # 使用AdamW优化器
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
class LatentDiffusion(DDPM):
    """LatentDiffusion 类继承自 DDPM 类"""
    """main class"""
    def make_cond_schedule(self, ):
        # 创建一个长度为 num_timesteps 的张量，填充值为 num_timesteps - 1，数据类型为 long
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        # 生成一个长度为 num_timesteps_cond 的张量，元素为从 0 到 num_timesteps - 1 的均匀间隔的整数，转换为 long 类型
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        # 将生成的 ids 赋值给 cond_ids 的前 num_timesteps_cond 个元素
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # 仅对第一个批次进行操作
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            # 断言确保不同时使用自定义缩放和标准缩放
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # 设置重新缩放权重为编码的标准差的倒数
            print("### USING STD-RESCALING ###")
            # 获取输入数据 x，并转移到设备上
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            # 对输入数据进行编码
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            # 删除之前的 scale_factor，并注册新的 scale_factor 为 1. / z 的标准差
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        # 调用父类的 register_schedule 方法
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        # 判断是否需要缩短条件调度
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            # 如果需要缩短条件调度，则调用 make_cond_schedule 方法
            self.make_cond_schedule()
    # 实例化第一阶段模型，根据配置信息创建模型实例，并设置为评估模式
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        # 禁用第一阶段模型的训练模式
        self.first_stage_model.train = disabled_train
        # 将第一阶段模型的所有参数设置为不需要梯度计算
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    # 实例化条件阶段模型，根据配置信息创建模型实例
    def instantiate_cond_stage(self, config):
        # 如果条件阶段模型不可训练
        if not self.cond_stage_trainable:
            # 如果配置为使用第一阶段模型作为条件阶段模型
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                # 将第一阶段模型作为条件阶段模型
                self.cond_stage_model = self.first_stage_model
            # 如果配置为无条件模型
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                # 将条件阶段模型设置为None
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                # 根据配置信息创建模型实例，并设置为评估模式
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                # 禁用条件阶段模型的训练模式
                self.cond_stage_model.train = disabled_train
                # 将条件阶段模型的所有参数设置为不需要梯度计算
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            # 断言条件阶段模型配置不为使用第一阶段模型或无条件模型
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            # 根据配置信息创建模型实例
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    # 从样本列表中获取去噪后的行数据
    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        # 初始化一个空列表用于存储去噪后的行数据
        denoise_row = []
        # 遍历样本列表，对每个样本进行第一阶段解码，并添加到去噪行列表中
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device), force_not_quantize=force_no_decoder_quantization))
        # 获取每行图片数量
        n_imgs_per_row = len(denoise_row)
        # 将去噪行数据转换为张量，维度为 n_log_step, n_row, C, H, W
        denoise_row = torch.stack(denoise_row)
        # 重新排列去噪行数据的维度，变为 b n c h w
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        # 重新排列去噪网格的维度，变为 (b n) c h w
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        # 创建一个网格，展示去噪后的图片，每行显示 n_imgs_per_row 张图片
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    # 获取第一阶段编码结果
    def get_first_stage_encoding(self, encoder_posterior):
        # 如果 encoder_posterior 是 DiagonalGaussianDistribution 类型，则采样 z
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        # 如果 encoder_posterior 是 torch.Tensor 类型，则直接使用
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        # 返回缩放后的 z
        return self.scale_factor * z

    # 获取学习到的条件
    def get_learned_conditioning(self, c):
        # 如果 cond_stage_forward 为空
        if self.cond_stage_forward is None:
            # 如果 cond_stage_model 具有 'encode' 方法且可调用，则使用 encode 方法对 c 进行编码
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                # 如果 c 是 DiagonalGaussianDistribution 类型，则取众数
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                # 否则，使用 cond_stage_model 对 c 进行处理
                c = self.cond_stage_model(c)
        else:
            # 否则，确保 cond_stage_model 具有 cond_stage_forward 属性
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            # 使用 cond_stage_forward 方法对 c 进行处理
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    # 创建一个网格，用于生成坐标网格
    def meshgrid(self, h, w):
        # 创建 y 坐标网格
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        # 创建 x 坐标网格
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        # 将 y 和 x 坐标合并为一个坐标数组
        arr = torch.cat([y, x], dim=-1)
        return arr
    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        # 创建一个张量表示图像右下角的坐标
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        # 生成一个网格，并将其归一化
        arr = self.meshgrid(h, w) / lower_right_corner
        # 计算左上角到边界的最小距离
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        # 计算右下角到边界的最小距离
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        # 计算边界距离，取左上角和右下角的最小值
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        # 获取边界距离权重
        weighting = self.delta_border(h, w)
        # 将权重限制在指定范围内
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        # 重复权重以匹配输出尺寸
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        # 如果需要使用 tie breaker
        if self.split_input_params["tie_braker"]:
            # 获取 L 图像的边界距离权重
            L_weighting = self.delta_border(Ly, Lx)
            # 将 L 图像的权重限制在指定范围内
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])
            # 重复 L 图像的权重以匹配输出尺寸
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            # 将权重乘以 L 图像的权重
            weighting = weighting * L_weighting
        return weighting

    @torch.no_grad()
    # 从父类中获取输入数据，根据批次和索引获取输入数据
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        # 调用父类方法获取输入数据
        x = super().get_input(batch, k)
        # 如果 bs 不为空，则截取输入数据的前 bs 个样本
        if bs is not None:
            x = x[:bs]
        # 将输入数据移动到指定设备上
        x = x.to(self.device)
        # 对输入数据进行第一阶段编码
        encoder_posterior = self.encode_first_stage(x)
        # 获取第一阶段编码的结果，并且将其 detach，即不参与梯度计算
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        # 如果 cond_key 为空，则使用默认的条件键
        cond_key = cond_key or self.cond_stage_key
        # 从父类中获取输入数据，根据批次和条件键获取输入数据
        xc = super().get_input(batch, cond_key)
        # 如果 bs 不为空，则截取条件输入数据的前 bs 个样本
        if bs is not None:
            xc["c_crossattn"] = xc["c_crossattn"][:bs]
            xc["c_concat"] = xc["c_concat"][:bs]
        # 初始化条件字典
        cond = {}

        # 支持无分类器引导，随机丢弃文本条件 5%，图像条件 5%，以及两者都丢弃 5%
        random = torch.rand(x.size(0), device=x.device)
        # 创建文本条件掩码和输入数据掩码
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")

        # 获取学习到的空条件
        null_prompt = self.get_learned_conditioning([""])
        # 将文本条件和输入数据应用掩码，并将其 detach
        cond["c_crossattn"] = [torch.where(prompt_mask, null_prompt, self.get_learned_conditioning(xc["c_crossattn"]).detach())]
        cond["c_concat"] = [input_mask * self.encode_first_stage((xc["c_concat"].to(self.device))).mode().detach()]

        # 将编码结果和条件字典组成输出列表
        out = [z, cond]
        # 如果需要返回第一阶段的输出
        if return_first_stage_outputs:
            # 对编码结果进行解码，将解码结果添加到输出列表中
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        # 如果需要返回原始条件数据
        if return_original_cond:
            out.append(xc)
        # 返回输出列表
        return out

    # 禁止梯度计算的装饰器
    @torch.no_grad()
    # 与上面相同，但没有装饰器
    @torch.no_grad()
    # 对输入数据进行第一阶段编码
    def encode_first_stage(self, x):
        # 检查是否存在split_input_params属性
        if hasattr(self, "split_input_params"):
            # 如果split_input_params中包含patch_distributed_vq字段
            if self.split_input_params["patch_distributed_vq"]:
                # 获取ks参数，表示卷积核大小，例如(128, 128)
                ks = self.split_input_params["ks"]
                # 获取stride参数，表示步长大小，例如(64, 64)
                stride = self.split_input_params["stride"]
                # 获取df参数，表示vqf
                df = self.split_input_params["vqf"]
                # 将原始图像大小保存在split_input_params中
                self.split_input_params['original_image_size'] = x.shape[-2:]
                # 获取输入数据的形状信息
                bs, nc, h, w = x.shape
                # 如果卷积核大小大于输入数据的高度或宽度，则调整卷积核大小
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                # 如果步长大小大于输入数据的高度或宽度，则调整步长大小
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                # 获取fold, unfold, normalization, weighting参数
                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                # 对输入数据进行展开操作
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # 重新调整形状为图像形状
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 对每个切片进行编码
                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                # 将编码结果堆叠在一起
                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # 恢复原始形状
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # 拼接切片
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                # 如果不使用分布式vq，则直接对输入数据进行编码
                return self.first_stage_model.encode(x)
        else:
            # 如果不存在split_input_params属性，则直接对输入数据进行编码
            return self.first_stage_model.encode(x)

    # 共享步骤，用于计算损失
    def shared_step(self, batch, **kwargs):
        # 获取输入数据和条件
        x, c = self.get_input(batch, self.first_stage_key)
        # 计算损失
        loss = self(x, c)
        return loss
    # 前向传播函数，接受输入 x、条件 c，以及其他参数和关键字参数
    def forward(self, x, c, *args, **kwargs):
        # 生成一个在 [0, self.num_timesteps) 范围内的随机整数张量，作为时间步长
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # 如果模型的 conditioning_key 不为空
        if self.model.conditioning_key is not None:
            # 断言条件 c 不为空
            assert c is not None
            # 如果条件阶段可训练
            if self.cond_stage_trainable:
                # 获取学习到的条件信息
                c = self.get_learned_conditioning(c)
            # 如果缩短条件调度
            if self.shorten_cond_schedule:  # TODO: drop this option
                # 获取对应时间步长的条件索引
                tc = self.cond_ids[t].to(self.device)
                # 从条件中采样，生成新的条件 c
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        # 返回 p_losses 函数的结果
        return self.p_losses(x, c, t, *args, **kwargs)

    # 根据输入 x_t、时间步长 t 和预测的起始值 pred_xstart，预测 eps
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        # 计算 eps
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # 计算先验的 KL 项，用于变分下界，以 bits-per-dim 为单位
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        # 获取批量大小
        batch_size = x_start.shape[0]
        # 创建时间步长张量 t，全为 self.num_timesteps - 1
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        # 计算条件分布的均值和对数方差
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        # 计算先验 KL 项
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        # 对 KL 值取平均，再除以 log(2.0)，得到 bits 单位的 KL 值
        return mean_flat(kl_prior) / np.log(2.0)
    # 计算潜在变量的损失函数
    def p_losses(self, x_start, cond, t, noise=None):
        # 如果没有提供噪声，则生成一个与 x_start 相同形状的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 在给定条件下从 Q 分布中采样得到噪声数据
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 应用模型得到输出
        model_output = self.apply_model(x_noisy, t, cond)

        # 初始化损失字典
        loss_dict = {}
        # 根据模型是否处于训练状态确定前缀
        prefix = 'train' if self.training else 'val'

        # 根据参数化方式确定目标数据
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            # 抛出未实现的错误
            raise NotImplementedError()

        # 计算简单损失
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        # 更新损失字典
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # 将时间步 t 对应的 logvar 转移到设备上
        logvar_t = self.logvar[t].to(self.device)
        # 计算总损失
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # 如果需要学习 logvar，则更新损失字典
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        # 计算最终损失
        loss = self.l_simple_weight * loss.mean()

        # 计算 VLBO 损失
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # 更新损失字典
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        # 更新总损失
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        # 返回总损失和损失字典
        return loss, loss_dict
    # 计算给定输入 x 的均值、后验方差和后验对数方差
    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        # 复制时间步 t
        t_in = t
        # 使用模型对输入 x 进行处理，返回模型输出
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)
    
        # 如果存在评分校正器，则修改模型输出的分数
        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
    
        # 如果需要返回码书 ID，则提取模型输出和对数概率
        if return_codebook_ids:
            model_out, logits = model_out
    
        # 根据参数化方式重建输入 x
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
    
        # 如果需要对重建后的输入进行裁剪，则将其限制在 [-1, 1] 范围内
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        # 如果需要对重建后的输入进行量化，则调用第一阶段模型进行量化
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        # 计算模型均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        # 根据需要返回的内容进行返回
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    
    # 禁用梯度计算
    @torch.no_grad()
    # 定义一个函数，用于生成样本
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        # 获取输入张量 x 的形状和设备信息
        b, *_, device = *x.shape, x.device
        # 调用 p_mean_variance 函数计算均值和方差
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        # 根据是否需要返回 codebook ids 进行不同处理
        if return_codebook_ids:
            # 抛出弃用警告
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        # 生成噪声
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        # 如果设置了 noise_dropout，则对噪声进行 dropout 处理
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # 当 t == 0 时，不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # 根据是否需要返回 codebook ids 进行不同返回
        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 禁用梯度计算
    @torch.no_grad()
    @torch.no_grad()
    # 定义一个函数，用于执行采样循环
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        # 如果未指定日志间隔，则使用默认值
        if not log_every_t:
            log_every_t = self.log_every_t
        # 获取设备信息
        device = self.betas.device
        # 获取 batch 大小
        b = shape[0]
        # 如果 x_T 为空，则生成一个形状为 shape 的随机张量
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        # 初始化 intermediates 列表，用于存储中间结果
        intermediates = [img]
        # 如果未指定时间步数，则使用默认值
        if timesteps is None:
            timesteps = self.num_timesteps

        # 如果指定了 start_T，则将时间步数限制在 start_T 之内
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        # 创建迭代器，用于迭代时间步数
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        # 如果存在 mask，则确保 x0 不为空且空间大小匹配
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        # 遍历时间步数
        for i in iterator:
            # 创建一个张量，用于存储时间步数
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            # 如果采样循环被缩短，则根据条件生成新的条件
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            # 执行 p_sample 函数，生成新的图像
            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            # 如果存在 mask，则根据 mask 对图像进行处理
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            # 如果达到日志间隔或是最后一个时间步，则将图像添加到 intermediates 中
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            # 如果存在回调函数，则执行回调函数
            if callback:
                callback(i)
            # 如果存在图像回调函数，则执行图像回调函数
            if img_callback:
                img_callback(img, i)

        # 如果需要返回中间结果，则返回图像和 intermediates 列表
        if return_intermediates:
            return img, intermediates
        # 否则只返回图像
        return img
    # 使用 torch.no_grad() 装饰器，确保在该函数中不会进行梯度计算
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        # 如果未指定 shape，则默认为(batch_size, self.channels, self.image_size, self.image_size)
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        # 处理条件 cond
        if cond is not None:
            # 如果 cond 是字典类型
            if isinstance(cond, dict):
                # 对字典中的每个值进行处理，确保长度不超过 batch_size
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                [x[:batch_size] for x in cond[key]] for key in cond}
            else:
                # 如果 cond 不是字典类型，则确保长度不超过 batch_size
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        # 调用 p_sample_loop 方法进行采样
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)
    
    # 使用 torch.no_grad() 装饰器，确保在该函数中不会进行梯度计算
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):
        # 如果 ddim 为真
        if ddim:
            # 创建 DDIMSampler 对象
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            # 调用 ddim_sampler 的 sample 方法进行采样
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)
        else:
            # 调用当前对象的 sample 方法进行采样
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)
        # 返回采样结果和中间结果
        return samples, intermediates
    
    # 使用 torch.no_grad() 装饰器，确保在该函数中不会进行梯度计算
    # 配置优化器，设置学习率为 self.learning_rate
    def configure_optimizers(self):
        lr = self.learning_rate
        # 获取模型的参数列表
        params = list(self.model.parameters())
        # 如果条件阶段可训练，将条件模型的参数也添加到参数列表中
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        # 如果需要学习 logvar，将 logvar 参数添加到参数列表中
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        # 使用 AdamW 优化器初始化
        opt = torch.optim.AdamW(params, lr=lr)
        # 如果使用调度器
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            # 实例化调度器
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            # 设置 LambdaLR 调度器
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        # 返回优化器
        return opt

    # 将输入 x 转换为 RGB 格式
    @torch.no_grad()
    def to_rgb(self, x):
        # 将输入 x 转换为浮点数类型
        x = x.float()
        # 如果对象中没有 colorize 属性，随机初始化一个 3x(x.shape[1])x1x1 的张量
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        # 使用卷积操作将 x 转换为 RGB 格式
        x = nn.functional.conv2d(x, weight=self.colorize)
        # 将 x 标准化到 [-1, 1] 范围内
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        # 返回转换后的 x
        return x
class DiffusionWrapper(pl.LightningModule):
    # 定义 DiffusionWrapper 类，继承自 pl.LightningModule
    def __init__(self, diff_model_config, conditioning_key):
        # 初始化方法，接受 diff_model_config 和 conditioning_key 两个参数
        super().__init__()
        # 调用父类的初始化方法
        self.diffusion_model = instantiate_from_config(diff_model_config)
        # 根据配置实例化扩散模型
        self.conditioning_key = conditioning_key
        # 设置条件键值
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']
        # 断言条件键值只能是给定的几种情况

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        # 前向传播方法，接受输入 x、时间 t，以及两个条件列表 c_concat 和 c_crossattn
        if self.conditioning_key is None:
            # 如果条件键值为 None
            out = self.diffusion_model(x, t)
            # 使用扩散模型处理输入 x 和时间 t
        elif self.conditioning_key == 'concat':
            # 如果条件键值为 'concat'
            xc = torch.cat([x] + c_concat, dim=1)
            # 将输入 x 和条件列表 c_concat 进行拼接
            out = self.diffusion_model(xc, t)
            # 使用扩散模型处理拼接后的输入 xc 和时间 t
        elif self.conditioning_key == 'crossattn':
            # 如果条件键值为 'crossattn'
            cc = torch.cat(c_crossattn, 1)
            # 将条件列表 c_crossattn 进行拼接
            out = self.diffusion_model(x, t, context=cc)
            # 使用扩散模型处理输入 x、时间 t 和上下文 cc
        elif self.conditioning_key == 'hybrid':
            # 如果条件键值为 'hybrid'
            xc = torch.cat([x] + c_concat, dim=1)
            # 将输入 x 和条件列表 c_concat 进行拼接
            cc = torch.cat(c_crossattn, 1)
            # 将条件列表 c_crossattn 进行拼接
            out = self.diffusion_model(xc, t, context=cc)
            # 使用扩散模型处理拼接后的输入 xc、时间 t 和上下文 cc
        elif self.conditioning_key == 'adm':
            # 如果条件键值为 'adm'
            cc = c_crossattn[0]
            # 获取条件列表 c_crossattn 的第一个元素
            out = self.diffusion_model(x, t, y=cc)
            # 使用扩散模型处理输入 x、时间 t 和 y=cc
        else:
            # 如果条件键值不在给定的几种情况中
            raise NotImplementedError()
            # 抛出未实现错误

        return out
        # 返回输出结果


class Layout2ImgDiffusion(LatentDiffusion):
    # 定义 Layout2ImgDiffusion 类，继承自 LatentDiffusion
    # TODO: move all layout-specific hacks to this class
    # TODO: 将所有与布局相关的 hack 移动到这个类中
    def __init__(self, cond_stage_key, *args, **kwargs):
        # 初始化方法，接受 cond_stage_key 和其他参数
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        # 断言 cond_stage_key 必须为 'coordinates_bbox'
        super().__init__(*args, cond_stage_key=cond_stage_key, **kwargs)
        # 调用父类的初始化方法，传入参数 cond_stage_key 和其他参数
    # 重写父类方法，记录图像信息
    def log_images(self, batch, N=8, *args, **kwargs):
        # 调用父类方法记录图像信息
        logs = super().log_images(*args, batch=batch, N=N, **kwargs)

        # 根据当前是训练还是验证阶段确定关键字
        key = 'train' if self.training else 'validation'
        # 获取数据集
        dset = self.trainer.datamodule.datasets[key]
        # 获取条件构建器
        mapper = dset.conditional_builders[self.cond_stage_key]

        # 初始化存储边界框图像的列表
        bbox_imgs = []
        # 定义映射函数，将类别编号映射为文本标签
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        # 遍历批次中的边界框数据，绘制边界框图像并添加到列表中
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        # 将边界框图像堆叠在一起
        cond_img = torch.stack(bbox_imgs, dim=0)
        # 将边界框图像添加到日志中
        logs['bbox_image'] = cond_img
        # 返回日志
        return logs
```