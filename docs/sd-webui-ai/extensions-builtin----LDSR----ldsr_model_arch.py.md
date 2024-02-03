# `stable-diffusion-webui\extensions-builtin\LDSR\ldsr_model_arch.py`

```
# 导入所需的库
import os
import gc
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf
import safetensors.torch

# 导入自定义模块
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, ismap
from modules import shared, sd_hijack, devices

# 全局变量，用于缓存模型
cached_ldsr_model: torch.nn.Module = None

# 创建 LDSR 类
class LDSR:
    # 从配置中加载模型
    def load_model_from_config(self, half_attention):
        global cached_ldsr_model

        # 如果启用了缓存且缓存中有模型，则从缓存中加载模型
        if shared.opts.ldsr_cached and cached_ldsr_model is not None:
            print("Loading model from cache")
            model: torch.nn.Module = cached_ldsr_model
        else:
            # 否则从指定路径加载模型
            print(f"Loading model from {self.modelPath}")
            _, extension = os.path.splitext(self.modelPath)
            # 根据文件扩展名加载模型参数
            if extension.lower() == ".safetensors":
                pl_sd = safetensors.torch.load_file(self.modelPath, device="cpu")
            else:
                pl_sd = torch.load(self.modelPath, map_location="cpu")
            sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
            config = OmegaConf.load(self.yamlPath)
            config.model.target = "ldm.models.diffusion.ddpm.LatentDiffusionV1"
            model: torch.nn.Module = instantiate_from_config(config.model)
            model.load_state_dict(sd, strict=False)
            model = model.to(shared.device)
            # 如果需要使用半精度注意力，则将模型转换为半精度
            if half_attention:
                model = model.half()
            # 如果指定了通道优先存储，则将模型转换为通道优先存储格式
            if shared.cmd_opts.opt_channelslast:
                model = model.to(memory_format=torch.channels_last)

            # 对模型进行优化
            sd_hijack.model_hijack.hijack(model)
            model.eval()

            # 如果启用了缓存，则将加载的模型缓存起来
            if shared.opts.ldsr_cached:
                cached_ldsr_model = model

        return {"model": model}

    # 初始化方法，接收模型路径和配置文件路径
    def __init__(self, model_path, yaml_path):
        self.modelPath = model_path
        self.yamlPath = yaml_path

    @staticmethod
    # 对输入的图像进行超分辨率处理，返回处理后的图像
    def super_resolution(self, image, steps=100, target_scale=2, half_attention=False):
        # 从配置加载模型
        model = self.load_model_from_config(half_attention)

        # 设置扩散步数
        diffusion_steps = int(steps)
        # 设置 eta 值
        eta = 1.0

        # 执行垃圾回收
        gc.collect()
        # 执行 torch 垃圾回收
        devices.torch_gc()

        # 保存原始图像信息
        im_og = image
        width_og, height_og = im_og.size
        # 计算降采样率
        down_sample_rate = target_scale / 4
        wd = width_og * down_sample_rate
        hd = height_og * down_sample_rate
        width_downsampled_pre = int(np.ceil(wd))
        height_downsampled_pre = int(np.ceil(hd))

        # 如果需要降采样，则进行降采样
        if down_sample_rate != 1:
            print(
                f'Downsampling from [{width_og}, {height_og}] to [{width_downsampled_pre}, {height_downsampled_pre}]')
            im_og = im_og.resize((width_downsampled_pre, height_downsampled_pre), Image.LANCZOS)
        else:
            print(f"Down sample rate is 1 from {target_scale} / 4 (Not downsampling)")

        # 将宽度和高度填充为 64 的倍数，使用图像的边缘值进行填充以避免伪影
        pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(im_og.size) / 64).astype(int)), axis=0) * 64 - im_og.size
        im_padded = Image.fromarray(np.pad(np.array(im_og), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))

        # 运行模型
        logs = self.run(model["model"], im_padded, diffusion_steps, eta)

        # 处理输出样本
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        a = Image.fromarray(sample[0])

        # 去除填充
        a = a.crop((0, 0) + tuple(np.array(im_og.size) * 4))

        # 释放模型内存
        del model
        # 执行垃圾回收
        gc.collect()
        # 执行 torch 垃圾回收
        devices.torch_gc()

        # 返回处理后的图像
        return a
# 获取条件信息，返回一个示例字典
def get_cond(selected_path):
    # 创建一个空字典作为示例
    example = {}
    # 设置上采样因子为4
    up_f = 4
    # 将选定路径的图像转换为 RGB 格式
    c = selected_path.convert('RGB')
    # 将 RGB 图像转换为 PyTorch 张量，并添加一个维度
    c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
    # 对图像进行上采样
    c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
    # 重新排列张量维度
    c_up = rearrange(c_up, '1 c h w -> 1 h w c')
    c = rearrange(c, '1 c h w -> 1 h w c')
    # 对图像进行归一化处理
    c = 2. * c - 1.
    
    # 将图像数据移动到指定设备上
    c = c.to(shared.device)
    # 将处理后的 LR 图像和上采样后的图像添加到示例字典中
    example["LR_image"] = c
    example["image"] = c_up

    return example


# 使用 DDIMSampler 模型进行采样
@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, temperature=1., score_corrector=None,
                    corrector_kwargs=None, x_t=None
                    ):
    # 创建 DDIMSampler 对象
    ddim = DDIMSampler(model)
    # 获取批次大小
    bs = shape[0]
    # 获取图像形状
    shape = shape[1:]
    # 打印采样参数信息
    print(f"Sampling with eta = {eta}; steps: {steps}")
    # 使用 DDIMSampler 进行采样
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_t=x_t)

    return samples, intermediates


# 生成卷积采样
@torch.no_grad()
def make_convolutional_sample(batch, model, custom_steps=None, eta=1.0, quantize_x0=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, ddim_use_x0_pred=False):
    # 创建一个空字典用于记录日志信息
    log = {}
    # 从模型中获取输入数据和条件数据，返回第一阶段输出，如果需要的话
    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    # 如果有自定义形状，则生成符合该形状的随机张量
    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    # 初始化变量 z0
    z0 = None

    # 将输入数据和重构数据记录到日志中
    log["input"] = x
    log["reconstruction"] = xrec

    # 如果条件数据是地图，则将原始条件数据转换为 RGB 格式
    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        # 如果模型有条件阶段关键字，则记录到日志中
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        # 如果条件数据不是地图，则将原始条件数据记录到日志中，如果为空则使用与 x 相同形状的零张量
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        # 如果模型有条件阶段模型，则记录到日志中
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            # 如果条件阶段关键字是 'class_label'，则记录到日志中
            if model.cond_stage_key == 'class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    # 在模型的 EMA 范围内进行绘图
    with model.ema_scope("Plotting"):
        t0 = time.time()

        # 使用模型进行卷积采样，返回采样结果和中间结果
        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, mask=None, x0=z0,
                                                temperature=temperature, score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_t=x_T)
        t1 = time.time()

        # 如果使用 x0 预测，则将采样结果替换为预测的 x0
        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    # 使用模型解码第一阶段的采样结果
    x_sample = model.decode_first_stage(sample)
    # 尝试使用模型的第一阶段解码样本，强制不进行量化
    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        # 将不进行量化的样本添加到日志中
        log["sample_noquant"] = x_sample_noquant
        # 计算不进行量化样本与原始样本的绝对差值，并添加到日志中
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    # 捕获任何异常
    except Exception:
        # 如果出现异常则不执行任何操作
        pass

    # 将原始样本添加到日志中
    log["sample"] = x_sample
    # 计算并记录执行时间
    log["time"] = t1 - t0

    # 返回日志
    return log
```