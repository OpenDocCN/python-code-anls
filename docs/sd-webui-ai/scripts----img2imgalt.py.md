# `stable-diffusion-webui\scripts\img2imgalt.py`

```py
# 导入必要的模块
from collections import namedtuple
import numpy as np
from tqdm import trange
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, sd_samplers_common
import torch
import k_diffusion as K

# 定义函数，用于为给定图像查找噪声
def find_noise_for_image(p, cond, uncond, cfg_scale, steps):
    # 初始化潜在向量
    x = p.init_latent

    # 创建与 x 相同形状的全为 1 的张量
    s_in = x.new_ones([x.shape[0]])

    # 根据模型参数化方式选择不同的去噪器
    if shared.sd_model.parameterization == "v":
        dnw = K.external.CompVisVDenoiser(shared.sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(shared.sd_model)
        skip = 0

    # 获取噪声标准差并翻转
    sigmas = dnw.get_sigmas(steps).flip(0)

    # 设置全局状态的采样步数
    shared.state.sampling_steps = steps

    # 循环处理每个标准差
    for i in trange(1, len(sigmas)):
        # 更新全局状态的采样步数
        shared.state.sampling_step += 1

        # 复制 x 和标准差
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        # 准备条件输入
        image_conditioning = torch.cat([p.image_conditioning] * 2)
        cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}

        # 获取缩放系数
        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]
        t = dnw.sigma_to_t(sigma_in)

        # 应用模型获取噪声
        eps = shared.sd_model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        # 计算去噪后的图像
        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        # 计算梯度
        d = (x - denoised) / sigmas[i]
        dt = sigmas[i] - sigmas[i - 1]

        # 更新潜在向量
        x = x + d * dt

        # 存储潜在向量
        sd_samplers_common.store_latent(x)

        # 释放内存，解决一些 VRAM 问题
        del x_in, sigma_in, cond_in, c_out, c_in, t,
        del eps, denoised_uncond, denoised_cond, denoised, d, dt

    # 更新全局状态的下一个任务
    shared.state.nextjob()

    # 返回标准化后的潜在向量
    return x / x.std()

# 定义命名元组 Cached
Cached = namedtuple("Cached", ["noise", "cfg_scale", "steps", "latent", "original_prompt", "original_negative_prompt", "sigma_adjustment"])
# 根据 briansemrau 在 https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/736 中提出的建议进行更改
def find_noise_for_image_sigma_adjustment(p, cond, uncond, cfg_scale, steps):
    # 初始化潜在变量 x
    x = p.init_latent

    # 创建一个与 x 形状相同的全为 1 的张量 s_in
    s_in = x.new_ones([x.shape[0]])
    
    # 根据共享的 sd_model 的参数化方式选择不同的 CompVisDenoiser 对象和 skip 值
    if shared.sd_model.parameterization == "v":
        dnw = K.external.CompVisVDenoiser(shared.sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(shared.sd_model)
        skip = 0
    
    # 获取噪声水平 sigmas，并翻转顺序
    sigmas = dnw.get_sigmas(steps).flip(0)

    # 设置共享状态的采样步数为 steps
    shared.state.sampling_steps = steps

    # 遍历 sigmas
    for i in trange(1, len(sigmas)):
        # 更新共享状态的采样步数
        shared.state.sampling_step += 1

        # 复制 x 两次，构成 x_in
        x_in = torch.cat([x] * 2)
        # 复制 sigmas[i - 1] * s_in 两次，构成 sigma_in
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        # 拼接 cond 和 uncond，构成 cond_in
        cond_in = torch.cat([uncond, cond])

        # 复制 p.image_conditioning 两次，构成 image_conditioning
        image_conditioning = torch.cat([p.image_conditioning] * 2)
        # 构建 cond_in 字典
        cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}

        # 获取 dnw 的缩放系数 c_out 和 c_in
        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]

        # 根据不同情况计算 t
        if i == 1:
            t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
        else:
            t = dnw.sigma_to_t(sigma_in)

        # 应用模型生成噪声 eps
        eps = shared.sd_model.apply_model(x_in * c_in, t, cond=cond_in)
        # 分离无条件和有条件的去噪结果
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        # 计算去噪后的结果 denoised
        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        # 根据不同情况计算梯度 d
        if i == 1:
            d = (x - denoised) / (2 * sigmas[i])
        else:
            d = (x - denoised) / sigmas[i - 1]

        # 计算 sigmas[i] 和 sigmas[i - 1] 之间的差值 dt
        dt = sigmas[i] - sigmas[i - 1]
        # 更新 x
        x = x + d * dt

        # 存储潜在变量 x
        sd_samplers_common.store_latent(x)

        # 释放内存，避免 VRAM 问题
        del x_in, sigma_in, cond_in, c_out, c_in, t,
        del eps, denoised_uncond, denoised_cond, denoised, d, dt

    # 更新共享状态的下一个任务
    shared.state.nextjob()

    # 返回 x 除以最后一个 sigmas 的结果
    return x / sigmas[-1]


# Script 类继承自 scripts.Script
class Script(scripts.Script):
    def __init__(self):
        # 初始化缓存为 None
        self.cache = None
    # 返回标题字符串
    def title(self):
        return "img2img alternative test"

    # 返回是否为img2img的布尔值
    def show(self, is_img2img):
        return is_img2img

    # 创建UI元素
    def ui(self, is_img2img):
        # 创建Markdown信息
        info = gr.Markdown('''
        * `CFG Scale` should be 2 or lower.
        ''')

        # 创建复选框元素
        override_sampler = gr.Checkbox(label="Override `Sampling method` to Euler?(this method is built for it)", value=True, elem_id=self.elem_id("override_sampler"))

        # 创建复选框元素和文本框元素
        override_prompt = gr.Checkbox(label="Override `prompt` to the same value as `original prompt`?(and `negative prompt`)", value=True, elem_id=self.elem_id("override_prompt"))
        original_prompt = gr.Textbox(label="Original prompt", lines=1, elem_id=self.elem_id("original_prompt"))
        original_negative_prompt = gr.Textbox(label="Original negative prompt", lines=1, elem_id=self.elem_id("original_negative_prompt"))

        # 创建复选框元素和滑块元素
        override_steps = gr.Checkbox(label="Override `Sampling Steps` to the same value as `Decode steps`?", value=True, elem_id=self.elem_id("override_steps"))
        st = gr.Slider(label="Decode steps", minimum=1, maximum=150, step=1, value=50, elem_id=self.elem_id("st"))

        # 创建复选框元素
        override_strength = gr.Checkbox(label="Override `Denoising strength` to 1?", value=True, elem_id=self.elem_id("override_strength"))

        # 创建滑块元素
        cfg = gr.Slider(label="Decode CFG scale", minimum=0.0, maximum=15.0, step=0.1, value=1.0, elem_id=self.elem_id("cfg"))
        randomness = gr.Slider(label="Randomness", minimum=0.0, maximum=1.0, step=0.01, value=0.0, elem_id=self.elem_id("randomness"))
        sigma_adjustment = gr.Checkbox(label="Sigma adjustment for finding noise for image", value=False, elem_id=self.elem_id("sigma_adjustment"))

        # 返回UI元素列表
        return [
            info,
            override_sampler,
            override_prompt, original_prompt, original_negative_prompt,
            override_steps, st,
            override_strength,
            cfg, randomness, sigma_adjustment,
        ]
```