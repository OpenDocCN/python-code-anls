# `stable-diffusion-webui\modules\sd_samplers_timesteps.py`

```py
# 导入 torch 模块
import torch
# 导入 inspect 模块
import inspect
# 导入 sys 模块
import sys
# 从 modules 模块中导入 devices, sd_samplers_common, sd_samplers_timesteps_impl 模块
from modules import devices, sd_samplers_common, sd_samplers_timesteps_impl
# 从 modules.sd_samplers_cfg_denoiser 模块中导入 CFGDenoiser 类
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
# 从 modules.script_callbacks 模块中导入 ExtraNoiseParams, extra_noise_callback 函数
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
# 从 modules.shared 模块中导入 opts 变量
from modules.shared import opts
# 将 modules.shared 模块重命名为 shared
import modules.shared as shared

# 定义 samplers_timesteps 列表，包含元组，每个元组包含采样器名称、采样器函数、别名列表和选项字典
samplers_timesteps = [
    ('DDIM', sd_samplers_timesteps_impl.ddim, ['ddim'], {}),
    ('PLMS', sd_samplers_timesteps_impl.plms, ['plms'], {}),
    ('UniPC', sd_samplers_timesteps_impl.unipc, ['unipc'], {}),
]

# 定义 samplers_data_timesteps 列表，使用列表推导式生成，每个元素是 SamplerData 对象
samplers_data_timesteps = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: CompVisSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_timesteps
]

# 定义 CompVisTimestepsDenoiser 类，继承自 torch.nn.Module 类
class CompVisTimestepsDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def forward(self, input, timesteps, **kwargs):
        return self.inner_model.apply_model(input, timesteps, **kwargs)

# 定义 CompVisTimestepsVDenoiser 类，继承自 torch.nn.Module 类
class CompVisTimestepsVDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    # 定义 predict_eps_from_z_and_v 方法，根据输入 x_t, t, v 预测 eps
    def predict_eps_from_z_and_v(self, x_t, t, v):
        return self.inner_model.sqrt_alphas_cumprod[t.to(torch.int), None, None, None] * v + self.inner_model.sqrt_one_minus_alphas_cumprod[t.to(torch.int), None, None, None] * x_t

    def forward(self, input, timesteps, **kwargs):
        model_output = self.inner_model.apply_model(input, timesteps, **kwargs)
        e_t = self.predict_eps_from_z_and_v(input, timesteps, model_output)
        return e_t

# 定义 CFGDenoiserTimesteps 类，继承自 CFGDenoiser 类
class CFGDenoiserTimesteps(CFGDenoiser):

    def __init__(self, sampler):
        super().__init__(sampler)

        # 初始化 alphas 属性为 shared.sd_model.alphas_cumprod
        self.alphas = shared.sd_model.alphas_cumprod
        # 设置 mask_before_denoising 属性为 True
        self.mask_before_denoising = True
    # 根据输入、输出和标准差计算时间步长
    def get_pred_x0(self, x_in, x_out, sigma):
        ts = sigma.to(dtype=int)

        # 获取时间步长对应的 alpha 值
        a_t = self.alphas[ts][:, None, None, None]
        # 计算 1 - alpha_t 的平方根
        sqrt_one_minus_at = (1 - a_t).sqrt()

        # 预测 x0 值
        pred_x0 = (x_in - sqrt_one_minus_at * x_out) / a_t.sqrt()

        return pred_x0

    # 获取内部模型
    @property
    def inner_model(self):
        # 如果模型封装对象为空，则根据参数化类型选择对应的去噪器
        if self.model_wrap is None:
            denoiser = CompVisTimestepsVDenoiser if shared.sd_model.parameterization == "v" else CompVisTimestepsDenoiser
            # 创建模型封装对象
            self.model_wrap = denoiser(shared.sd_model)

        return self.model_wrap
# 定义一个名为 CompVisSampler 的类，继承自 sd_samplers_common.Sampler 类
class CompVisSampler(sd_samplers_common.Sampler):
    # 初始化方法，接受函数名和 sd_model 作为参数
    def __init__(self, funcname, sd_model):
        # 调用父类的初始化方法
        super().__init__(funcname)

        # 设置 eta_option_field 字段为 'eta_ddim'
        self.eta_option_field = 'eta_ddim'
        # 设置 eta_infotext_field 字段为 'Eta DDIM'
        self.eta_infotext_field = 'Eta DDIM'
        # 设置 eta_default 字段为 0.0
        self.eta_default = 0.0

        # 创建一个 CFGDenoiserTimesteps 对象并赋值给 model_wrap_cfg 字段
        self.model_wrap_cfg = CFGDenoiserTimesteps(self)

    # 定义一个名为 get_timesteps 的方法，接受 p 和 steps 作为参数
    def get_timesteps(self, p, steps):
        # 检查是否应该丢弃倒数第二个 sigma
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        # 如果总是丢弃倒数第二个 sigma 且之前未设置丢弃倒数第二个 sigma，则设置为 True
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            # 在额外生成参数中添加 "Discard penultimate sigma" 字段并设置为 True
            p.extra_generation_params["Discard penultimate sigma"] = True

        # 如果需要丢弃倒数第二个 sigma，则步数加一，否则不变
        steps += 1 if discard_next_to_last_sigma else 0

        # 生成一个包含步长的张量，范围从 0 到 999，根据步数均匀分布
        timesteps = torch.clip(torch.asarray(list(range(0, 1000, 1000 // steps)), device=devices.device) + 1, 0, 999)

        # 返回生成的步长张量
        return timesteps
    # 定义一个函数，用于生成图像到图像的样本
    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        # 设置步骤数和时间编码
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        # 获取时间步长
        timesteps = self.get_timesteps(p, steps)
        # 获取时间步长调度
        timesteps_sched = timesteps[:t_enc]

        # 获取累积 alpha 的平方根
        alphas_cumprod = shared.sd_model.alphas_cumprod
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[timesteps[t_enc]])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps[t_enc]])

        # 计算 xi
        xi = x * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod

        # 如果存在额外噪音
        if opts.img2img_extra_noise > 0:
            # 设置额外生成参数中的额外噪音
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            # 创建额外噪音参数对象
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            # 调用额外噪音回调函数
            extra_noise_callback(extra_noise_params)
            # 更新噪音
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise * sqrt_alpha_cumprod

        # 初始化额外参数关键字参数
        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        # 如果参数中包含 'timesteps'
        if 'timesteps' in parameters:
            extra_params_kwargs['timesteps'] = timesteps_sched
        # 如果参数中包含 'is_img2img'
        if 'is_img2img' in parameters:
            extra_params_kwargs['is_img2img'] = True

        # 设置模型包装配置的初始潜在变量和最后潜在变量
        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        # 设置采样器额外参数
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        # 启动采样过程
        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        # 如果模型包装配置中存在填充条件和无条件
        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        # 返回样本
        return samples
    # 定义一个方法用于生成样本
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        # 如果未指定步数，则使用默认步数
        steps = steps or p.steps
        # 获取时间步长
        timesteps = self.get_timesteps(p, steps)

        # 初始化额外参数
        extra_params_kwargs = self.initialize(p)
        # 获取函数参数列表
        parameters = inspect.signature(self.func).parameters

        # 如果函数参数中包含 'timesteps'，则将 timesteps 添加到额外参数中
        if 'timesteps' in parameters:
            extra_params_kwargs['timesteps'] = timesteps

        # 设置最后一个潜在变量为 x
        self.last_latent = x
        # 设置采样器的额外参数
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }
        # 启动采样过程，调用函数生成样本
        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        # 如果模型配置中包含填充条件和非条件，则将额外生成参数中的 "Pad conds" 设置为 True
        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        # 返回生成的样本
        return samples
# 将当前模块添加到 sys.modules 字典中，键为 'modules.sd_samplers_compvis'，值为当前模块对象
sys.modules['modules.sd_samplers_compvis'] = sys.modules[__name__]
# 将 CompVisSampler 赋值给 VanillaStableDiffusionSampler，用于与旧扩展的临时兼容性
VanillaStableDiffusionSampler = CompVisSampler
```