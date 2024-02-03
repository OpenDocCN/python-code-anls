# `stable-diffusion-webui\modules\sd_samplers_kdiffusion.py`

```py
# 导入 torch 库
import torch
# 导入 inspect 模块
import inspect
# 导入 k_diffusion.sampling 模块
import k_diffusion.sampling
# 从 modules 中导入 sd_samplers_common, sd_samplers_extra, sd_samplers_cfg_denoiser 模块
from modules import sd_samplers_common, sd_samplers_extra, sd_samplers_cfg_denoiser
# 从 modules.sd_samplers_cfg_denoiser 模块中导入 CFGDenoiser 类
from modules.sd_samplers_cfg_denoiser import CFGDenoiser  # noqa: F401
# 从 modules.script_callbacks 模块中导入 ExtraNoiseParams, extra_noise_callback 函数
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback

# 从 modules.shared 模块中导入 opts 变量
from modules.shared import opts
# 将 modules.shared 模块重命名为 shared
import modules.shared as shared

# 定义 samplers_k_diffusion 列表，包含多个元组，每个元组表示一个采样器的配置信息
samplers_k_diffusion = [
    ('DPM++ 2M Karras', 'sample_dpmpp_2m', ['k_dpmpp_2m_ka'], {'scheduler': 'karras'}),
    ('DPM++ SDE Karras', 'sample_dpmpp_sde', ['k_dpmpp_sde_ka'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE Exponential', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_exp'], {'scheduler': 'exponential', "brownian_noise": True}),
    ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {'scheduler': 'karras', "brownian_noise": True}),
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
    ('Euler', 'sample_euler', ['k_euler'], {}),
    ('LMS', 'sample_lms', ['k_lms'], {}),
    ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'discard_next_to_last_sigma': True, "second_order": True}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {"uses_ensd": True, "second_order": True}),
    ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {}),
    ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {"second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {"brownian_noise": True}),
    ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {"brownian_noise": True, "solver_type": "heun"}),
    # 创建元组，包含模型名称、模型文件夹名称、模型文件列表、模型参数字典
    ('DPM++ 2M SDE Heun Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun_ka'], {'scheduler': 'karras', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 2M SDE Heun Exponential', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun_exp'], {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM++ 3M SDE Karras', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM++ 3M SDE Exponential', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde_exp'], {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
    ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
    ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a_ka'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    ('Restart', sd_samplers_extra.restart_sampler, ['restart'], {'scheduler': 'karras', "second_order": True}),
# 定义一个列表，包含了KDiffusion采样器的相关数据
samplers_data_k_diffusion = [
    # 使用SamplerData类创建采样器数据对象，包括标签、函数名、别名和选项
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
    # 遍历samplers_k_diffusion列表中的元素，筛选出符合条件的元素
    for label, funcname, aliases, options in samplers_k_diffusion
    if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
]

# 定义一个字典，包含了各个采样器的额外参数
sampler_extra_params = {
    'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_heun': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_fast': ['s_noise'],
    'sample_dpm_2_ancestral': ['s_noise'],
    'sample_dpmpp_2s_ancestral': ['s_noise'],
    'sample_dpmpp_sde': ['s_noise'],
    'sample_dpmpp_2m_sde': ['s_noise'],
    'sample_dpmpp_3m_sde': ['s_noise'],
}

# 创建一个字典，将采样器名称映射到采样器对象
k_diffusion_samplers_map = {x.name: x for x in samplers_data_k_diffusion}

# 定义一个字典，包含了不同采样器调度器的函数
k_diffusion_scheduler = {
    'Automatic': None,
    'karras': k_diffusion.sampling.get_sigmas_karras,
    'exponential': k_diffusion.sampling.get_sigmas_exponential,
    'polyexponential': k_diffusion.sampling.get_sigmas_polyexponential
}

# 定义一个CFGDenoiserKDiffusion类，继承自CFGDenoiser类
class CFGDenoiserKDiffusion(sd_samplers_cfg_denoiser.CFGDenoiser):
    # 定义inner_model属性，用于获取内部模型
    @property
    def inner_model(self):
        # 如果模型封装为空，则根据参数化类型选择合适的去噪器
        if self.model_wrap is None:
            denoiser = k_diffusion.external.CompVisVDenoiser if shared.sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
            self.model_wrap = denoiser(shared.sd_model, quantize=shared.opts.enable_quantization)
        # 返回模型封装
        return self.model_wrap

# 定义一个KDiffusionSampler类，继承自Sampler类
class KDiffusionSampler(sd_samplers_common.Sampler):
    # 初始化函数，接受函数名、sd_model和选项作为参数
    def __init__(self, funcname, sd_model, options=None):
        # 调用父类的初始化函数
        super().__init__(funcname)

        # 获取特定采样器的额外参数
        self.extra_params = sampler_extra_params.get(funcname, [])

        # 设置选项，如果选项为空则使用默认值
        self.options = options or {}
        # 如果函数名可调用，则直接使用，否则获取k_diffusion.sampling中的函数
        self.func = funcname if callable(funcname) else getattr(k_diffusion.sampling, self.funcname)

        # 创建CFGDenoiserKDiffusion对象，并传入当前对象
        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        # 获取内部模型
        self.model_wrap = self.model_wrap_cfg.inner_model
    # 定义一个方法用于生成样本
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        # 如果未指定步数，则使用默认步数
        steps = steps or p.steps

        # 获取每个步骤的标准差
        sigmas = self.get_sigmas(p, steps)

        # 如果设置了噪声乘数，则在输入数据上应用噪声乘数
        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]

        # 初始化额外参数的关键字参数
        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        # 如果函数参数中包含'n'，则将步数添加到额外参数中
        if 'n' in parameters:
            extra_params_kwargs['n'] = steps

        # 如果函数参数中包含'sigma_min'，则将最小和最大标准差添加到额外参数中
        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()

        # 如果函数参数中包含'sigmas'，则将所有标准差添加到额外参数中
        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigmas

        # 如果配置中设置了'brownian_noise'，则创建噪声采样器并添加到额外参数中
        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        # 如果配置中设置了'solver_type'为'heun'，则将求解器类型添加到额外参数中
        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        # 保存最后一个潜在变量
        self.last_latent = x
        # 设置采样器的额外参数
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        # 启动采样过程，生成样本
        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        # 如果模型配置中设置了'padded_cond_uncond'，则将'Pad conds'标记添加到额外生成参数中
        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        # 返回生成的样本
        return samples
```