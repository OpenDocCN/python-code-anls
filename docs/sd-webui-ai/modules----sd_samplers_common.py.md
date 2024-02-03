# `stable-diffusion-webui\modules\sd_samplers_common.py`

```
# 导入 inspect 模块，用于检查对象
import inspect
# 导入 namedtuple 类型，用于创建命名元组
from collections import namedtuple
# 导入 numpy 库，用于科学计算
import numpy as np
# 导入 torch 库，用于深度学习
import torch
# 导入 PIL 库，用于图像处理
from PIL import Image
# 导入自定义模块
from modules import devices, images, sd_vae_approx, sd_samplers, sd_vae_taesd, shared, sd_models
# 从 modules.shared 模块中导入 opts, state 变量
from modules.shared import opts, state
# 导入 k_diffusion.sampling 模块
import k_diffusion.sampling

# 创建命名元组 SamplerDataTuple，包含字段 name, constructor, aliases, options
SamplerDataTuple = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])

# 定义类 SamplerData，继承自 SamplerDataTuple
class SamplerData(SamplerDataTuple):
    # 定义方法 total_steps，根据选项计算总步数
    def total_steps(self, steps):
        # 如果选项中包含 "second_order"，则总步数乘以 2
        if self.options.get("second_order", False):
            steps = steps * 2

        return steps

# 定义函数 setup_img2img_steps，设置图像到图像的步数
def setup_img2img_steps(p, steps=None):
    # 如果 opts.img2img_fix_steps 为真或者 steps 不为空
    if opts.img2img_fix_steps or steps is not None:
        # 计算请求的步数
        requested_steps = (steps or p.steps)
        # 根据去噪强度计算步数
        steps = int(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc

# 创建字典 approximation_indexes，包含不同近似方法的索引
approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2, "TAESD": 3}

# 定义函数 samples_to_images_tensor，将 4 通道潜在空间图像转换为 3 通道 RGB 图像张量
def samples_to_images_tensor(sample, approximation=None, model=None):
    """Transforms 4-channel latent space images into 3-channel RGB image tensors, with values in range [-1, 1]."""

    # 如果未指定近似方法或者状态被中断且 live_preview_fast_interrupt 为真
    if approximation is None or (shared.state.interrupted and opts.live_preview_fast_interrupt):
        # 获取近似方法的索引
        approximation = approximation_indexes.get(opts.show_progress_type, 0)

        # 导入 lowvram 模块
        from modules import lowvram
        # 如果近似方法为 Full 且低 VRAM 模式已启用且 live_preview_allow_lowvram_full 为假
        if approximation == 0 and lowvram.is_enabled(shared.sd_model) and not shared.opts.live_preview_allow_lowvram_full:
            approximation = 1

    # 根据不同的近似方法处理样本
    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample)
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype)).detach()
    elif approximation == 3:
        x_sample = sd_vae_taesd.decoder_model()(sample.to(devices.device, devices.dtype)).detach()
        x_sample = x_sample * 2 - 1
    else:
        # 如果模型为空，则使用共享的 sd_model
        if model is None:
            model = shared.sd_model
        # 关闭自动转换，修复在 fp32 下仍不稳定的 VAE 问题
        with devices.without_autocast(): 
            # 使用模型的第一阶段数据类型解码样本
            x_sample = model.decode_first_stage(sample.to(model.first_stage_model.dtype))

    # 返回解码后的样本
    return x_sample
# 将单个样本转换为图像
def single_sample_to_image(sample, approximation=None):
    # 将样本转换为图像张量，并将其范围从[-1, 1]映射到[0, 1]
    x_sample = samples_to_images_tensor(sample.unsqueeze(0), approximation)[0] * 0.5 + 0.5

    # 将图像张量的值限制在[0, 1]范围内
    x_sample = torch.clamp(x_sample, min=0.0, max=1.0)
    # 将图像张量转换为 numpy 数组，并重新排列维度
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    # 将图像数组转换为无符号整数类型
    x_sample = x_sample.astype(np.uint8)

    # 返回图像对象
    return Image.fromarray(x_sample)


# 解码第一阶段
def decode_first_stage(model, x):
    # 将输入数据转换为指定设备的数据类型
    x = x.to(devices.dtype_vae)
    # 获取逼近索引
    approx_index = approximation_indexes.get(opts.sd_vae_decode_method, 0)
    # 返回样本转换为图像的结果
    return samples_to_images_tensor(x, approx_index, model)


# 将样本转换为图像
def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


# 将多个样本转换为图像网格
def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


# 将图像张量转换为样本
def images_tensor_to_samples(image, approximation=None, model=None):
    '''image[0, 1] -> latent'''
    # 如果逼近方法未指定，则使用默认的逼近方法
    if approximation is None:
        approximation = approximation_indexes.get(opts.sd_vae_encode_method, 0)

    # 根据不同的逼近方法进行处理
    if approximation == 3:
        # 将图像转换为指定设备和数据类型
        image = image.to(devices.device, devices.dtype)
        # 使用编码器模型获取潜在表示
        x_latent = sd_vae_taesd.encoder_model()(image)
    else:
        # 如果模型未指定，则使用共享的模型
        if model is None:
            model = shared.sd_model
        # 将第一阶段模型转换为指定数据类型
        model.first_stage_model.to(devices.dtype_vae)

        # 将图像转换为指定设备和数据类型，并将值范围映射到[-1, 1]
        image = image.to(shared.device, dtype=devices.dtype_vae)
        image = image * 2 - 1
        # 如果图像数量大于1，则对每个图像进行处理
        if len(image) > 1:
            x_latent = torch.stack([
                model.get_first_stage_encoding(
                    model.encode_first_stage(torch.unsqueeze(img, 0))
                )[0]
                for img in image
            ])
        else:
            x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))

    # 返回潜在表示
    return x_latent


# 存储潜在表示
def store_latent(decoded):
    state.current_latent = decoded
    # 检查是否启用实时预览、是否设置了每隔多少步显示进度、当前采样步数是否符合显示进度的条件
    if opts.live_previews_enable and opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % opts.show_progress_every_n_steps == 0:
        # 如果不允许并行处理，则将解码后的样本分配给当前图像
        if not shared.parallel_processing_allowed:
            shared.state.assign_current_image(sample_to_image(decoded))
# 判断配置中的采样器是否使用 eta 噪声种子增量来创建图像
def is_sampler_using_eta_noise_seed_delta(p):
    # 查找采样器配置
    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)

    # 获取 eta 参数
    eta = p.eta

    # 如果 eta 为 None 且采样器不为 None，则使用采样器中的 eta 参数
    if eta is None and p.sampler is not None:
        eta = p.sampler.eta

    # 如果 eta 为 None 且采样器配置不为 None，则根据配置设置 eta 参数
    if eta is None and sampler_config is not None:
        eta = 0 if sampler_config.options.get("default_eta_is_0", False) else 1.0

    # 如果 eta 为 0，则返回 False
    if eta == 0:
        return False

    # 返回采样器配置中是否使用 eta 噪声种子增量的信息
    return sampler_config.options.get("uses_ensd", False)


# 定义中断异常类
class InterruptedException(BaseException):
    pass


# 替换 torchsde 中的 brownian_interval 模块
def replace_torchsde_browinan():
    import torchsde._brownian.brownian_interval

    # 定义替换的随机数生成函数
    def torchsde_randn(size, dtype, device, seed):
        return devices.randn_local(seed, size).to(device=device, dtype=dtype)

    # 替换 brownian_interval 模块中的 _randn 函数
    torchsde._brownian.brownian_interval._randn = torchsde_randn


# 应用精化器配置
def apply_refiner(cfg_denoiser):
    # 计算完成比例
    completed_ratio = cfg_denoiser.step / cfg_denoiser.total_steps
    refiner_switch_at = cfg_denoiser.p.refiner_switch_at
    refiner_checkpoint_info = cfg_denoiser.p.refiner_checkpoint_info

    # 如果到达精化器切换比例，则返回 False
    if refiner_switch_at is not None and completed_ratio < refiner_switch_at:
        return False

    # 如果没有精化器检查点信息或者当前检查点信息与共享模型的检查点信息相同，则返回 False
    if refiner_checkpoint_info is None or shared.sd_model.sd_checkpoint_info == refiner_checkpoint_info:
        return False

    # 如果启用高分辨率生成器，则根据配置设置额外生成参数
    if getattr(cfg_denoiser.p, "enable_hr", False):
        is_second_pass = cfg_denoiser.p.is_hr_pass

        if opts.hires_fix_refiner_pass == "first pass" and is_second_pass:
            return False

        if opts.hires_fix_refiner_pass == "second pass" and not is_second_pass:
            return False

        if opts.hires_fix_refiner_pass != "second pass":
            cfg_denoiser.p.extra_generation_params['Hires refiner'] = opts.hires_fix_refiner_pass

    # 设置额外生成参数中的精化器信息和切换比例信息
    cfg_denoiser.p.extra_generation_params['Refiner'] = refiner_checkpoint_info.short_title
    cfg_denoiser.p.extra_generation_params['Refiner switch at'] = refiner_switch_at
    # 使用上下文管理器跳过写入配置的操作
    with sd_models.SkipWritingToConfig():
        # 重新加载模型权重
        sd_models.reload_model_weights(info=refiner_checkpoint_info)

    # 执行 Torch 的垃圾回收
    devices.torch_gc()
    # 配置去噪器的条件
    cfg_denoiser.p.setup_conds()
    # 更新内部模型
    cfg_denoiser.update_inner_model()

    # 返回 True 表示执行成功
    return True
class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, p):
        # 初始化 TorchHijack 类，接收参数 p
        self.rng = p.rng

    def __getattr__(self, item):
        # 如果属性为 'randn_like'，则返回自定义的 randn_like 方法
        if item == 'randn_like':
            return self.randn_like

        # 如果属性存在于 torch 模块中，则返回对应属性
        if hasattr(torch, item):
            return getattr(torch, item)

        # 抛出属性错误
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        # 返回随机数生成器的下一个值
        return self.rng.next()


class Sampler:
    def __init__(self, funcname):
        # 初始化 Sampler 类，接收参数 funcname
        self.funcname = funcname
        self.func = funcname
        self.extra_params = []
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config: SamplerData = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None
        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')
        self.s_noise = 1.0

        self.eta_option_field = 'eta_ancestral'
        self.eta_infotext_field = 'Eta'
        self.eta_default = 1.0

        self.conditioning_key = shared.sd_model.model.conditioning_key

        self.p = None
        self.model_wrap_cfg = None
        self.sampler_extra_args = None
        self.options = {}

    def callback_state(self, d):
        # 获取当前步数
        step = d['i']

        # 如果设置了停止步数，并且当前步数大于停止步数，则抛出中断异常
        if self.stop_at is not None and step > self.stop_at:
            raise InterruptedException

        # 更新状态的采样步数
        state.sampling_step = step
        # 更新总进度条
        shared.total_tqdm.update()
    # 启动采样过程，设置步数和函数
    def launch_sampling(self, steps, func):
        # 设置模型包装配置的步数
        self.model_wrap_cfg.steps = steps
        # 计算总步数
        self.model_wrap_cfg.total_steps = self.config.total_steps(steps)
        # 设置采样步数
        state.sampling_steps = steps
        # 设置采样步骤为0
        state.sampling_step = 0

        try:
            # 调用给定的函数进行采样
            return func()
        except RecursionError:
            # 处理递归错误
            print(
                'Encountered RecursionError during sampling, returning last latent. '
                'rho >5 with a polyexponential scheduler may cause this error. '
                'You should try to use a smaller rho value instead.'
            )
            # 返回上一个潜在值
            return self.last_latent
        except InterruptedException:
            # 处理中断异常
            return self.last_latent

    # 返回所需噪声的数量
    def number_of_needed_noises(self, p):
        return p.steps

    # 创建噪声采样器
    def create_noise_sampler(self, x, sigmas, p):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        # 如果禁用 DPM++ SDE 批次确定性，则返回 None
        if shared.opts.no_dpmpp_sde_batch_determinism:
            return None

        # 导入 BrownianTreeNoiseSampler 类
        from k_diffusion.sampling import BrownianTreeNoiseSampler
        # 计算 sigma 的最小值和最大值
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        # 计算当前迭代的种子
        current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        # 返回 BrownianTreeNoiseSampler 实例
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)

    # 对样本进行采样
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    # 对图像进行图像采样
    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()
```