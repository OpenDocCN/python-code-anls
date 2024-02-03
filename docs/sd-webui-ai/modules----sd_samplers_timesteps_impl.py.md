# `stable-diffusion-webui\modules\sd_samplers_timesteps_impl.py`

```
# 导入 torch 库
import torch
# 导入 tqdm 库
import tqdm
# 导入 k_diffusion.sampling 模块
import k_diffusion.sampling
# 导入 numpy 库
import numpy as np

# 从 modules 模块中导入 shared
from modules import shared
# 从 modules.models.diffusion.uni_pc 模块中导入 uni_pc

# 使用 torch.no_grad() 装饰器，表示不需要计算梯度
@torch.no_grad()
# 定义函数 ddim，接受模型、输入数据 x、时间步长 timesteps、额外参数 extra_args、回调函数 callback、禁用 disable、eta 参数
def ddim(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    # 获取模型内部的 alphas_cumprod
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
    # 根据时间步长获取 alphas
    alphas = alphas_cumprod[timesteps]
    # 根据时间步长获取 alphas_prev
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(torch.float64 if x.device.type != 'mps' and x.device.type != 'xpu' else torch.float32)
    # 计算 sigmas
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

    # 如果 extra_args 为 None，则设置为空字典
    extra_args = {} if extra_args is None else extra_args
    # 初始化 s_in 和 s_x
    s_in = x.new_ones((x.shape[0]))
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    # 遍历时间步长
    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        # 计算索引
        index = len(timesteps) - 1 - i

        # 获取模型输出 e_t
        e_t = model(x, timesteps[index].item() * s_in, **extra_args)

        # 计算 a_t、a_prev、sigma_t、sqrt_one_minus_at
        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sigma_t = sigmas[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        # 计算预测值 pred_x0、方向 dir_xt、噪声 noise、更新 x
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * k_diffusion.sampling.torch.randn_like(x)
        x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # 如果存在回调函数，则调用回调函数
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    # 返回更新后的 x
    return x

# 使用 torch.no_grad() 装饰器，表示不需要计算梯度
@torch.no_grad()
# 定义函数 plms，接受模型、输入数据 x、时间步长 timesteps、额外参数 extra_args、回调函数 callback、禁用 disable 参数
def plms(model, x, timesteps, extra_args=None, callback=None, disable=None):
    # 获取模型内部的 alphas_cumprod
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
    # 根据时间步长获取 alphas
    alphas = alphas_cumprod[timesteps]
    # 根据时间步长获取 alphas_prev
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(torch.float64 if x.device.type != 'mps' and x.device.type != 'xpu' else torch.float32)
    # 计算 alphas 对应的 1 - alphas 的平方根
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)

    # 如果额外参数为 None，则设置为空字典，否则使用传入的额外参数
    extra_args = {} if extra_args is None else extra_args
    # 创建与输入张量 x 形状相同的全为 1 的张量 s_in
    s_in = x.new_ones([x.shape[0]])
    # 创建与输入张量 x 形状相同的全为 1 的张量 s_x
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    # 存储旧的 epsilon 值的列表
    old_eps = []

    # 定义函数，用于获取 x_prev 和 pred_x0
    def get_x_prev_and_pred_x0(e_t, index):
        # 选择与当前考虑的时间步对应的参数
        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        # 计算 x_0 的当前预测值
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # 指向 x_t 的方向
        dir_xt = (1. - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, pred_x0
    # 使用 tqdm.trange 迭代器遍历时间步长列表，从最后一个时间步开始，到第一个时间步结束
    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        # 计算当前时间步的索引
        index = len(timesteps) - 1 - i
        # 获取当前时间步和下一个时间步的时间值
        ts = timesteps[index].item() * s_in
        t_next = timesteps[max(index - 1, 0)].item() * s_in

        # 使用模型预测当前时间步的值
        e_t = model(x, ts, **extra_args)

        # 根据历史预测值计算当前时间步的估计值
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = model(x_prev, t_next, **extra_args)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        else:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        # 获取当前时间步的预测值和真实值
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        # 将当前时间步的预测值加入历史预测值列表
        old_eps.append(e_t)
        # 如果历史预测值列表长度超过4，则移除最早的预测值
        if len(old_eps) >= 4:
            old_eps.pop(0)

        # 更新当前时间步的值
        x = x_prev

        # 如果存在回调函数，则调用回调函数
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    # 返回最终的结果值
    return x
# 定义 UniPCCFG 类，继承自 uni_pc.UniPC 类
class UniPCCFG(uni_pc.UniPC):
    # 初始化方法，接受 cfg_model、extra_args、callback 等参数
    def __init__(self, cfg_model, extra_args, callback, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(None, *args, **kwargs)

        # 定义一个在更新后执行的函数
        def after_update(x, model_x):
            # 调用回调函数，传入相关参数
            callback({'x': x, 'i': self.index, 'sigma': 0, 'sigma_hat': 0, 'denoised': model_x})
            # 更新索引
            self.index += 1

        # 初始化实例变量
        self.cfg_model = cfg_model
        self.extra_args = extra_args
        self.callback = callback
        self.index = 0
        self.after_update = after_update

    # 获取模型输入时间
    def get_model_input_time(self, t_continuous):
        return (t_continuous - 1. / self.noise_schedule.total_N) * 1000.

    # 模型方法，接受输入 x 和时间 t
    def model(self, x, t):
        # 获取模型输入时间
        t_input = self.get_model_input_time(t)

        # 调用 cfg_model 方法，传入相关参数
        res = self.cfg_model(x, t_input, **self.extra_args)

        return res


# 定义 unipc 函数，接受 model、x、timesteps 等参数
def unipc(model, x, timesteps, extra_args=None, callback=None, disable=None, is_img2img=False):
    # 获取模型内部的 alphas_cumprod 属性
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod

    # 创建 NoiseScheduleVP 实例
    ns = uni_pc.NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
    # 计算 t_start 值
    t_start = timesteps[-1] / 1000 + 1 / 1000 if is_img2img else None  # this is likely off by a bit - if someone wants to fix it please by all means
    # 创建 UniPCCFG 实例
    unipc_sampler = UniPCCFG(model, extra_args, callback, ns, predict_x0=True, thresholding=False, variant=shared.opts.uni_pc_variant)
    # 调用 unipc_sampler 的 sample 方法，传入相关参数
    x = unipc_sampler.sample(x, steps=len(timesteps), t_start=t_start, skip_type=shared.opts.uni_pc_skip_type, method="multistep", order=shared.opts.uni_pc_order, lower_order_final=shared.opts.uni_pc_lower_order_final)

    return x
```