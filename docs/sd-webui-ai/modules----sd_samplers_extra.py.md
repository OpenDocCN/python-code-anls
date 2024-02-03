# `stable-diffusion-webui\modules\sd_samplers_extra.py`

```
# 导入 torch 库
import torch
# 导入 tqdm 库
import tqdm
# 导入 k_diffusion.sampling 模块
import k_diffusion.sampling

# 使用 torch.no_grad() 装饰器，表示在该函数中不需要计算梯度
def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., restart_list=None):
    """Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    # 如果 extra_args 为 None，则设置为一个空字典
    extra_args = {} if extra_args is None else extra_args
    # 创建一个全为 1 的张量 s_in，与输入张量 x 具有相同的设备
    s_in = x.new_ones([x.shape[0]])
    # 初始化步数为 0
    step_id = 0
    # 从 k_diffusion.sampling 模块导入 to_d 和 get_sigmas_karras 函数

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        # 使用 nonlocal 关键字声明 step_id 变量为外部函数中的变量
        nonlocal step_id
        # 使用模型 model 对输入 x 进行处理，得到去噪后的结果 denoised
        denoised = model(x, old_sigma * s_in, **extra_args)
        # 计算输入 x 和去噪结果 denoised 之间的差异 d
        d = to_d(x, old_sigma, denoised)
        # 如果存在回调函数 callback，则调用回调函数
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        # 计算新旧 sigma 之间的差值 dt
        dt = new_sigma - old_sigma
        # 如果新 sigma 为 0 或者不使用二阶方法
        if new_sigma == 0 or not second_order:
            # 使用欧拉方法更新 x
            x = x + d * dt
        else:
            # 使用 Heun's 方法更新 x
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        # 步数加一
        step_id += 1
        return x

    # 计算 sigma 的步数
    steps = sigmas.shape[0] - 1
    # 如果 restart_list 为 None
    if restart_list is None:
        # 如果步数大于等于 20
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            # 如果步数大于等于 36
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            # 根据步数计算新的 sigma 序列
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device)
            # 设置默认的 restart_list
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            # 如果步数小于 20，则设置 restart_list 为空字典
            restart_list = {}
    # 根据 sigmas 和 key 计算最接近 key 的索引，构建索引到 value 的字典
    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    # 初始化步长列表
    step_list = []
    # 遍历 sigmas 列表
    for i in range(len(sigmas) - 1):
        # 将相邻的 sigmas 组成步长元组，添加到步长列表中
        step_list.append((sigmas[i], sigmas[i + 1]))
        # 如果当前索引需要重启
        if i + 1 in restart_list:
            # 获取重启所需的参数
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            # 计算最小索引和最大索引
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            # 如果最大索引小于最小索引
            if max_idx < min_idx:
                # 根据重启参数获取新的 sigmas 列表
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device)[:-1]
                # 执行重启次数
                while restart_times > 0:
                    restart_times -= 1
                    # 将新的 sigmas 列表添加到步长列表中
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    # 初始化上一个 sigma 为 None
    last_sigma = None
    # 遍历步长列表
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        # 如果上一个 sigma 为 None，则将当前 sigma 赋值给上一个 sigma
        if last_sigma is None:
            last_sigma = old_sigma
        # 如果上一个 sigma 小于当前 sigma
        elif last_sigma < old_sigma:
            # 根据公式更新 x
            x = x + k_diffusion.sampling.torch.randn_like(x) * s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
        # 执行 Heun 步骤
        x = heun_step(x, old_sigma, new_sigma)
        # 更新上一个 sigma
        last_sigma = new_sigma

    # 返回更新后的 x
    return x
```