# `.\cogvideo-finetune\sat\sgm\modules\diffusionmodules\sampling_utils.py`

```py
# 导入 PyTorch 库
import torch
# 从 scipy 导入积分功能
from scipy import integrate

# 从上层模块导入 append_dims 函数
from ...util import append_dims
# 从 einops 导入 rearrange 函数，用于重排张量
from einops import rearrange


# 定义无动态阈值类
class NoDynamicThresholding:
    # 定义调用方法，接受无条件输入、条件输入和缩放因子
    def __call__(self, uncond, cond, scale):
        # 如果 scale 是张量，则调整其维度以匹配条件的维度
        scale = append_dims(scale, cond.ndim) if isinstance(scale, torch.Tensor) else scale
        # 返回无条件输入和缩放后的条件与无条件差值之和
        return uncond + scale * (cond - uncond)


# 定义静态阈值类
class StaticThresholding:
    # 定义调用方法，接受无条件输入、条件输入和缩放因子
    def __call__(self, uncond, cond, scale):
        # 计算无条件输入和缩放后的条件与无条件差值之和
        result = uncond + scale * (cond - uncond)
        # 将结果限制在 -1.0 到 1.0 之间
        result = torch.clamp(result, min=-1.0, max=1.0)
        # 返回处理后的结果
        return result


# 定义动态阈值函数
def dynamic_threshold(x, p=0.95):
    # 获取输入张量的维度
    N, T, C, H, W = x.shape
    # 将张量重排为适合计算的形状
    x = rearrange(x, "n t c h w -> n c (t h w)")
    # 计算给定分位数的左侧和右侧阈值
    l, r = x.quantile(q=torch.tensor([1 - p, p], device=x.device), dim=-1, keepdim=True)
    # 计算阈值的最大值
    s = torch.maximum(-l, r)
    # 创建阈值掩码，用于过滤
    threshold_mask = (s > 1).expand(-1, -1, H * W * T)
    # 如果阈值掩码中有任何 True 值，则应用阈值处理
    if threshold_mask.any():
        x = torch.where(threshold_mask, x.clamp(min=-1 * s, max=s), x)
    # 恢复张量的原始形状
    x = rearrange(x, "n c (t h w) -> n t c h w", t=T, h=H, w=W)
    # 返回处理后的张量
    return x


# 定义第二种动态阈值处理函数
def dynamic_thresholding2(x0):
    p = 0.995  # 参考论文“Imagen”中的超参数
    # 保存输入张量的原始数据类型
    origin_dtype = x0.dtype
    # 将输入张量转换为浮点数
    x0 = x0.to(torch.float32)
    # 计算输入张量绝对值的分位数
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    # 将分位数与 1 进行比较并调整维度
    s = append_dims(torch.maximum(s, torch.ones_like(s).to(s.device)), x0.dim())
    # 限制输入张量的值在 -s 和 s 之间
    x0 = torch.clamp(x0, -s, s)  # / s
    # 返回转换为原始数据类型的张量
    return x0.to(origin_dtype)


# 定义潜在动态阈值处理函数
def latent_dynamic_thresholding(x0):
    p = 0.9995  # 参考论文中的超参数
    # 保存输入张量的原始数据类型
    origin_dtype = x0.dtype
    # 将输入张量转换为浮点数
    x0 = x0.to(torch.float32)
    # 计算输入张量绝对值的分位数
    s = torch.quantile(torch.abs(x0), p, dim=2)
    # 调整分位数的维度以匹配输入张量
    s = append_dims(s, x0.dim())
    # 限制输入张量的值在 -s 和 s 之间，并进行归一化处理
    x0 = torch.clamp(x0, -s, s) / s
    # 返回转换为原始数据类型的张量
    return x0.to(origin_dtype)


# 定义第三种动态阈值处理函数
def dynamic_thresholding3(x0):
    p = 0.995  # 参考论文“Imagen”中的超参数
    # 保存输入张量的原始数据类型
    origin_dtype = x0.dtype
    # 将输入张量转换为浮点数
    x0 = x0.to(torch.float32)
    # 计算输入张量绝对值的分位数
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    # 将分位数与 1 进行比较并调整维度
    s = append_dims(torch.maximum(s, torch.ones_like(s).to(s.device)), x0.dim())
    # 限制输入张量的值在 -s 和 s 之间
    x0 = torch.clamp(x0, -s, s)  # / s
    # 返回转换为原始数据类型的张量
    return x0.to(origin_dtype)


# 定义动态阈值类
class DynamicThresholding:
    # 定义调用方法，接受无条件输入、条件输入和缩放因子
    def __call__(self, uncond, cond, scale):
        # 计算无条件输入的均值和标准差
        mean = uncond.mean()
        std = uncond.std()
        # 计算无条件输入和缩放后的条件与无条件差值之和
        result = uncond + scale * (cond - uncond)
        # 计算结果的均值和标准差
        result_mean, result_std = result.mean(), result.std()
        # 标准化结果，使其具有相同的标准差
        result = (result - result_mean) / result_std * std
        # result = dynamic_thresholding3(result)  # 可选的进一步处理
        # 返回处理后的结果
        return result


# 定义动态阈值版本1类
class DynamicThresholdingV1:
    # 初始化时接收缩放因子
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    # 定义一个函数，接受三个参数 uncond, cond, scale
    def __call__(self, uncond, cond, scale):
        # 计算结果，根据公式 uncond + scale * (cond - uncond)
        result = uncond + scale * (cond - uncond)
        # 对结果进行反缩放，除以缩放因子
        unscaled_result = result / self.scale_factor
        # 获取结果的形状信息，分别为 Batch size, Time steps, Channels, Height, Width
        B, T, C, H, W = unscaled_result.shape
        # 将结果重新排列成 "b t c h w" 的形式
        flattened = rearrange(unscaled_result, "b t c h w -> b c (t h w)")
        # 计算每个通道的均值，并在第二维度上增加一个维度
        means = flattened.mean(dim=2).unsqueeze(2)
        # 对结果进行重新中心化，减去均值
        recentered = flattened - means
        # 计算每个通道的绝对值的最大值
        magnitudes = recentered.abs().max()
        # 对结果进行归一化，除以最大值
        normalized = recentered / magnitudes
        # 对结果进行动态阈值处理
        thresholded = latent_dynamic_thresholding(normalized)
        # 对结果进行反归一化，乘以最大值
        denormalized = thresholded * magnitudes
        # 对结果进行重新中心化，加上均值
        uncentered = denormalized + means
        # 将结果重新排列成 "b c (t h w)" 的形式
        unflattened = rearrange(uncentered, "b c (t h w) -> b t c h w", t=T, h=H, w=W)
        # 对结果进行缩放，乘以缩放因子
        scaled_result = unflattened * self.scale_factor
        # 返回缩放后的结果
        return scaled_result
# 定义一个动态阈值处理类
class DynamicThresholdingV2:
    # 定义类的调用方法，接受无条件值、条件值和缩放比例作为参数
    def __call__(self, uncond, cond, scale):
        # 获取无条件值的形状信息
        B, T, C, H, W = uncond.shape
        # 计算条件值和无条件值的差值
        diff = cond - uncond
        # 计算最小目标值
        mim_target = uncond + diff * 4.0
        # 计算配置目标值
        cfg_target = uncond + diff * 8.0

        # 将最小目标值展平为二维数组
        mim_flattened = rearrange(mim_target, "b t c h w -> b c (t h w)")
        # 将配置目标值展平为二维数组
        cfg_flattened = rearrange(cfg_target, "b t c h w -> b c (t h w)")
        # 计算最小目标值的均值
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        # 计算配置目标值的均值
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        # 计算最小目标值的中心化值
        mim_centered = mim_flattened - mim_means
        # 计算配置目标值的中心化值
        cfg_centered = cfg_flattened - cfg_means

        # 计算最小目标值的标准差
        mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
        # 计算配置目标值的标准差
        cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)

        # 对配置目标值进行重新归一化
        cfg_renormalized = cfg_centered / cfg_scaleref * mim_scaleref

        # 将结果还原为原始形状
        result = cfg_renormalized + cfg_means
        unflattened = rearrange(result, "b c (t h w) -> b t c h w", t=T, h=H, w=W)

        return unflattened


# 定义一个线性多步系数函数
def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    # 如果阶数大于i，则抛出异常
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    # 定义一个内部函数，接受tau作为参数
    def fn(tau):
        prod = 1.0
        # 遍历阶数
        for k in range(order):
            # 如果j等于k，则跳过
            if j == k:
                continue
            # 计算乘积
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    # 对内部函数进行积分
    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


# 定义一个获取祖先步长的函数
def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    # 如果eta为假，则返回sigma_to和0.0
    if not eta:
        return sigma_to, 0.0
    # 计算上行步长
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    # 计算下行步长
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


# 定义一个转换为d的函数
def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


# 定义一个转换为负对数sigma的函数
def to_neg_log_sigma(sigma):
    return sigma.log().neg()


# 定义一个转换为sigma的函数
def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
```