# `.\cogview3-finetune\sat\sgm\modules\diffusionmodules\sampling_utils.py`

```
# 导入数学库以进行数学运算
import math
# 导入 PyTorch 库以进行张量操作
import torch
# 从 SciPy 库导入积分函数
from scipy import integrate

# 从上级目录的 util 模块导入 append_dims 函数
from ...util import append_dims

# 定义一个不进行动态阈值处理的类
class NoDynamicThresholding:
    # 定义类的调用方法
    def __call__(self, uncond, cond, scale, **kwargs):
        # 返回无条件和有条件的加权差值
        return uncond + scale * (cond - uncond)

# 定义一个重新缩放阈值处理的类
class RescaleThresholding:
    # 初始化类，设置 phi 的默认值
    def __init__(self, phi=0.7):
        # 保存 phi 参数
        self.phi = phi

    # 定义类的调用方法
    def __call__(self, uncond, cond, scale, **kwargs):
        # 计算去噪后的配置
        denoised_cfg = uncond + scale * (cond - uncond)
        # 计算条件和去噪配置的标准差
        sigma_pos, sigma_cfg = cond.std(), denoised_cfg.std()
        # 计算缩放因子
        factor = self.phi * sigma_pos / sigma_cfg + (1 - self.phi)
        # 根据因子调整去噪结果
        denoised_final = denoised_cfg * factor
        # 返回最终去噪结果
        return denoised_final

# 定义一个动态阈值处理的类
class DynamicThresholding:
    # 定义可选模式的列表
    Modes = ["Constant", "Linear Up", "Linear Down", "Half Cosine Up", "Half Cosine Down", "Power Up", "Power Down", "Cosine Down","Cosine Up"]
    # 初始化类并设置参数
    def __init__(self, interpret_mode, 
                 scale_min = 3,
                 mimic_interpret_mode = 'Constant',
                 mimic_scale = 3, 
                 mimic_scale_min = 3, 
                 threshold_percentile = 1.0,
                 phi = 1.0,
                 separate_feature_channels = True,
                 measure = 'AD',
                 scaling_startpoint = 'ZERO',
                 ):
        # 验证解释模式是否在可选模式中
        assert interpret_mode in self.Modes
        # 验证模仿解释模式是否在可选模式中
        assert mimic_interpret_mode in self.Modes
        # 验证测量方法是否合法
        assert measure in ['AD', 'STD']
        # 验证缩放起点是否合法
        assert scaling_startpoint in ['ZERO', 'MEAN']
        # 保存各种初始化参数
        self.mode = interpret_mode
        self.mimic_mode = mimic_interpret_mode
        self.scale_min = scale_min
        self.mimic_scale = mimic_scale
        self.mimic_scale_min = mimic_scale_min
        self.threshold_percentile = threshold_percentile
        self.phi = phi
        self.separate_feature_channels = separate_feature_channels
        self.measure = measure
        self.scaling_startpoint = scaling_startpoint
    
    # 定义解释缩放的方法
    def interpret_scale(self, mode, scale, scale_min, step, num_steps):
        """
        num_steps = 50
        step from 0 to 50.
        """
        # 将缩放值减去最小缩放值
        scale -= scale_min
        # 计算当前步骤的比例
        frac = step / num_steps
        # 根据模式调整缩放值
        if mode == 'Constant':
            pass
        elif mode == "Linear Up":
            scale *= frac
        elif mode == "Linear Down":
            scale *= 1.0 - frac
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos(frac)
        elif mode == "Half Cosine Down":
            scale *= math.cos(frac)
        elif mode == "Cosine Down":
            scale *= math.cos(frac * 1.5707)
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos(frac * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(frac, 2.0)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(frac, 2.0)
        # 将调整后的缩放值加回最小缩放值
        scale += scale_min
        # 返回最终的缩放值
        return scale
    # 定义调用方法，接受无条件和条件输入，以及缩放和步骤参数
    def __call__(self, uncond, cond, scale, step, num_steps):
        # 根据当前模式解释缩放参数，计算 cfg_scale
        cfg_scale = self.interpret_scale(self.mode, scale, self.scale_min, step, num_steps)
        # 根据模拟模式解释缩放参数，计算 mimic_cfg_scale
        mimic_cfg_scale = self.interpret_scale(self.mimic_mode, self.mimic_scale, self.mimic_scale_min, step, num_steps)
    
        # 计算 x，作为无条件输入和条件输入之间的线性插值
        x = uncond + cfg_scale*(cond - uncond)
        # 计算 mimic_x，作为无条件输入和条件输入之间的线性插值，使用 mimic_cfg_scale
        mimic_x = uncond + mimic_cfg_scale*(cond - uncond)  
    
        # 将 x 展平，以便于后续操作
        x_flattened = x.flatten(2)
        # 将 mimic_x 展平，以便于后续操作
        mimic_x_flattened = mimic_x.flatten(2)
        
        # 根据缩放起始点的选择，计算均值并中心化
        if self.scaling_startpoint == 'MEAN':
            # 计算 x 的均值，保留维度
            x_means = x_flattened.mean(dim=2, keepdim = True)
            # 计算 mimic_x 的均值，保留维度
            mimic_x_means = mimic_x_flattened.mean(dim=2, keepdim = True)
            # 通过均值中心化 x
            x_centered = x_flattened - x_means
            # 通过均值中心化 mimic_x
            mimic_x_centered = mimic_x_flattened - mimic_x_means
        else:
            # 如果不使用均值中心化，直接赋值
            x_centered = x_flattened
            mimic_x_centered = mimic_x_flattened
                
        # 根据是否分开特征通道的选项，计算尺度参考
        if self.separate_feature_channels:
            # 如果测量方式为绝对差异
            if self.measure == 'AD':
                # 计算 x 的绝对差异的分位数作为尺度参考
                x_scaleref = torch.quantile(x_centered.abs(), self.threshold_percentile, dim=2, keepdim = True)
                # 计算 mimic_x 的绝对差异的最大值作为尺度参考
                mimic_x_scaleref = mimic_x_centered.abs().max(dim=2, keepdim = True).values
            # 如果测量方式为标准差
            elif self.measure == 'STD':
                # 计算 x 的标准差作为尺度参考
                x_scaleref = x_centered.std(dim=2, keepdim = True)
                # 计算 mimic_x 的标准差作为尺度参考
                mimic_x_scaleref = mimic_x_centered.std(dim=2, keepdim = True)
        else:
            # 如果不分开特征通道
            if self.measure == 'AD':
                # 计算 x 的绝对差异的分位数作为尺度参考
                x_scaleref = torch.quantile(x_centered.abs(), self.threshold_percentile)
                # 计算 mimic_x 的绝对差异的最大值作为尺度参考
                mimic_x_scaleref = mimic_x_centered.abs().max()
            # 如果测量方式为标准差
            elif self.measure == 'STD':
                # 计算 x 的标准差作为尺度参考
                x_scaleref = x_centered.std()
                # 计算 mimic_x 的标准差作为尺度参考
                mimic_x_scaleref = mimic_x_centered.std()
            
        # 根据测量方式调整 x 的尺度
        if self.measure == 'AD':
            # 计算 x_scaleref 和 mimic_x_scaleref 的最大值
            max_scaleref = torch.maximum(x_scaleref, mimic_x_scaleref)
            # 限制 x_centered 的值在 [-max_scaleref, max_scaleref] 范围内
            x_clamped = x_centered.clamp(-max_scaleref, max_scaleref)
            # 重新归一化 x
            x_renormed = x_clamped * (mimic_x_scaleref / max_scaleref)
        elif self.measure == 'STD':
            # 重新归一化 x
            x_renormed = x_centered * (mimic_x_scaleref / x_scaleref)
        
        # 根据缩放起始点选择调整 x_dyn 的值
        if self.scaling_startpoint == 'MEAN':
            # 将均值与重新归一化的结果相加
            x_dyn = x_means + x_renormed
        else:
            # 直接使用重新归一化的结果
            x_dyn = x_renormed
        # 反展平 x_dyn，恢复原始形状
        x_dyn = x_dyn.unflatten(2, x.shape[2:])
        # 返回加权和结果
        return self.phi*x_dyn + (1-self.phi)*x
# 定义一个线性多步系数函数，接受阶数、时间点和索引等参数
def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    # 检查阶数是否超出当前步骤的限制，超出则抛出错误
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    # 定义内部函数，用于计算特定tau下的乘积
    def fn(tau):
        prod = 1.0  # 初始化乘积为1
        # 遍历从0到阶数的每个k
        for k in range(order):
            # 跳过与j相等的k
            if j == k:
                continue
            # 计算乘积，基于给定的tau和时间点
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod  # 返回计算得到的乘积

    # 使用数值积分计算fn在时间点[i]到[i+1]之间的积分，返回积分值
    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


# 定义一个函数，用于计算祖先步骤的sigma值
def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    # 如果eta为0，则返回sigma_to和0.0
    if not eta:
        return sigma_to, 0.0
    # 计算sigma_up，限制为sigma_to与特定表达式的较小值
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    # 计算sigma_down，基于sigma_to和sigma_up
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    # 返回计算得到的sigma_down和sigma_up
    return sigma_down, sigma_up


# 定义一个函数，将去噪后的图像与输入图像进行处理，返回标准化结果
def to_d(x, sigma, denoised):
    # 计算去噪后的结果与输入的差异，并除以sigma的扩展维度
    return (x - denoised) / append_dims(sigma, x.ndim)


# 定义一个函数，将sigma转换为负对数形式
def to_neg_log_sigma(sigma):
    # 计算sigma的对数并取负值
    return sigma.log().neg()


# 定义一个函数，将负对数sigma转换为sigma
def to_sigma(neg_log_sigma):
    # 取负值并计算指数，得到sigma
    return neg_log_sigma.neg().exp()
```