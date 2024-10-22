# `.\diffusers\schedulers\scheduling_vq_diffusion.py`

```py
# 版权所有 2024 Microsoft 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件在许可证下按“原样”分发，
# 不附有任何形式的保证或条件，无论是明示还是暗示的。
# 有关许可证所适用的权限和限制，请参见许可证。

# 从数据类导入装饰器
from dataclasses import dataclass
# 从类型导入可选类型、元组和联合
from typing import Optional, Tuple, Union

# 导入 NumPy 库并命名为 np
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F

# 从配置实用工具中导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从实用程序中导入 BaseOutput
from ..utils import BaseOutput
# 从调度实用工具中导入 SchedulerMixin
from .scheduling_utils import SchedulerMixin


# 定义调度器输出类，继承自 BaseOutput
@dataclass
class VQDiffusionSchedulerOutput(BaseOutput):
    """
    调度器步骤函数输出的输出类。

    参数：
        prev_sample (`torch.LongTensor`，形状为 `(batch size, num latent pixels)`):
            上一时间步的计算样本 x_{t-1}。`prev_sample` 应用作去噪循环中的下一个模型输入。
    """

    # 上一时间步的样本
    prev_sample: torch.LongTensor


# 将类索引批次转换为批次的对数 onehot 向量
def index_to_log_onehot(x: torch.LongTensor, num_classes: int) -> torch.Tensor:
    """
    将类索引的向量批次转换为对数 onehot 向量批次。

    参数：
        x (`torch.LongTensor`，形状为 `(batch size, vector length)`):
            类索引的批次

        num_classes (`int`):
            用于 onehot 向量的类别数量

    返回：
        `torch.Tensor`，形状为 `(batch size, num classes, vector length)`:
            对数 onehot 向量
    """
    # 将类索引转换为 onehot 向量
    x_onehot = F.one_hot(x, num_classes)
    # 重新排列维度
    x_onehot = x_onehot.permute(0, 2, 1)
    # 计算对数并防止出现无穷大
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    # 返回对数 onehot 向量
    return log_x


# 对 logits 应用 Gumbel 噪声
def gumbel_noised(logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """
    对 `logits` 应用 Gumbel 噪声
    """
    # 生成与 logits 形状相同的均匀随机数
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    # 计算 Gumbel 噪声
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    # 将噪声添加到 logits
    noised = gumbel_noise + logits
    # 返回加噪后的 logits
    return noised


# 定义累积和非累积的 alpha 调度
def alpha_schedules(num_diffusion_timesteps: int, alpha_cum_start=0.99999, alpha_cum_end=0.000009):
    """
    累积和非累积的 alpha 调度。

    请参阅第 4.1 节。
    """
    # 计算 alpha 调度值
    att = (
        np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1) * (alpha_cum_end - alpha_cum_start)
        + alpha_cum_start
    )
    # 在前面添加 1
    att = np.concatenate(([1], att))
    # 计算非累积 alpha
    at = att[1:] / att[:-1]
    # 在最后添加 1
    att = np.concatenate((att[1:], [1]))
    # 返回非累积和累积的 alpha
    return at, att


# 定义累积和非累积的 gamma 调度
def gamma_schedules(num_diffusion_timesteps: int, gamma_cum_start=0.000009, gamma_cum_end=0.99999):
    """
    累积和非累积的 gamma 调度。

    请参阅第 4.1 节。
    """
    # 计算归一化的累积gamma值，范围从gamma_cum_start到gamma_cum_end
        ctt = (
            np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1) * (gamma_cum_end - gamma_cum_start)
            + gamma_cum_start
        )
        # 在计算后的数组前添加0，以形成新数组ctt
        ctt = np.concatenate(([0], ctt))
        # 计算1减去ctt的值
        one_minus_ctt = 1 - ctt
        # 计算一阶差分，即当前值与前一个值的比值
        one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
        # 计算ct的值，即1减去one_minus_ct
        ct = 1 - one_minus_ct
        # 重新组织ctt数组，将最后一个元素替换为0
        ctt = np.concatenate((ctt[1:], [0]))
        # 返回ct和ctt两个数组
        return ct, ctt
# 定义一个用于向量量化扩散调度的类，继承自 SchedulerMixin 和 ConfigMixin
class VQDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    向量量化扩散的调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关库实现的所有调度器的通用方法，请查看超类文档。

    参数：
        num_vec_classes (`int`):
            潜在像素的向量嵌入的类别数量，包括被遮罩潜在像素的类别。
        num_train_timesteps (`int`, defaults to 100):
            训练模型的扩散步骤数量。
        alpha_cum_start (`float`, defaults to 0.99999):
            起始累积 alpha 值。
        alpha_cum_end (`float`, defaults to 0.00009):
            结束累积 alpha 值。
        gamma_cum_start (`float`, defaults to 0.00009):
            起始累积 gamma 值。
        gamma_cum_end (`float`, defaults to 0.99999):
            结束累积 gamma 值。
    """

    # 设置调度器的顺序
    order = 1

    # 注册初始化方法到配置中
    @register_to_config
    def __init__(
        self,
        num_vec_classes: int,  # 向量类别数量
        num_train_timesteps: int = 100,  # 默认训练时间步数
        alpha_cum_start: float = 0.99999,  # 默认起始累积 alpha
        alpha_cum_end: float = 0.000009,  # 默认结束累积 alpha
        gamma_cum_start: float = 0.000009,  # 默认起始累积 gamma
        gamma_cum_end: float = 0.99999,  # 默认结束累积 gamma
    ):
        # 保存向量嵌入类别数量
        self.num_embed = num_vec_classes

        # 根据约定，遮罩类的索引为最后一个类别索引
        self.mask_class = self.num_embed - 1

        # 计算 alpha 和 gamma 的调度值
        at, att = alpha_schedules(num_train_timesteps, alpha_cum_start=alpha_cum_start, alpha_cum_end=alpha_cum_end)
        ct, ctt = gamma_schedules(num_train_timesteps, gamma_cum_start=gamma_cum_start, gamma_cum_end=gamma_cum_end)

        # 计算非遮罩类的数量
        num_non_mask_classes = self.num_embed - 1
        # 计算 bt 和 btt 值
        bt = (1 - at - ct) / num_non_mask_classes
        btt = (1 - att - ctt) / num_non_mask_classes

        # 转换为张量，确保数据类型为 float64
        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)  # 计算 at 的对数
        log_bt = torch.log(bt)  # 计算 bt 的对数
        log_ct = torch.log(ct)  # 计算 ct 的对数

        # 转换 att, btt 和 ctt 为张量
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)  # 计算 att 的对数
        log_cumprod_bt = torch.log(btt)  # 计算 btt 的对数
        log_cumprod_ct = torch.log(ctt)  # 计算 ctt 的对数

        # 将对数值转换为 float 类型并保存
        self.log_at = log_at.float()
        self.log_bt = log_bt.float()
        self.log_ct = log_ct.float()
        self.log_cumprod_at = log_cumprod_at.float()
        self.log_cumprod_bt = log_cumprod_bt.float()
        self.log_cumprod_ct = log_cumprod_ct.float()

        # 可设置的推理步骤数量
        self.num_inference_steps = None
        # 创建时间步张量，逆序排列
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
    # 定义设置离散时间步的函数，用于扩散链（在推断之前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        # 文档字符串，说明函数的参数和用途
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
    
        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps and diffusion process parameters (alpha, beta, gamma) should be moved
                to.
        """
        # 将输入的推断步数赋值给实例变量
        self.num_inference_steps = num_inference_steps
        # 创建一个从 num_inference_steps 到 0 的倒序时间步数组
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        # 将时间步数组转换为 PyTorch 张量并移动到指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
        # 将日志参数移动到指定设备
        self.log_at = self.log_at.to(device)
        self.log_bt = self.log_bt.to(device)
        self.log_ct = self.log_ct.to(device)
        self.log_cumprod_at = self.log_cumprod_at.to(device)
        self.log_cumprod_bt = self.log_cumprod_bt.to(device)
        self.log_cumprod_ct = self.log_cumprod_ct.to(device)
    
    # 定义步骤函数，用于执行单次推断
    def step(
        # 输入的模型输出张量
        model_output: torch.Tensor,
        # 当前的时间步长
        timestep: torch.long,
        # 样本张量
        sample: torch.LongTensor,
        # 可选的随机数生成器
        generator: Optional[torch.Generator] = None,
        # 返回字典的标志
        return_dict: bool = True,
    ) -> Union[VQDiffusionSchedulerOutput, Tuple]:
        """
        从前一个时间步预测样本，通过反向转移分布。有关如何计算分布的更多细节，请参见
        [`~VQDiffusionScheduler.q_posterior`]。

        参数：
            log_p_x_0: (`torch.Tensor`，形状为`(batch size, num classes - 1, num latent pixels)`):
                初始潜在像素预测类别的对数概率。不包括被遮盖类别的预测，因为初始未噪声图像不能被遮盖。
            t (`torch.long`):
                确定使用哪些转移矩阵的时间步。
            x_t (`torch.LongTensor`，形状为`(batch size, num latent pixels)`):
                时间`t`时每个潜在像素的类别。
            generator (`torch.Generator`，或`None`):
                用于在从`p(x_{t-1} | x_t)`中采样之前施加的噪声的随机数生成器。
            return_dict (`bool`，*可选*，默认值为`True`):
                是否返回[`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`]或`tuple`。

        返回：
            [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`]或`tuple`：
                如果`return_dict`为`True`，则返回[`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`]，
                否则返回一个元组，元组的第一个元素是样本张量。
        """
        # 如果时间步为0，则将模型输出赋值给log_p_x_t_min_1
        if timestep == 0:
            log_p_x_t_min_1 = model_output
        else:
            # 使用后验分布函数计算log_p_x_t_min_1
            log_p_x_t_min_1 = self.q_posterior(model_output, sample, timestep)

        # 对log_p_x_t_min_1施加Gumbel噪声
        log_p_x_t_min_1 = gumbel_noised(log_p_x_t_min_1, generator)

        # 找到log_p_x_t_min_1中最大值的索引，得到x_t_min_1
        x_t_min_1 = log_p_x_t_min_1.argmax(dim=1)

        # 如果return_dict为False，则返回x_t_min_1的元组
        if not return_dict:
            return (x_t_min_1,)

        # 返回一个VQDiffusionSchedulerOutput对象，包含前一个样本x_t_min_1
        return VQDiffusionSchedulerOutput(prev_sample=x_t_min_1)

    # 定义一个方法，用于计算从已知类别到已知类别的转移概率的对数
    def log_Q_t_transitioning_to_known_class(
        self, *, t: torch.int, x_t: torch.LongTensor, log_onehot_x_t: torch.Tensor, cumulative: bool
    # 定义一个方法，应用累积转移
    def apply_cumulative_transitions(self, q, t):
        # 获取输入q的批大小
        bsz = q.shape[0]
        # 获取时间步t的对数累积概率
        a = self.log_cumprod_at[t]
        b = self.log_cumprod_bt[t]
        c = self.log_cumprod_ct[t]

        # 获取潜在像素的数量
        num_latent_pixels = q.shape[2]
        # 扩展c的形状以匹配bsz和num_latent_pixels
        c = c.expand(bsz, 1, num_latent_pixels)

        # 将q与a相加并进行logaddexp操作与b合并
        q = (q + a).logaddexp(b)
        # 在最后一个维度上将c与q进行拼接
        q = torch.cat((q, c), dim=1)

        # 返回更新后的q
        return q
```