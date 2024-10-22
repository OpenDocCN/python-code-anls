# `.\diffusers\pipelines\shap_e\renderer.py`

```py
# 版权信息，声明此代码的版权归 Open AI 和 HuggingFace 团队所有
# 
# 根据 Apache License 2.0 版本（“许可证”）进行许可；
# 您不得在不遵守许可证的情况下使用此文件。
# 您可以在以下网址获得许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，根据许可证分发的软件是在“按现状”基础上分发的，
# 不附带任何明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和限制的具体说明。

# 导入数学模块
import math
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入字典、可选类型和元组
from typing import Dict, Optional, Tuple

# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch
# 导入 PyTorch 的功能模块
import torch.nn.functional as F
# 从 torch 库导入 nn 模块
from torch import nn

# 从配置工具导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从模型模块导入 ModelMixin
from ...models import ModelMixin
# 从工具模块导入 BaseOutput
from ...utils import BaseOutput
# 从当前目录的 camera 模块导入 create_pan_cameras 函数
from .camera import create_pan_cameras


def sample_pmf(pmf: torch.Tensor, n_samples: int) -> torch.Tensor:
    r"""
    从给定的离散概率分布中进行有替换的采样。

    假定第 i 个箱子的质量为 pmf[i]。

    参数：
        pmf: [batch_size, *shape, n_samples, 1]，其中 (pmf.sum(dim=-2) == 1).all()
        n_samples: 采样的数量

    返回：
        用替换方式采样的索引
    """

    # 获取 pmf 的形状，并提取支持大小和最后一个维度
    *shape, support_size, last_dim = pmf.shape
    # 确保最后一个维度为 1
    assert last_dim == 1

    # 计算 pmf 的累积分布函数（CDF）
    cdf = torch.cumsum(pmf.view(-1, support_size), dim=1)
    # 在 CDF 中查找随机数的索引
    inds = torch.searchsorted(cdf, torch.rand(cdf.shape[0], n_samples, device=cdf.device))

    # 返回形状调整后的索引，并限制在有效范围内
    return inds.view(*shape, n_samples, 1).clamp(0, support_size - 1)


def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    按照 NeRF 的方式将 x 及其位置编码进行连接。

    参考文献： https://arxiv.org/pdf/2210.04628.pdf
    """
    # 如果最小和最大角度相同，则直接返回 x
    if min_deg == max_deg:
        return x

    # 生成尺度，范围为 [min_deg, max_deg)
    scales = 2.0 ** torch.arange(min_deg, max_deg, dtype=x.dtype, device=x.device)
    # 获取 x 的形状和维度
    *shape, dim = x.shape
    # 将 x 重新形状并与尺度相乘，然后调整形状
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    # 确保 xb 的最后一个维度与预期相符
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    # 计算正弦值并进行连接
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    # 返回原始 x 和位置编码的连接
    return torch.cat([x, emb], dim=-1)


def encode_position(position):
    # 使用 posenc_nerf 函数对位置进行编码
    return posenc_nerf(position, min_deg=0, max_deg=15)


def encode_direction(position, direction=None):
    # 如果未提供方向，则返回与位置编码相同形状的零张量
    if direction is None:
        return torch.zeros_like(posenc_nerf(position, min_deg=0, max_deg=8))
    else:
        # 使用 posenc_nerf 函数对方向进行编码
        return posenc_nerf(direction, min_deg=0, max_deg=8)


def _sanitize_name(x: str) -> str:
    # 替换字符串中的点为双下划线
    return x.replace(".", "__")


def integrate_samples(volume_range, ts, density, channels):
    r"""
    集成模型输出的函数。

    参数：
        volume_range: 指定积分范围 [t0, t1]
        ts: 时间步
        density: torch.Tensor [batch_size, *shape, n_samples, 1]
        channels: torch.Tensor [batch_size, *shape, n_samples, n_channels]
    # 返回值说明
    returns:
        # channels: 集成的 RGB 输出权重，类型为 torch.Tensor，形状为 [batch_size, *shape, n_samples, 1] 
        # (density * transmittance)[i] 表示在 [...] 中每个 RGB 输出的权重 
        # transmittance: 表示此体积的透射率
    )

    # 1. 计算权重
    # 使用 volume_range 对象的 partition 方法划分 ts，得到三个输出值，前两个被忽略
    _, _, dt = volume_range.partition(ts)
    # 计算密度与时间间隔 dt 的乘积，得到每个体素的密度变化
    ddensity = density * dt

    # 对 ddensity 进行累加，计算质量随深度的变化
    mass = torch.cumsum(ddensity, dim=-2)
    # 计算体积的透射率，使用指数衰减公式
    transmittance = torch.exp(-mass[..., -1, :])

    # 计算 alpha 值，表示每个体素的透明度
    alphas = 1.0 - torch.exp(-ddensity)
    # 计算 T 值，表示光通过每个体素的概率，使用累积质量
    Ts = torch.exp(torch.cat([torch.zeros_like(mass[..., :1, :]), -mass[..., :-1, :]], dim=-2))
    # 这是光在深度 [..., i, :] 上击中并反射的概率
    weights = alphas * Ts

    # 2. 集成通道
    # 计算每个通道的加权和，得到最终的 RGB 输出
    channels = torch.sum(channels * weights, dim=-2)

    # 返回计算得到的通道、权重和透射率
    return channels, weights, transmittance
# 定义一个函数，查询体积内的点坐标
def volume_query_points(volume, grid_size):
    # 创建一个张量，包含从0到grid_size^3-1的索引，设备为volume的最小边界设备
    indices = torch.arange(grid_size**3, device=volume.bbox_min.device)
    # 计算每个索引在grid_size维度上的z坐标
    zs = indices % grid_size
    # 计算每个索引在grid_size维度上的y坐标
    ys = torch.div(indices, grid_size, rounding_mode="trunc") % grid_size
    # 计算每个索引在grid_size维度上的x坐标
    xs = torch.div(indices, grid_size**2, rounding_mode="trunc") % grid_size
    # 将x, y, z坐标组合成一个张量，维度为(数量, 3)
    combined = torch.stack([xs, ys, zs], dim=1)
    # 归一化坐标并转换为体积的坐标范围
    return (combined.float() / (grid_size - 1)) * (volume.bbox_max - volume.bbox_min) + volume.bbox_min


# 定义一个函数，将sRGB颜色值转换为线性颜色值
def _convert_srgb_to_linear(u: torch.Tensor):
    # 使用条件语句，按照sRGB到线性空间的转换公式进行转换
    return torch.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


# 定义一个函数，创建平面边缘索引
def _create_flat_edge_indices(
    flat_cube_indices: torch.Tensor,
    grid_size: Tuple[int, int, int],
):
    # 计算在x方向上的索引数量
    num_xs = (grid_size[0] - 1) * grid_size[1] * grid_size[2]
    # 计算y方向的偏移量
    y_offset = num_xs
    # 计算在y方向上的索引数量
    num_ys = grid_size[0] * (grid_size[1] - 1) * grid_size[2]
    # 计算z方向的偏移量
    z_offset = num_xs + num_ys
    # 将一组张量堆叠成一个新的张量，指定最后一个维度
    return torch.stack(
        [
            # 表示跨越 x 轴的边
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]  # 计算 x 轴索引的基值
            + flat_cube_indices[:, 1] * grid_size[2]  # 加上 y 轴索引的偏移
            + flat_cube_indices[:, 2],  # 加上 z 轴索引
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]  # 计算 x 轴索引的基值
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]  # 加上 y 轴索引偏移（+1）
            + flat_cube_indices[:, 2],  # 加上 z 轴索引
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]  # 计算 x 轴索引的基值
            + flat_cube_indices[:, 1] * grid_size[2]  # 加上 y 轴索引的偏移
            + flat_cube_indices[:, 2]  # 加上 z 轴索引
            + 1,  # 取下一个 z 轴的索引
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]  # 计算 x 轴索引的基值
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]  # 加上 y 轴索引偏移（+1）
            + flat_cube_indices[:, 2]  # 加上 z 轴索引
            + 1,  # 取下一个 z 轴的索引
            # 表示跨越 y 轴的边
            (
                y_offset  # y 轴的偏移量
                + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]  # 计算 x 轴索引的基值
                + flat_cube_indices[:, 1] * grid_size[2]  # 加上 y 轴索引的偏移
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
            ),
            (
                y_offset  # y 轴的偏移量
                + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]  # 计算 x 轴索引的基值（+1）
                + flat_cube_indices[:, 1] * grid_size[2]  # 加上 y 轴索引的偏移
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
            ),
            (
                y_offset  # y 轴的偏移量
                + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]  # 计算 x 轴索引的基值
                + flat_cube_indices[:, 1] * grid_size[2]  # 加上 y 轴索引的偏移
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
                + 1  # 取下一个 z 轴的索引
            ),
            (
                y_offset  # y 轴的偏移量
                + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]  # 计算 x 轴索引的基值（+1）
                + flat_cube_indices[:, 1] * grid_size[2]  # 加上 y 轴索引的偏移
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
                + 1  # 取下一个 z 轴的索引
            ),
            # 表示跨越 z 轴的边
            (
                z_offset  # z 轴的偏移量
                + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)  # 计算 x 轴索引的基值
                + flat_cube_indices[:, 1] * (grid_size[2] - 1)  # 加上 y 轴索引的偏移
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
            ),
            (
                z_offset  # z 轴的偏移量
                + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)  # 计算 x 轴索引的基值（+1）
                + flat_cube_indices[:, 1] * (grid_size[2] - 1)  # 加上 y 轴索引的偏移
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
            ),
            (
                z_offset  # z 轴的偏移量
                + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)  # 计算 x 轴索引的基值
                + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)  # 加上 y 轴索引的偏移（+1）
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
            ),
            (
                z_offset  # z 轴的偏移量
                + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)  # 计算 x 轴索引的基值（+1）
                + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)  # 加上 y 轴索引的偏移（+1）
                + flat_cube_indices[:, 2]  # 加上 z 轴索引
            ),
        ],
        dim=-1,  # 指定堆叠的维度
    )
# 定义一个名为 VoidNeRFModel 的类，继承自 nn.Module
class VoidNeRFModel(nn.Module):
    """
    实现默认的空空间模型，所有查询渲染为背景。
    """

    # 初始化方法，接收背景和通道缩放参数
    def __init__(self, background, channel_scale=255.0):
        # 调用父类的初始化方法
        super().__init__()
        # 将背景数据转换为张量并归一化
        background = nn.Parameter(torch.from_numpy(np.array(background)).to(dtype=torch.float32) / channel_scale)
        # 注册背景为模型的缓冲区
        self.register_buffer("background", background)

    # 前向传播方法，接收位置参数
    def forward(self, position):
        # 将背景张量扩展到与输入位置相同的设备
        background = self.background[None].to(position.device)
        # 获取位置的形状，除去最后一维
        shape = position.shape[:-1]
        # 创建一个与 shape 维度相同的 ones 列表
        ones = [1] * (len(shape) - 1)
        # 获取背景的通道数
        n_channels = background.shape[-1]
        # 将背景张量广播到与位置相同的形状
        background = torch.broadcast_to(background.view(background.shape[0], *ones, n_channels), [*shape, n_channels])
        # 返回背景张量
        return background


@dataclass
# 定义一个数据类 VolumeRange，包含 t0、t1 和 intersected
class VolumeRange:
    t0: torch.Tensor
    t1: torch.Tensor
    intersected: torch.Tensor

    # 后置初始化方法，检查张量形状是否一致
    def __post_init__(self):
        assert self.t0.shape == self.t1.shape == self.intersected.shape

    # 分区方法，将 t0 和 t1 分成 n_samples 区间
    def partition(self, ts):
        """
        将 t0 和 t1 分区成 n_samples 区间。

        参数:
            ts: [batch_size, *shape, n_samples, 1]

        返回:
            lower: [batch_size, *shape, n_samples, 1] upper: [batch_size, *shape, n_samples, 1] delta: [batch_size,
            *shape, n_samples, 1]

        其中
            ts \\in [lower, upper] deltas = upper - lower
        """

        # 计算 ts 的中间值
        mids = (ts[..., 1:, :] + ts[..., :-1, :]) * 0.5
        # 将 t0 和中间值拼接形成 lower
        lower = torch.cat([self.t0[..., None, :], mids], dim=-2)
        # 将中间值和 t1 拼接形成 upper
        upper = torch.cat([mids, self.t1[..., None, :]], dim=-2)
        # 计算 upper 和 lower 之间的差值
        delta = upper - lower
        # 确保 lower、upper 和 delta 的形状一致
        assert lower.shape == upper.shape == delta.shape == ts.shape
        # 返回 lower、upper 和 delta
        return lower, upper, delta


# 定义一个名为 BoundingBoxVolume 的类，继承自 nn.Module
class BoundingBoxVolume(nn.Module):
    """
    由两个对角点定义的轴对齐边界框。
    """

    # 初始化方法，接收边界框的最小和最大角点
    def __init__(
        self,
        *,
        bbox_min,
        bbox_max,
        min_dist: float = 0.0,
        min_t_range: float = 1e-3,
    ):
        """
        参数:
            bbox_min: 边界框的左/底角点
            bbox_max: 边界框的另一角点
            min_dist: 所有光线应至少从该距离开始。
        """
        # 调用父类的初始化方法
        super().__init__()
        # 保存最小距离和最小 t 范围
        self.min_dist = min_dist
        self.min_t_range = min_t_range
        # 将最小和最大边界框角点转换为张量
        self.bbox_min = torch.tensor(bbox_min)
        self.bbox_max = torch.tensor(bbox_max)
        # 堆叠最小和最大角点形成边界框
        self.bbox = torch.stack([self.bbox_min, self.bbox_max])
        # 确保边界框形状正确
        assert self.bbox.shape == (2, 3)
        # 确保最小距离和最小 t 范围有效
        assert min_dist >= 0.0
        assert min_t_range > 0.0

    # 相交方法，接收光线的原点和方向
    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        epsilon=1e-6,
    # 定义文档字符串，描述函数参数和返回值的格式
        ):
            """
            Args:
                origin: [batch_size, *shape, 3] 原点坐标的张量，表示光线起始点
                direction: [batch_size, *shape, 3] 方向向量的张量，表示光线方向
                t0_lower: Optional [batch_size, *shape, 1] 可选参数，表示相交体积时 t0 的下界
                params: Optional meta parameters in case Volume is parametric 可选元参数，用于参数化体积
                epsilon: to stabilize calculations 计算时用于稳定的小常数
    
            Return:
                A tuple of (t0, t1, intersected) 返回一个元组，包含 t0, t1 和交集信息
            """
    
            # 获取 origin 张量的 batch_size 和形状，忽略最后一个维度
            batch_size, *shape, _ = origin.shape
            # 创建一个与 shape 长度相同的列表，填充 1
            ones = [1] * len(shape)
            # 将边界框转换为与 origin 设备相同的张量，形状为 [1, *ones, 2, 3]
            bbox = self.bbox.view(1, *ones, 2, 3).to(origin.device)
    
            # 定义安全除法函数，避免除以零的情况
            def _safe_divide(a, b, epsilon=1e-6):
                return a / torch.where(b < 0, b - epsilon, b + epsilon)
    
            # 计算 t 的值，表示光线与边界框的交点
            ts = _safe_divide(bbox - origin[..., None, :], direction[..., None, :], epsilon=epsilon)
    
            # 考虑光线与边界框相交的不同情况
            #
            #   1. t1 <= t0: 光线未通过 AABB。
            #   2. t0 < t1 <= 0: 光线相交，但边界框在原点后面。
            #   3. t0 <= 0 <= t1: 光线从边界框内部开始。
            #   4. 0 <= t0 < t1: 光线不在内部并且与边界框相交两次。
            #
            # 情况 1 和 4 已通过 t0 < t1 处理。
            # 通过将 t0 至少设为 min_dist (>= 0) 处理情况 2 和 3。
            t0 = ts.min(dim=-2).values.max(dim=-1, keepdim=True).values.clamp(self.min_dist)
            # 计算 t1，取 ts 的最大值
            t1 = ts.max(dim=-2).values.min(dim=-1, keepdim=True).values
            # 断言 t0 和 t1 的形状相同
            assert t0.shape == t1.shape == (batch_size, *shape, 1)
            # 如果 t0_lower 不为空，则取 t0 和 t0_lower 的最大值
            if t0_lower is not None:
                assert t0.shape == t0_lower.shape
                t0 = torch.maximum(t0, t0_lower)
    
            # 计算光线是否与体积相交
            intersected = t0 + self.min_t_range < t1
            # 如果相交，保持 t0 否则设为零
            t0 = torch.where(intersected, t0, torch.zeros_like(t0))
            # 如果相交，保持 t1 否则设为一
            t1 = torch.where(intersected, t1, torch.ones_like(t1))
    
            # 返回一个包含 t0, t1 和相交信息的 VolumeRange 对象
            return VolumeRange(t0=t0, t1=t1, intersected=intersected)
# 定义一个分层射线采样器类，继承自 nn.Module
class StratifiedRaySampler(nn.Module):
    """
    在每个间隔内随机均匀地抽样，而不是使用固定的间隔。
    """

    # 初始化方法，接受一个深度模式参数，默认为线性
    def __init__(self, depth_mode: str = "linear"):
        """
        :param depth_mode: 线性样本在深度上线性分布。谐波模式确保
            更靠近的点被更密集地采样。
        """
        # 保存深度模式参数
        self.depth_mode = depth_mode
        # 确保深度模式是允许的选项之一
        assert self.depth_mode in ("linear", "geometric", "harmonic")

    # 定义采样方法
    def sample(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        n_samples: int,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        """
        Args:
            t0: 开始时间，形状为 [batch_size, *shape, 1]
            t1: 结束时间，形状为 [batch_size, *shape, 1]
            n_samples: 要采样的时间戳数量
        Return:
            采样的时间戳，形状为 [batch_size, *shape, n_samples, 1]
        """
        # 创建一个列表，长度为 t0 形状的维度减一，元素全为 1
        ones = [1] * (len(t0.shape) - 1)
        # 创建从 0 到 1 的线性间隔，并调整形状以适应 t0 的维度
        ts = torch.linspace(0, 1, n_samples).view(*ones, n_samples).to(t0.dtype).to(t0.device)

        # 根据深度模式计算时间戳
        if self.depth_mode == "linear":
            # 线性插值计算时间戳
            ts = t0 * (1.0 - ts) + t1 * ts
        elif self.depth_mode == "geometric":
            # 对数插值计算时间戳，使用 clamp 限制最小值
            ts = (t0.clamp(epsilon).log() * (1.0 - ts) + t1.clamp(epsilon).log() * ts).exp()
        elif self.depth_mode == "harmonic":
            # 原始 NeRF 推荐的插值方案，适用于球形场景
            ts = 1.0 / (1.0 / t0.clamp(epsilon) * (1.0 - ts) + 1.0 / t1.clamp(epsilon) * ts)

        # 计算中间时间戳
        mids = 0.5 * (ts[..., 1:] + ts[..., :-1])
        # 创建上界和下界，分别为中间时间戳和结束时间、开始时间
        upper = torch.cat([mids, t1], dim=-1)
        lower = torch.cat([t0, mids], dim=-1)
        # yiyi 注释：这里添加一个随机种子以便于测试，记得在生产中移除
        torch.manual_seed(0)
        # 生成与 ts 形状相同的随机数
        t_rand = torch.rand_like(ts)

        # 根据随机数计算最终的时间戳
        ts = lower + (upper - lower) * t_rand
        # 返回增加了一个维度的时间戳
        return ts.unsqueeze(-1)


# 定义一个重要性射线采样器类，继承自 nn.Module
class ImportanceRaySampler(nn.Module):
    """
    根据初始密度估计，从预期有物体的区域/箱中进行更多采样。
    """

    # 初始化方法，接受多个参数
    def __init__(
        self,
        volume_range: VolumeRange,
        ts: torch.Tensor,
        weights: torch.Tensor,
        blur_pool: bool = False,
        alpha: float = 1e-5,
    ):
        """
        Args:
            volume_range: 射线与给定体积相交的范围。
            ts: 来自粗渲染步骤的早期采样
            weights: 密度 * 透射率的离散版本
            blur_pool: 如果为真，则使用来自 mip-NeRF 的 2-tap 最大 + 2-tap 模糊滤波器。
            alpha: 添加到权重的小值。
        """
        # 保存体积范围
        self.volume_range = volume_range
        # 克隆并分离传入的时间戳
        self.ts = ts.clone().detach()
        # 克隆并分离传入的权重
        self.weights = weights.clone().detach()
        # 保存是否使用模糊池的标志
        self.blur_pool = blur_pool
        # 保存 alpha 参数
        self.alpha = alpha

    # 标记方法为不需要计算梯度
    @torch.no_grad()
    # 定义一个名为 sample 的方法，接受两个张量 t0 和 t1 以及样本数量 n_samples
    def sample(self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Args:
            t0: start time has shape [batch_size, *shape, 1]
            t1: finish time has shape [batch_size, *shape, 1]
            n_samples: number of ts to sample
        Return:
            sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        # 从 volume_range 获取 t 的范围，分割成 lower 和 upper
        lower, upper, _ = self.volume_range.partition(self.ts)

        # 获取输入张量 ts 的批大小、形状和粗样本数量
        batch_size, *shape, n_coarse_samples, _ = self.ts.shape

        # 获取权重
        weights = self.weights
        # 如果启用了模糊池，进行权重的处理
        if self.blur_pool:
            # 在权重的前后各添加一层，以便进行边界处理
            padded = torch.cat([weights[..., :1, :], weights, weights[..., -1:, :]], dim=-2)
            # 计算相邻权重的最大值
            maxes = torch.maximum(padded[..., :-1, :], padded[..., 1:, :])
            # 更新权重为相邻最大值的平均
            weights = 0.5 * (maxes[..., :-1, :] + maxes[..., 1:, :])
        # 在权重上加上 alpha 值
        weights = weights + self.alpha
        # 计算权重的概率质量函数 (pmf)
        pmf = weights / weights.sum(dim=-2, keepdim=True)
        # 根据 pmf 进行采样，获取样本索引
        inds = sample_pmf(pmf, n_samples)
        # 确保索引的形状符合预期
        assert inds.shape == (batch_size, *shape, n_samples, 1)
        # 确保索引在有效范围内
        assert (inds >= 0).all() and (inds < n_coarse_samples).all()

        # 生成与索引形状相同的随机数
        t_rand = torch.rand(inds.shape, device=inds.device)
        # 根据索引从 lower 和 upper 中获取对应的值
        lower_ = torch.gather(lower, -2, inds)
        upper_ = torch.gather(upper, -2, inds)

        # 根据随机数和上下限计算采样时间
        ts = lower_ + (upper_ - lower_) * t_rand
        # 对采样结果进行排序
        ts = torch.sort(ts, dim=-2).values
        # 返回采样后的时间序列
        return ts
# 定义一个数据类，用于存储三维三角网格及其可选的顶点和面数据
@dataclass
class MeshDecoderOutput(BaseOutput):
    """
    A 3D triangle mesh with optional data at the vertices and faces.

    Args:
        verts (`torch.Tensor` of shape `(N, 3)`):
            array of vertext coordinates
        faces (`torch.Tensor` of shape `(N, 3)`):
            array of triangles, pointing to indices in verts.
        vertext_channels (Dict):
            vertext coordinates for each color channel
    """

    # 顶点坐标的张量
    verts: torch.Tensor
    # 三角形面索引的张量
    faces: torch.Tensor
    # 每个颜色通道的顶点坐标字典
    vertex_channels: Dict[str, torch.Tensor]


# 定义一个神经网络模块，用于通过有符号距离函数构建网格
class MeshDecoder(nn.Module):
    """
    Construct meshes from Signed distance functions (SDFs) using marching cubes method
    """

    # 初始化方法，构建基本组件
    def __init__(self):
        super().__init__()
        # 创建一个大小为 (256, 5, 3) 的零张量，用于存储网格案例
        cases = torch.zeros(256, 5, 3, dtype=torch.long)
        # 创建一个大小为 (256, 5) 的零布尔张量，用于存储掩码
        masks = torch.zeros(256, 5, dtype=torch.bool)

        # 将案例和掩码注册为模块的缓冲区
        self.register_buffer("cases", cases)
        self.register_buffer("masks", masks)

# 定义一个数据类，用于存储MLP NeRF模型的输出
@dataclass
class MLPNeRFModelOutput(BaseOutput):
    # 存储密度的张量
    density: torch.Tensor
    # 存储有符号距离的张量
    signed_distance: torch.Tensor
    # 存储通道的张量
    channels: torch.Tensor
    # 存储时间步长的张量
    ts: torch.Tensor


# 定义一个混合模型类，用于构建MLP NeRF
class MLPNeRSTFModel(ModelMixin, ConfigMixin):
    @register_to_config
    # 初始化方法，接受多个参数以配置模型
    def __init__(
        self,
        d_hidden: int = 256,
        n_output: int = 12,
        n_hidden_layers: int = 6,
        act_fn: str = "swish",
        insert_direction_at: int = 4,
    ):
        super().__init__()

        # 实例化MLP

        # 创建一个单位矩阵以找到编码位置和方向的维度
        dummy = torch.eye(1, 3)
        # 编码位置的维度
        d_posenc_pos = encode_position(position=dummy).shape[-1]
        # 编码方向的维度
        d_posenc_dir = encode_direction(position=dummy).shape[-1]

        # 根据隐藏层数量设置MLP宽度
        mlp_widths = [d_hidden] * n_hidden_layers
        # 输入宽度由编码位置的维度和隐藏层宽度组成
        input_widths = [d_posenc_pos] + mlp_widths
        # 输出宽度由隐藏层宽度和输出数量组成
        output_widths = mlp_widths + [n_output]

        # 如果需要，在输入宽度中插入方向的维度
        if insert_direction_at is not None:
            input_widths[insert_direction_at] += d_posenc_dir

        # 创建线性层的模块列表
        self.mlp = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(input_widths, output_widths)])

        # 根据激活函数选择设置
        if act_fn == "swish":
            # 定义Swish激活函数
            self.activation = lambda x: F.silu(x)
        else:
            # 如果激活函数不被支持，则抛出错误
            raise ValueError(f"Unsupported activation function {act_fn}")

        # 定义不同的激活函数
        self.sdf_activation = torch.tanh
        self.density_activation = torch.nn.functional.relu
        self.channel_activation = torch.sigmoid

    # 将索引映射到键的函数
    def map_indices_to_keys(self, output):
        # 创建一个映射表，将键映射到输出的切片
        h_map = {
            "sdf": (0, 1),
            "density_coarse": (1, 2),
            "density_fine": (2, 3),
            "stf": (3, 6),
            "nerf_coarse": (6, 9),
            "nerf_fine": (9, 12),
        }

        # 根据映射表生成新的输出字典
        mapped_output = {k: output[..., start:end] for k, (start, end) in h_map.items()}

        # 返回映射后的输出
        return mapped_output
    # 定义前向传播方法，接受位置、方向、时间戳等参数
    def forward(self, *, position, direction, ts, nerf_level="coarse", rendering_mode="nerf"):
        # 对输入位置进行编码
        h = encode_position(position)

        # 初始化激活值的预激活变量
        h_preact = h
        # 初始化无方向的激活值变量
        h_directionless = None
        # 遍历多层感知机中的每一层
        for i, layer in enumerate(self.mlp):
            # 检查当前层是否为插入方向的层
            if i == self.config.insert_direction_at:  # 4 in the config
                # 保存当前的预激活值作为无方向的激活值
                h_directionless = h_preact
                # 对位置和方向进行编码，得到方向编码
                h_direction = encode_direction(position, direction=direction)
                # 将位置编码和方向编码在最后一维进行拼接
                h = torch.cat([h, h_direction], dim=-1)

            # 将当前的激活值输入到当前层进行处理
            h = layer(h)

            # 更新预激活值为当前激活值
            h_preact = h

            # 如果不是最后一层，则应用激活函数
            if i < len(self.mlp) - 1:
                h = self.activation(h)

        # 将最后一层的激活值赋给最终激活值
        h_final = h
        # 如果无方向的激活值仍为 None，则赋值为当前的预激活值
        if h_directionless is None:
            h_directionless = h_preact

        # 将激活值映射到相应的键
        activation = self.map_indices_to_keys(h_final)

        # 根据 nerf_level 选择粗糙或细致的密度
        if nerf_level == "coarse":
            h_density = activation["density_coarse"]
        else:
            h_density = activation["density_fine"]

        # 根据渲染模式选择相应的通道
        if rendering_mode == "nerf":
            if nerf_level == "coarse":
                h_channels = activation["nerf_coarse"]
            else:
                h_channels = activation["nerf_fine"]

        # 如果渲染模式为 stf，选择相应的通道
        elif rendering_mode == "stf":
            h_channels = activation["stf"]

        # 对密度进行激活处理
        density = self.density_activation(h_density)
        # 对有符号距离进行激活处理
        signed_distance = self.sdf_activation(activation["sdf"])
        # 对通道进行激活处理
        channels = self.channel_activation(h_channels)

        # yiyi notes: I think signed_distance is not used
        # 返回 MLPNeRFModelOutput 对象，包含密度、有符号距离和通道信息
        return MLPNeRFModelOutput(density=density, signed_distance=signed_distance, channels=channels, ts=ts)
# 定义一个名为 ChannelsProj 的类，继承自 nn.Module
class ChannelsProj(nn.Module):
    # 初始化方法，接受参数以定义投影的特性
    def __init__(
        self,
        *,
        vectors: int,  # 设定向量的数量
        channels: int,  # 设定通道的数量
        d_latent: int,  # 设定潜在特征的维度
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 定义线性层，将 d_latent 映射到 vectors * channels
        self.proj = nn.Linear(d_latent, vectors * channels)
        # 定义层归一化，用于标准化每个通道
        self.norm = nn.LayerNorm(channels)
        # 保存潜在特征的维度
        self.d_latent = d_latent
        # 保存向量的数量
        self.vectors = vectors
        # 保存通道的数量
        self.channels = channels

    # 前向传播方法，定义输入张量的处理方式
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量赋值给 x_bvd，便于后续使用
        x_bvd = x
        # 将权重重新形状为 (vectors, channels, d_latent)
        w_vcd = self.proj.weight.view(self.vectors, self.channels, self.d_latent)
        # 将偏置重新形状为 (1, vectors, channels)
        b_vc = self.proj.bias.view(1, self.vectors, self.channels)
        # 计算爱因斯坦求和，将输入与权重相乘并累加
        h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        # 对计算结果进行层归一化
        h = self.norm(h)

        # 将偏置添加到归一化后的结果
        h = h + b_vc
        # 返回最终的输出
        return h


# 定义一个名为 ShapEParamsProjModel 的类，继承自 ModelMixin 和 ConfigMixin
class ShapEParamsProjModel(ModelMixin, ConfigMixin):
    """
    将 3D 资产的潜在表示投影，以获取多层感知器（MLP）的权重。

    更多细节见原始论文：
    """

    # 注册到配置中
    @register_to_config
    def __init__(
        self,
        *,
        param_names: Tuple[str] = (  # 定义参数名称的元组
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = (  # 定义参数形状的元组
            (256, 93),
            (256, 256),
            (256, 256),
            (256, 256),
        ),
        d_latent: int = 1024,  # 设置潜在特征的维度，默认值为 1024
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查输入参数的有效性
        if len(param_names) != len(param_shapes):
            # 如果参数名称与形状数量不一致，抛出错误
            raise ValueError("Must provide same number of `param_names` as `param_shapes`")
        # 初始化一个空的模块字典，用于存储投影层
        self.projections = nn.ModuleDict({})
        # 遍历参数名称和形状，并为每一对创建一个 ChannelsProj 实例
        for k, (vectors, channels) in zip(param_names, param_shapes):
            self.projections[_sanitize_name(k)] = ChannelsProj(
                vectors=vectors,  # 设置向量数量
                channels=channels,  # 设置通道数量
                d_latent=d_latent,  # 设置潜在特征的维度
            )

    # 前向传播方法
    def forward(self, x: torch.Tensor):
        out = {}  # 初始化输出字典
        start = 0  # 初始化起始索引
        # 遍历参数名称和形状
        for k, shape in zip(self.config.param_names, self.config.param_shapes):
            vectors, _ = shape  # 获取当前参数的向量数量
            end = start + vectors  # 计算结束索引
            x_bvd = x[:, start:end]  # 从输入中切片提取相关部分
            # 将切片后的输入通过对应的投影层处理，并调整形状
            out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape(len(x), *shape)
            start = end  # 更新起始索引为结束索引
        # 返回包含所有输出的字典
        return out


# 定义一个名为 ShapERenderer 的类，继承自 ModelMixin 和 ConfigMixin
class ShapERenderer(ModelMixin, ConfigMixin):
    # 注册到配置中
    @register_to_config
    # 初始化方法，用于创建类的实例
    def __init__(
        self,
        *,  # 指定后续参数为关键字参数
        param_names: Tuple[str] = (  # 定义参数名称的元组，默认值为特定的权重名称
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = (  # 定义参数形状的元组，默认值为特定的形状
            (256, 93),
            (256, 256),
            (256, 256),
            (256, 256),
        ),
        d_latent: int = 1024,  # 定义潜在维度的整数，默认值为1024
        d_hidden: int = 256,  # 定义隐藏层维度的整数，默认值为256
        n_output: int = 12,  # 定义输出层的神经元数量，默认值为12
        n_hidden_layers: int = 6,  # 定义隐藏层的层数，默认值为6
        act_fn: str = "swish",  # 定义激活函数的名称，默认值为"swish"
        insert_direction_at: int = 4,  # 定义插入方向的索引，默认值为4
        background: Tuple[float] = (  # 定义背景颜色的元组，默认值为白色
            255.0,
            255.0,
            255.0,
        ),
    ):
        super().__init__()  # 调用父类的初始化方法

        # 创建参数投影模型，传入参数名称、形状和潜在维度
        self.params_proj = ShapEParamsProjModel(
            param_names=param_names,
            param_shapes=param_shapes,
            d_latent=d_latent,
        )
        # 创建多层感知机模型，传入隐藏层维度、输出层数量、隐藏层层数、激活函数和插入方向
        self.mlp = MLPNeRSTFModel(d_hidden, n_output, n_hidden_layers, act_fn, insert_direction_at)
        # 创建空的神经辐射场模型，传入背景颜色和通道缩放
        self.void = VoidNeRFModel(background=background, channel_scale=255.0)
        # 创建包围盒体积模型，定义最大和最小边界
        self.volume = BoundingBoxVolume(bbox_max=[1.0, 1.0, 1.0], bbox_min=[-1.0, -1.0, -1.0])
        # 创建网格解码器
        self.mesh_decoder = MeshDecoder()

    @torch.no_grad()  # 禁用梯度计算，提高推理性能
    @torch.no_grad()  # 冗余的禁用梯度计算装饰器
    def decode_to_image(
        self,
        latents,  # 输入的潜在变量
        device,  # 指定计算设备（如CPU或GPU）
        size: int = 64,  # 输出图像的尺寸，默认值为64
        ray_batch_size: int = 4096,  # 每批次光线的数量，默认值为4096
        n_coarse_samples=64,  # 粗采样的数量，默认值为64
        n_fine_samples=128,  # 精细采样的数量，默认值为128
    ):
        # 从生成的潜在变量投影参数
        projected_params = self.params_proj(latents)

        # 更新渲染器的MLP层
        for name, param in self.mlp.state_dict().items():  # 遍历MLP模型的所有参数
            if f"nerstf.{name}" in projected_params.keys():  # 检查投影参数是否存在于MLP参数中
                param.copy_(projected_params[f"nerstf.{name}"].squeeze(0))  # 更新MLP参数

        # 创建相机对象
        camera = create_pan_cameras(size)  # 生成全景相机
        rays = camera.camera_rays  # 获取相机射线
        rays = rays.to(device)  # 将射线移动到指定设备
        n_batches = rays.shape[1] // ray_batch_size  # 计算总批次数量

        coarse_sampler = StratifiedRaySampler()  # 创建粗采样器

        images = []  # 初始化图像列表

        for idx in range(n_batches):  # 遍历每个批次
            rays_batch = rays[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]  # 获取当前批次的射线

            # 使用粗糙的分层采样渲染射线
            _, fine_sampler, coarse_model_out = self.render_rays(rays_batch, coarse_sampler, n_coarse_samples)
            # 然后使用附加的重要性加权射线样本进行渲染
            channels, _, _ = self.render_rays(
                rays_batch, fine_sampler, n_fine_samples, prev_model_out=coarse_model_out
            )

            images.append(channels)  # 将渲染结果添加到图像列表

        images = torch.cat(images, dim=1)  # 在维度1上拼接所有图像
        images = images.view(*camera.shape, camera.height, camera.width, -1).squeeze(0)  # 调整图像的形状

        return images  # 返回渲染出的图像

    @torch.no_grad()  # 禁用梯度计算，提高推理性能
    def decode_to_mesh(
        self,
        latents,  # 输入的潜在变量
        device,  # 指定计算设备（如CPU或GPU）
        grid_size: int = 128,  # 网格大小，默认值为128
        query_batch_size: int = 4096,  # 每批次查询的数量，默认值为4096
        texture_channels: Tuple = ("R", "G", "B"),  # 纹理通道，默认值为RGB
```