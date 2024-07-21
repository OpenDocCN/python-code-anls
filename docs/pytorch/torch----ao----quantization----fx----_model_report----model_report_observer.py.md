# `.\pytorch\torch\ao\quantization\fx\_model_report\model_report_observer.py`

```
# mypy: allow-untyped-defs
# 引入 PyTorch 库
import torch
# 从 torch.ao.quantization.observer 模块中导入 ObserverBase 类
from torch.ao.quantization.observer import ObserverBase

# 定义 ModelReportObserver 类，继承自 ObserverBase 类
class ModelReportObserver(ObserverBase):
    r"""This observer is used to record additional information regarding keeping track
    of S = average_batch_activation_range/epoch_activation_range.

    The purpose of this information is to prepare a report to present to users on whether
    Dynamic or Static Quantization is more appropriate for their model given the general
    distributions of their data.

    Args:
        ch_axis (int, optional): The channel axis for which the range and outlier stats are computed
            Default: 1
        comp_percentile (float, optional): The percentile to compare against 100 percentile to find outliers
            Should be between 0 and 1 exclusive
            Default: 0.9

    * :attr:`num_batches_tracked` specifies number of batches passed through the observer

    * :attr:`average_batch_activation_range` defines average across the ranges of each batch passed through

    * :attr:`epoch_activation_min` defines the minimum value passed through the observer

    * :attr:`epoch_activation_max` defines the maximum value passed through the observer

    * :attr:`ch_axis` defines the channel being used to compute per channel min max stats

    * :attr:`min_val` defines the per channel minimum values passed through

    * :attr:`max_val` defines the per channel maximum values passed through

    * :attr:`comp_percentile` defines comparison percentile to find outliers

    * :attr:`average_percentile_ratio` defines the per channel average percentile ratios

    * :attr:`percentile_batches_tracked` defines the number of percentile batches tracked for each channel

    * :attr:`constant_channels` defines the number of batches that aren't constant channels per channel

    Note: this tool is meant for FX Graph Mode Quantization
    """

    # 定义 epoch_activation_min 属性，用于存储通过观察器传递的最小值的张量
    epoch_activation_min: torch.Tensor
    # 定义 epoch_activation_max 属性，用于存储通过观察器传递的最大值的张量
    epoch_activation_max: torch.Tensor
    # 定义 min_val 属性，用于存储通过观察器传递的每个通道的最小值的张量
    min_val: torch.Tensor
    # 定义 max_val 属性，用于存储通过观察器传递的每个通道的最大值的张量
    max_val: torch.Tensor
    # 定义 comp_percentile 属性，用于存储用于查找异常值的比较百分位数的张量
    comp_percentile: torch.Tensor
    # 定义 average_percentile_ratio 属性，用于存储每个通道的平均百分位比率的张量
    average_percentile_ratio: torch.Tensor
    # 定义 percentile_batches_tracked 属性，用于存储每个通道跟踪的百分位数批次数的张量
    percentile_batches_tracked: torch.Tensor
    # 定义 constant_channels 属性，用于存储每个通道中非常量通道批次数的张量
    constant_channels: torch.Tensor
    def __init__(self, ch_axis: int = 1, comp_percentile: float = 0.9):
        # 调用父类的初始化方法，使用 torch.qint8 类型
        super().__init__(torch.qint8)
        # 初始化批次追踪计数器为 0
        self.num_batches_tracked = 0

        # 初始化平均批次和整个 epoch 范围的最小和最大值
        self.average_batch_activation_range: torch.Tensor = torch.tensor(float(0))
        self.register_buffer("epoch_activation_min", torch.tensor(float("inf")))
        self.register_buffer("epoch_activation_max", torch.tensor(float("-inf")))

        # 使用给定的通道轴初始化通道轴属性
        self.ch_axis: int = ch_axis
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))

        # 使用给定的压缩百分位初始化压缩百分位属性
        self.register_buffer("comp_percentile", torch.tensor([comp_percentile]))
        self.register_buffer("average_percentile_ratio", torch.tensor([]))
        self.register_buffer("percentile_batches_tracked", torch.tensor([]))
        self.register_buffer("constant_channels", torch.tensor([]))

    def forward(self, x):
        # 分离输入张量 x，避免保留自动求导记录
        x_copy = x.detach()
        # 将分离后的张量 x_copy 转换为与 epoch_activation_min 相同的数据类型
        x_copy = x_copy.to(self.epoch_activation_min.dtype)

        # 计算并更新范围统计信息
        x_copy = self._calculate_range_stats(x_copy)
        # 计算并更新最小-最大值统计信息
        x_copy = self._calculate_min_max_stats(x_copy)
        # 计算并更新百分位统计信息
        x_copy = self._calculate_percentile_stats(x_copy)

        # 返回原始输入值 x
        return x

    def _calculate_range_stats(self, x_copy):
        r"""Calculates and stores range stats with forward values.

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # 获取数据 x_copy 的最小值和最大值
        min_val_cur, max_val_cur = torch.aminmax(x_copy)

        # 计算新的 epoch 范围值
        epoch_min_val = torch.min(self.epoch_activation_min, min_val_cur)
        epoch_max_val = torch.max(self.epoch_activation_max, max_val_cur)

        # 更新 epoch_activation_min 和 epoch_activation_max 的值
        self.epoch_activation_min.copy_(epoch_min_val)
        self.epoch_activation_max.copy_(epoch_max_val)

        # 计算平均批次激活范围
        current_batch_range = max_val_cur - min_val_cur
        new_range = (
            self.average_batch_activation_range * self.num_batches_tracked
            + current_batch_range
        ) / (self.num_batches_tracked + 1)

        # 更新 average_batch_activation_range 和 num_batches_tracked
        self.average_batch_activation_range = new_range
        self.num_batches_tracked += 1  # 处理了新的批次

        return x_copy
    def _calculate_min_max_stats(self, x_copy):
        r"""Calculates and stores the per_channel min, max stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # 获取当前的最小值和最大值
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x_copy.size()

        # 创建新的轴列表，重新排列数据以便按通道轴进行计算
        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)

        # 确保更新缓冲区时数据类型匹配
        y = y.to(self.min_val.dtype)

        # 将张量展平以便进行统计计算
        y = torch.flatten(y, start_dim=1)

        # 如果当前的最小值或最大值为空，则计算新的最小值和最大值
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            # 否则，更新当前的最小值和最大值
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)

        # 更新对象中的最小值和最大值
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        return x_copy

    @torch.jit.export
    def get_batch_to_epoch_ratio(self):
        # 计算批次与时期之间的比率
        epoch_activation_range = self.epoch_activation_max - self.epoch_activation_min

        # 如果时期的活动范围为零，则引发值错误
        if epoch_activation_range == torch.tensor(float(0)):
            raise ValueError("Range for Epoch is 0")
        # 如果时期的活动范围为无穷大，则引发值错误
        elif epoch_activation_range == torch.tensor(float("inf")):
            raise ValueError(
                "No data has been run through observer or infinity value present"
            )
        else:
            # 否则，返回批次平均激活范围与时期活动范围的比率
            return self.average_batch_activation_range / epoch_activation_range

    @torch.jit.export
    def reset_batch_and_epoch_values(self):
        # 将所有值重置为新时期的原始默认值
        # 保留设备信息
        device = self.max_val.device
        self.num_batches_tracked = 0
        self.average_batch_activation_range = torch.tensor(float(0), device=device)
        self.epoch_activation_min = torch.tensor(float("inf"), device=device)
        self.epoch_activation_max = torch.tensor(float("-inf"), device=device)
        self.min_val = torch.tensor([], device=device)
        self.max_val = torch.tensor([], device=device)
        self.average_percentile_ratio = torch.tensor([], device=device)
        self.percentile_batches_tracked = torch.tensor([], device=device)
        self.constant_channels = torch.tensor([], device=device)

    @torch.jit.export
    def calculate_qparams(self):
        # ModelReportObserver 不应调用此方法，抛出异常
        raise Exception(
            "calculate_qparams should not be called for ModelReportObserver"
        )
```