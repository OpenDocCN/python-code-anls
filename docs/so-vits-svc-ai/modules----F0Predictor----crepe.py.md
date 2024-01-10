# `so-vits-svc\modules\F0Predictor\crepe.py`

```
# 导入必要的类型提示
from typing import Optional, Union

# 尝试导入 Literal 类型提示，如果失败则导入 typing_extensions 中的 Literal
try:
    from typing import Literal
except Exception:
    from typing_extensions import Literal

# 导入 numpy 和 torch 库
import numpy as np
import torch
import torchcrepe
from torch import nn
from torch.nn import functional as F

# 导入 repeat_expand 函数
#from:https://github.com/fishaudio/fish-diffusion

# 定义 repeat_expand 函数，用于将内容重复扩展到目标长度
def repeat_expand(
    content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """

    # 获取内容的维度
    ndim = content.ndim

    # 如果内容的维度为 1，则将其扩展为二维
    if content.ndim == 1:
        content = content[None, None]
    # 如果内容的维度为 2，则将其扩展为三维
    elif content.ndim == 2:
        content = content[None]

    # 断言内容的维度为三维
    assert content.ndim == 3

    # 检查内容是否为 numpy 数组，如果是则转换为 torch.Tensor
    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    # 使用 torch.nn.functional.interpolate 函数对内容进行插值，使其长度达到目标长度
    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    # 如果内容为 numpy 数组，则将结果转换为 numpy 数组
    if is_np:
        results = results.numpy()

    # 根据原始内容的维度返回结果
    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]

# 定义 BasePitchExtractor 类
class BasePitchExtractor:
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        keep_zeros: bool = True,
    ):
        """Base pitch extractor.

        Args:
            hop_length (int, optional): Hop length. Defaults to 512.
            f0_min (float, optional): Minimum f0. Defaults to 50.0.
            f0_max (float, optional): Maximum f0. Defaults to 1100.0.
            keep_zeros (bool, optional): Whether keep zeros in pitch. Defaults to True.
        """

        # 初始化基本音高提取器的参数
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros
    # 定义一个方法，用于处理输入的音频信号，提取音高信息
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        # 抛出未实现的错误，提示子类需要实现该方法
        raise NotImplementedError("BasePitchExtractor is not callable.")

    # 对提取的音高信息进行后处理
    def post_process(self, x, sampling_rate, f0, pad_to):
        # 如果音高信息是 NumPy 数组，则转换为 PyTorch 张量，并移动到相同的设备上
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        # 如果没有指定填充到的长度，则直接返回音高信息
        if pad_to is None:
            return f0

        # 将音高信息重复扩展到指定的长度
        f0 = repeat_expand(f0, pad_to)

        # 如果需要保留0频率的信息，则直接返回音高信息
        if self.keep_zeros:
            return f0
        
        # 创建一个与音高信息相同大小的零张量
        vuv_vector = torch.zeros_like(f0)
        # 根据音高信息的值，将对应位置的零张量值设为1
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # 去掉0频率, 并线性插值
        # 找到非零频率的索引
        nzindex = torch.nonzero(f0).squeeze()
        # 根据非零频率的索引，提取对应的频率值和时间信息
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate
        
        # 使用线性插值方法，将频率信息插值到指定长度
        vuv_vector = F.interpolate(vuv_vector[None,None,:],size=pad_to)[0][0]

        # 如果频率信息为空，则返回全零的频率信息和对应的声门开合向量
        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device),vuv_vector.cpu().numpy()
        # 如果频率信息只有一个值，则返回填充到指定长度的频率信息和对应的声门开合向量
        if f0.shape[0] == 1:
            return torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0],vuv_vector.cpu().numpy()
    
        # 使用 NumPy 的插值方法，将频率信息插值到指定长度
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        #vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        # 返回插值后的频率信息和对应的声门开合向量
        return f0,vuv_vector.cpu().numpy()
class MaskedAvgPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        # 调用父类的构造函数
        super(MaskedAvgPool1d, self).__init__()
        # 设置池化窗口的大小
        self.kernel_size = kernel_size
        # 如果未指定步长，则默认为池化窗口的大小
        self.stride = stride or kernel_size
        # 设置池化窗口的填充大小
        self.padding = padding
    # 定义一个前向传播函数，接受输入张量 x 和掩码 mask（可选）
    def forward(self, x, mask=None):
        # 获取输入张量的维度
        ndim = x.dim()
        # 如果输入张量维度为 2，则在第一维度上增加一个维度
        if ndim == 2:
            x = x.unsqueeze(1)

        # 断言输入张量的维度为 3，如果不是则抛出异常
        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        # 如果没有提供掩码，则创建一个掩码，将被掩盖的元素设为零，或将 NaN 设为零
        if mask is None:
            mask = ~torch.isnan(x)

        # 确保掩码和输入张量具有相同的形状
        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        # 使用掩码对输入张量进行掩盖操作，将被掩盖的元素设为零
        masked_x = torch.where(mask, x, torch.zeros_like(x))
        # 创建一个与输入张量具有相同通道数的全为 1 的卷积核
        ones_kernel = torch.ones(x.size(1), 1, self.kernel_size, device=x.device)

        # 执行池化操作，对掩盖后的张量进行求和池化
        sum_pooled = nn.functional.conv1d(
            masked_x,
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )

        # 计算每个池化窗口中非掩盖（有效）元素的数量
        valid_count = nn.functional.conv1d(
            mask.float(),
            ones_kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.size(1),
        )
        valid_count = valid_count.clamp(min=1)  # 避免除以零

        # 执行掩盖平均池化操作
        avg_pooled = sum_pooled / valid_count

        # 将零值替换为 NaN
        avg_pooled[avg_pooled == 0] = float("nan")

        # 如果输入张量维度为 2，则压缩第一维度后返回结果
        if ndim == 2:
            return avg_pooled.squeeze(1)

        # 返回平均池化结果
        return avg_pooled
class MaskedMedianPool1d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """

        # 调用父类的构造函数
        super(MaskedMedianPool1d, self).__init__()
        # 设置中值池化窗口的大小
        self.kernel_size = kernel_size
        # 如果未指定步幅，则默认为窗口大小
        self.stride = stride or kernel_size
        # 设置中值池化窗口的填充大小
        self.padding = padding
    # 定义一个前向传播函数，接受输入张量 x 和掩码 mask（可选）
    def forward(self, x, mask=None):
        # 获取输入张量 x 的维度
        ndim = x.dim()
        # 如果输入张量 x 是二维的，则在第一维度上添加一个维度
        if ndim == 2:
            x = x.unsqueeze(1)

        # 断言输入张量 x 的维度为 3，如果不是则抛出异常
        assert (
            x.dim() == 3
        ), "Input tensor must have 2 or 3 dimensions (batch_size, channels, width)"

        # 如果没有提供掩码 mask，则创建一个掩码，将输入张量 x 中的 NaN 值标记为 False
        if mask is None:
            mask = ~torch.isnan(x)

        # 断言输入张量 x 和掩码 mask 的形状相同，如果不同则抛出异常
        assert x.shape == mask.shape, "Input tensor and mask must have the same shape"

        # 根据掩码 mask 对输入张量 x 进行遮盖，将掩码为 False 的位置置为 0
        masked_x = torch.where(mask, x, torch.zeros_like(x))

        # 在输入张量 x 上进行反射填充，填充大小为 self.padding
        x = F.pad(masked_x, (self.padding, self.padding), mode="reflect")
        # 在掩码 mask 上进行常数填充，填充大小为 self.padding，填充值为 0
        mask = F.pad(
            mask.float(), (self.padding, self.padding), mode="constant", value=0
        )

        # 在第三维度上对输入张量 x 和掩码 mask 进行滑动窗口操作，窗口大小为 self.kernel_size，滑动步长为 self.stride
        x = x.unfold(2, self.kernel_size, self.stride)
        mask = mask.unfold(2, self.kernel_size, self.stride)

        # 将输入张量 x 和掩码 mask 进行连续化，并将掩码 mask 移动到与输入张量 x 相同的设备上
        x = x.contiguous().view(x.size()[:3] + (-1,))
        mask = mask.contiguous().view(mask.size()[:3] + (-1,)).to(x.device)

        # 将掩码 mask 应用到输入张量 x 上，将掩码为 False 的位置置为正无穷
        x_masked = torch.where(mask.bool(), x, torch.FloatTensor([float("inf")]).to(x.device))

        # 沿着最后一个维度对遮盖后的张量进行排序
        x_sorted, _ = torch.sort(x_masked, dim=-1)

        # 计算非遮盖（有效）值的数量
        valid_count = mask.sum(dim=-1)

        # 计算每个池化窗口中位数值的索引
        median_idx = (torch.div((valid_count - 1), 2, rounding_mode='trunc')).clamp(min=0)

        # 使用计算出的索引来收集中位数值
        median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)

        # 将无穷大的值填充为 NaN
        median_pooled[torch.isinf(median_pooled)] = float("nan")
        
        # 如果输入张量 x 是二维的，则压缩第一维度并返回中位数池化结果
        if ndim == 2:
            return median_pooled.squeeze(1)

        # 返回中位数池化结果
        return median_pooled
# 定义 CrepePitchExtractor 类，继承自 BasePitchExtractor 类
class CrepePitchExtractor(BasePitchExtractor):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        hop_length: int = 512,  # 设置默认值为 512 的 hop_length 参数
        f0_min: float = 50.0,  # 设置默认值为 50.0 的 f0_min 参数
        f0_max: float = 1100.0,  # 设置默认值为 1100.0 的 f0_max 参数
        threshold: float = 0.05,  # 设置默认值为 0.05 的 threshold 参数
        keep_zeros: bool = False,  # 设置默认值为 False 的 keep_zeros 参数
        device = None,  # 默认值为 None 的 device 参数
        model: Literal["full", "tiny"] = "full",  # 设置默认值为 "full" 的 model 参数，取值为 "full" 或 "tiny"
        use_fast_filters: bool = True,  # 设置默认值为 True 的 use_fast_filters 参数
        decoder="viterbi"  # 默认值为 "viterbi" 的 decoder 参数
    ):
        # 调用父类的初始化方法，传入 hop_length, f0_min, f0_max, keep_zeros 参数
        super().__init__(hop_length, f0_min, f0_max, keep_zeros)
        # 根据 decoder 参数的值选择相应的解码器
        if decoder == "viterbi":
            self.decoder = torchcrepe.decode.viterbi
        elif decoder == "argmax":
            self.decoder = torchcrepe.decode.argmax
        elif decoder == "weighted_argmax":
            self.decoder = torchcrepe.decode.weighted_argmax
        else:
            raise "Unknown decoder"  # 如果 decoder 参数值未知，则抛出异常
        self.threshold = threshold  # 设置对象的 threshold 属性为传入的 threshold 参数值
        self.model = model  # 设置对象的 model 属性为传入的 model 参数值
        self.use_fast_filters = use_fast_filters  # 设置对象的 use_fast_filters 属性为传入的 use_fast_filters 参数值
        self.hop_length = hop_length  # 设置对象的 hop_length 属性为传入的 hop_length 参数值
        if device is None:  # 如果 device 参数为 None
            # 判断是否有可用的 CUDA 设备，如果有则使用 CUDA，否则使用 CPU
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)  # 设置对象的 dev 属性为传入的 device 参数值
        if self.use_fast_filters:  # 如果 use_fast_filters 为 True
            # 创建并将 MaskedMedianPool1d 对象移动到指定设备
            self.median_filter = MaskedMedianPool1d(3, 1, 1).to(device)
            # 创建并将 MaskedAvgPool1d 对象移动到指定设备
            self.mean_filter = MaskedAvgPool1d(3, 1, 1).to(device)
    # 定义一个方法，用于使用 crepe 提取音高
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using crepe.


        Args:
            x (torch.Tensor): Audio signal, shape (1, T).  # 输入参数 x，音频信号，形状为 (1, T)
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.  # 采样率，默认为 44100
            pad_to (int, optional): Pad to length. Defaults to None.  # 填充到指定长度，默认为 None

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).  # 返回音高，形状为 (T // hop_length,)
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."  # 断言，确保输入的张量是二维的
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."  # 断言，确保输入的张量是单声道的

        x = x.to(self.dev)  # 将输入张量移动到指定的设备上
        f0, pd = torchcrepe.predict(
            x,
            sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            pad=True,
            model=self.model,
            batch_size=1024,
            device=x.device,
            return_periodicity=True,
            decoder=self.decoder
        )

        # 过滤、去除静音、设置 UV 阈值，参考原始仓库的 readme
        if self.use_fast_filters:
            pd = self.median_filter(pd)  # 使用中值滤波器对 pd 进行滤波
        else:
            pd = torchcrepe.filter.median(pd, 3)  # 使用中值滤波器对 pd 进行滤波

        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, sampling_rate, self.hop_length)  # 使用 Silence 阈值对 pd 进行处理
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)  # 使用指定阈值对 f0 进行处理
        
        if self.use_fast_filters:
            f0 = self.mean_filter(f0)  # 使用均值滤波器对 f0 进行滤波
        else:
            f0 = torchcrepe.filter.mean(f0, 3)  # 使用均值滤波器对 f0 进行滤波

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)[0]  # 将 f0 中的 NaN 值替换为 0

        if torch.all(f0 == 0):  # 如果 f0 全部为 0
            rtn = f0.cpu().numpy() if pad_to is None else np.zeros(pad_to)  # 返回全为 0 的数组
            return rtn,rtn  # 返回结果数组
        
        return self.post_process(x, sampling_rate, f0, pad_to)  # 调用 post_process 方法处理结果
```