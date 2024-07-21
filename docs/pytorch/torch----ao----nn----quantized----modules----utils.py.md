# `.\pytorch\torch\ao\nn\quantized\modules\utils.py`

```
# 导入必要的模块和函数
import abc  # 导入抽象基类模块
import torch  # 导入PyTorch库
import itertools  # 导入迭代工具模块
import collections  # 导入集合模块
from torch.nn.modules.module import _addindent  # 从PyTorch的神经网络模块中导入_addindent函数

# 定义模块公开的接口
__all__ = [
    "WeightedQuantizedModule",
]

class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """包装量化模块，可从参考模块降低。"""
    
    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point):
        raise NotImplementedError

def _get_weight_observer(observer):
    # 获取权重观察器
    # 如果observer具有"activation_post_process"属性，则将observer设置为activation_post_process属性
    if hasattr(observer, "activation_post_process"):
        observer = observer.activation_post_process
    # 返回UniformQuantizationObserverBase观察器
    return observer

def _needs_weight_clamping(observer, dtype):
    # 检查是否需要对权重进行夹紧操作
    observer = _get_weight_observer(observer)
    if dtype in [torch.qint8, torch.quint8, torch.qint32]:
        info = torch.iinfo(dtype)
        return observer.quant_min > info.min or observer.quant_max < info.max
    return False

def _clamp_weights(qweight, observer, scale, zp):
    # 对权重进行夹紧操作
    if not _needs_weight_clamping(observer, qweight.dtype):
        return qweight

    observer = _get_weight_observer(observer)
    min_, max_ = observer.quant_min, observer.quant_max

    # 由于当前无法使用torch.ops.quantized.clamp()对每通道量化方案进行操作
    qw_int_max = torch.clone(qweight.int_repr()).fill_(max_)
    qw_int_min = torch.clone(qweight.int_repr()).fill_(min_)
    qw_int = torch.minimum(torch.maximum(qweight.int_repr(), qw_int_min), qw_int_max)

    if observer.qscheme in [torch.per_tensor_symmetric,
                            torch.per_tensor_affine]:
        qweight = torch._make_per_tensor_quantized_tensor(qw_int, scale.item(), zp.item())
    elif observer.qscheme in [torch.per_channel_symmetric,
                              torch.per_channel_affine,
                              torch.per_channel_affine_float_qparams]:
        qweight = torch._make_per_channel_quantized_tensor(qw_int, scale, zp, axis=observer.ch_axis)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight

def _quantize_weight(float_wt, observer):
    # 对权重进行量化
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), torch.qint8)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, torch.qint8)
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    # 如果观察器的量化方案在 [torch.per_channel_affine_float_qparams] 中
    elif observer.qscheme in [torch.per_channel_affine_float_qparams]:
        # 对权重进行通道间量化，使用观察器的缩放因子和零点进行量化
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float), wt_zp.to(torch.float), observer.ch_axis, observer.dtype)
        # 对量化后的权重进行截断处理，确保在观察器指定的范围内
        qweight = _clamp_weights(qweight, observer, wt_scale, wt_zp)
    # 如果不在预期的量化方案中，则抛出值错误异常
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    # 返回量化后的权重
    return qweight
def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""
    # 定义一个函数 parse，用于将输入 x 转换为长度为 n 的元组，第一个元素重复 n 次
    def parse(x):
        # 只要 x 是 collections.abc.Sequence 的实例就一直循环
        while isinstance(x, collections.abc.Sequence):
            # 如果 x 的长度已经是 n，则停止循环
            if len(x) == n:
                break
            # 否则取 x 的第一个元素继续循环
            x = x[0]
        # 返回由 x 的第一个元素重复 n 次组成的元组
        return tuple(itertools.repeat(x, n))
    # 返回 parse 函数
    return parse

def _hide_packed_params_repr(self, params):
    # We don't want to show `PackedParams` children, hence custom
    # `__repr__`. This is the same as nn.Module.__repr__, except the check
    # for the `params module`.
    
    # 初始化额外的行为空列表
    extra_lines = []
    # 调用 self 的 extra_repr 方法获取额外的描述信息
    extra_repr = self.extra_repr()
    # 如果 extra_repr 不为空
    if extra_repr:
        # 将 extra_repr 按换行符分割成列表赋给 extra_lines
        extra_lines = extra_repr.split('\n')
    
    # 初始化子模块行列表为空
    child_lines = []
    # 遍历 self 的子模块字典 self._modules
    for key, module in self._modules.items():
        # 如果 module 是 params 的实例，则跳过
        if isinstance(module, params):
            continue
        # 获得 module 的字符串表示
        mod_str = repr(module)
        # 调用 _addindent 函数，将 mod_str 缩进 2 个空格
        mod_str = _addindent(mod_str, 2)
        # 将格式化后的子模块字符串添加到 child_lines 列表中
        child_lines.append('(' + key + '): ' + mod_str)
    
    # 将额外的行和子模块行合并为总行列表
    lines = extra_lines + child_lines

    # 初始化主要字符串为 self 的名称 + '('
    main_str = self._get_name() + '('
    # 如果 lines 列表不为空
    if lines:
        # 如果额外行只有一行且没有子模块行，则简单地将这一行作为主要字符串的一部分
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            # 否则，格式化输出额外行和子模块行，并添加到主要字符串中
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    # 将 ')' 添加到主要字符串的末尾，形成最终的字符串表示
    main_str += ')'
    # 返回最终的字符串表示
    return main_str

# 创建一个 _pair_from_first 函数，使用 _ntuple_from_first(2) 返回的函数来处理参数为 2 的情况
_pair_from_first = _ntuple_from_first(2)
```