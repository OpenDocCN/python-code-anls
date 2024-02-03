# `bitsandbytes\bitsandbytes\nn\modules.py`

```py
# 导入必要的库和模块
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings

import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F

import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer

# 定义类型变量
T = TypeVar("T", bound="torch.nn.Module")

# 创建一个稳定的嵌入层类，继承自torch.nn.Embedding
class StableEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )
        # 创建一个LayerNorm层，用于规范化嵌入向量
        self.norm = torch.nn.LayerNorm(embedding_dim, device=device)
        # 注册模块覆盖，设置优化位数为32
        GlobalOptimManager.get_instance().register_module_override(
            self, "weight", {"optim_bits": 32}
        )

    # 重置参数方法
    def reset_parameters(self) -> None:
        # 使用Xavier初始化方法初始化权重
        torch.nn.init.xavier_uniform_(self.weight)
        # 将填充索引位置的嵌入向量设为零向量
        self._fill_padding_idx_with_zero()
    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """
    
    # 重新定义 _fill_padding_idx_with_zero 方法，以使该层与 PyTorch < 1.9 兼容。
    # 这意味着如果将来 PyTorch 发布中发生更改，这里也需要相应更改，这很麻烦。
    # 然而，通过这样做，我们可以确保与之前的 PyTorch 发布版本的兼容性。

    def _fill_padding_idx_with_zero(self) -> None:
        # 如果存在填充索引
        if self.padding_idx is not None:
            # 使用 torch.no_grad() 上下文管理器，确保不会计算梯度
            with torch.no_grad():
                # 将填充索引位置的权重值填充为 0
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        # 使用 F.embedding 方法进行嵌入操作
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        # 始终在全精度下应用层归一化
        emb = emb.to(torch.get_default_dtype())

        # 对嵌入结果进行归一化操作，并转换为权重的数据类型
        return self.norm(emb).to(self.weight.dtype)
class Embedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device: Optional[device] = None,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device=device
        )
        # 注册模块覆盖，设置优化位数为32
        GlobalOptimManager.get_instance().register_module_override(
            self, "weight", {"optim_bits": 32}
        )

    def reset_parameters(self) -> None:
        # 使用 xavier_uniform_ 方法初始化权重
        torch.nn.init.xavier_uniform_(self.weight)
        # 将填充索引位置的权重值设为0
        self._fill_padding_idx_with_zero()

    """ !!! This is a redefinition of _fill_padding_idx_with_zero in torch.nn.Embedding
        to make the Layer compatible with Pytorch < 1.9.
        This means that if this changes in future PyTorch releases this need to change too
        which is cumbersome. However, with this we can ensure compatibility with previous
        PyTorch releases.
    """

    def _fill_padding_idx_with_zero(self) -> None:
        # 如果存在填充索引
        if self.padding_idx is not None:
            # 使用 torch.no_grad() 上下文管理器，填充索引位置的权重值为0
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        # 使用 F.embedding 方法进行前向传播计算
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return emb


class Params4bit(torch.nn.Parameter):
    # 定义一个新的类构造函数，用于创建 Params4bit 对象
    def __new__(
            cls,
            data: Optional[torch.Tensor] = None,
            requires_grad=True,
            quant_state: Optional[QuantState] = None,
            blocksize: int = 64,
            compress_statistics: bool = True,
            quant_type: str = 'fp4',
            quant_storage: torch.dtype = torch.uint8,
            module: Optional["Linear4bit"] = None,
            bnb_quantized: bool = False
    ) -> "Params4bit":
        # 如果没有传入数据，则创建一个空的 Tensor 对象
        if data is None:
            data = torch.empty(0)

        # 创建一个子类对象，继承自 Tensor 类
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        # 设置对象的属性
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self.data = data
        self.module = module
        # 返回创建的对象
        return self

    # 定义一个类方法，用于从预量化数据创建 Params4bit 对象
    @classmethod
    def from_prequantized(cls, data: torch.Tensor, quantized_stats: Dict[str, Any], requires_grad: bool = False, device='cuda', **kwargs) -> "Params4bit":
        # 创建一个子类对象，继承自 Tensor 类，并将数据移动到指定设备上
        self = torch.Tensor._make_subclass(cls, data.to(device))
        self.requires_grad = requires_grad
        # 从字典中创建量化状态对象
        self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
        self.blocksize = self.quant_state.blocksize
        self.compress_statistics = self.quant_state.nested
        self.quant_type = self.quant_state.quant_type
        self.bnb_quantized = True
        # 返回创建的对象
        return self
    # 将参数量化为4位，使用指定的设备
    def _quantize(self, device):
        # 将数据转换为连续的张量，并移动到指定设备上
        w = self.data.contiguous().cuda(device)
        # 对数据进行4位量化，并返回量化后的数据和量化状态
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                                           quant_type=self.quant_type, quant_storage=self.quant_storage)
        # 更新数据为量化后的数据
        self.data = w_4bit
        # 更新量化状态
        self.quant_state = quant_state
        # 如果存在模块，则更新模块的量化状态
        if self.module is not None:
            self.module.quant_state = quant_state
        # 标记为已量化
        self.bnb_quantized = True
        # 返回自身
        return self

    # 将参数移动到指定设备
    def cuda(self, device: Optional[Union[int, device, str]] = None, non_blocking: bool = False):
        return self.to(device='cuda' if device is None else device, non_blocking=non_blocking)

    # 将参数移动到指定设备和数据类型
    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ..., non_blocking: bool = ...,) -> T:
        ...

    # 将参数移动到指定数据类型
    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    # 将参数移动到与给定张量相同的设备
    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    # 将参数移动到指定设备和数据类型
    def to(self, *args, **kwargs):
        # 解析参数，获取设备、数据类型、是否非阻塞、是否转换格式
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        # 如果设备不为空且为cuda类型且未量化，则进行量化
        if (device is not None and device.type == "cuda" and not self.bnb_quantized):
            return self._quantize(device)
        else:
            # 如果存在量化状态，则将量化状态移动到指定设备
            if self.quant_state is not None:
                self.quant_state.to(device)

            # 创建新的参数对象，包含指定设备、数据类型、量化状态等信息
            new_param = Params4bit(super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                                   requires_grad=self.requires_grad, quant_state=self.quant_state,
                                   blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                   quant_type=self.quant_type)

            # 返回新的参数对象
            return new_param
class Linear4bit(nn.Linear):

    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4', quant_storage=torch.uint8, device=None):
        # 调用父类的初始化方法，设置输入特征数、输出特征数、是否包含偏置、设备
        super().__init__(input_features, output_features, bias, device)
        # 初始化权重参数为4位压缩的参数，不需要梯度，设置是否压缩统计信息、量化类型、量化存储类型、模块
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type, quant_storage=quant_storage, module=self)
        # 设置计算数据类型
        self.compute_dtype = compute_dtype
        # 计算数据类型是否已设置的标志
        self.compute_type_is_set = False
        # 量化状态
        self.quant_state = None
        # 量化存储类型
        self.quant_storage = quant_storage

    def set_compute_type(self, x):
        # 如果输入数据类型是torch.float32或torch.bfloat16，则设置计算数据类型为输入数据类型
        if x.dtype in [torch.float32, torch.bfloat16]:
            # 输入数据类型安全，可以在此类型上进行计算，为了速度和稳定性，切换到此类型
            self.compute_dtype = x.dtype
        # 如果输入数据类型是torch.float16
        elif x.dtype == torch.float16:
            # 如果计算数据类型是torch.float32且输入数据元素数量等于最后一个维度的数量
            if self.compute_dtype == torch.float32 and (x.numel() == x.shape[-1]):
                # 单批次推断，输入为torch.float16，计算数据类型为float32 -> 推断速度慢
                # 提醒用户
                warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference.')
                warnings.filterwarnings('ignore', message='.*inference.')
            # 如果计算数据类型是torch.float32且输入数据元素数量不等于最后一个维度的数量
            if self.compute_dtype == torch.float32 and (x.numel() != x.shape[-1]):
                # 提醒用户
                warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.')
                warnings.filterwarnings('ignore', message='.*inference or training')
    # 将权重和偏置保存到状态字典中，并填充状态字典的组件与量化状态的内容
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        save weight and bias,
        then fill state_dict with components of quant_state
        """
        # 调用父类方法保存权重和偏置
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # 如果权重具有量化状态，则将其内容填充到状态字典中
        if getattr(self.weight, "quant_state", None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    # 前向传播函数
    def forward(self, x: torch.Tensor):
        # 权重会自动转换为 Int8Params，但偏置需要手动转换
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # 如果权重没有量化状态
        if getattr(self.weight, 'quant_state', None) is None:
            if getattr(self, 'quant_state', None) is not None:
                # 当参数转换时，量化状态丢失。例如，对于 fsdp
                # 由于我们注册了模块，因此可以在此处恢复状态
                assert self.weight.shape[1] == 1
                if not isinstance(self.weight, Params4bit):
                    self.weight = Params4bit(self.weight, quant_storage=self.quant_storage)
                self.weight.quant_state = self.quant_state
            else:
                print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        
        # 如果计算类型未设置，则设置计算类型
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        # 如果计算数据类型不为 None，则将输入数据转换为该数据类型
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        # 如果偏置不为 None，则将其转换为计算数据类型
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        # 进行矩阵乘法运算，使用 4 位量化参数
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)

        # 将输出转换回输入数据类型
        out = out.to(inp_dtype)

        return out
# 定义一个继承自Linear4bit的类LinearFP4
class LinearFP4(Linear4bit):
    # 初始化函数，设置输入特征数、输出特征数、是否包含偏置、计算数据类型、是否压缩统计信息、量化存储类型、设备
    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_storage=torch.uint8, device=None):
        # 调用父类的初始化函数，设置参数
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'fp4', quant_storage, device)


# 定义一个继承自Linear4bit的类LinearNF4
class LinearNF4(Linear4bit):
    ''' Implements the NF4 data type.

        Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
        is normalized into the range [-1, 1].

        For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

        Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
        the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
    '''
    # 初始化函数，设置输入特征数、输出特征数、是否包含偏置、计算数据类型、是否压缩统计信息、量化存储类型、设备
    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_storage=torch.uint8, device=None):
        # 调用父类的初始化函数，设置参数
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'nf4', quant_storage, device)


# 定义一个继承自torch.nn.Parameter的类Int8Params
class Int8Params(torch.nn.Parameter):
    # 定义__new__方法，用于创建新的实例
    def __new__(
        cls,
        data=None,
        requires_grad=True,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    ):
        # 设置类属性has_fp16_weights为传入的值
        cls.has_fp16_weights = has_fp16_weights
        # 初始化类属性CB和SCB为None
        cls.CB = None
        cls.SCB = None
        # 如果data为None，则创建一个空的Tensor
        if data is None:
            data = torch.empty(0)
        # 返回一个新的Tensor实例
        return torch.Tensor._make_subclass(cls, data, requires_grad)
    # 将参数对象移动到指定设备上，如果具有 fp16 权重，则调用父类的 cuda 方法
    def cuda(self, device):
        if self.has_fp16_weights:
            return super().cuda(device)
        else:
            # 存储 8 位行主权重
            # 在第一次推理过程中，将此权重转换为转向/安培权重
            B = self.data.contiguous().half().cuda(device)
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            # 释放不再需要的变量
            del CBt
            del SCBt
            self.data = CB
            self.CB = CB
            self.SCB = SCB

        return self

    # 重载 to 方法，支持多种参数组合
    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[dtype, str]] = ...,
        non_blocking: bool = ...,
    ) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    # to 方法根据参数类型解析设备、数据类型、非阻塞标志等参数
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        # 如果设备不为空且为 cuda 类型，且当前数据在 CPU 上，则调用 cuda 方法
        if (
            device is not None
            and device.type == "cuda"
            and self.data.device.type == "cpu"
        ):
            return self.cuda(device)
        else:
            # 创建新的 Int8Params 对象，将参数转移到指定设备上
            new_param = Int8Params(
                super().to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                ),
                requires_grad=self.requires_grad,
                has_fp16_weights=self.has_fp16_weights,
            )
            new_param.CB = self.CB
            new_param.SCB = self.SCB

            return new_param
# 可能重新排列权重的函数，用于处理不同格式的权重数据
def maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # 获取指定前缀下的权重数据
    weight = state_dict.get(f"{prefix}weight")
    # 如果权重数据为空，则不进行任何操作
    if weight is None:
        return
    # 获取权重数据的格式，默认为"row"
    weight_format = state_dict.pop(f"{prefix}weight_format", "row")

    # 如果权重数据格式不是"row"，则获取瓦片索引并对权重数据进行布局还原
    if weight_format != "row":
        tile_indices = get_tile_inds(weight_format, weight.device)
        state_dict[f"{prefix}weight"] = undo_layout(weight, tile_indices)


# 继承自 nn.Linear 的自定义类 Linear8bitLt
class Linear8bitLt(nn.Linear):
    # 初始化函数，定义了一些参数和属性
    def __init__(self, input_features, output_features, bias=True, has_fp16_weights=True,
                       memory_efficient_backward=False, threshold=0.0, index=None, device=None):
        # 调用父类的初始化函数
        super().__init__(input_features, output_features, bias, device)
        # 断言，确保 memory_efficient_backward 参数不再需要，并在未来版本中将被移除
        assert not memory_efficient_backward, "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        # 初始化 MatmulLtState 对象和 index 属性
        self.state = bnb.MatmulLtState()
        self.index = index

        # 设置 state 对象的一些属性值
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        # 如果阈值大于0且没有使用 fp16 权重，则设置 use_pool 为 True
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        # 初始化 Int8Params 对象，处理权重数据
        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        # 注册加载状态字典前钩子函数 maybe_rearrange_weight
        self._register_load_state_dict_pre_hook(maybe_rearrange_weight)
    # 调用父类方法，将当前对象的状态保存到目标字典中
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # 定义 SCB 参数的名称
        scb_name = "SCB"

        # 情况1：调用了.cuda()方法，SCB 参数存储在self.weight中
        param_from_weight = getattr(self.weight, scb_name)
        # 情况2：调用了self.init_8bit_state方法，SCB 参数存储在self.state中
        param_from_state = getattr(self.state, scb_name)
        # 情况3：SCB 参数存储在self.state中，在第一次前向传播后，权重布局被重新排序
        layout_reordered = self.state.CxB is not None

        # 定义保存到目标字典中的键名
        key_name = prefix + f"{scb_name}"
        format_name = prefix + "weight_format"

        # 如果模型没有使用fp16权重
        if not self.state.has_fp16_weights:
            # 如果param_from_weight不为空，则将其保存到目标字典中，根据keep_vars决定是否保留梯度信息
            if param_from_weight is not None:
                destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
                destination[format_name] = "row"
            # 如果param_from_state不为空且布局未重新排序，则将其保存到目标字典中，根据keep_vars决定是否保留梯度信息
            elif param_from_state is not None and not layout_reordered:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = "row"
            # 如果param_from_state不为空，则将其保存到目标字典中，根据keep_vars决定是否保留梯度信息，并保存权重格式信息
            elif param_from_state is not None:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = self.state.formatB
    # 从给定的 state_dict 中加载模型参数，根据前缀 prefix 进行匹配
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 调用父类的加载函数，传入相应参数
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        # 复制一份未预期的键列表，以便在循环中修改
        unexpected_copy = list(unexpected_keys)

        # 遍历未预期的键列表
        for key in unexpected_copy:
            # 获取输入名称，去除前缀
            input_name = key[len(prefix):]
            # 如果输入名称为 "SCB"
            if input_name == "SCB":
                # 如果权重的 SCB 属性为 None
                if self.weight.SCB is None:
                    # 抛出异常，提示无法加载量化检查点到非量化的 Linear8bitLt 模型中
                    raise RuntimeError("Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                                       "not supported. Please call module.cuda() before module.load_state_dict()")

                # 获取输入参数，并将其复制给权重的 SCB 属性
                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)

                # 如果状态的 SCB 属性不为 None，则将其设置为权重的 SCB 属性
                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB

                # 从未预期的键列表中移除当前键
                unexpected_keys.remove(key)

    # 初始化 8 位状态
    def init_8bit_state(self):
        # 将状态的 CB 属性设置为权重的 CB 属性
        self.state.CB = self.weight.CB
        # 将状态的 SCB 属性设置为权重的 SCB 属性
        self.state.SCB = self.weight.SCB
        # 将权重的 CB 属性设置为 None
        self.weight.CB = None
        # 将权重的 SCB 属性设置为 None
        self.weight.SCB = None
    # 前向传播函数，接受输入张量 x
    def forward(self, x: torch.Tensor):
        # 设置当前状态为训练状态
        self.state.is_training = self.training
        # 如果权重的 CB 属性不为空，则初始化 8 位状态
        if self.weight.CB is not None:
            self.init_8bit_state()

        # 权重会自动转换为 Int8Params，但是偏置需要手动转换
        if self.bias is not None and self.bias.dtype != x.dtype:
            # 将偏置数据转换为与输入张量相同的数据类型
            self.bias.data = self.bias.data.to(x.dtype)

        # 使用 bnb.matmul 函数进行矩阵乘法运算，包括权重和偏置，同时传入状态信息
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        # 如果状态中没有 fp16 权重
        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # 在第一次推理过程中，我们将 8 位行主格式转换为图灵/安培格式
                # 不再需要行主格式的权重
                del self.state.CB
                # 更新权重数据为转换后的权重数据
                self.weight.data = self.state.CxB
        # 返回输出结果
        return out
# 创建一个自定义的线性层，继承自 nn.Linear
class OutlierAwareLinear(nn.Linear):
    # 初始化函数，定义输入特征数、输出特征数、是否包含偏置项、设备
    def __init__(self, input_features, output_features, bias=True, device=None):
        # 调用父类的初始化函数
        super().__init__(input_features, output_features, bias, device)
        # 初始化异常值维度为 None
        self.outlier_dim = None
        # 初始化是否量化为 False
        self.is_quantized = False

    # 带异常值的前向传播函数，需要被重写
    def forward_with_outliers(self, x, outlier_idx):
        raise NotImplementedError('Please override the `forward_with_outliers(self, x, outlier_idx)` function')

    # 量化权重的函数，需要被重写
    def quantize_weight(self, w, outlier_idx):
        raise NotImplementedError('Please override the `quantize_weights(self, w, outlier_idx)` function')

    # 前向传播函数
    def forward(self, x):
        # 如果异常值维度为 None
        if self.outlier_dim is None:
            # 获取 OutlierTracer 的实例
            tracer = OutlierTracer.get_instance()
            # 如果未初始化
            if not tracer.is_initialized():
                # 打印提示信息
                print('Please use OutlierTracer.initialize(model) before using the OutlierAwareLinear layer')
            # 获取权重的异常值索引
            outlier_idx = tracer.get_outliers(self.weight)
            # 将异常值索引和权重的 H 值打印出来
            #print(outlier_idx, tracer.get_hvalue(self.weight))
            # 将异常值索引赋值给异常值维度
            self.outlier_dim = outlier_idx

        # 如果未量化
        if not self.is_quantized:
            # 量化权重
            w = self.quantize_weight(self.weight, self.outlier_dim)
            # 将量化后的权重赋值给权重
            self.weight.data.copy_(w)
            # 设置为已量化
            self.is_quantized = True

# 创建一个带开关的线性层，继承自 nn.Linear
class SwitchBackLinearBnb(nn.Linear):
    # 初始化函数，定义输入特征数、输出特征数、是否包含偏置项、是否有 fp16 权重、是否内存高效反向传播、阈值、索引、设备
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
        device=None
    # 定义一个继承自父类的构造函数，初始化输入特征、输出特征、偏置和设备
    def __init__(
        input_features, output_features, bias, device
    ):
        # 调用父类的构造函数，初始化输入特征、输出特征、偏置和设备
        super().__init__(
            input_features, output_features, bias, device
        )
        # 初始化状态对象
        self.state = bnb.MatmulLtState()
        # 初始化索引
        self.index = index

        # 设置阈值
        self.state.threshold = threshold
        # 设置是否有 FP16 权重
        self.state.has_fp16_weights = has_fp16_weights
        # 设置是否内存高效反向传播
        self.state.memory_efficient_backward = memory_efficient_backward
        # 如果阈值大于0且没有 FP16 权重，则使用池化
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        # 初始化 Int8Params 对象
        self.weight = Int8Params(
            self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights
        )

    # 初始化8位状态
    def init_8bit_state(self):
        # 设置状态的 CB 属性
        self.state.CB = self.weight.CB
        # 设置状态的 SCB 属性
        self.state.SCB = self.weight.SCB
        # 将权重的 CB 属性设置为 None
        self.weight.CB = None
        # 将权重的 SCB 属性设置为 None
        self.weight.SCB = None

    # 前向传播函数
    def forward(self, x):
        # 设置状态的训练状态为当前模型的训练状态
        self.state.is_training = self.training

        # 如果权重的 CB 属性不为 None，则初始化8位状态
        if self.weight.CB is not None:
            self.init_8bit_state()

        # 执行混合矩阵乘法操作，得到输出并加上偏置
        out = bnb.matmul_mixed(x.half(), self.weight.half(), bias=None, state=self.state) + self.bias
```