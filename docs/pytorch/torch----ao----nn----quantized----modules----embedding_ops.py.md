# `.\pytorch\torch\ao\nn\quantized\modules\embedding_ops.py`

```py
# mypy: allow-untyped-defs
# 导入PyTorch库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入Tensor类型
from torch import Tensor  # noqa: F401
# 导入Optional和List类型
from torch._jit_internal import Optional, List  # noqa: F401

# 导入本地的工具函数
from .utils import _hide_packed_params_repr
from .utils import _quantize_weight

# 定义模块导出的符号列表
__all__ = ['EmbeddingPackedParams', 'Embedding', 'EmbeddingBag']

# 定义EmbeddingPackedParams类，继承自torch.nn.Module
class EmbeddingPackedParams(torch.nn.Module):
    # 类版本号
    _version = 1

    # 初始化方法，接受num_embeddings（嵌入数量）、embedding_dim（嵌入维度）、dtype（数据类型，默认为torch.quint8）
    def __init__(self, num_embeddings, embedding_dim, dtype=torch.quint8):
        super().__init__()
        # 设置数据类型
        self.dtype = dtype
        # 如果数据类型是torch.quint8或torch.quint4x2
        if self.dtype in [torch.quint8, torch.quint4x2]:
            # 创建用于量化的刻度和零点张量
            scales = torch.ones(num_embeddings, dtype=torch.float)
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)
            # 创建量化权重张量
            wq = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim], scales=scales,
                                                           zero_points=zero_points,
                                                           axis=0, dtype=self.dtype)
            # 设置权重
            self.set_weight(wq)
        else:
            # 抛出错误，不支持的数据类型
            raise NotImplementedError(f'Unsupported dtype on quantized embedding! Supports quint8 and quint4x2. Got dtype: {dtype}')

    # 设置权重的方法，接受权重张量，并且没有返回值
    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None:
        # 如果数据类型是torch.quint8或torch.quint4x2
        if self.dtype in [torch.quint8, torch.quint4x2]:
            # 使用PyTorch运算符进行量化嵌入预打包
            self._packed_weight = torch.ops.quantized.embedding_bag_prepack(weight)
        else:
            # 抛出错误，不支持的数据类型
            raise NotImplementedError('Unsupported dtype for quantized embedding prepack! Supports quint8 and quint4x2.')

    # 获取权重的方法，返回权重张量
    @torch.jit.export
    def _weight(self):
        # 如果数据类型是torch.quint8或torch.quint4x2
        if self.dtype in [torch.quint8, torch.quint4x2]:
            # 使用PyTorch运算符进行量化嵌入解包
            return torch.ops.quantized.embedding_bag_unpack(self._packed_weight)
        else:
            # 抛出错误，不支持的数据类型
            raise NotImplementedError('Unsupported dtype for quantized embedding unpack! Supports quint8 and quint4x2.')

    # 前向传播方法，直接返回输入张量x
    def forward(self, x):
        return x

    # 版本1
    #   self
    #   |--- _packed_weight : 表示EmbeddingPackedParamsBase的权重张量
    #   |--- dtype : torch.dtype

    # 将模块保存到状态字典的方法，保存dtype和权重
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + '_packed_weight'] = self._weight()

    # 从状态字典加载模块的方法，加载dtype和权重
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.dtype = state_dict[prefix + 'dtype']
        state_dict.pop(prefix + 'dtype')

        weight = state_dict[prefix + '_packed_weight']
        state_dict.pop(prefix + '_packed_weight')
        self.set_weight(weight)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    # 返回对象的字符串表示形式，使用权重的字符串表示形式
    def __repr__(self):
        return self._weight().__repr__()

# 定义Embedding类，继承自torch.nn.Module
class Embedding(torch.nn.Module):
    r"""
    A quantized Embedding module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html for documentation.

    Similar to :class:`~torch.nn.Embedding`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.Embedding(num_embeddings=10, embedding_dim=12)
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        >>> output = m(indices)
        >>> print(output.size())
        torch.Size([9, 12])

    """
    _version = 1  # 模块版本号

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, dtype=torch.quint8) -> None:
        super().__init__()  # 调用父类构造函数初始化

        self.num_embeddings = num_embeddings  # 设置嵌入矩阵的行数（嵌入的数量）
        self.embedding_dim = embedding_dim  # 设置每个嵌入向量的维度
        self.dtype = dtype  # 设置数据类型，默认为量化8位整数

        if _weight is None:
            scales = torch.ones(num_embeddings, dtype=torch.float)  # 创建全为1的缩放因子张量
            zero_points = torch.zeros(num_embeddings, dtype=torch.float)  # 创建全为0的零点张量
            qweight = torch._empty_per_channel_affine_quantized([num_embeddings, embedding_dim],
                                                                scales=scales, zero_points=zero_points,
                                                                axis=0, dtype=torch.quint8)  # 创建空的通道仿射量化权重
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            qweight = _weight  # 使用给定的权重初始化量化权重

        self._packed_params = EmbeddingPackedParams(num_embeddings, embedding_dim, dtype)  # 创建嵌入参数包装对象
        self._packed_params.set_weight(qweight)  # 设置嵌入参数包装对象的权重

    def forward(self, indices: Tensor) -> Tensor:
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_4bit(self._packed_params._packed_weight, indices)  # 使用4位量化执行嵌入操作
        else:
            return torch.ops.quantized.embedding_byte(self._packed_params._packed_weight, indices)  # 使用8位量化执行嵌入操作

    def _get_name(self):
        return 'QuantizedEmbedding'  # 返回模块名称字符串

    def __repr__(self):
        return _hide_packed_params_repr(self, EmbeddingPackedParams)  # 返回包含嵌入参数的字符串表示形式

    def extra_repr(self):
        extra_repr_str = (f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, '
                          f'dtype={self._packed_params.dtype}, qscheme={self.weight().qscheme()}')  # 返回额外的字符串表示形式，包括嵌入矩阵的维度、数据类型和量化方案
        return extra_repr_str

    def set_weight(self, w: torch.Tensor) -> None:
        self._packed_params.set_weight(w)  # 设置嵌入参数包装对象的权重

    def weight(self):
        return self._packed_params._weight()  # 返回嵌入参数包装对象的权重

    @classmethod
    # 从浮点模块创建一个量化的嵌入模块
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
        # 如果浮点模块有 'weight_fake_quant' 属性
        if hasattr(mod, 'weight_fake_quant'):
            # 断言模块类型为 torch.ao.nn.qat.Embedding，只有这种类型支持带有伪量化的情况
            assert type(mod) == torch.ao.nn.qat.Embedding, 'nnq.' + cls.__name__ + '.from_float ' + \
                'with fake quant only works for ' + torch.ao.nn.qat.Embedding.__name__
            weight_observer = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            # 断言模块类型为 nn.Embedding，只有这种类型支持没有伪量化的情况
            assert type(mod) == nn.Embedding, 'nnq.' + cls.__name__ + '.from_float only works for ' + \
                nn.Embedding.__name__
            # 断言浮点模块有 'qconfig' 属性，用于定义量化配置
            assert hasattr(mod, 'qconfig'), 'Embedding input float module must have qconfig defined'
            from torch.ao.quantization import float_qparams_weight_only_qconfig
            # 如果模块的 qconfig 存在且 weight 不为 None
            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                weight_observer = float_qparams_weight_only_qconfig.weight()

        dtype = weight_observer.dtype
        is_float_qparams_qconfig = weight_observer.qscheme == torch.per_channel_affine_float_qparams
        # 断言使用的量化配置为 float_qparams_weight_only_qconfig
        assert is_float_qparams_qconfig, \
            'Embedding quantization is only supported with float_qparams_weight_only_qconfig.'

        # 断言权重的数据类型为 torch.quint8 或 torch.quint4x2
        assert dtype == torch.quint8 or dtype == torch.quint4x2, \
            f'The only supported dtype for nnq.Embedding is torch.quint8 and torch.quint4x2, got {dtype}'

        # 运行观察器以计算量化参数
        weight_observer(mod.weight)
        # 对权重进行量化
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # 创建量化的嵌入模块并传入量化的权重
        qembedding = Embedding(mod.num_embeddings, mod.embedding_dim)
        qembedding.set_weight(qweight)
        return qembedding

    @classmethod
    def from_reference(cls, ref_embedding):
        # 从参考嵌入模块创建一个新的嵌入模块
        qembedding = cls(
            ref_embedding.num_embeddings,
            ref_embedding.embedding_dim,
            ref_embedding.padding_idx,
            ref_embedding.max_norm,
            ref_embedding.norm_type,
            ref_embedding.scale_grad_by_freq,
            ref_embedding.sparse,
            ref_embedding.get_quantized_weight(),
            ref_embedding.weight_dtype,
        )
        return qembedding
class EmbeddingBag(Embedding):
    r"""
    A quantized EmbeddingBag module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html for documentation.

    Similar to :class:`~torch.nn.EmbeddingBag`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{num\_embeddings}, \text{embedding\_dim})`.

    Examples::
        >>> m = nn.quantized.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12])

    """
    _version = 1

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 mode: str = 'sum', sparse: bool = False, _weight: Optional[Tensor] = None,
                 include_last_offset: bool = False, dtype=torch.quint8) -> None:
        # 调用父类的构造函数，初始化 Embedding 模块
        super().__init__(num_embeddings, embedding_dim, _weight=_weight, dtype=dtype)

        # 设置模式
        self.mode = mode
        # 是否对权重进行剪枝
        self.pruned_weights = False
        # 是否包含最后一个偏移量
        self.include_last_offset = include_last_offset
        # 数据类型
        self.dtype = dtype

    def forward(self, indices: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None,
                compressed_indices_mapping: Optional[Tensor] = None) -> Tensor:
        # 根据数据类型选择不同的量化方法
        if self.dtype == torch.quint4x2:
            return torch.ops.quantized.embedding_bag_4bit(self._packed_params._packed_weight, indices, offsets, False, 0,
                                                          self.pruned_weights, per_sample_weights, compressed_indices_mapping,
                                                          self.include_last_offset)
        else:
            return torch.ops.quantized.embedding_bag_byte(self._packed_params._packed_weight, indices, offsets, False, 0,
                                                          self.pruned_weights, per_sample_weights, compressed_indices_mapping,
                                                          self.include_last_offset)

    def _get_name(self):
        return 'QuantizedEmbeddingBag'

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
            use_precomputed_fake_quant (bool, optional): Whether to use precomputed fake quantization. Default is False.
        """
        # 检查输入的浮点模块是否已经定义了 weight_fake_quant 属性
        if hasattr(mod, 'weight_fake_quant'):
            # 如果定义了，使用 mod 的 weight_fake_quant 作为 weight_observer
            weight_observer = mod.weight_fake_quant
        else:
            # 如果未定义，确保 mod 是 nn.EmbeddingBag 类型的模块
            assert type(mod) == nn.EmbeddingBag, 'nnq.' + cls.__name__ + '.from_float only works for ' + \
                nn.EmbeddingBag.__name__
            # 确保 mod 的 qconfig 属性已定义
            assert hasattr(mod, 'qconfig'), 'EmbeddingBag input float module must have qconfig defined'
            # 如果 mod 的 qconfig 属性为 weight 不为 None，使用 mod.qconfig.weight() 创建 weight_observer
            from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig
            if mod.qconfig is not None and mod.qconfig.weight is not None:  # type: ignore[union-attr]
                weight_observer = mod.qconfig.weight()  # type: ignore[union-attr, operator]
            else:
                # 否则，使用 float_qparams_weight_only_qconfig.weight() 创建 weight_observer
                weight_observer = float_qparams_weight_only_qconfig.weight()

        # 获取 weight_observer 的数据类型
        dtype = weight_observer.dtype
        # 确保 weight_observer 使用的是 torch.per_channel_affine_float_qparams 的量化方案
        is_float_qparams_qconfig = weight_observer.qscheme == torch.per_channel_affine_float_qparams
        assert is_float_qparams_qconfig, \
            'EmbeddingBag quantization is only supported with float_qparams_weight_only_qconfig.'

        # 确保 dtype 是 torch.quint8 或 torch.quint4x2
        assert dtype == torch.quint8 or dtype == torch.quint4x2, \
            f'The only supported dtype for nnq.EmbeddingBag is torch.quint8 and torch.quint4x2, got {dtype}'

        # 运行 observer 来计算量化参数
        weight_observer(mod.weight)
        # 对 mod.weight 进行量化得到 qweight
        qweight = _quantize_weight(mod.weight.float(), weight_observer)

        # 创建量化的 EmbeddingBag 模块并传入量化的权重
        qembedding_bag = EmbeddingBag(mod.num_embeddings, mod.embedding_dim, dtype=dtype)
        qembedding_bag.set_weight(qweight)
        return qembedding_bag

    @classmethod
    def from_reference(cls, ref_embedding_bag):
        # 使用参考的 ref_embedding_bag 创建一个新的量化 EmbeddingBag 模块
        qembedding_bag = cls(
            ref_embedding_bag.num_embeddings,
            ref_embedding_bag.embedding_dim,
            ref_embedding_bag.max_norm,
            ref_embedding_bag.norm_type,
            ref_embedding_bag.scale_grad_by_freq,
            ref_embedding_bag.mode,
            ref_embedding_bag.sparse,
            ref_embedding_bag.get_quantized_weight(),
            ref_embedding_bag.include_last_offset,
            ref_embedding_bag.weight_dtype,
        )
        return qembedding_bag
```