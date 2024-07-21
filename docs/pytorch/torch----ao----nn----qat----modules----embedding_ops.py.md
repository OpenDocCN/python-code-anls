# `.\pytorch\torch\ao\nn\qat\modules\embedding_ops.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 导入 Tensor 数据类型
from torch import Tensor
# 导入神经网络模块
import torch.nn as nn
# 导入 PyTorch 函数库
import torch.nn.functional as F

# 定义模块公开接口列表
__all__ = ['Embedding', 'EmbeddingBag']

# 定义 Embedding 类，继承自 nn.Embedding
class Embedding(nn.Embedding):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
    for documentation.

    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    # 定义 FLOAT_MODULE 类属性，指定基础的 nn.Embedding 类
    _FLOAT_MODULE = nn.Embedding

    # 初始化函数，设置 Embedding 层的各种参数
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, device=None, dtype=None, qconfig=None) -> None:
        # 根据设备和数据类型创建工厂参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类 nn.Embedding 的初始化方法
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight,
                         **factory_kwargs)
        # 断言检查是否提供了 qconfig 参数
        assert qconfig, 'qconfig must be provided for QAT module'
        # 断言检查权重的量化模式是否为 per_channel_affine_float_qparams
        assert qconfig.weight().qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(qconfig.weight().qscheme)
        # 保存 qconfig 参数
        self.qconfig = qconfig
        # 使用 qconfig 创建权重的 FakeQuantize 模块
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    # 前向传播方法，接受输入 input，返回 Tensor
    def forward(self, input) -> Tensor:
        # 调用 F.embedding 函数，应用权重的 FakeQuantize 模块
        return F.embedding(input, self.weight_fake_quant(self.weight), self.padding_idx,
                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)

    # 类方法，用于创建 Embedding 实例
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        # 断言输入的模块类型必须与类属性 _FLOAT_MODULE 相同，否则抛出异常
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        # 断言输入的浮点模块必须有 qconfig 属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        # 断言 qconfig 属性不能为空
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        # 获取模块的量化配置中的权重量化方案
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        # 断言权重量化方案必须为 torch.per_channel_affine_float_qparams，否则抛出异常
        assert weight_qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(weight_qscheme)

        # 将输入模块的 qconfig 赋值给 qconfig 变量
        qconfig = mod.qconfig
        # 使用类构造函数创建量化感知训练模块 qat_embedding_bag
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.padding_idx,
                                mod.max_norm, mod.norm_type, mod.scale_grad_by_freq,
                                mod.sparse, mod.weight, qconfig=qconfig)

        # 返回构造的量化感知训练模块 qat_embedding_bag
        return qat_embedding_bag

    def to_float(self):
        # 创建一个新的浮点数嵌入模块 embedding_bag，其参数从当前量化感知训练模块中复制而来
        embedding_bag = torch.nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx,
                                           self.max_norm, self.norm_type, self.scale_grad_by_freq,
                                           self.sparse, None)
        # 将当前量化感知训练模块的权重作为浮点数嵌入模块的参数，并且将其设置为不可训练
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        # 设置浮点数嵌入模块的训练状态与当前模块一致
        embedding_bag.train(self.training)
        # 返回转换后的浮点数嵌入模块 embedding_bag
        return embedding_bag
# 定义一个继承自 nn.EmbeddingBag 的类 EmbeddingBag，用于支持量化感知训练，包含伪量化模块用于权重。

r"""
一个附带了 FakeQuantize 模块的嵌入包模块，用于量化感知训练。

我们采用与 `torch.nn.EmbeddingBag` 相同的接口，请参见
https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag
获取文档。

与 `torch.nn.EmbeddingBag` 类似，但 FakeQuantize 模块被初始化为默认值。

属性:
    weight: 权重的伪量化模块
"""

# 浮点数模块设为 nn.EmbeddingBag
_FLOAT_MODULE = nn.EmbeddingBag

def __init__(self, num_embeddings, embedding_dim, max_norm=None,
             norm_type=2.0, scale_grad_by_freq=False, mode='mean',
             sparse=False, _weight=None, include_last_offset=False,
             padding_idx=None, qconfig=None, device=None, dtype=None) -> None:
    # 用于创建张量的工厂参数
    factory_kwargs = {'device': device, 'dtype': dtype}
    # 调用父类构造函数初始化模块
    super().__init__(num_embeddings, embedding_dim, max_norm, norm_type,
                     scale_grad_by_freq, mode, sparse, _weight,
                     include_last_offset, padding_idx, **factory_kwargs)
    # 断言检查，确保 QAT 模块需要提供 qconfig
    assert qconfig, 'qconfig must be provided for QAT module'
    # 断言检查，确保 Embedding Bag 的权重需要 torch.per_channel_affine_float_qparams 的 qscheme
    assert qconfig.weight().qscheme == torch.per_channel_affine_float_qparams, \
        'Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
        str(qconfig.weight().qscheme)
    # 设置模块的量化配置
    self.qconfig = qconfig
    # 初始化权重伪量化模块
    self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor:
    # 调用 F.embedding_bag 函数进行前向传播计算
    return F.embedding_bag(input, self.weight_fake_quant(self.weight), offsets,
                           self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.mode, self.sparse,
                           per_sample_weights, self.include_last_offset,
                           self.padding_idx)

@classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        # 断言输入的模块类型必须是指定的浮点模块类型
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        # 断言输入的浮点模块必须定义了 qconfig 属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        # 断言 qconfig 属性不能为空
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        # 获取模块的权重量化方案
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        # 断言权重量化方案必须是 torch.per_channel_affine_float_qparams
        assert weight_qscheme == torch.per_channel_affine_float_qparams, \
            'Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got ' + \
            str(weight_qscheme)

        # 将输入模块的 qconfig 赋值给 qconfig 变量
        qconfig = mod.qconfig
        # 创建一个 qat 模块的嵌入包对象，使用输入模块的参数和 qconfig
        qat_embedding_bag = cls(mod.num_embeddings, mod.embedding_dim, mod.max_norm, mod.norm_type,
                                mod.scale_grad_by_freq, mod.mode, mod.sparse, mod.weight,
                                mod.include_last_offset, mod.padding_idx, qconfig=qconfig)

        return qat_embedding_bag

    def to_float(self):
        # 创建一个浮点数嵌入包对象，使用当前对象的参数
        embedding_bag = torch.nn.EmbeddingBag(self.num_embeddings, self.embedding_dim, self.max_norm,
                                              self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                              None, self.include_last_offset, self.padding_idx)
        # 将当前对象的权重作为浮点数参数的参数
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        # 设置嵌入包对象的训练状态为当前对象的训练状态
        embedding_bag.train(self.training)
        return embedding_bag
```