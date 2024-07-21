# `.\pytorch\torch\ao\nn\sparse\quantized\linear.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Optional

# 导入 PyTorch 库
import torch
# 导入量化模块相关的工具函数
from torch.ao.nn.quantized.modules.utils import _quantize_weight, _hide_packed_params_repr

# 定义模块导出的符号列表
__all__ = ['LinearPackedParams', 'Linear']

# TODO (zaf): Inherit from `quantized.LinearPackedParams` (T83294430)
# 稀疏量化线性参数的类，继承自 torch.nn.Module
class LinearPackedParams(torch.nn.Module):
    _version = 1

    # 初始化函数，设置行块大小、列块大小和数据类型（默认为 torch.qint8）
    def __init__(self, row_block_size=1, col_block_size=4, dtype=torch.qint8):
        super().__init__()

        # 如果数据类型不是 qint8，抛出 NotImplementedError
        if dtype != torch.qint8:
            raise NotImplementedError("Linear prepacking only supports QINT8")
        self.dtype = dtype
        # 创建一个空的量化张量 wq
        wq = torch._empty_affine_quantized([1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        # 设置权重和偏置（此处偏置为 None），调用 set_weight_bias 方法
        self.set_weight_bias(wq, None, row_block_size, col_block_size)

    # 返回模块的名称
    def _get_name(self):
        return "SparseQuantizedLinearPackedParams"

    # 设置权重和偏置的方法，调用了 sparse.qlinear_prepack 运算符
    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor],
                        row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
        assert row_block_size is not None and col_block_size is not None
        # 将权重和偏置预打包为 packed_params
        self._packed_params = torch.ops.sparse.qlinear_prepack(weight, bias, row_block_size, col_block_size)

    # 返回权重和偏置的方法，调用了 sparse.qlinear_unpack 运算符
    @torch.jit.export
    def _weight_bias(self):
        # 解包 _packed_params 得到权重、偏置和块大小
        (weight, bias, block_sizes) = torch.ops.sparse.qlinear_unpack(self._packed_params)
        return (weight, bias, block_sizes[0], block_sizes[1])

    # 前向传播方法，简单地返回输入 x
    def forward(self, x):
        return x

    # 将模块的状态字典保存到 destination 中，包括 dtype 和 _packed_params
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + '_packed_params'] = self._weight_bias()

    # 从状态字典中加载模块的状态，包括 dtype 和 _packed_params
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        assert version <= self._version

        self.dtype = state_dict.pop(prefix + 'dtype')
        # 从状态字典中加载权重、偏置和块大小，并调用 set_weight_bias 方法
        weight, bias, row_block_size, col_block_size = state_dict.pop(prefix + '_packed_params')
        self.set_weight_bias(weight, bias, row_block_size, col_block_size)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    # 返回模块的序列化状态，包括 _packed_params、training 和 dtype
    @torch.jit.export
    def __getstate__(self):
        return self._packed_params, self.training, self.dtype

    # 设置模块的状态，接收状态元组 state，包括 _packed_params、training 和 dtype
    @torch.jit.export
    def __setstate__(self, state):
        (self._packed_params, self.training, self.dtype) = state

    # 返回模块的字符串表示形式，调用 _weight_bias 方法
    def __repr__(self):
        return self._weight_bias().__repr__()

# TODO (zaf): Inherit from `quantized.Linear` (T83294430)
# 稀疏量化线性模块，继承自 torch.nn.Module
class Linear(torch.nn.Module):
    r"""
    A quantized sparse linear module with quantized tensor as inputs and outputs.
    """
    _version = 1
    _FLOAT_MODULE = torch.nn.Linear
    def __init__(self, in_features, out_features, row_block_size, col_block_size, bias=True, dtype=torch.qint8):
        super().__init__()  # 调用父类的构造函数

        if dtype != torch.qint8:
            raise NotImplementedError("Only QINT8 is supported for Sparse Quantized Linear")

        self.in_features = in_features  # 初始化输入特征数
        self.out_features = out_features  # 初始化输出特征数

        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)  # 如果有偏置，则初始化为全零张量
        else:
            bias = None  # 如果没有偏置，则设为None

        qweight = torch._empty_affine_quantized([out_features, in_features],  # 创建空的量化权重张量
                                                scale=1, zero_point=0, dtype=torch.qint8)
        self._packed_params = LinearPackedParams(row_block_size=row_block_size,  # 创建线性层的打包参数对象
                                                 col_block_size=col_block_size,
                                                 dtype=dtype)
        self._packed_params.set_weight_bias(qweight, bias, row_block_size, col_block_size)  # 设置权重和偏置到打包参数对象中
        self.scale = 1.0  # 初始化量化参数：缩放因子
        self.zero_point = 0  # 初始化量化参数：零点

    @classmethod
    def _get_name(cls):
        return 'SparseQuantizedLinear'  # 返回类的名称字符串

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}, '  # 返回额外的字符串表示，包括输入、输出特征数以及量化参数
                f'zero_point={self.zero_point}, qscheme={self.weight().qscheme()}')

    def __repr__(self):
        return _hide_packed_params_repr(self, LinearPackedParams)  # 返回对象的字符串表示形式，隐藏打包参数的具体表示

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparse.qlinear(x, self._packed_params._packed_params, self.scale, self.zero_point)  # 前向传播函数，调用稀疏量化线性操作

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)  # 调用父类的保存状态字典方法

        destination[prefix + 'scale'] = torch.tensor(self.scale)  # 将缩放因子保存到状态字典中
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)  # 将零点保存到状态字典中

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict[prefix + 'scale'])  # 从状态字典中加载并设置缩放因子
        state_dict.pop(prefix + 'scale')  # 从状态字典中移除缩放因子项

        self.zero_point = int(state_dict[prefix + 'zero_point'])  # 从状态字典中加载并设置零点
        state_dict.pop(prefix + 'zero_point')  # 从状态字典中移除零点项

        op_type = int(state_dict[prefix + 'op_type'])  # 从状态字典中加载操作类型（未使用）
        state_dict.pop(prefix + 'op_type')  # 从状态字典中移除操作类型项

        version = local_metadata.get('version', None)  # 获取本地元数据中的版本信息
        assert version <= self._version  # 断言当前版本不超过类的版本

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)  # 调用父类的从状态字典加载方法

    def _weight_bias(self):
        return self._packed_params._weight_bias()  # 返回打包参数对象中的权重和偏置

    def weight(self):
        return self._weight_bias()[0]  # 返回打包参数对象中的权重

    def bias(self):
        return self._weight_bias()[1]  # 返回打包参数对象中的偏置
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor],
                        row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
        # 确保 row_block_size 和 col_block_size 都不为 None
        assert row_block_size is not None and col_block_size is not None
        # 调用内部的 _packed_params 对象的方法，设置权重和偏置以及块大小
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized sparse module from a float module.

        We only care about the convert at this stage, no need for observers just yet.

        TODO(zaf): Need to add the sparse params to the qconfig
        """
        # 断言输入的 mod 类型必须是 cls._FLOAT_MODULE
        assert type(mod) == cls._FLOAT_MODULE, cls._get_name() + \
            '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        # 断言输入的 mod 必须具有 'sparse_params' 属性
        assert hasattr(mod, 'sparse_params'), \
            ('Expecting the Linear to have `sparse_params`. Make sure you have provided arguments '
             'in the `sparsifier.squash_mask(params_to_save=("sparse_block_shape",))` method.')
        # 获取稀疏块形状参数
        sparse_block_shape = mod.sparse_params.get('sparse_block_shape', None)  # type: ignore[operator, union-attr]
        # 确保稀疏块形状是 tuple 或 list，且长度为 2
        assert isinstance(sparse_block_shape, (tuple, list))
        assert len(sparse_block_shape) == 2
        # TODO: 需要向 qconfig 添加选项以避免校准。
        # TODO: 为稀疏性添加校准
        # 断言输入的 mod 必须具有 'qconfig' 属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        # 获取激活后处理器
        activation_post_process = mod.activation_post_process
        # 获取权重后处理器
        weight_post_process = mod.qconfig.weight()  # type: ignore[operator, union-attr]

        # 假设权重已经被 `sparsifier.convert` 稀疏化
        weight = mod.weight

        # 对权重应用权重后处理器
        weight_post_process(weight)
        # 获取权重的数据类型
        dtype = weight_post_process.dtype
        # 计算激活后处理器的量化参数
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[operator, union-attr]
        # 断言权重数据类型必须是 torch.qint8
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        # 获取权重后处理器的量化参数
        w_sc, w_zp = weight_post_process.calculate_qparams()
        # 如果 w_zp 是 torch.Tensor，则确保所有的权重零点映射到 0
        if isinstance(w_zp, torch.Tensor):
            assert not torch.any(w_zp.bool()), "All weight zero points must map to 0"
        else:
            assert w_zp == 0, 'Weight zero point must map to 0'
        # 对权重进行量化
        qweight = _quantize_weight(weight.float(), weight_post_process)

        # 获取稀疏块的行块大小和列块大小
        row_block_size = mod.sparse_params['sparse_block_shape'][0]  # type: ignore[index]
        col_block_size = mod.sparse_params['sparse_block_shape'][1]  # type: ignore[index]
        # 创建一个量化稀疏线性模块实例
        qlinear = cls(mod.in_features,
                      mod.out_features,
                      row_block_size,
                      col_block_size,
                      dtype=dtype)
        # 设置权重和偏置以及块大小到量化稀疏线性模块中
        qlinear.set_weight_bias(qweight, mod.bias,
                                row_block_size, col_block_size)  # type: ignore[arg-type]
        # 设置量化线性模块的缩放因子和零点
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        # 返回量化稀疏线性模块
        return qlinear
```