# `.\pytorch\torch\ao\nn\sparse\quantized\dynamic\linear.py`

```
# mypy: allow-untyped-defs
# 引入类型提示模块
from typing import Optional

# 引入 PyTorch 库
import torch
import torch.ao.nn.intrinsic as nni

# 引入稀疏量化线性模块相关库
from torch.ao.nn.sparse.quantized import linear
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern
from torch.ao.nn.quantized.modules.utils import _quantize_weight, _hide_packed_params_repr

# 模块中可导出的类名
__all__ = ['Linear']

# 稀疏动态量化线性模块
class Linear(torch.nn.Module):
    r"""
    一个使用浮点张量作为输入和输出的动态稀疏量化线性模块。
    """
    _version = 1
    _op_type = "sparse_dynamic"  # 操作类型为动态稀疏量化

    # 基于浮点数的线性模块
    _FLOAT_MODULE = torch.nn.Linear

    def __init__(self, in_features, out_features, row_block_size, col_block_size, bias=True, dtype=torch.qint8):
        super().__init__()

        # 仅支持 QINT8 类型的稀疏量化线性模块
        if dtype != torch.qint8:
            raise NotImplementedError("Only QINT8 is supported for Sparse Quantized Linear Dynamic")

        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数

        # 如果有偏置，则初始化为零张量，否则设为 None
        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)
        else:
            bias = None

        # 创建空的仿射量化权重
        qweight = torch._empty_affine_quantized([out_features, in_features],
                                                scale=1, zero_point=0, dtype=torch.qint8)

        # 初始化稀疏线性模块的压缩参数
        self._packed_params = linear.LinearPackedParams(row_block_size=row_block_size,
                                                        col_block_size=col_block_size,
                                                        dtype=dtype)
        self._packed_params.set_weight_bias(qweight, bias, row_block_size, col_block_size)

    # 获取模块的名称
    def _get_name(self):
        return 'SparseQuantizedDynamicLinear'

    # 额外的表示信息，包括输入特征数、输出特征数和量化方案
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, qscheme={self.weight().qscheme()}'

    # 返回对象的字符串表示，隐藏了压缩参数的表示
    def __repr__(self):
        return _hide_packed_params_repr(self, linear.LinearPackedParams)

    # 前向传播函数，调用了稀疏动态量化线性运算
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparse.qlinear_dynamic(x, self._packed_params._packed_params)

    # 将模块的状态字典保存到 destination 中，保留变量名
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'op_type'] = self._op_type
    # 从给定的状态字典中加载模型参数，更新当前对象的状态
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 获取操作类型并转换为整数
        op_type = int(state_dict[prefix + 'op_type'])
        # 断言操作类型为稀疏类型
        assert op_type == 'sparse', \
            f"Cannot load from op_type [{op_type}], expecting [{self._op_type}]"
        # 移除状态字典中的操作类型项
        state_dict.pop(prefix + 'op_type')

        # 获取本地元数据中的版本信息
        version = local_metadata.get('version', None)
        # 断言版本号不大于当前对象的版本号
        assert version <= self._version

        # 在旧的量化中，似乎使用这段代码加载旧模型
        # 从状态字典中弹出权重和偏置项
        weight = state_dict.pop(prefix + 'weight')
        bias = state_dict.pop(prefix + 'bias')
        # 更新状态字典，将权重和偏置项添加到打包参数中
        state_dict.update({prefix + '_packed_params.weight': weight,
                           prefix + '_packed_params.bias': bias})

        # 调用父类的_load_from_state_dict方法，加载状态字典的其余部分
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    # 返回打包参数对象的权重和偏置项
    def _weight_bias(self):
        return self._packed_params._weight_bias()

    # 返回打包参数对象的权重
    def weight(self):
        return self._weight_bias()[0]

    # 返回打包参数对象的偏置项
    def bias(self):
        return self._weight_bias()[1]

    # 设置打包参数对象的权重和偏置项，并指定行块大小和列块大小
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor],
                        row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
        # 断言行块大小和列块大小不为空
        assert row_block_size is not None and col_block_size is not None
        # 设置输出特征数量和输入特征数量
        self.out_features = w.shape[0]
        self.in_features = w.shape[1]
        # 调用打包参数对象的设置权重和偏置项方法
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized sparse dynamic module from a float module.

        We only care about the convert at this stage, no need for observers just yet.
        """
        # 断言输入的模块类型必须与类的_FLOAT_MODULE相同，否则抛出异常
        assert type(mod) == cls._FLOAT_MODULE, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        # 断言输入的浮点模块必须有定义 qconfig 属性
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'

        # 如果模块类型是 nni.LinearReLU，则取其第一个模块
        if type(mod) == nni.LinearReLU:
            mod = mod[0]

        # 根据模块的 qconfig 来选择权重观察器，如果没有指定，则使用默认的动态量化配置
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # 解决循环导入问题，延迟导入 torch.ao.quantization.qconfig 中的 default_dynamic_qconfig
            from torch.ao.quantization.qconfig import default_dynamic_qconfig
            weight_observer = default_dynamic_qconfig.weight()

        # 在调用 weight_observer 之前，乘以模块的掩码（如果存在）
        weight = mod.weight
        if getattr(mod.qconfig, 'mask', False):
            weight = mod.qconfig.mask * mod.weight

        # 调用 weight_observer 观察权重，获取数据类型
        weight_observer(weight)
        dtype = weight_observer.dtype

        # 断言权重观察器的数据类型必须是 torch.qint8
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'

        # 计算量化参数 w_sc（缩放因子）和 w_zp（零点）
        w_sc, w_zp = weight_observer.calculate_qparams()

        # 如果 w_zp 是 torch.Tensor 类型，则所有的零点必须映射到0
        if isinstance(w_zp, torch.Tensor):
            assert not torch.any(w_zp.bool()), "All weight zero points must map to 0"
        else:
            assert w_zp == 0, 'Weight zero point must map to 0'

        # 对权重进行量化
        qweight = _quantize_weight(weight.float(), weight_observer)

        # 获取行块和列块的大小
        row_block_size, col_block_size = LinearBlockSparsePattern.block_size()

        # 使用类的构造函数创建量化后的稀疏动态模块 qlinear
        qlinear = cls(mod.in_features,
                      mod.out_features,
                      row_block_size,
                      col_block_size,
                      dtype=dtype)

        # 设置 qlinear 的权重和偏置
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)

        # 返回创建的量化稀疏动态模块 qlinear
        return qlinear
```