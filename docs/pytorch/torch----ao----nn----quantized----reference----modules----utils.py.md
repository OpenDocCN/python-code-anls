# `.\pytorch\torch\ao\nn\quantized\reference\modules\utils.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库和 typing 模块
import torch
import typing

# 定义模块的公开接口列表
__all__ = [
    "ReferenceQuantizedModule",
]

# 定义参考量化模块类
class ReferenceQuantizedModule(torch.nn.Module):
    def get_weight(self):
        """
        Fake quantize (quantize and dequantize) the weight with
        the quantization parameters for weight, this is used to
        simulate the numerics for the quantized weight in a quantized
        model
        """
        # 确保 self.weight_scale 和 self.weight_zero_point 是 torch.Tensor 类型
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        
        # 如果模块是分解的
        if self.is_decomposed:
            # 调用 _quantize_and_dequantize_weight_decomposed 函数，将权重进行伪量化
            return _quantize_and_dequantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max)
        else:
            # 调用 _quantize_and_dequantize_weight 函数，将权重进行伪量化
            return _quantize_and_dequantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int)

    def get_quantized_weight(self):
        # 确保 self.weight_scale 和 self.weight_zero_point 是 torch.Tensor 类型
        assert isinstance(self.weight_scale, torch.Tensor)
        assert isinstance(self.weight_zero_point, torch.Tensor)
        
        # 如果模块是分解的
        if self.is_decomposed:
            # 调用 _quantize_weight_decomposed 函数，将权重进行量化
            return _quantize_weight_decomposed(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max)
        else:
            # 调用 _quantize_weight 函数，将权重进行量化
            return _quantize_weight(
                self.weight,  # type: ignore[arg-type]
                self.weight_qscheme,
                self.weight_dtype,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # 调用父类的 _save_to_state_dict 方法，保存模块的状态字典
        super()._save_to_state_dict(destination, prefix, keep_vars)
        
        # 调用 _save_weight_qparams 函数，保存权重的量化参数到状态字典中
        _save_weight_qparams(
            destination, prefix, self.weight_qscheme, self.weight_dtype,
            self.weight_scale, self.weight_zero_point, self.weight_axis)
    # 从给定的状态字典中加载模型的权重和量化参数
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 获取需要加载的权重和量化参数的键列表
        for key in _get_weight_qparam_keys(state_dict, prefix):
            # 将状态字典中的值设置为当前对象的属性，并从状态字典中移除已加载的键值对
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        # 调用父类的_load_from_state_dict方法，传递必要的参数和设置strict为False
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)
# 对权重进行分解量化，返回量化后的权重张量
def _quantize_weight_decomposed(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis: int,
        weight_quant_min: typing.Optional[int],
        weight_quant_max: typing.Optional[int],
) -> torch.Tensor:
    # 定义不同数据类型的量化值范围
    _DTYPE_TO_QVALUE_BOUNDS = {
        torch.uint8: (0, 255),
        torch.int8: (-128, 127),
        torch.int32: (-(2**31), 2**31 - 1),
    }
    # TODO: 添加一个实用函数，用于将量化数据类型转换为标准数据类型

    # 如果量化方案为每张量基准
    if weight_qscheme == torch.per_tensor_affine:
        # 如果权重数据类型在支持的量化类型内
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 获取底层整数表示的数据类型
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            # 如果未提供量化范围，则使用默认范围
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
            # 调用量化函数，返回量化后的权重张量
            weight = torch.ops.quantized_decomposed.quantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_
            )
            return weight
    # 如果量化方案为每通道基准或浮点量化参数的每通道基准
    elif weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        # TODO: 不支持 torch.quint4x2
        # 如果权重数据类型在支持的量化类型内
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 获取底层整数表示的数据类型
            weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
            # 如果未提供量化范围，则使用默认范围
            if weight_quant_min is None or weight_quant_max is None:
                weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
            # 调用每通道量化函数，返回量化后的权重张量
            weight = torch.ops.quantized_decomposed.quantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_)  # type: ignore[arg-type]
            return weight
    # 如果既不是每张量基准也不是每通道基准，抛出错误
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")

# 对分解量化后的权重进行反量化，返回反量化后的权重张量
def _dequantize_weight_decomposed(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis: int,
        weight_quant_min: typing.Optional[int],
        weight_quant_max: typing.Optional[int],
) -> torch.Tensor:
    # TODO: 从激活后处理中获取量化最小值和最大值
    _DTYPE_TO_QVALUE_BOUNDS = {
        torch.uint8: (0, 255),
        torch.int8: (-128, 127),
        torch.int32: (-(2**31), 2**31 - 1),
    }
    # TODO: 添加一个实用函数，用于将量化数据类型转换为标准数据类型
    # 定义一个字典，将量化数据类型映射到其对应的底层整数表示数据类型
    _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }
    # 将传入的量化权重数据类型映射到其底层整数表示数据类型
    weight_dtype_ = _QDTYPE_TO_UNDERLYING_INT_REPR_DTYPE[weight_dtype]
    # 如果未提供量化权重的最小值或最大值，则使用预定义的边界
    if weight_quant_min is None or weight_quant_max is None:
        weight_quant_min, weight_quant_max = _DTYPE_TO_QVALUE_BOUNDS[weight_dtype_]
    # 如果权重的量化方案为每张量仿射，则执行如下操作
    if weight_qscheme == torch.per_tensor_affine:
        # 如果权重数据类型为 quint8、qint8 或 qint32 中的一种，则进行解量化操作
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                weight,
                weight_scale,
                weight_zero_point,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_
            )
            # 返回解量化后的权重张量
            return weight
    # 如果权重的量化方案为每通道仿射或每通道仿射浮点量化参数，则执行如下操作
    elif weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        # TODO: 不支持 torch.quint4x2
        # 如果权重数据类型为 quint8、qint8 或 qint32 中的一种，则进行每通道解量化操作
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.ops.quantized_decomposed.dequantize_per_channel(
                weight,
                weight_scale,
                weight_zero_point,
                weight_axis,
                weight_quant_min,
                weight_quant_max,
                weight_dtype_)  # type: ignore[arg-type]
            # 返回解量化后的权重张量
            return weight
    # 如果不支持的组合出现，则引发 ValueError 异常
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")
def _quantize_weight(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis_int: int
) -> torch.Tensor:
    """ 对权重进行量化处理。

    Args:
        weight: 待量化的权重张量
        weight_qscheme: 量化方案，可以是 torch.per_tensor_affine, torch.per_channel_affine 或 torch.per_channel_affine_float_qparams
        weight_dtype: 权重的数据类型
        weight_scale: 量化的比例因子张量
        weight_zero_point: 量化的零点张量
        weight_axis_int: 量化轴的整数值

    Returns:
        torch.Tensor: 量化后的权重张量
    """
    if weight_dtype == torch.float16:
        # 如果权重数据类型为 torch.float16，则将权重转换为 torch.float16 后返回
        weight = weight.to(weight_dtype)
        return weight

    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 如果量化方案为 per_tensor_affine，且权重数据类型符合条件，则进行张量级别的量化
            weight = torch.quantize_per_tensor(weight, weight_scale, weight_zero_point, weight_dtype)
            return weight
    elif weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        if weight_dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
            # 如果量化方案为 per_channel_affine 或 per_channel_affine_float_qparams，且权重数据类型符合条件，
            # 则进行通道级别的量化
            weight = torch.quantize_per_channel(
                weight, weight_scale, weight_zero_point, weight_axis_int, weight_dtype)  # type: ignore[arg-type]
            return weight

    # 如果未匹配到支持的量化方案和数据类型组合，则抛出异常
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")

def _quantize_and_dequantize_weight_decomposed(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis_int: int,
        weight_quant_min: typing.Optional[int],
        weight_quant_max: typing.Optional[int],
) -> torch.Tensor:
    """ 根据量化参数对权重进行分解后的量化和反量化处理。

    Args:
        weight: 待量化的权重张量
        weight_qscheme: 量化方案，可以是 torch.per_tensor_affine, torch.per_channel_affine 或 torch.per_channel_affine_float_qparams
        weight_dtype: 权重的数据类型
        weight_scale: 量化的比例因子张量
        weight_zero_point: 量化的零点张量
        weight_axis_int: 量化轴的整数值
        weight_quant_min: 量化的最小值（可选）
        weight_quant_max: 量化的最大值（可选）

    Returns:
        torch.Tensor: 反量化后的权重张量
    """
    if weight_qscheme in [
            torch.per_tensor_affine,
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams]:
        # 根据不同的量化方案，调用相应的分解后量化和反量化函数
        weight_quant = _quantize_weight_decomposed(
            weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis_int,
            weight_quant_min, weight_quant_max)
        weight_dequant = _dequantize_weight_decomposed(
            weight_quant, weight_qscheme, weight_dtype, weight_scale, weight_zero_point,
            weight_axis_int, weight_quant_min, weight_quant_max)
    else:
        # 如果量化方案不在支持的列表中，则直接返回原始权重张量
        weight_dequant = weight

    return weight_dequant

def _quantize_and_dequantize_weight(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis_int: int
) -> torch.Tensor:
    """ 根据量化参数对权重进行量化和反量化处理。

    Args:
        weight: 待量化的权重张量
        weight_qscheme: 量化方案，可以是 torch.per_tensor_affine, torch.per_channel_affine 或 torch.per_channel_affine_float_qparams
        weight_dtype: 权重的数据类型
        weight_scale: 量化的比例因子张量
        weight_zero_point: 量化的零点张量
        weight_axis_int: 量化轴的整数值

    Returns:
        torch.Tensor: 反量化后的权重张量
    """
    if weight_qscheme in [
            torch.per_tensor_affine,
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams]:
        # 根据不同的量化方案，调用相应的量化函数，并对量化后的张量进行反量化操作
        weight_quant = _quantize_weight(
            weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis_int)
        weight_dequant = weight_quant.dequantize()
    else:
        # 如果量化方案不在支持的列表中，则直接返回原始权重张量
        weight_dequant = weight

    return weight_dequant
# 将权重量化参数保存到目标字典中
def _save_weight_qparams(destination, prefix, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis):
    # 保存权重量化方案到目标字典
    destination[prefix + "weight_qscheme"] = weight_qscheme
    # 保存权重数据类型到目标字典
    destination[prefix + "weight_dtype"] = weight_dtype
    # 如果权重量化方案不为None，则继续保存相关参数
    if weight_qscheme is not None:
        # 保存权重缩放因子到目标字典
        destination[prefix + "weight_scale"] = weight_scale
        # 保存权重零点到目标字典
        destination[prefix + "weight_zero_point"] = weight_zero_point
        # 如果权重量化方案为每通道仿射，则保存权重轴信息到目标字典
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis

# 获取权重量化参数的键列表
def _get_weight_qparam_keys(
        state_dict: typing.Dict[str, typing.Any],
        prefix: str):
    # 初始化键列表，包含固定的两个键
    keys = ["weight_qscheme", "weight_dtype"]
    # 从状态字典中获取权重量化方案
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    # 如果权重量化方案不为None，则继续添加相关键
    if weight_qscheme is not None:
        # 添加权重缩放因子的键
        keys.append("weight_scale")
        # 添加权重零点的键
        keys.append("weight_zero_point")
        # 如果权重量化方案为每通道量化，则添加权重轴的键
        if weight_qscheme == torch.quantize_per_channel:
            keys.append("weight_axis")
    # 返回最终的键列表
    return keys
```