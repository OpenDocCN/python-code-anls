# `.\pytorch\torch\distributed\algorithms\_quantization\quantization.py`

```py
# 设置一个全局变量，用于存储半精度浮点数的最小值
TORCH_HALF_MIN = torch.finfo(torch.float16).min
# 设置一个全局变量，用于存储半精度浮点数的最大值
TORCH_HALF_MAX = torch.finfo(torch.float16).max


# 定义一个枚举类 DQuantType，用于表示不同的量化方法
class DQuantType(Enum):
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """

    # 表示 FP16 量化方法
    FP16 = ("fp16",)
    # 表示 BFP16 量化方法
    BFP16 = "bfp16"

    def __str__(self) -> str:
        return self.value


# 将输入的 torch.Tensor 转换为半精度浮点数，并进行上下限的裁剪
def _fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


# 根据指定的量化类型对输入的 tensor 进行量化处理
def _quantize_tensor(tensor, qtype):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_quantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(tensor)
    elif qtype == DQuantType.BFP16:
        return torch.ops.quantization._FloatToBfloat16Quantized(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


# 对输入的 tensor 列表中的每个 tensor 进行量化处理
def _quantize_tensor_list(tensor_list, qtype):
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_quantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    quantized_tensor_list = [_quantize_tensor(t, qtype) for t in tensor_list]
    return quantized_tensor_list


# 根据指定的量化类型对输入的 tensor 进行反量化处理
def _dequantize_tensor(tensor, qtype, quant_loss=None):
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"_dequantize_tensor expecting torch.Tensor as input but found {type(tensor)}"
        )
    if qtype == DQuantType.FP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        elif tensor.dtype == torch.float16 and quant_loss is None:
            return tensor.float()
        else:
            return tensor.float() / quant_loss
    elif qtype == DQuantType.BFP16:
        if tensor.dtype != torch.float16:
            raise RuntimeError(
                f"tensor dtype is {tensor.dtype} while expected to be FP16."
            )
        else:
            return torch.ops.quantization._Bfloat16QuantizedToFloat(tensor)
    else:
        raise RuntimeError(f"Quantization type {qtype} is not supported")


# 对输入的 tensor 列表中的每个 tensor 进行反量化处理
def _dequantize_tensor_list(tensor_list, qtype, quant_loss=None):
    if not isinstance(tensor_list, list) or not all(
        isinstance(p, torch.Tensor) for p in tensor_list
    ):
        raise RuntimeError(
            f"_dequantize_tensor_list expecting list of torch.Tensor as input but found {type(tensor_list)}"
        )
    dequantized_tensor_list = [_dequantize_tensor(t, qtype) for t in tensor_list]
    return dequantized_tensor_list
# 定义自动量化装饰器函数，用于量化输入张量并执行集合操作后再反量化输出

def auto_quantize(func, qtype, quant_loss=None):
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 从关键字参数中获取 group 参数，默认为 None
        group = kwargs.get("group", None)
        # 从关键字参数中获取 async_op 参数，默认为 False
        async_op = kwargs.get("async_op", False)
        
        # 如果 async_op 设置为 True，抛出运行时错误，暂不支持异步操作
        if async_op is True:
            raise RuntimeError("The async_op=True mode is not supported yet.")
        
        # 如果 func 是 dist.all_gather 函数
        if func == dist.all_gather:
            # 第一个参数是 tensors 列表
            tensors = args[0]
            # 将第二个参数 input_tensors 量化
            input_tensors = _quantize_tensor(args[1], qtype)
            # 将 tensors 列表中的所有张量进行量化
            out_tensors = _quantize_tensor_list(tensors, qtype)
            # 调用 dist.all_gather 执行全局聚合操作
            dist.all_gather(out_tensors, input_tensors, group=group, async_op=async_op)
            # 将输出张量列表 out_tensors 反量化，并更新到输入 tensors 列表中
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        # 如果 func 是 dist.all_to_all 函数
        elif func == dist.all_to_all:
            # 第一个参数是 tensors 列表
            tensors = args[0]
            # 将第二个参数 input_tensors 列表中的所有张量进行量化
            input_tensors = _quantize_tensor_list(args[1], qtype)
            # 将 tensors 列表中的所有张量进行量化
            out_tensors = _quantize_tensor_list(tensors, qtype)
            # 调用 dist.all_to_all 执行全局交换操作
            dist.all_to_all(out_tensors, input_tensors, group=group, async_op=async_op)
            # 将输出张量列表 out_tensors 反量化，并更新到输入 tensors 列表中
            for i, t in enumerate(
                _dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        # 如果 func 是 dist.all_to_all_single 函数
        elif func == dist.all_to_all_single:
            # 第一个参数是 tensors 列表
            tensors = args[0]
            # 从关键字参数中获取 out_splits 和 in_splits 参数
            out_splits = kwargs.get("out_splits", None)
            in_splits = kwargs.get("in_splits", None)
            # 将第二个参数 input_tensors 量化
            input_tensors = _quantize_tensor(args[1], qtype)
            # 将 tensors 列表中的所有张量进行量化
            out_tensors = _quantize_tensor(tensors, qtype)
            # 调用 dist.all_to_all_single 执行单一全局交换操作
            dist.all_to_all_single(
                out_tensors, input_tensors, out_splits, in_splits, group=group
            )
            # 将输出张量列表 out_tensors 反量化，并更新到输入 tensors 列表中
            for i, t in enumerate(
                _dequantize_tensor(out_tensors, qtype, quant_loss=quant_loss)
            ):
                tensors[i] = t

        # 如果 func 不是支持的集合操作函数，抛出运行时错误
        else:
            raise RuntimeError(f"The collective op {func} is not supported yet")

    return wrapper
```