# `.\pytorch_utils.py`

```
# 导入inspect模块，用于获取对象信息
import inspect
# 导入类型提示模块
from typing import Callable, List, Optional, Set, Tuple, Union

# 导入PyTorch库
import torch
# 导入版本管理模块
from packaging import version
# 导入safetensors库中的相关函数
from safetensors.torch import storage_ptr, storage_size
# 导入PyTorch的神经网络模块
from torch import nn

# 导入本地的is_torch_xla_available和logging函数
from .utils import is_torch_xla_available, logging

# 定义一个包含nn.LayerNorm的列表
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 解析当前使用的PyTorch版本的基础版本号
parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

# 检查当前PyTorch版本是否大于等于2.2
is_torch_greater_or_equal_than_2_2 = parsed_torch_version_base >= version.parse("2.2")
# 检查当前PyTorch版本是否大于等于2.1
is_torch_greater_or_equal_than_2_1 = parsed_torch_version_base >= version.parse("2.1")
# 检查当前PyTorch版本是否大于等于2.0
is_torch_greater_or_equal_than_2_0 = parsed_torch_version_base >= version.parse("2.0")
# 检查当前PyTorch版本是否大于等于1.13
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")
# 检查当前PyTorch版本是否大于等于1.12
is_torch_greater_or_equal_than_1_12 = parsed_torch_version_base >= version.parse("1.12")


def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    调用内部的`_softmax_backward_data` PyTorch方法，并根据检测到的torch版本调整参数。
    
    Args:
        parent: 父对象
        grad_output: 梯度输出
        output: 输出
        dim: 维度
        self: 当前对象

    Returns:
        返回内部方法`_softmax_backward_data`的调用结果
    """
    from torch import _softmax_backward_data

    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    对线性层进行修剪，仅保留index中的条目。

    用于移除神经网络中的头部。

    Args:
        layer (`torch.nn.Linear`): 需要修剪的线性层。
        index (`torch.LongTensor`): 要在层中保留的索引。
        dim (`int`, *可选*, 默认为0): 在哪个维度上保留索引。

    Returns:
        `torch.nn.Linear`: 作为新层的修剪后的层，具有`requires_grad=True`。
    """
    # 将索引移到与权重张量相同的设备上
    index = index.to(layer.weight.device)
    # 从权重张量中选择指定维度上的索引，并进行克隆和分离
    W = layer.weight.index_select(dim, index).clone().detach()
    # 如果存在偏置，则根据维度选择偏置，并进行克隆和分离
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    # 创建新层，其尺寸与权重相同，但在指定维度上为索引长度
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    # 设置新层的权重不需要梯度，并复制修剪后的权重
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True

    return new_layer
    # 检查原始层是否具有偏置项
    if layer.bias is not None:
        # 将新层的偏置项设置为不需要梯度计算
        new_layer.bias.requires_grad = False
        # 将新层的偏置项赋值为现有偏置项的连续拷贝
        new_layer.bias.copy_(b.contiguous())
        # 设置新层的偏置项为需要梯度计算
        new_layer.bias.requires_grad = True
    # 返回已经设置好的新层对象
    return new_layer
    def apply_chunking_to_forward(
        forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors: Any
    ) -> torch.Tensor:
        """
        This function applies chunking to a forward function to allow for large tensor inputs that
        exceed memory capacity.

        Args:
            forward_fn (`Callable[..., torch.Tensor]`): The forward function of the model.
            chunk_size (`int`): The size of each chunk in the specified dimension.
            chunk_dim (`int`): The dimension along which to chunk the input tensors.
            *input_tensors (`Any`): Input tensors to the forward function.

        Returns:
            `torch.Tensor`: The result tensor from the forward function after applying chunking.
        """
        assert isinstance(chunk_size, int) and chunk_size > 0
        assert isinstance(chunk_dim, int) and chunk_dim < len(input_tensors[0].size())

        chunked_input_tensors = list(zip(*map(lambda x: x.chunk(chunk_size, dim=chunk_dim), input_tensors)))
        outputs = []

        for chunked_inputs in chunked_input_tensors:
            outputs.append(forward_fn(*chunked_inputs))

        if len(outputs) == 1:
            return outputs[0]
        else:
            return torch.cat(outputs, dim=chunk_dim)
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors


# forward_fn: Callable[..., torch.Tensor] 定义了一个名为 forward_fn 的参数，它是一个可调用对象，接受任意数量和类型的参数并返回 torch.Tensor 类型的对象。
# chunk_size: int 是一个整数类型的参数，用于指定数据块的大小。
# chunk_dim: int 是一个整数类型的参数，表示数据块在输入张量中的维度。
# *input_tensors 指定了一个可变数量的输入张量参数，它们将被传递给 forward_fn 函数。
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied.

    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    # Check if there are input tensors provided
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # Determine the number of arguments expected by the forward function
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)

    # Validate that the number of input tensors matches the number of expected arguments in forward_fn
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )
    # 如果指定了有效的块大小
    if chunk_size > 0:
        # 获取输入张量列表中第一个张量在指定维度上的形状
        tensor_shape = input_tensors[0].shape[chunk_dim]
        
        # 遍历输入张量列表，检查它们在指定维度上的形状是否与第一个张量相同
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                # 如果形状不同，则抛出数值错误异常
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        # 检查第一个张量在指定维度上的大小是否能被块大小整除
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            # 如果不能整除，则抛出数值错误异常
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        # 计算需要分块的数量
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # 将每个输入张量按指定维度分块成元组
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        
        # 对每个元组应用前向函数，并生成输出块的元组
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        
        # 在指定维度上连接输出块，生成最终的输出张量
        return torch.cat(output_chunks, dim=chunk_dim)

    # 如果未指定有效的块大小，则直接将前向函数应用于输入张量并返回结果
    return forward_fn(*input_tensors)
# 定义一个函数，用于查找可以裁剪的头部索引及其位置
def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    找到需要裁剪的头部及其索引，考虑到已经裁剪的头部。

    Args:
        heads (`List[int]`): 需要裁剪的头部索引列表。
        n_heads (`int`): 模型中头部的数量。
        head_size (`int`): 每个头部的大小。
        already_pruned_heads (`Set[int]`): 已经裁剪的头部集合。

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: 返回一个元组，包含考虑到 `already_pruned_heads` 后需要裁剪的头部索引，
        以及层权重中需要保留的行/列索引。
    """
    mask = torch.ones(n_heads, head_size)  # 创建一个全为1的掩码
    heads = set(heads) - already_pruned_heads  # 转换为集合并移除已经裁剪的头部
    for head in heads:
        # 计算在当前头部之前有多少已经裁剪的头部，并相应地调整索引
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0  # 将对应头部的掩码置为0，表示需要裁剪
    mask = mask.view(-1).contiguous().eq(1)  # 将掩码展平为一维，并保留值为1的位置
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()  # 获取需要保留的索引
    return heads, index  # 返回需要裁剪的头部索引和需要保留的索引


# 定义一个函数，用于创建网格
def meshgrid(
    *tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: Optional[str] = None
) -> Tuple[torch.Tensor, ...]:
    """
    对 torch.meshgrid 的包装，以避免关于引入的 `indexing` 参数的警告信息。

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    return torch.meshgrid(*tensors, indexing=indexing)


# 定义一个函数，用于获取张量的存储信息
def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    唯一标识符，用于标识张量的存储。多个不同的张量可以共享相同的底层存储。
    例如，“meta”张量共享相同的存储，因此它们的标识符将相等。
    此标识符在张量的生命周期内保证是唯一且常量的。两个存储生命周期不重叠的张量可能具有相同的id。
    """
    if tensor.device.type == "xla" and is_torch_xla_available():
        # 注意：xla 张量没有存储
        # 使用其他唯一标识符来区分。
        # 这是一个 XLA 张量，必须使用 torch_xla 的设备创建。
        # 所以以下导入是安全的：
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)
```