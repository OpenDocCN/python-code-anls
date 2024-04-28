# `.\transformers\pytorch_utils.py`

```py
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库和模块
import inspect
from typing import Callable, List, Optional, Set, Tuple, Union
import torch
from packaging import version
from safetensors.torch import storage_ptr, storage_size
from torch import nn
from .utils import is_torch_tpu_available, logging

# 定义全局变量
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
logger = logging.get_logger(__name__)

# 解析当前 torch 版本
parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_2_1 = parsed_torch_version_base >= version.parse("2.1")
is_torch_greater_or_equal_than_2_0 = parsed_torch_version_base >= version.parse("2.0")
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")
is_torch_greater_or_equal_than_1_12 = parsed_torch_version_base >= version.parse("1.12")

# 定义 softmax_backward_data 函数
def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    A function that calls the internal `_softmax_backward_data` PyTorch method and that adjusts the arguments according
    to the torch version detected.
    """
    from torch import _softmax_backward_data
    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)

# 定义 prune_linear_layer 函数
def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    # 将索引转移到与权重相同的设备上
    index = index.to(layer.weight.device)
    # 选择指定维度上的索引，并克隆权重
    W = layer.weight.index_select(dim, index).clone().detach()
    # 如果存在偏置项，根据维度选择偏置
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    # 创建新的线性层，保留指定索引的权重
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    # 检查原始层是否有偏置项
    if layer.bias is not None:
        # 设置新层的偏置项不需要梯度计算
        new_layer.bias.requires_grad = False
        # 将原始层的偏置项拷贝到新层
        new_layer.bias.copy_(b.contiguous())
        # 设置新层的偏置项需要梯度计算
        new_layer.bias.requires_grad = True
    # 返回新层
    return new_layer
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        # 初始化 Conv1D 类的对象
        self.nf = nf
        # 定义权重参数，参数为可训练参数
        self.weight = nn.Parameter(torch.empty(nx, nf))
        # 定义偏置参数，参数为可训练参数
        self.bias = nn.Parameter(torch.zeros(nf))
        # 使用正态分布初始化权重参数
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # 计算输出尺寸
        size_out = x.size()[:-1] + (self.nf,)
        # 进行线性变换
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # 将结果reshape成指定尺寸
        x = x.view(size_out)
        return x


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~pytorch_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    # 将索引转移到与权重参数相同的设备上
    index = index.to(layer.weight.device)
    # 根据索引选取权重参数，创建新的权重参数
    W = layer.weight.index_select(dim, index).clone().detach()
    # 根据维度选择偏置参数
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    # 创建新的 Conv1D 层对象
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(
    layer: Union[nn.Linear, Conv1D], index: torch.LongTensor, dim: Optional[int] = None
) -> Union[nn.Linear, Conv1D]:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`Union[torch.nn.Linear, Conv1D]`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear` or [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    # 根据层的类型调用对应的剪枝函数
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError(f"Can't prune layer of class {layer.__class__}")


def apply_chunking_to_forward(
    # 定义一个参数 forward_fn，类型为 Callable[..., torch.Tensor]，表示一个可调用对象，接受任意参数并返回 torch.Tensor 类型的数据
    # 定义一个参数 chunk_size，表示分块的大小
    # 定义一个参数 chunk_dim，表示分块的维度
    # 定义可变参数 *input_tensors，表示接受任意数量的输入张量
# 定义一个函数，将输入张量按照指定大小和维度分块，然后对每个块独立应用给定的前向函数，以节省内存
def apply_chunking_to_forward(forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, input_tensors: Tuple[torch.Tensor]) -> torch.Tensor:
    # 断言输入张量列表不为空
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # 检查前向函数的参数个数是否与输入张量列表长度相同
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )
    # 如果 chunk_size 大于 0
    if chunk_size > 0:
        # 获取输入张量的形状
        tensor_shape = input_tensors[0].shape[chunk_dim]
        # 遍历输入张量
        for input_tensor in input_tensors:
            # 检查每个输入张量的指定维度是否与第一个张量相同
            if input_tensor.shape[chunk_dim] != tensor_shape:
                # 如果不同，抛出数值错误
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        # 检查指定维度是否可以被 chunk_size 整除
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            # 如果不能整除，抛出数值错误
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        # 计算分块的数量
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # 将输入张量分块成元组
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # 对每个元组应用前向函数
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # 在相同维度上连接输出
        return torch.cat(output_chunks, dim=chunk_dim)

    # 如果 chunk_size 不大于 0，直接应用前向函数到输入张量
    return forward_fn(*input_tensors)
# 找到可以修剪的头部和它们的索引，考虑到已经修剪的头部
def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    # 创建一个全为1的掩码，表示所有头部均未修剪
    mask = torch.ones(n_heads, head_size)
    # 将heads转换为集合，并移除已经修剪的头部
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    # 遍历剩余的头部
    for head in heads:
        # 计算在该头部之前已经修剪的头部数量，并相应地移动索引
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        # 将该头部对应的掩码置为0，表示要修剪
        mask[head] = 0
    # 将掩码展平为一维，并返回非零元素的索引
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


# 创建一个meshgrid的包装函数，避免关于引入的`indexing`参数的警告消息
def meshgrid(
    *tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: Optional[str] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    return torch.meshgrid(*tensors, indexing=indexing)


# 返回一个张量存储的唯一标识符。多个不同的张量可以共享相同的底层存储。
# 例如，"meta"张量共享相同的存储，因此它们的标识符将全部相等。
# 保证这个标识符在张量的生命周期内是唯一且不变的。
# 具有非重叠生命周期的两个张量存储可能具有相同的id。
def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    # 如果张量是XLA张量，并且torch_xla可用
    if tensor.device.type == "xla" and is_torch_tpu_available():
        # 注意：xla张量没有存储
        # 使用其他唯一id进行区分。
        # 这是一个XLA张量，必须使用torch_xla的设备创建。所以以下导入是安全的：
        import torch_xla

        # 使用torch_xla库获取张量的唯一id
        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        # 使用内置的函数获取张量的存储指针作为唯一id
        unique_id = storage_ptr(tensor)

    # 返回张量的设备，唯一id和存储大小
    return tensor.device, unique_id, storage_size(tensor)
```