# `.\pytorch\torch\nn\utils\rnn.py`

```
import warnings
from typing import Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import _VF, Tensor

# 导入所需的模块和类

__all__ = [
    "PackedSequence",
    "invert_permutation",
    "pack_padded_sequence",
    "pad_packed_sequence",
    "pad_sequence",
    "unpad_sequence",
    "pack_sequence",
    "unpack_sequence",
]

# 定义公开的符号列表

class PackedSequence_(NamedTuple):
    data: torch.Tensor
    batch_sizes: torch.Tensor
    sorted_indices: Optional[torch.Tensor]
    unsorted_indices: Optional[torch.Tensor]

# 命名元组 PackedSequence_，用于表示打包的序列及其批次信息

def bind(optional, fn):
    if optional is None:
        return None
    return fn(optional)

# 定义绑定函数 bind，用于将可选值传入指定函数进行处理

class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a :class:`PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    """

    def __new__(
        cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None
    ):
        return super().__new__(
            cls,
            *_packed_sequence_init_args(
                data, batch_sizes, sorted_indices, unsorted_indices
            ),
        )

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").

    # 定义 __new__ 方法，用于创建新的 PackedSequence 实例，基于给定的参数初始化
    def pin_memory(self):
        # 将当前对象进行内存固定化，返回新的固定化对象
        # 为什么不转换 `batch_sizes`？
        # 参见注释 [ PackedSequence 的设备和数据类型 ]
        return type(self)(
            self.data.pin_memory(),  # 固定化 self.data 的内存
            self.batch_sizes,  # 不变的 batch_sizes
            bind(self.sorted_indices, lambda t: t.pin_memory()),  # 固定化 sorted_indices 的内存
            bind(self.unsorted_indices, lambda t: t.pin_memory()),  # 固定化 unsorted_indices 的内存
        )

    def cuda(self, *args, **kwargs):
        # 检查是否应该在 kwargs 中添加 'cuda'
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.is_cuda:  # 如果 ex 在 CUDA 上
            return self.to(*args, **kwargs)  # 使用给定的 args 和 kwargs 转换到 CUDA
        return self.to(*args, device="cuda", **kwargs)  # 否则使用 'cuda' 设备转换到 CUDA

    def cpu(self, *args, **kwargs):
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.device.type == "cpu":  # 如果 ex 在 CPU 上
            return self.to(*args, **kwargs)  # 使用给定的 args 和 kwargs 转换到 CPU
        return self.to(*args, device="cpu", **kwargs)  # 否则使用 'cpu' 设备转换到 CPU

    def double(self):
        # 转换为 torch.double 类型
        return self.to(dtype=torch.double)

    def float(self):
        # 转换为 torch.float 类型
        return self.to(dtype=torch.float)

    def half(self):
        # 转换为 torch.half 类型
        return self.to(dtype=torch.half)

    def long(self):
        # 转换为 torch.long 类型
        return self.to(dtype=torch.long)

    def int(self):
        # 转换为 torch.int 类型
        return self.to(dtype=torch.int)

    def short(self):
        # 转换为 torch.short 类型
        return self.to(dtype=torch.short)

    def char(self):
        # 转换为 torch.int8 类型
        return self.to(dtype=torch.int8)

    def byte(self):
        # 转换为 torch.uint8 类型
        return self.to(dtype=torch.uint8)

    def to(self, *args, **kwargs):
        r"""对 `self.data` 执行数据类型和/或设备转换。

        其签名与 :meth:`torch.Tensor.to` 类似，除了非必需的参数如 `non_blocking` 和 `copy` 应作为 kwargs 而不是 args，否则它们将不适用于索引张量。

        .. 注意::

            如果 ``self.data`` 张量已经具有正确的 :class:`torch.dtype` 和 :class:`torch.device`，则返回 ``self``。
            否则，返回一个具有所需配置的副本。
        """
        # 为什么不转换 `batch_sizes`？
        # 参见注释 [ PackedSequence 的设备和数据类型 ]
        data = self.data.to(*args, **kwargs)  # 转换 self.data 的数据类型和设备
        if data is self.data:  # 如果没有进行任何转换
            return self
        else:
            # 不转发设备或数据类型的参数/关键字参数，设备从 data.device 设置
            kwargs = dict(
                filter(lambda t: t[0] != "device" and t[0] != "dtype", kwargs.items())
            )
            sorted_indices = bind(
                self.sorted_indices, lambda t: t.to(data.device, **kwargs)
            )  # 转换 sorted_indices 到与 data 相同的设备
            unsorted_indices = bind(
                self.unsorted_indices, lambda t: t.to(data.device, **kwargs)
            )  # 转换 unsorted_indices 到与 data 相同的设备
            return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        r"""如果 `self.data` 存储在 GPU 上，则返回 True。"""
        return self.data.is_cuda
    def is_pinned(self):
        r"""Return true if `self.data` stored on in pinned memory."""
        # 检查对象中的数据是否存储在固定内存中，并返回相应的布尔值
        return self.data.is_pinned()
# TorchScript 不支持命名元组的构造函数，因此我们使用这个辅助方法来构造 PackedSequence
def _packed_sequence_init_args(
    data: Tensor,
    batch_sizes: Optional[Tensor] = None,
    sorted_indices: Optional[Tensor] = None,
    unsorted_indices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    # 如果提供了 unsorted_indices，则它应该是 sorted_indices 的反向置换。
    # 在这里不进行断言，因为 PackedSequence 构造函数应该只在内部使用。

    if unsorted_indices is None:
        unsorted_indices = invert_permutation(sorted_indices)

    # 如果 batch_sizes 不为 None，则支持调用形式 `PackedSequence(data, batch_sizes, sorted_indices)`
    if batch_sizes is not None:
        # TODO: 重新启用此检查（在 TorchScript 中不支持 .type）
        if batch_sizes.device.type != "cpu":
            raise ValueError(
                "batch_sizes 应该始终在 CPU 上。"
                "不应手动创建 PackedSequence 实例。"
                "应该通过像 nn.utils.rnn 中的 pack_sequence 和 pack_padded_sequences 等函数来实例化。"
                "https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence"
            )
        return data, batch_sizes, sorted_indices, unsorted_indices

    # 如果作为 `PackedSequence((data, batch_sizes), *, sorted_indices)` 形式调用
    else:
        assert isinstance(data, (list, tuple)) and len(data) == 2
        return data[0], data[1], sorted_indices, unsorted_indices


def _packed_sequence_init(
    data: Tensor,
    batch_sizes: Optional[Tensor] = None,
    sorted_indices: Optional[Tensor] = None,
    unsorted_indices: Optional[Tensor] = None,
) -> PackedSequence:
    # 使用 _packed_sequence_init_args 函数获取参数
    data, batch_sizes, sorted_indices, unsorted_indices = _packed_sequence_init_args(
        data, batch_sizes, sorted_indices, unsorted_indices
    )
    # 返回通过参数构造的 PackedSequence 对象
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


def invert_permutation(permutation: Optional[Tensor]) -> Optional[Tensor]:
    if permutation is None:
        return None
    # 创建一个与 permutation 张量相同形状的空张量 output
    output = torch.empty_like(permutation, memory_format=torch.legacy_contiguous_format)
    # 使用 scatter_ 函数将序号按 permutation 张量的值散布到 output 张量中
    output.scatter_(
        0, permutation, torch.arange(0, permutation.numel(), device=permutation.device)
    )
    return output


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""对包含变长填充序列的张量进行打包。

    :attr:`input` 可以是大小为 ``T x B x *`` 的张量，其中 ``T`` 是最长序列的长度，``B`` 是批量大小，``*`` 可以是任意数量的维度（包括 0）。
    如果 :attr:`batch_first` 是 ``False``，则期望 ``T x B x *`` 的 :attr:`input`，否则期望 ``B x T x *``。

    对于未排序的序列，请使用 `enforce_sorted = False`。如果 :attr:`enforce_sorted` 是
    """
    Pack the padded batch of variable length sequences into a PackedSequence object.

    If `enforce_sorted = True`, the sequences should be sorted by length in a decreasing order,
    i.e., `input[:,0]` should be the longest sequence and `input[:,B-1]` the shortest one.
    `enforce_sorted = True` is only necessary for ONNX export.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its `.data` attribute.

    Args:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor or list(int)): list of sequence lengths of each batch
            element (must be on the CPU if provided as a tensor).
        batch_first (bool, optional): if `True`, the input is expected in `B x T x *`
            format, `T x B x *` otherwise.
        enforce_sorted (bool, optional): if `True`, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            `False`, the input will get sorted unconditionally. Default: `True`.

    Returns:
        a :class:`PackedSequence` object
    """
    # Convert lengths to a torch.Tensor if it's originally provided as a list
    if not isinstance(lengths, torch.Tensor):
        # Issue a warning if tracing is active and lengths is provided as a Python list
        if torch._C._get_tracing_state():
            warnings.warn(
                "pack_padded_sequence has been called with a Python list of "
                "sequence lengths. The tracer cannot track the data flow of Python "
                "values, and it will treat them as constants, likely rendering "
                "the trace incorrect for any other combination of lengths.",
                stacklevel=2,
            )
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
    else:
        # Convert lengths to torch.int64 if it's a different dtype
        lengths = lengths.to(dtype=torch.int64)

    # Sort lengths if enforce_sorted is False and reindex input accordingly
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    # Pack the padded sequence using the C++ extension _VF._pack_padded_sequence
    data, batch_sizes = _VF._pack_padded_sequence(input, lengths, batch_first)
    
    # Initialize and return a PackedSequence object
    return _packed_sequence_init(data, batch_sizes, sorted_indices, None)
def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Pad a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) , where ``T`` is the length of the longest
    sequence and ``B`` is the batch size.

    Example:
        >>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> seq = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
        >>> lens = [2, 1, 3]
        >>> packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        >>> packed
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                       sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[1, 2, 0],
                [3, 0, 0],
                [4, 5, 6]])
        >>> lens_unpacked
        tensor([2, 1, 3])

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`this FAQ section <pack-rnn-unpack-with-data-parallelism>` for
        details.

    Args:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): values for padded elements.
        total_length (int, optional): if not ``None``, the output will be padded to
            have length :attr:`total_length`. This method will throw :class:`ValueError`
            if :attr:`total_length` is less than the max sequence length in
            :attr:`sequence`.

    Returns:
        Tuple of Tensor containing the padded sequence, and a Tensor
        containing the list of lengths of each sequence in the batch.
        Batch elements will be re-ordered as they were ordered originally when
        the batch was passed to ``pack_padded_sequence`` or ``pack_sequence``.

    """
    # 获取批次中最长序列的长度
    max_seq_length = sequence.batch_sizes.size(0)
    
    # 如果指定了 total_length，确保要求的总长度不小于批次中最长序列的长度
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError(
                "Expected total_length to be at least the length "
                "of the longest sequence in input, but got "
                f"total_length={total_length} and max sequence length being {max_seq_length}"
            )
        max_seq_length = total_length
    # 调用 _VF._pad_packed_sequence 函数，解压压缩的序列数据并获取填充后的输出和长度信息
    padded_output, lengths = _VF._pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length
    )
    # 获取未排序的索引信息
    unsorted_indices = sequence.unsorted_indices
    # 如果存在未排序的索引
    if unsorted_indices is not None:
        # 根据 batch_first 决定批次维度的位置
        batch_dim = 0 if batch_first else 1
        # 返回根据未排序索引重新排序后的填充输出和对应长度
        return (
            padded_output.index_select(batch_dim, unsorted_indices),
            lengths[unsorted_indices.cpu()],
        )
    # 如果不存在未排序的索引，直接返回填充后的输出和对应长度
    return padded_output, lengths
# NOTE: .pyi stub allows Iterable[Tensor], but for JIT-compatibility we need to be more restrictive here.
def pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including 0). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    # Check if JIT is not tracing or scripting
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`, so raise an error if sequences is not Iterable
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)

        # In non-JIT context, convert sequences to a tuple
        sequences = tuple(sequences)
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]], so unbind if sequences is Tensor
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(sequences, batch_first, padding_value)


def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
) -> List[Tensor]:
    r"""Unpad padded Tensor into a list of variable length Tensors.

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.
    """
    # 初始化一个空列表，用于存储未填充的序列
    unpadded_sequences = []
    
    # 如果参数 batch_first 为 False，则交换填充后序列的维度，将批次维度移动到第一维
    if not batch_first:
        padded_sequences.transpose_(0, 1)
    
    # 获取填充后序列的最大长度
    max_length = padded_sequences.shape[1]
    
    # 创建一个索引张量，长度与填充前序列的设备一致
    idx = torch.arange(max_length, device=lengths.device)
    
    # 遍历填充后的序列和其对应的长度
    for seq, length in zip(padded_sequences, lengths):
        # 创建一个布尔掩码，标记出有效数据的部分（即长度范围内的数据）
        mask = idx < length
        # 根据掩码从填充后的序列中获取有效数据，并添加到未填充序列列表中
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)
    
    # 返回未填充的序列列表
    return unpadded_sequences
def pack_sequence(
    sequences: List[Tensor],
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a list of variable length Tensors.

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including zero.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> pack_sequence([a, b, c])
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)


    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    # 计算每个序列的长度
    lengths = torch.as_tensor([v.size(0) for v in sequences])
    # 调用pad_sequence函数填充序列，并调用pack_padded_sequence打包填充后的序列
    return pack_padded_sequence(
        pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted
    )


def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]:
    r"""Unpack PackedSequence into a list of variable length Tensors.

    ``packed_sequences`` should be a PackedSequence object.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> sequences = [a, b, c]
        >>> print(sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]
        >>> packed_sequences = pack_sequence(sequences)
        >>> print(packed_sequences)
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        >>> print(unpacked_sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]


    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """
    # 使用pad_packed_sequence解包填充的序列，设置batch_first=True
    padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first=True)
    # 使用unpad_sequence还原填充前的原始序列
    unpacked_sequences = unpad_sequence(padded_sequences, lengths, batch_first=True)
    return unpacked_sequences
```