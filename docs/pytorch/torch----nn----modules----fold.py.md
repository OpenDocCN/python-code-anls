# `.\pytorch\torch\nn\modules\fold.py`

```
# 导入 torch.nn.functional 中的 F 模块，用于调用神经网络函数
# 导入 Tensor 类型，用于定义函数的参数和返回类型
# 导入 torch.nn.common_types 模块中的 _size_any_t，用于定义函数参数类型
from torch import Tensor
from torch.nn.common_types import _size_any_t

# 从当前目录下的 module.py 文件中导入 Module 类
from .module import Module

# 定义 __all__ 列表，指定了当前模块导出的所有公共接口
__all__ = ["Fold", "Unfold"]


# 定义 Fold 类，继承自 Module 类
class Fold(Module):
    # 文档字符串，描述了 Fold 类的作用和使用方法
    r"""Combines an array of sliding local blocks into a large containing tensor.
    
    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel\_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel\_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel\_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~torch.nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~torch.nn.Unfold`, the
    arguments must satisfy
    
    .. math::
        L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,
    
    where :math:`d` is over all spatial dimensions.
    
    * :attr:`output_size` describes the spatial shape of the large containing
      tensor of the sliding local blocks. It is useful to resolve the ambiguity
      when multiple input shapes map to same number of sliding blocks, e.g.,
      with ``stride > 0``.
    
    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.
    
    * :attr:`stride` controls the stride for the sliding blocks.
    
    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.
    
    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
    
    Args:
        output_size (int or tuple): the shape of the spatial dimensions of the
                                    output (i.e., ``output.sizes()[2:]``)
        kernel_size (int or tuple): the size of the sliding blocks
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        stride (int or tuple): the stride of the sliding blocks in the input
                               spatial dimensions. Default: 1
    """

    # 在这里定义类的方法和属性，实现具体功能
    """
    If :attr:`output_size`, :attr:`kernel_size`, :attr:`dilation`,
    :attr:`padding` or :attr:`stride` is an int or a tuple of length 1 then
    their values will be replicated across all spatial dimensions.
    
    For the case of two output spatial dimensions this operation is sometimes
    called ``col2im``.
    
    .. note::
        :class:`~torch.nn.Fold` calculates each combined value in the resulting
        large tensor by summing all values from all containing blocks.
        :class:`~torch.nn.Unfold` extracts the values in the local blocks by
        copying from the large tensor. So, if the blocks overlap, they are not
        inverses of each other.
    
        In general, folding and unfolding operations are related as
        follows. Consider :class:`~torch.nn.Fold` and
        :class:`~torch.nn.Unfold` instances created with the same
        parameters:
    
        >>> fold_params = dict(kernel_size=..., dilation=..., padding=..., stride=...)
        >>> fold = nn.Fold(output_size=..., **fold_params)
        >>> unfold = nn.Unfold(**fold_params)
    
        Then for any (supported) ``input`` tensor the following
        equality holds:
    
        ::
    
            fold(unfold(input)) == divisor * input
    
        where ``divisor`` is a tensor that depends only on the shape
        and dtype of the ``input``:
    
        >>> # xdoctest: +SKIP
        >>> input_ones = torch.ones(input.shape, dtype=input.dtype)
        >>> divisor = fold(unfold(input_ones))
    
        When the ``divisor`` tensor contains no zero elements, then
        ``fold`` and ``unfold`` operations are inverses of each
        other (up to constant divisor).
    
    .. warning::
        Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.
    
    Shape:
        - Input: :math:`(N, C \times \prod(\text{kernel\_size}), L)` or :math:`(C \times \prod(\text{kernel\_size}), L)`
        - Output: :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
          or :math:`(C, \text{output\_size}[0], \text{output\_size}[1], \dots)` as described above
    
    Examples::
    
        >>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
        >>> input = torch.randn(1, 3 * 2 * 2, 12)
        >>> output = fold(input)
        >>> output.size()
        torch.Size([1, 3, 4, 5])
    
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    """
    
    __constants__ = ["output_size", "kernel_size", "dilation", "padding", "stride"]
    output_size: _size_any_t
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t
    
    def __init__(
        self,
        output_size: _size_any_t,
        kernel_size: _size_any_t,
        dilation: _size_any_t = 1,
        padding: _size_any_t = 0,
        stride: _size_any_t = 1,
    ) -> None:
        # 构造函数初始化方法
        super().__init__()
        # 设置输出大小
        self.output_size = output_size
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置膨胀率
        self.dilation = dilation
        # 设置填充
        self.padding = padding
        # 设置步长
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        # 调用 F.fold 函数进行折叠操作
        return F.fold(
            input,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

    def extra_repr(self) -> str:
        # 返回一个描述网络结构的字符串，包括输出大小、卷积核大小、膨胀率、填充和步长
        return (
            "output_size={output_size}, kernel_size={kernel_size}, "
            "dilation={dilation}, padding={padding}, stride={stride}".format(
                **self.__dict__
            )
        )
# 定义一个继承自 Module 的类 Unfold，用于从批量输入张量中提取滑动局部块。

r"""Extracts sliding local blocks from a batched input tensor.

Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`,
where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
and :math:`*` represent arbitrary spatial dimensions. This operation flattens
each sliding :attr:`kernel_size`-sized block within the spatial dimensions
of :attr:`input` into a column (i.e., last dimension) of a 3-D :attr:`output`
tensor of shape :math:`(N, C \times \prod(\text{kernel\_size}), L)`, where
:math:`C \times \prod(\text{kernel\_size})` is the total number of values
within each block (a block has :math:`\prod(\text{kernel\_size})` spatial
locations each containing a :math:`C`-channeled vector), and :math:`L` is
the total number of such blocks:

.. math::
    L = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d] %
        - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

where :math:`\text{spatial\_size}` is formed by the spatial dimensions
of :attr:`input` (:math:`*` above), and :math:`d` is over all spatial
dimensions.

Therefore, indexing :attr:`output` at the last dimension (column dimension)
gives all values within a certain block.

The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
how the sliding blocks are retrieved.

* :attr:`stride` controls the stride for the sliding blocks.

* :attr:`padding` controls the amount of implicit zero-paddings on both
  sides for :attr:`padding` number of points for each dimension before
  reshaping.

* :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.
  It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

Args:
    kernel_size (int or tuple): the size of the sliding blocks
    dilation (int or tuple, optional): a parameter that controls the
                                       stride of elements within the
                                       neighborhood. Default: 1
    padding (int or tuple, optional): implicit zero padding to be added on
                                      both sides of input. Default: 0
    stride (int or tuple, optional): the stride of the sliding blocks in the input
                                     spatial dimensions. Default: 1

* If :attr:`kernel_size`, :attr:`dilation`, :attr:`padding` or
  :attr:`stride` is an int or a tuple of length 1, their values will be
  replicated across all spatial dimensions.

* For the case of two input spatial dimensions this operation is sometimes
  called ``im2col``.
    # 定义常量列表，这些常量描述了卷积操作的核大小、膨胀、填充和步幅
    __constants__ = ["kernel_size", "dilation", "padding", "stride"]
    
    # 定义卷积核的大小，可以是一个整数或者元组，描述了卷积核的空间大小
    kernel_size: _size_any_t
    
    # 定义卷积核的膨胀参数，可以是一个整数或者元组，描述了卷积核元素之间的间隔
    dilation: _size_any_t
    
    # 定义填充大小，可以是一个整数或者元组，描述了输入的每一条边补充0的层数
    padding: _size_any_t
    
    # 定义步幅大小，可以是一个整数或者元组，描述了卷积操作每次在输入数据的哪一维度移动
    stride: _size_any_t
    # 初始化函数，用于设置卷积操作的参数
    def __init__(
        self,
        kernel_size: _size_any_t,  # 卷积核大小，可以是任意尺寸类型
        dilation: _size_any_t = 1,  # 膨胀率，默认为1
        padding: _size_any_t = 0,   # 填充大小，默认为0
        stride: _size_any_t = 1,    # 步幅大小，默认为1
    ) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.dilation = dilation        # 设置膨胀率
        self.padding = padding          # 设置填充大小
        self.stride = stride            # 设置步幅大小

    # 前向传播函数，调用 PyTorch 的 F 模块中的 unfold 函数实现
    def forward(self, input: Tensor) -> Tensor:
        return F.unfold(
            input, self.kernel_size, self.dilation, self.padding, self.stride
        )

    # 返回额外的表示信息，描述卷积层的参数
    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, dilation={dilation}, padding={padding},"
            " stride={stride}".format(**self.__dict__)
        )
```