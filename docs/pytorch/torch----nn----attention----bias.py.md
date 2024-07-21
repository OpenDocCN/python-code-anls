# `.\pytorch\torch\nn\attention\bias.py`

```py
# 设置全局类型提示允许未定义的函数
# 引入需要的库和模块
from enum import auto, IntEnum  # 自动命名，整数枚举
from typing import Optional  # 导入可选类型
from warnings import warn  # 导入警告模块

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数模块
from torch.backends.cuda import (  # 导入CUDA后端相关模块
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)
from torch.nn.attention import _raise_kernel_warnings  # 导入注意力模块的警告函数
from torch.nn.attention._utils import (  # 导入注意力模块的辅助函数
    _calculate_scale,
    _input_requires_grad,
    _postprocess_flash_output,
    _validate_sdpa_input,
)

# 设置公开的模块接口列表
__all__ = ["causal_upper_left", "causal_lower_right", "CausalVariant", "CausalBias"]

# 设置全局允许Flash注意力的使用
torch._dynamo.allow_in_graph(can_use_flash_attention)
# 设置全局允许高效注意力的使用
torch._dynamo.allow_in_graph(can_use_efficient_attention)
# 设置全局允许SDPA参数的使用
torch._dynamo.allow_in_graph(SDPAParams)

# 定义用于注意力机制中因果变体的枚举类
class CausalVariant(IntEnum):
    """
    枚举因果注意力机制中使用的因果变体。

    定义了两种因果偏置类型：

    `UPPER_LEFT`: 用于标准因果注意力的左上三角形偏置。
    相应的PyTorch代码为构造此偏置：

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    例如，对于 `shape=(3,4)`，生成的偏置张量为：

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]


    `LOWER_RIGHT`: 代表右下角三角形偏置，包含的值对齐于矩阵的右下角。

    相应的PyTorch代码为构造此偏置：

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    例如，对于 `shape=(3,4)`，生成的偏置张量为：

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    注意，当查询和键/值张量的序列长度相等时，这些变体是等效的，因为三角形矩阵是方阵。

    .. warning:: 此枚举类是原型，可能会发生变化。
    """

    UPPER_LEFT = auto()  # 自动分配值为1
    LOWER_RIGHT = auto()  # 自动分配值为2


# 定义表示因果注意力模式的偏置类
class CausalBias(torch.Tensor):
    """
    代表因果注意力模式的偏置类。有关偏置结构的概述，请参阅 :class:`CausalVariant` 枚举类。

    此类用于定义因果（三角形）注意力偏置。有两个工厂函数可用于构建偏置：:func:`causal_upper_left` 和 :func:`causal_lower_right`。

    示例：
        from torch.nn.attention.bias import causal_lower_right  # 导入 causal_lower_right 函数，用于生成下三角形式的因果偏置矩阵

        bsz, num_heads, seqlen_q, seqlen_kv, head_dim = 32, 8, 4, 12, 8

        # 创建一个下三角形式的因果偏置矩阵
        attn_bias = causal_lower_right(seqlen_q, seqlen_kv)

        # 在 CUDA 上生成随机的查询（q）、键（k）、值（v）张量，使用半精度浮点数类型
        q = torch.randn(bsz, num_heads, seqlen_q, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16)

        # 调用 scaled_dot_product_attention 函数，使用生成的因果偏置矩阵 attn_bias
        out = F.scaled_dot_product_attention(q, k, v, attn_bias)

    """
    .. warning:: This class is a prototype and subject to change.
    """

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int):
        """
        Initializes the CausalBias instance with a specified variant and sequence lengths.

        Args:
            variant (CausalVariant): The type of causal bias to use (either UPPER_LEFT or LOWER_RIGHT).
            seq_len_q (int): The sequence length of the query tensor.
            seq_len_kv (int): The sequence length of the key/value tensor.

        Raises a warning if the LOWER_RIGHT variant is used with seq_len_q > seq_len_kv, as it may produce NaNs.
        """
        assert isinstance(variant, CausalVariant)
        self.variant = variant  # 将传入的 variant 参数赋值给实例变量 self.variant
        self.seq_len_q = seq_len_q  # 将传入的 seq_len_q 参数赋值给实例变量 self.seq_len_q
        self.seq_len_kv = seq_len_kv  # 将传入的 seq_len_kv 参数赋值给实例变量 self.seq_len_kv
        if seq_len_q > seq_len_kv and variant == CausalVariant.LOWER_RIGHT:
            warn(
                "Lower right causal bias will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

    def _upper_left(self, device: torch.device) -> torch.Tensor:
        """Upper left causal bias"""
        # 创建一个上三角形式的因果偏置矩阵，并返回
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
        )

    def _lower_right(self, device: torch.device) -> torch.Tensor:
        """Lower right causal bias"""
        diagonal_offset = self.seq_len_kv - self.seq_len_q  # 计算对角线偏移量
        # 创建一个下三角形式的因果偏置矩阵，并返回，指定对角线偏移量
        return torch.tril(
            torch.ones(
                self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool
            ),
            diagonal=diagonal_offset,
        )
    def _materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Materializes the causal bias into a tensor form.

        Depending on the variant, this method generates either an upper-left or lower-right
        triangular matrix to represent the causal bias.

        Args:
            device (Optional[torch.device]): The device on which to create the tensor. Defaults to CPU.

        Returns:
            torch.Tensor: The materialized bias tensor.
        """
        # 检查设备是否为空，如果为空则默认为 CPU
        if device is None:
            device = torch.device("cpu")
        # 根据不同的变体类型调用相应的方法生成上左或下右三角形矩阵来表示因果偏置
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left(device)
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right(device)

    @staticmethod
    def _dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "CausalBias",
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        """
        Static method to dispatch the computation to the appropriate method based on the function and arguments.

        Args:
            query (torch.Tensor): Tensor representing query.
            key (torch.Tensor): Tensor representing key.
            value (torch.Tensor): Tensor representing value.
            attn_mask (CausalBias): Instance of CausalBias representing attention mask.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            is_causal (bool, optional): Whether the attention is causal. Defaults to False.
            scale (Optional[float], optional): Scaling factor. Defaults to None.

        Returns:
            torch.Tensor: Result of the dispatched computation.
        """
        if kwargs is None:
            kwargs = {}
        # 如果调用的函数不是 scaled_dot_product_attention，则抛出未实现的错误
        if func != torch.nn.functional.scaled_dot_product_attention:
            raise NotImplementedError(
                "CausalBias only supports scaled_dot_product_attention"
            )
        # 调用 _dispatch 方法，传递参数和关键字参数
        return cls._dispatch(*args, **kwargs)

    def __repr__(self):
        """
        Returns a string representation of the CausalBias instance.

        Returns:
            str: String representation of the object.
        """
        # 返回 _materialize 方法生成的对象的字符串表示形式
        return self._materialize().__repr__()
# 创建一个上左三角形因果偏置
def causal_upper_left(*size) -> CausalBias:
    """
    创建一个上左三角形因果偏置。

    此函数生成一个上左三角形矩阵，用于表示因果注意力偏置，对角线偏移设置为使包含的值与矩阵的左上角对齐。
    这相当于 `scaled_dot_product_attention` 中 `is_causal=True` 参数的效果。

    用于构建此偏置的等效的 PyTorch 代码是：

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    例如，对于 `shape=(3,4)`，生成的偏置张量如下所示：

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]

    Args:
        size: 偏置矩阵的大小。

    Returns:
        CausalBias: 上左三角形因果偏置的变体。
    """
    assert len(size) == 2, "causal_upper_left 只支持 2D 张量"
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.UPPER_LEFT, seq_len_q, seq_len_kv)


# 创建一个下右三角形因果偏置
def causal_lower_right(*size) -> CausalBias:
    """
    创建一个下右三角形因果偏置。

    此函数生成一个下右三角形矩阵，用于表示因果注意力偏置，对角线偏移设置为使包含的值与矩阵的右下角对齐。

    用于构建此偏置的等效的 PyTorch 代码是：

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    例如，对于 `shape=(3,4)`，生成的偏置张量如下所示：

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Args:
        size: 偏置矩阵的大小。

    Returns:
        CausalBias: 下右三角形因果偏置的变体。
    """
    assert len(size) == 2, "causal_lower_right 只支持 2D 张量"
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.LOWER_RIGHT, seq_len_q, seq_len_kv)
```