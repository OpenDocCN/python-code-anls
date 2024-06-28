# `.\modeling_attn_mask_utils.py`

```py
# 引入必要的库
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# 导入PyTorch库
import torch

# 定义一个数据类，用于处理注意力掩码转换的实用工具
@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    # 定义类的属性，控制注意力掩码的类型和参数
    is_causal: bool
    sliding_window: int

    # 类的初始化方法，设定初始参数并验证滑动窗口参数的有效性
    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        # 如果定义了滑动窗口参数，确保其为正整数
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    # 方法：生成单向（因果）的四维注意力掩码
    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        # 如果不是因果关系，抛出数值错误异常
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # 如果形状未被缓存，创建一个新的因果遮罩并缓存它
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # 创建因果遮罩
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        # 计算输入形状，即(batch_size, query_length)
        input_shape = (attention_mask_2d.shape[0], query_length)

        # 创建因果(mask)
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            # 计算过去键值长度
            past_key_values_length = key_value_length - query_length
            # 生成因果(mask)
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            # 抛出未实现错误，滑动窗口目前仅支持因果掩蔽
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # 扩展注意力(mask)
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        # 如果存在因果(mask)，则用大负数填充未注意的位置
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # 扩展后的注意力(mask)可能会导致溢出
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
        """
        Make causal mask used for bi-directional self-attention.
        """
        # 获取输入的张量形状信息，包括批大小和目标序列长度
        bsz, tgt_len = input_ids_shape

        # 创建一个与目标长度相同的方形矩阵，用极小的浮点数填充，设备为指定的设备
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

        # 创建一个条件张量，范围是目标长度的整数序列
        mask_cond = torch.arange(mask.size(-1), device=device)

        # 使用条件张量来生成一个下三角矩阵，将其对角线上的元素保持为0
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        # 将掩码转换为指定的数据类型
        mask = mask.to(dtype)

        # 如果过去键值长度大于0，将0填充的过去键值长度张量连接到现有掩码之前
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # 如果需要，添加下三角滑动窗口掩码
        if sliding_window is not None:
            # 计算对角线的值，用于下三角滑动窗口掩码
            diagonal = past_key_values_length - sliding_window - 1

            # 创建一个下三角矩阵，掩盖大于对角线值的元素
            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)

            # 使用极小的浮点数填充掩码中被标记的位置
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        # 将掩码扩展为四维张量：[bsz, 1, tgt_len, tgt_len + past_key_values_length]
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        # 获取输入掩码的形状信息，包括批大小和源序列长度
        bsz, src_len = mask.size()

        # 如果未提供目标序列长度，则使用源序列长度作为目标序列长度
        tgt_len = tgt_len if tgt_len is not None else src_len

        # 将二维掩码扩展为四维张量，增加两个额外的维度，将其转换为指定的数据类型
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        # 创建一个反掩码，用1减去扩展的掩码
        inverted_mask = 1.0 - expanded_mask

        # 使用极小的浮点数填充反掩码中被标记的位置
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.FloatTensor,
        min_dtype: float,
        device: Optional[torch.device] = None
    ):
        """
        Unmasks the unattended positions in the attention matrix.
        """
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        """
        # fmt: on
        # 检查 expanded_mask 的数据类型是否为 torch.bool，若是则抛出 ValueError
        if expanded_mask.dtype == torch.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )

        # 返回一个修改过的 expanded_mask，其中未被完全掩盖的行将保持不变，其他行将被置为零
        return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))
    # 创建一个用于 SDPA（scaled_dot_product_attention）的 4D 因果注意力掩码，形状为 `(batch_size, 1, query_length, key_value_length)`
    # 从形状为 `(batch_size, key_value_length)` 的 2D 注意力掩码创建
def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    # 创建一个 AttentionMaskConverter 对象，用于生成因果关系的注意力掩码
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    # 计算 key_value_length，这里是输入形状的最后一个维度加上过去的键值长度
    key_value_length = input_shape[-1] + past_key_values_length

    # 如果输入的 attention_mask 不为空且是 2D 的
    if attention_mask is not None and len(attention_mask.shape) == 2:
        # 将 2D 的 attention_mask 转换成 4D 的
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    # 如果 attention_mask 不为空且是 4D 的
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        # 检查 attention_mask 的形状是否符合预期
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            # 如果不符合预期，抛出 ValueError
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # 如果 4D 的 attention_mask 形状正确，则反转它并用负无穷填充
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        # 如果 attention_mask 为空或者不是 2D 或 4D 的，则使用 AttentionMaskConverter 生成因果 4D 注意力掩码
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    # 返回生成的 attention_mask
    return attention_mask


# Adapted from _prepare_4d_causal_attention_mask
# 根据 _prepare_4d_causal_attention_mask 适配
def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.

    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    """
    # 创建一个注意力掩码转换器对象，设定为因果关系模式，并指定是否使用滑动窗口
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    # 计算键值对的长度，包括输入形状的最后一个维度和过去键值对的长度
    key_value_length = input_shape[-1] + past_key_values_length
    # 获取输入形状中的批处理大小和查询长度
    batch_size, query_length = input_shape

    # 检查是否处于追踪状态，如果是，则需要使用SDPA的`attn_mask`参数而不是自定义的`attention_mask`
    is_tracing = (
        torch.jit.is_tracing()
        or isinstance(inputs_embeds, torch.fx.Proxy)
        or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
    )

    # 如果传入了注意力掩码
    if attention_mask is not None:
        # 如果注意力掩码是4维的
        if len(attention_mask.shape) == 4:
            # 验证注意力掩码的形状是否符合预期
            expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
            if tuple(attention_mask.shape) != expected_shape:
                # 抛出值错误，指出注意力掩码的形状不正确
                raise ValueError(
                    f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
            else:
                # 如果4维掩码形状正确，反转掩码并用负无穷填充
                inverted_mask = 1.0 - attention_mask.to(inputs_embeds.dtype)
                attention_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
                )
                return attention_mask

        # 如果不处于追踪状态，并且所有的注意力掩码值都是1
        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1:
                # 当查询长度为1时，因果关注和双向关注是相同的
                attention_mask = None
            elif key_value_length == query_length:
                # 当键值对的长度等于查询长度时，不需要注意力掩码
                attention_mask = None
            else:
                # 对于查询长度大于1且键值长度不等于查询长度的情况，无法忽略注意力掩码，需要特别处理
                pass

    # 如果没有传入注意力掩码，并且查询长度大于1且键值长度不等于查询长度
    elif query_length > 1 and key_value_length != query_length:
        # 将注意力掩码设为True，以便在后续控制流中转到`to_causal_4d`
        attention_mask = True
    # 如果正在进行跟踪（tracing），且未提供注意力掩码（attention_mask），则抛出值错误异常
    elif is_tracing:
        raise ValueError(
            'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
        )

    # 如果没有提供 attention_mask，则将 expanded_4d_mask 设置为 None
    if attention_mask is None:
        expanded_4d_mask = None
    # 如果 attention_mask 设置为 True，则通过 attn_mask_converter.to_causal_4d 函数生成扩展后的 4D 注意力掩码
    elif attention_mask is True:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    # 否则，根据给定的 attention_mask 使用 attn_mask_converter.to_4d 函数生成扩展后的 4D 注意力掩码
    else:
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # 如果不是在跟踪模式下，并且 expanded_4d_mask 存在且在 CUDA 设备上，
        # 则调用 AttentionMaskConverter._unmask_unattended 函数，处理未注意的部分
        if not is_tracing and expanded_4d_mask.device.type == "cuda":
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, min_dtype=torch.finfo(inputs_embeds.dtype).min
            )

    # 返回生成或处理后的 expanded_4d_mask 注意力掩码
    return expanded_4d_mask
# 创建一个非因果关系的四维注意力掩码，其形状为 `(batch_size, 1, query_length, key_value_length)`，从形状为 `(batch_size, key_value_length)` 的二维掩码创建
def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    # 调用内部函数 `_expand_mask` 扩展掩码至四维并返回
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


# 为 SDPA（Scaled Dot-Product Attention）创建非因果四维注意力掩码，形状为 `(batch_size, 1, query_length, key_value_length)`，从形状为 `(batch_size, key_value_length)` 的二维掩码创建
def _prepare_4d_attention_mask_for_sdpa(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    # 获取 batch_size 和 key_value_length 的尺寸
    batch_size, key_value_length = mask.shape
    # 如果未提供 tgt_len，则默认为 key_value_length
    tgt_len = tgt_len if tgt_len is not None else key_value_length

    # 检查是否处于追踪模式
    is_tracing = (
        torch.jit.is_tracing()
        or isinstance(mask, torch.fx.Proxy)
        or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
    )

    # 如果掩码中所有元素均为 1
    if torch.all(mask == 1):
        if is_tracing:
            pass  # 如果处于追踪模式，不做任何操作
        elif tgt_len == 1:
            # 对于 query_length == 1，因果和双向注意力相同，返回 None
            return None
        elif key_value_length == tgt_len:
            # 如果 key_value_length 等于 tgt_len，返回 None
            return None
        else:
            # 对于 query_length > 1 且 key_value_length != query_length 的情况，
            # 我们不能忽略注意力掩码，因为 SDPA 因果掩码的生成可能会出错，
            # 我们在 SDPA 中将 is_causal=False，并依赖于 Transformers 的 attention_mask，因此在这里不设置为 None。
            # 参考: https://github.com/pytorch/pytorch/issues/108108
            return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
    else:
        # 对于其他情况，调用 `_expand_mask` 扩展掩码并返回
        return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
    # device: torch.device,
    # 定义一个参数 `device`，类型为 `torch.device`，用于指定张量运算的设备（如CPU或GPU）

    # past_key_values_length: int = 0,
    # 定义一个参数 `past_key_values_length`，类型为 `int`，默认值为 `0`，用于指定过去的键值对长度

    # sliding_window: Optional[int] = None,
    # 定义一个参数 `sliding_window`，类型为可选的 `int`，默认值为 `None`，用于指定滑动窗口的大小
# 创建一个形状为 `(batch_size, 1, query_length, key_value_length)` 的因果性四维掩码

def create_causal_mask(
    input_shape: Union[tuple[int], list[int], torch.Size],
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None
) -> Optional[torch.Tensor]:
    """
    创建一个形状为 `(batch_size, 1, query_length, key_value_length)` 的因果性四维掩码

    Args:
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            输入形状应为定义 `(batch_size, query_length)` 的元组。
        dtype (`torch.dtype`):
            所创建掩码的 torch 数据类型。
        device (`torch.device`):
            所创建掩码的 torch 设备。
        sliding_window (`int`, *optional*):
            如果模型使用窗口化注意力，应传入一个滑动窗口大小。
    """
    # 创建一个注意力掩码转换器，设置为因果性，根据是否提供滑动窗口参数决定
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    # 计算 key_value_length，包括过去键值长度和输入形状的最后一个维度
    key_value_length = past_key_values_length + input_shape[-1]

    # 使用掩码转换器生成四维因果性掩码
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )

    # 返回生成的注意力掩码
    return attention_mask
```