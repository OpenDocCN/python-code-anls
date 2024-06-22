# `.\transformers\modeling_attn_mask_utils.py`

```py
# 导入所需模块和类
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
# 导入 PyTorch 库
import torch

# 定义一个数据类，用于生成和转换注意力遮罩
@dataclass
class AttentionMaskConverter:
    """
    一个实用的注意力遮罩类，可以实现以下功能：
        - 创建一个因果关系的四维遮罩
        - 创建一个带有滑动窗口的因果关系的四维遮罩
        - 将一个二维的注意力遮罩（batch_size, query_length）转换为一个四维的注意力遮罩（batch_size, 1, query_length, key_value_length），该遮罩可以与注意力分数相乘

    例子:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```py

    参数:
        is_causal (`bool`):
            注意力遮罩是否应该是单向的（因果关系）或双向的。

        sliding_window (`int`, *可选*):
            如果定义了 `sliding_window` 为正整数，则可以创建滑动窗口遮罩。
    """

    is_causal: bool
    sliding_window: int

    # 初始化函数
    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        # 设置是否为因果关系的遮罩
        self.is_causal = is_causal
        # 设置滑动窗口的大小
        self.sliding_window = sliding_window

        # 如果滑动窗口大小不为 None 且小于等于 0，则抛出值错误异常
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    # 将输入的注意力遮罩转换为因果关系的四维遮罩
    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
``` 
    ) -> Optional[torch.Tensor]:
        """
        创建一个因果关系的 4D 掩码，形状为 (bsz, head_dim=1, query_length, key_value_length)，并向右上角的三角矩阵（因果关系掩码）添加大的负偏置。
        """
        if not self.is_causal:
            raise ValueError(f"请仅在 {self.__class__} 的 `is_causal` 设置为 True 时使用 `to_causal_4d`。")

        # 如果形状未缓存，则创建一个新的因果关系掩码并缓存它
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # 创建因果关系掩码
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
        # 定义输入形状为 (bsz, query_length)
        input_shape = (attention_mask_2d.shape[0], query_length)

        # 创建因果关系掩码
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        # 如果输入形状的最后一个维度大于1或者存在滑动窗口，并且是因果关系的
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            # 如果未传入 key_value_length，则抛出异常
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            # 计算过去键值的长度
            past_key_values_length = key_value_length - query_length
            # 创建因果关系的4D掩码
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        # 如果滑动窗口不为空，则抛出未实现的错误
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # 将2D注意力掩码扩展为4D注意力掩码
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        # 如果存在因果关系的4D掩码，则将其应用到扩展的注意力掩码上，并填充大的负数偏置到未注意的位置
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # 扩展的4D掩码
        expanded_4d_mask = expanded_attn_mask

        # 返回扩展的4D注意力掩码
        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        生成用于双向自注意力的因果掩码。
        """
        # 获取输入张量的形状
        bsz, tgt_len = input_ids_shape
        # 创建一个形状为(tgt_len, tgt_len)的张量，填充为dtype类型的最小值，存储在设备device上
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        # 创建一个与mask相同大小的张量，用于条件填充
        mask_cond = torch.arange(mask.size(-1), device=device)
        # 根据条件填充mask张量
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        # 将mask张量转换为dtype类型
        mask = mask.to(dtype)

        # 如果过去的键值长度大于0，则在mask张量的右侧添加0填充
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # 如果需要，添加下三角滑动窗口掩码
        if sliding_window is not None:
            # 计算对角线位置
            diagonal = past_key_values_length - sliding_window + 1
            # 创建上三角矩阵
            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            # 使用dtype类型的最小值填充mask张量
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        # 返回扩展后的mask张量，形状为[bsz, 1, tgt_len, tgt_len + past_key_values_length]
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        将attention_mask从`[bsz, seq_len]`扩展为`[bsz, 1, tgt_seq_len, src_seq_len]`。
        """
        # 获取mask张量的形状
        bsz, src_len = mask.size()
        # 如果tgt_len为None，则将其设置为src_len
        tgt_len = tgt_len if tgt_len is not None else src_len

        # 将mask张量扩展为形状为[bsz, 1, tgt_len, src_len]，并转换为dtype类型
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        # 创建反转的掩码，用于填充未被注意的位置
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]
# 准备 4D 因果注意力掩码，从一个形状为 `(batch_size, key_value_length)` 的 2D 掩码创建一个形状为 `(batch_size, 1, query_length, key_value_length)` 的掩码

def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],  # 注意力掩码，可以为 None
    input_shape: Union[torch.Size, Tuple, List],  # 输入形状，应为一个定义了 `(batch_size, query_length)` 的元组
    inputs_embeds: torch.Tensor,  # 嵌入的输入，作为 torch 张量
    past_key_values_length: int,  # 关键值缓存的长度
    sliding_window: Optional[int] = None,  # 如果模型使用窗口注意力，应传入一个滑动窗口大小，默认为 None
):

    # 创建注意力掩码转换器，设为因果模式，如果使用窗口注意力，传入滑动窗口大小
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    # 计算关键值长度，包括输入形状和过去关键值长度
    key_value_length = input_shape[-1] + past_key_values_length

    # 如果注意力掩码不为空且形状为 2D
    if attention_mask is not None and len(attention_mask.shape) == 2:
        # 调用掩码转换器，将 2D 控制器转换为 4D
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    # 如果注意力掩码不为空且形状为 4D
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        # 期望的形状应为 (batch_size, 1, query_length, key_value_length)
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        # 如果注意力掩码形状与期望的不同，则引发错误
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # 如果 4D 掩码形状正确，则反转它并用负无穷填充
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        # 如果注意力掩码为空，则调用掩码转换器，创建因果 4D 控制器
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    # 返回处理后的注意力掩码
    return attention_mask


# 从 _prepare_4d_causal_attention_mask 适配而来
def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],  # 注意力掩码，可以为 None
    input_shape: Union[torch.Size, Tuple, List],  # 输入形状，应为一个定义了 `(batch_size, query_length)` 的元组
    inputs_embeds: torch.Tensor,  # 嵌入的输入，作为 torch 张量
    past_key_values_length: int,  # 关键值缓存的长度
    sliding_window: Optional[int] = None,  # 如果模型使用窗口注意力，应传入一个滑动窗口大小，默认为 None
):
    """
    为 `torch.nn.functional.scaled_dot_product_attention` 准备正确的 `attn_mask` 参数。

    如果 `attention_mask` 参数中没有被掩码的标记，我们将在 `query_length == 1` 和...
    """
    """
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    # 创建一个AttentionMaskConverter对象，设置is_causal为True，sliding_window为sliding_window参数的值
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    # 计算key_value_length的值
    key_value_length = input_shape[-1] + past_key_values_length
    # 获取batch_size和query_length
    batch_size, query_length = input_shape

    # 检查是否处于追踪状态，如果是则设置is_tracing为True
    is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy)

    # 如果attention_mask不为None
    if attention_mask is not None:
        # 如果attention_mask的维度为4
        if len(attention_mask.shape) == 4:
            # 检查attention_mask的形状是否符合预期
            expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
            else:
                # 如果4D mask的形状正确 - 反转它并用负无穷填充
                inverted_mask = 1.0 - attention_mask.to(inputs_embeds.dtype)
                attention_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
                )
                return attention_mask

        # 如果不处于追踪状态且attention_mask全为1
        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1:
                # 对于query_length == 1，因果注意力和双向注意力是相同的
                attention_mask = None
            elif key_value_length == query_length:
                attention_mask = None
            else:
                # 不幸的是，对于query_length > 1且key_value_length != query_length，我们通常不能忽略注意力掩码，因为SDPA因果掩码生成可能是错误的。
                # 我们将在SDPA中设置`is_causal=False`，依赖于Transformers的attention_mask，因此在这里不将其设置为None。
                # 参考：https://github.com/pytorch/pytorch/issues/108108
                pass
    # 如果attention_mask为None且query_length > 1且key_value_length != query_length
    elif query_length > 1 and key_value_length != query_length:
        # 参考上面的注释（https://github.com/pytorch/pytorch/issues/108108）。
        # 丑陋的：我们在这里将其设置为True，以便在以下控制流中调度到`to_causal_4d`。
        attention_mask = True
    # 如果正在进行跟踪操作
    elif is_tracing:
        # 抛出值错误，提示不能在没有提供 attention_mask 的情况下使用 torch.jit.trace 跟踪 SDPA 注意力模块。
        # 解决此问题的方法是，要么在加载模型时使用参数 `attn_implementation="eager"`，要么在跟踪模型时传递 attention_mask 输入。
        raise ValueError(
            'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
        )

    # 如果 attention_mask 为 None
    if attention_mask is None:
        # 将扩展后的 4D 注意力掩码设置为 None
        expanded_4d_mask = None
    # 如果 attention_mask 为 True
    elif attention_mask is True:
        # 将扩展后的 4D 注意力掩码设置为按因果关系的形式，根据输入形状、key_value_length 和输入嵌入的 dtype、device 转换而来
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    # 如果 attention_mask 不为 None 且不为 True
    else:
        # 将扩展后的 4D 注意力掩码设置为根据输入的 attention_mask 转换而来，同时考虑 key_value_length 和输入嵌入的 dtype
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # 从 PyTorch 2.1 开始，使用内存效率较高的注意力后端的 F.scaled_dot_product_attention
        # 如果在注意力掩码中某些序列完全未被关注，将会产生 NaN 值。详情参见：https://github.com/pytorch/pytorch/issues/110213
        #
        # 如果查询长度大于 1，且不是在进行跟踪操作，则执行以下代码块
        if query_length > 1 and not is_tracing:
            # 调用 _unmask_unattended 方法，将未关注的序列从扩展后的 4D 注意力掩码中移除，并将其值设为 0.0
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, attention_mask, unmasked_value=0.0
            )

    # 返回扩展后的 4D 注意力掩码
    return expanded_4d_mask
# 创建一个非因果关系的4D掩码，形状为`(batch_size, 1, query_length, key_value_length)`，从形状为`(batch_size, key_value_length)`的2D掩码中创建

def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    创建一个非因果关系的4D掩码，形状为`(batch_size, 1, query_length, key_value_length)`，从形状为`(batch_size, key_value_length)`的2D掩码中创建

    Args:
        mask (`torch.Tensor` or `None`):
            形状为`(batch_size, key_value_length)`的2D注意力掩码
        dtype (`torch.dtype`):
            创建的掩码应具有的torch数据类型
        tgt_len (`int`):
            创建的掩码应具有的目标长度或查询长度
    """
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_attention_mask_for_sdpa(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    创建一个非因果关系的4D掩码，形状为`(batch_size, 1, query_length, key_value_length)`，从形状为`(batch_size, key_value_length)`的2D掩码中创建

    Args:
        mask (`torch.Tensor` or `None`):
            形状为`(batch_size, key_value_length)`的2D注意力掩码
        dtype (`torch.dtype`):
            创建的掩码应具有的torch数据类型
        tgt_len (`int`):
            创建的掩码应具有的目标长度或查询长度
    """
    batch_size, key_value_length = mask.shape
    tgt_len = tgt_len if tgt_len is not None else key_value_length

    # torch.jit.trace和torchdynamo无法捕获控制流`is_causal=attention_mask is None and q_len > 1`，作为SDPA参数使用。我们通过始终在追踪时使用SDPA的`attn_mask`参数来保持与这些追踪工具的兼容性。
    # 当使用torchdynamo和fullgraph=True时，也需要修复这个问题。
    is_tracing = torch.jit.is_tracing()

    if torch.all(mask == 1):
        if is_tracing:
            pass
        elif tgt_len == 1:
            # 对于query_length == 1，因果关系注意力和双向注意力是相同的。
            return None
        elif key_value_length == tgt_len:
            return None
        else:
            # 不幸的是，对于query_length > 1且key_value_length != query_length，我们通常不能忽略注意力掩码，因为SDPA因果掩码生成可能是错误的。我们将在SDPA中将is_causal=False，并依赖于Transformers的attention_mask，因此在这里不将其设置为None。
            # 参考：https://github.com/pytorch/pytorch/issues/108108
            return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
    else:
        return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _create_4d_causal_attention_mask(
    input_shape: Union[torch.Size, Tuple, List],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    # 创建一个因果关系的4D掩码，形状为`(batch_size, 1, query_length, key_value_length)`

    Args:
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            输入形状应该是一个定义了`(batch_size, query_length)`的元组。
        dtype (`torch.dtype`):
            创建的掩码应该具有的 torch 数据类型。
        device (`int`):
            创建的掩码应该存在的 torch 设备。
        sliding_window (`int`, *optional*):
            如果模型使用窗口化注意力，应传入一个滑动窗口值。
    """
    # 创建一个 AttentionMaskConverter 对象，is_causal 设置为 True，如果使用窗口化注意力，传入滑动窗口值
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    # 计算 key_value_length，为过去的键值长度加上输入形状的最后一个维度
    key_value_length = past_key_values_length + input_shape[-1]
    # 使用 attn_mask_converter 将掩码转换为因果关系的4D形状
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )

    # 返回生成的 attention_mask
    return attention_mask
```