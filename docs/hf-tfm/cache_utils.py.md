# `.\cache_utils.py`

```py
# 导入必要的模块和类
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# 导入日志模块中的日志记录器
from .configuration_utils import PretrainedConfig
from .utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个数据类，表示缓存的抽象基类
@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    # 更新缓存，存储新的键和值的状态到特定层的缓存中
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    # 获取缓存状态的序列长度
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    # 获取缓存状态的最大序列长度
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    # 根据新输入的序列长度返回可用的缓存长度
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    # 已弃用警告：`seen_tokens` 属性将在 v4.41 中移除，请使用 `cache_position` 模型输入代替
    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    """
    Concrete subclass of Cache representing a dynamic cache.
    """
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    # 初始化函数，设置空的缓存列表和初始的 tokens 计数
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []  # 用于存储每个层的 Key 状态的列表
        self.value_cache: List[torch.Tensor] = []  # 用于存储每个层的 Value 状态的列表
        self._seen_tokens = 0  # 在 `generate` 方法中用于记录缓存已见过的 tokens 数量的计数器

    # 支持通过索引访问 `past_key_value`，例如 `past_key_value[0][0].shape[2]` 获取序列长度
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    # 支持通过迭代访问 `past_key_value`，例如 `for x in past_key_value:` 迭代访问键和值
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    # 返回当前缓存的层数，支持 `len(past_key_value)` 的操作，对应模型中的层数
    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    # 更新缓存中特定层的 Key 和 Value 状态
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Update function to update the cache with new key and value states for a specific layer.
        """
        # 将新的 Key 和 Value 状态添加到指定层的缓存中
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        # 可选的其他缓存参数，这里可以用来扩展更新功能
        if cache_kwargs is not None:
            pass  # Placeholder for additional cache update logic
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        # 如果 layer_idx 为 0，则更新已见过的 token 数量
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        # 如果 key_cache 的长度小于等于 layer_idx，则将 key_states 和 value_states 添加到 cache 中
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 否则，在已有的 cache 中更新 layer_idx 对应的 key_states 和 value_states
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # 返回更新后的 key_states 和 value_states
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # 如果 key_cache 的长度小于等于 layer_idx，则返回 0
        if len(self.key_cache) <= layer_idx:
            return 0
        # 否则返回 key_cache 中 layer_idx 对应的 tensor 的第二维度的大小
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        # 返回 None，因为 DynamicCache 类型的缓存没有最大长度限制
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        # 遍历每个层的缓存，根据 beam_idx 重新排序 key_cache 和 value_cache
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        # 将 DynamicCache 实例转换成遗留缓存格式的等价表示，并返回为元组的形式
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        # 创建一个空的 DynamicCache 对象
        cache = cls()
        # 如果传入了过去的键-值对数据
        if past_key_values is not None:
            # 遍历过去的键-值对数据的每一层
            for layer_idx in range(len(past_key_values)):
                # 分别获取键状态和值状态
                key_states, value_states = past_key_values[layer_idx]
                # 将键状态和值状态更新到缓存中的指定层
                cache.update(key_states, value_states, layer_idx)
        # 返回转换后的 DynamicCache 对象
        return cache
# 定义一个名为 `SinkCache` 的类，继承自 `Cache` 类，实现了一个缓存机制，根据 [Attention Sinks paper](https://arxiv.org/abs/2309.17453) 描述的内容，
# 允许模型在超出其上下文窗口长度的情况下生成内容，同时保持对话的流畅性。当丢弃过去的标记时，模型将失去依赖于被丢弃上下文的标记生成能力。

class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        # 初始化空列表，用于存储每一层的 Key 状态和 Value 状态的张量
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # 设定上下文窗口长度和沉降标记的数量
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        # 缓存余弦和正弦值的字典
        self.cos_sin_cache = {}
        # 记录缓存已见标记的数量，在 `generate` 方法中使用，用于统计缓存处理的标记数
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    @staticmethod
    def _rotate_half(x):
        # 将张量 `x` 在最后一个维度上分成两半，进行半旋转操作
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # 对 Key 状态应用旋转位置嵌入，使用余弦和正弦值进行加权
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # 临时提升到 float32 类型以提高精度
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # 计算用于向前和向后旋转到序列中前一位置所需的余弦和正弦值
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            # 缓存计算结果，使用张量的数据类型，并扩展维度
            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # 如果 self.key_cache 的长度小于等于 layer_idx，返回 0
        if len(self.key_cache) <= layer_idx:
            return 0
        # 返回 self.key_cache[layer_idx] 张量的倒数第二个维度的长度
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        # 返回 window_length 属性的值，即缓存状态的最大序列长度
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Updates the cache with new key and value states for a specific layer."""
        # 更新指定层的缓存状态 key_cache 和 value_cache
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        # 针对 beam search 重新排序缓存状态
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            # 根据 beam_idx 重新排序 key_cache
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            # 根据 beam_idx 重新排序 value_cache
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the `max_position_embeddings`, `hidden_size` and `num_attention_heads`
            required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        # 设置最大批处理大小
        self.max_batch_size = max_batch_size
        # 设置最大缓存长度，如果未指定则使用配置文件中的最大位置嵌入数
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # 计算头部维度，如果配置中定义了自定义头部维度，则使用；否则根据隐藏层大小和注意力头数计算
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        # 设置数据类型，默认为 torch.float32
        self.dtype = dtype if dtype is not None else torch.float32
        # 设置键值头的数量，如果未指定则使用配置文件中的注意力头数
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        # 初始化键和值的缓存张量，形状为 (最大批处理大小, 键值头数, 最大缓存长度, 头部维度)
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.value_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    def update_cache(self, key_states: torch.Tensor, value_states: torch.Tensor,
                     layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for. Kept for backward compatibility
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` just needs the `q_len`
                to know how much of the cache it should overwrite.

        Return:
            A tuple containing the updated key and value states.
        """

        new_cache_positions = cache_kwargs.get("cache_position")  # 获取缓存位置参数
        k_out = self.key_cache  # 获取当前的键缓存
        v_out = self.value_cache  # 获取当前的值缓存

        k_out[:, :, new_cache_positions] = key_states  # 更新键缓存的指定位置的状态
        v_out[:, :, new_cache_positions] = value_states  # 更新值缓存的指定位置的状态

        return k_out, v_out  # 返回更新后的键和值缓存

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states that were seen by the model. `layer_idx` kept for BC
        """
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: This is error prone, a filled cache may be `0.0`. Let's use a stateless integer instead, after
        # https://github.com/pytorch/pytorch/issues/120248 is fixed
        return (self.key_cache[0, 0].any(dim=-1)).sum()  # 计算缓存中非零值的数量，用于表示序列长度

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return self.max_cache_len  # 返回缓存中的最大序列长度

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        Reorders the cache for beam search, given the selected beam indices.
        """
        device = self.key_cache.device  # 获取当前键缓存所在的设备
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))  # 根据beam索引重新排序键缓存
        device = self.value_cache.device  # 获取当前值缓存所在的设备
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))  # 根据beam索引重新排序值缓存

    def to_legacy_cache(self):
        """
        Dummy function for BC. We have to keep it because otherwise the call in the forward of models will break it
        """
        return None  # 返回空值，用于保持向后兼容
```