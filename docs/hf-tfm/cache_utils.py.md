# `.\transformers\cache_utils.py`

```py
from typing import Any, Dict, List, Optional, Tuple

import torch


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

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
        # Raises a NotImplementedError to remind subclasses to implement this method
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Raises a NotImplementedError to remind subclasses to implement this method
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        # Raises a NotImplementedError to remind subclasses to implement this method
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

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


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        # Initialize lists to store key and value states for each layer
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self.seen_tokens = 0  
``` 
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        # 如果 layer_idx 小于缓存中的层数，返回对应层的键值对
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            # 如果 layer_idx 超出缓存层数，抛出异常
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        # 迭代缓存中的层，每次返回一个键值对
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        # 返回缓存中的层数
        return len(self.key_cache)

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
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # 更新已见的令牌数
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        # 更新缓存
        if len(self.key_cache) <= layer_idx:
            # 如果 layer_idx 超出了缓存中的层数，添加新的层
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 如果 layer_idx 对应的层已存在，拼接新的状态到原有状态上
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # 返回指定层的序列长度，如果未指定层，默认为第 0 层
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        # 返回缓存状态的最大序列长度，对于 DynamicCache，没有最大长度，返回 None
        return None
    # 重新排列缓存以用于束搜索，根据选择的束索引
    def reorder_cache(self, beam_idx: torch.LongTensor):
        # 遍历每一层的缓存
        for layer_idx in range(len(self.key_cache)):
            # 获取当前缓存所在设备
            device = self.key_cache[layer_idx].device
            # 根据给定的索引重新排列键缓存
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            # 获取当前缓存所在设备
            device = self.value_cache[layer_idx].device
            # 根据给定的索引重新排列值缓存
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    # 将`DynamicCache`实例转换为其等效的传统缓存格式
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        # 初始化传统缓存
        legacy_cache = ()
        # 遍历每一层的缓存
        for layer_idx in range(len(self)):
            # 将每一层的键缓存和值缓存添加到传统缓存中
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        # 返回传统缓存
        return legacy_cache

    # 将传统缓存格式转换为等效的`DynamicCache`
    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        # 创建一个新的`DynamicCache`实例
        cache = cls()
        # 如果传统缓存不为空
        if past_key_values is not None:
            # 遍历每一层的传统缓存
            for layer_idx in range(len(past_key_values)):
                # 获取键状态和值状态
                key_states, value_states = past_key_values[layer_idx]
                # 更新`DynamicCache`实例的键缓存和值缓存
                cache.update(key_states, value_states, layer_idx)
        # 返回更新后的`DynamicCache`实例
        return cache
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
        # 初始化缓存列表，用于存储Key和Value状态的张量，每个层级一个张量
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # 上下文窗口的长度
        self.window_length = window_length
        # 沉没令牌的数量，详见原始论文
        self.num_sink_tokens = num_sink_tokens
        # 存储用于正弦和余弦位置编码的缓存
        self.cos_sin_cache = {}
        # 在`generate`中使用，用于记录缓存已经看到的令牌数量
        self.seen_tokens = 0  

    @staticmethod
    def _rotate_half(x):
        # 将张量x的后一半旋转到前一半，前一半旋转到后一半，并连接起来
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # 应用键的旋转位置嵌入，根据提供的余弦和正弦值
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # 临时将余弦和正弦值升级为float32以获得更好的精度
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # 计算后向和前向旋转到序列中早一位位置所需的余弦和正弦值
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            # 将计算结果存入缓存，并转换为key_states的数据类型，然后在第0维上增加维度
            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        # 返回缓存中的结果
        return self.cos_sin_cache[key_states.shape[-2]]
    # 返回缓存状态的序列长度。可选传入层索引。
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # 为了确保 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length 的临时解决方法
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    # 返回缓存状态的最大序列长度
    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    # 更新缓存状态
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    # 重新排序缓存状态，用于束搜索，给定选定的束索引
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
```