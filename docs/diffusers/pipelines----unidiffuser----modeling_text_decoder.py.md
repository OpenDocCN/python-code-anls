# `.\diffusers\pipelines\unidiffuser\modeling_text_decoder.py`

```py
# 从 typing 模块导入 Optional 类型
from typing import Optional

# 导入 numpy 库并命名为 np
import numpy as np
# 导入 PyTorch 库
import torch
# 从 PyTorch 中导入 nn 模块
from torch import nn
# 从 transformers 库导入 GPT2Config 和 GPT2LMHeadModel
from transformers import GPT2Config, GPT2LMHeadModel
# 从 transformers.modeling_utils 导入 ModuleUtilsMixin 类
from transformers.modeling_utils import ModuleUtilsMixin

# 从上层目录导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从上层目录导入 ModelMixin
from ...models import ModelMixin


# 从 https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py 修改而来
class UniDiffuserTextDecoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    """
    用于图像-文本 [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) 模型的文本解码器。此模型用于
    从 UniDiffuser 图像-文本嵌入生成文本。
    """

    # 在加载时忽略的意外键
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.bias", r"h\.\d+\.attn\.masked_bias"]

    # 注册为配置的初始化方法
    @register_to_config
    def __init__(
        # 前缀长度参数
        self,
        prefix_length: int,
        # 前缀内部维度参数
        prefix_inner_dim: int,
        # 可选的前缀隐藏维度参数
        prefix_hidden_dim: Optional[int] = None,
        # 词汇表大小，默认是 GPT2 配置参数
        vocab_size: int = 50257,  # Start of GPT2 config args
        # 位置数量参数
        n_positions: int = 1024,
        # 嵌入维度参数
        n_embd: int = 768,
        # 层数参数
        n_layer: int = 12,
        # 注意力头数量参数
        n_head: int = 12,
        # 可选的内部维度参数
        n_inner: Optional[int] = None,
        # 激活函数类型，默认是 "gelu_new"
        activation_function: str = "gelu_new",
        # 残差丢弃率参数
        resid_pdrop: float = 0.1,
        # 嵌入丢弃率参数
        embd_pdrop: float = 0.1,
        # 注意力丢弃率参数
        attn_pdrop: float = 0.1,
        # 层归一化的 epsilon 值
        layer_norm_epsilon: float = 1e-5,
        # 初始化范围参数
        initializer_range: float = 0.02,
        # 是否缩放注意力权重
        scale_attn_weights: bool = True,
        # 是否使用缓存
        use_cache: bool = True,
        # 是否按层索引的倒数缩放注意力
        scale_attn_by_inverse_layer_idx: bool = False,
        # 是否重排和上溯注意力
        reorder_and_upcast_attn: bool = False,
    ):
        # 初始化父类
        super().__init__()

        # 设置前缀长度
        self.prefix_length = prefix_length

        # 检查前缀内维度与嵌入维度是否一致，且前缀隐藏维度是否为 None
        if prefix_inner_dim != n_embd and prefix_hidden_dim is None:
            # 抛出错误，提示前缀隐藏维度不能为 None
            raise ValueError(
                f"`prefix_hidden_dim` cannot be `None` when `prefix_inner_dim`: {prefix_hidden_dim} and"
                f" `n_embd`: {n_embd} are not equal."
            )

        # 设置前缀内维度
        self.prefix_inner_dim = prefix_inner_dim
        # 设置前缀隐藏维度
        self.prefix_hidden_dim = prefix_hidden_dim

        # 创建编码前缀的线性层，如果前缀隐藏维度不为 None
        self.encode_prefix = (
            nn.Linear(self.prefix_inner_dim, self.prefix_hidden_dim)
            if self.prefix_hidden_dim is not None
            else nn.Identity()
        )
        # 创建解码前缀的线性层，如果前缀隐藏维度不为 None
        self.decode_prefix = (
            nn.Linear(self.prefix_hidden_dim, n_embd) if self.prefix_hidden_dim is not None else nn.Identity()
        )

        # 配置 GPT2 模型的参数
        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )
        # 创建 GPT2 语言模型头
        self.transformer = GPT2LMHeadModel(gpt_config)

    def forward(
        # 定义前向传播的输入参数
        input_ids: torch.Tensor,
        prefix_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids (`torch.Tensor` of shape `(N, max_seq_len)`):
                用于推理的文本标记。
            prefix_embeds (`torch.Tensor` of shape `(N, prefix_length, 768)`):
                预先附加到嵌入标记的前缀嵌入。
            attention_mask (`torch.Tensor` of shape `(N, prefix_length + max_seq_len, 768)`, *optional*):
                前缀嵌入的注意力掩码。
            labels (`torch.Tensor`, *optional*):
                用于语言建模的标签。
        """
        # 获取输入标记的嵌入表示
        embedding_text = self.transformer.transformer.wte(input_ids)
        # 编码前缀嵌入
        hidden = self.encode_prefix(prefix_embeds)
        # 解码前缀嵌入
        prefix_embeds = self.decode_prefix(hidden)
        # 拼接前缀嵌入和文本嵌入
        embedding_cat = torch.cat((prefix_embeds, embedding_text), dim=1)

        # 如果标签不为 None，处理标签
        if labels is not None:
            # 获取虚拟标记，用于标签拼接
            dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
            # 拼接虚拟标记和输入标记
            labels = torch.cat((dummy_token, input_ids), dim=1)
        # 使用 Transformer 进行前向传播
        out = self.transformer(inputs_embeds=embedding_cat, labels=labels, attention_mask=attention_mask)
        # 如果前缀隐藏维度不为 None，返回输出和隐藏状态
        if self.prefix_hidden_dim is not None:
            return out, hidden
        else:
            # 否则只返回输出
            return out
    # 获取一个全零的张量，形状为 (batch_size, prefix_length)，用于生成虚拟的输入令牌
        def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
            return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    # 编码给定的前缀，调用编码前缀的方法
        def encode(self, prefix):
            return self.encode_prefix(prefix)
    
    # 生成文本描述，采用无梯度计算以节省内存
        @torch.no_grad()
        def generate_captions(self, features, eos_token_id, device):
            """
            根据文本嵌入特征生成描述，返回字符串列表。
    
            参数：
                features (`torch.Tensor`，形状为 `(B, L, D)`):
                    用于生成描述的文本嵌入特征。
                eos_token_id (`int`):
                    文本解码模型的 EOS 令牌 ID。
                device:
                    进行文本生成的设备。
    
            返回：
                `List[str]`: 从解码模型生成的字符串列表。
            """
    
            # 将特征张量在第0维度分割成多个单独的特征
            features = torch.split(features, 1, dim=0)
            generated_tokens = []  # 存储生成的令牌
            generated_seq_lengths = []  # 存储生成序列的长度
            for feature in features:
                # 解码前缀特征，转换为 CLIP 特征
                feature = self.decode_prefix(feature.to(device))
                # 当前仅支持束搜索
                output_tokens, seq_lengths = self.generate_beam(
                    input_embeds=feature, device=device, eos_token_id=eos_token_id
                )
                # 添加生成的第一个令牌和序列长度到列表
                generated_tokens.append(output_tokens[0])
                generated_seq_lengths.append(seq_lengths[0])
            # 将生成的令牌和序列长度堆叠成张量
            generated_tokens = torch.stack(generated_tokens)
            generated_seq_lengths = torch.stack(generated_seq_lengths)
            # 返回生成的令牌和序列长度
            return generated_tokens, generated_seq_lengths
    
    # 生成束搜索的描述，采用无梯度计算以节省内存
        @torch.no_grad()
        def generate_beam(
            self,
            input_ids=None,  # 输入的 ID，默认为 None
            input_embeds=None,  # 输入的嵌入，默认为 None
            device=None,  # 设备，默认为 None
            beam_size: int = 5,  # 束搜索的大小，默认为 5
            entry_length: int = 67,  # 入口长度，默认为 67
            temperature: float = 1.0,  # 温度，默认为 1.0
            eos_token_id: Optional[int] = None,  # EOS 令牌 ID，默认为 None
```