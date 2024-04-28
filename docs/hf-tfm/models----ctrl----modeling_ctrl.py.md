# `.\models\ctrl\modeling_ctrl.py`

```
# 设置文件编码为UTF-8
# 版权声明
# 根据Apache许可证2.0，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则在“按原样”基础上分发软件
# 没有明示或暗示的任何保证或条件，包括但不限于
# 特定用途的保证或条件。请查看许可证以了解权限和限制
""" PyTorch CTRL模型。"""

# 导入必要的库和模块
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_ctrl import CTRLConfig

# 获取logger对象
logger = logging.get_logger(__name__)

# 文档配置的变量
_CONFIG_FOR_DOC = "CTRLConfig"

# 预训练模型存档列表
CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/ctrl"
    # 查看所有CTRL模型：https://huggingface.co/models?filter=ctrl
]


def angle_defn(pos, i, d_model_size):
    # 角度定义函数
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates


def positional_encoding(position, d_model_size, dtype):
    # 创建位置编码的正弦模式
    angle_rads = angle_defn(
        torch.arange(position, dtype=dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
        d_model_size,
    )

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # 计算注意力
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        nd, ns = scaled_attention_logits.size(-2), scaled_attention_logits.size(-1)
        scaled_attention_logits += mask[ns - nd : ns, :ns] * -1e4

    if attention_mask is not None:
        # 应用注意力掩码
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    # 如果需要的话掩盖头部
    if head_mask is not None:
        attention_weights = attention_weights * head_mask

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    # 初始化方法，设定模型大小和注意力头数
    def __init__(self, d_model_size, num_heads):
        # 调用父类初始化方法
        super().__init__()
        # 初始化注意力头数和模型大小
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        # 计算每个注意力头的深度
        self.depth = int(d_model_size / self.num_heads)

        # 初始化权重矩阵
        self.Wq = nn.Linear(d_model_size, d_model_size)
        self.Wk = nn.Linear(d_model_size, d_model_size)
        self.Wv = nn.Linear(d_model_size, d_model_size)

        # 初始化一个线性层
        self.dense = nn.Linear(d_model_size, d_model_size)
        # 初始化一个集合用于存储需要剔除的注意力头
        self.pruned_heads = set()

    # 剔除指定的注意力头
    def prune_heads(self, heads):
        # 计算每个注意力头的大小
        attention_head_size = self.d_model_size // self.num_heads
        # 如果没有需要剔除的注意力头，则直接返回
        if len(heads) == 0:
            return
        # 找到需要剔除的注意力头和其索引
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, attention_head_size, self.pruned_heads)

        # 剔除线性层对应的权重
        self.Wq = prune_linear_layer(self.Wq, index)
        self.Wk = prune_linear_layer(self.Wk, index)
        self.Wv = prune_linear_layer(self.Wv, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # 更新超参数
        self.num_heads = self.num_heads - len(heads)
        self.d_model_size = attention_head_size * self.num_heads
        # 将剔除的注意力头添加到已剔除的集合中
        self.pruned_heads = self.pruned_heads.union(heads)

    # 将输入向量分割成多个注意力头
    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    # 前向传播方法，实现注意力机制
    def forward(
        self,
        v,
        k,
        q,
        mask,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size = q.shape[0]

        # 计算查询、键、值的线性变换
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # 将查询、键、值分成多个注意力头
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        # 如果存在过去的键值对，则将当前键值对与过去的键值对合并
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # 如果需要缓存，则保存当前的键值对
        if use_cache is True:
            present = torch.stack((k, v))
        else:
            present = (None,)

        # 进行缩放点乘注意力计算
        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        # 将缩放后的注意力结果经过全连接层变换
        output = self.dense(original_size_attention)

        outputs = (output, present)
        # 如果需要返回注意力结果，则加入到输出中
        if output_attentions:
            outputs = outputs + (attn,)
        return outputs
def point_wise_feed_forward_network(d_model_size, dff):
    # 返回一个包含线性层、ReLU激活函数和再一个线性层的顺序结构用于点对点前馈网络
    return nn.Sequential(nn.Linear(d_model_size, dff), nn.ReLU(), nn.Linear(dff, d_model_size))


class EncoderLayer(nn.Module):
    def __init__(self, d_model_size, num_heads, dff, rate=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)

        self.layernorm1 = nn.LayerNorm(d_model_size, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model_size, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(
        self, x, mask, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False
    ):
        # 对输入数据进行 layer normalization
        normed = self.layernorm1(x)
        # 使用 MultiHeadAttention 处理并返回输出结果
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        # 对 MultiHeadAttention 的输出结果进行 dropout 处理
        attn_output = self.dropout1(attn_output)
        out1 = x + attn_output

        out2 = self.layernorm2(out1)
        # 使用 point_wise_feed_forward_network 处理并返回输出结果
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output

        outputs = (out2,) + attn_outputs[1:]
        return outputs


class CTRLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CTRLConfig
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # 对线性层和 Conv1D 进行参数初始化
            # 在 TF 版本中略有不同，使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对 Embedding 层进行参数初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层进行参数初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

CTRL_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    # 此代码是关于如何使用此模型的说明，建议参考 PyTorch 文档了解如何使用和正常行为。
    
    # 参数:
    #    config ([`CTRLConfig`]): 包含模型所有参数的模型配置类。
    #        使用配置文件进行初始化不会加载模型的权重，只会加载配置信息。
    #        可以使用 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
# 定义了一个命名为CTRLModel的类，继承自CTRLPreTrainedModel
@add_start_docstrings(
    "The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
    CTRL_START_DOCSTRING,
)
class CTRLModel(CTRLPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 从config中获取模型的参数
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        # 根据模型参数创建位置编码
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)

        # 创建词嵌入层
        self.w = nn.Embedding(config.vocab_size, config.n_embd)

        # 创建dropout层
        self.dropout = nn.Dropout(config.embd_pdrop)

        # 创建多层EncoderLayer，形成Encoder
        self.h = nn.ModuleList(
            [EncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop) for _ in range(config.n_layer)]
        )

        # 创建LayerNormalization层
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回词嵌入层
    def get_input_embeddings(self):
        return self.w

    # 设置词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    # 剪枝模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].multi_head_attention.prune_heads(heads)

    # 运行模型的前向传播
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    CTRL_START_DOCSTRING,
)
class CTRLLMHeadModel(CTRLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建CTRLModel的实例
        self.transformer = CTRLModel(config)
        # 创建线性层，作为语言模型的头部
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输出的词嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的词嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 为了生成准备输入数据，在给定参数的情况下，只返回最后一个 token 的 input_ids
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        # 如果 past_key_values 不为 None
        if past_key_values is not None:
            # 获取 past_key_values 第一个元素的第三个维度的大小作为 past_length
            past_length = past_key_values[0][0].shape[2]

            # 如果 input_ids 的第二个维度大于 past_length
            if input_ids.shape[1] > past_length:
                # 设置 remove_prefix_length 为 past_length
                remove_prefix_length = past_length
            else:
                # 否则设置 remove_prefix_length 为 input_ids 的第二个维度减一
                remove_prefix_length = input_ids.shape[1] - 1

            # 重新定义 input_ids 为从 remove_prefix_length 开始到结尾的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含修改后的 input_ids, past_key_values 和 use_cache 的字典
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache}

    # 将指定文档字符串添加到模型前向传播的注解
    # 将返回值的文档字符串替换为 CausalLMOutputWithPast 类型的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    
    # 重新排序缓存 past_key_values，以匹配每个生成步骤的 beam_idx
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 对 past_key_values 进行重新排序，以匹配每个生成步骤的 beam_idx
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 导入函数装饰器，用于添加文档字符串
@add_start_docstrings(
    """
    The CTRL Model transformer with a sequence classification head on top (linear layer).
    [`CTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last
    token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in
    each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last
    value in each row of the batch).
    """,
    CTRL_START_DOCSTRING,
)
# 定义一个类，继承自 CTRLPreTrainedModel，用于序列分类任务
class CTRLForSequenceClassification(CTRLPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels
        # 创建 CTRLModel 实例，用于序列转换
        self.transformer = CTRLModel(config)
        # 创建线性层，用于分类
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 定义参数列表
```