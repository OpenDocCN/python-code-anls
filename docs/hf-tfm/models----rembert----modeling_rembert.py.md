# `.\models\rembert\modeling_rembert.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可证信息
# 此代码使用 Apache License, Version 2.0 许可证，详细信息可查阅 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何形式的担保或条件
# 请查阅许可证了解更多信息
""" PyTorch RemBERT 模型。"""

# 导入必要的库和模块
import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_rembert import RemBertConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 文档中的配置和检查点
_CONFIG_FOR_DOC = "RemBertConfig"
_CHECKPOINT_FOR_DOC = "google/rembert"

# RemBERT 预训练模型的存档列表
REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/rembert",
    # 查看所有 RemBERT 模型：https://huggingface.co/models?filter=rembert
]


def load_tf_weights_in_rembert(model, config, tf_checkpoint_path):
    """从 TensorFlow checkpoints 中加载权重到 PyTorch 模型中。"""
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入 NumPy 模块
        import tensorflow as tf  # 导入 TensorFlow 模块
    except ImportError:
        logger.error(
            "在 PyTorch 中加载 TensorFlow 模型需要安装 TensorFlow。请访问 "
            "https://www.tensorflow.org/install/ 获取安装指南。"
        )
        raise  # 抛出 ImportError 异常
    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow checkpoints 的绝对路径
    logger.info(f"从 {tf_path} 转换 TensorFlow checkpoints")  # 记录日志信息
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)  # 获取 TensorFlow checkpoints 的变量列表
    names = []  # 初始化空列表存储变量名
    arrays = []  # 初始化空列表存储权重数组
    for name, shape in init_vars:
        # 检查点占用12Gb，通过不加载无用变量来节省内存
        # 输出嵌入和cls在分类时会被重置
        if any(deny in name for deny in ("adam_v", "adam_m", "output_embedding", "cls")):
            # 如果变量名包含"adam_v", "adam_m", "output_embedding", "cls"中的任意一个，跳过加载
            # logger.info("Skipping loading of %s", name)
            continue
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用TensorFlow的函数加载变量
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        # 将名称中的前缀"bert/"替换为"rembert/"
        name = name.replace("bert/", "rembert/")
        # "pooler"是一个线性层
        # 如果名称包含"pooler/dense"，则替换为"pooler"
        # name = name.replace("pooler/dense", "pooler")

        # 将名称按"/"分割
        name = name.split("/")
        # "adam_v"和"adam_m"是AdamWeightDecayOptimizer中用于计算m和v的变量，预训练模型不需要这些变量
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            # 如果变量名符合形如"A-Za-z+_\d+"的正则表达式，分割出作用域名称和数字
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据作用域名称选择指针位置
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            # 如果作用域名称长度大于等于2，选择指定数字位置的指针
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果变量名以"_embeddings"结尾，选择权重指针
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            # 如果变量名是"kernel"，转置数组
            array = np.transpose(array)
        try:
            # 检查指针和数组的形状是否匹配，如果不匹配抛出异常
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        # 将NumPy数组转换为PyTorch张量，初始化权重指针
        pointer.data = torch.from_numpy(array)
    return model
# 定义一个名为 RemBertEmbeddings 的神经网络模块，用于构建来自单词、位置和标记类型嵌入的嵌入向量。
class RemBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建单词嵌入层，vocab_size 表示词汇表大小，input_embedding_size 表示嵌入向量的维度，
        # padding_idx 指定了填充标记的索引，以便在计算时将其置零。
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.input_embedding_size, padding_idx=config.pad_token_id
        )
        # 创建位置嵌入层，max_position_embeddings 表示最大的位置编码数量，input_embedding_size 表示嵌入向量的维度。
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.input_embedding_size)
        # 创建标记类型嵌入层，type_vocab_size 表示标记类型的数量，input_embedding_size 表示嵌入向量的维度。
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.input_embedding_size)

        # 使用 LayerNorm 进行归一化处理，保持与 TensorFlow 模型变量名的一致性，
        # 并能够加载任意 TensorFlow 检查点文件。
        self.LayerNorm = nn.LayerNorm(config.input_embedding_size, eps=config.layer_norm_eps)
        # Dropout 层，用于在训练过程中随机将一部分输入单元置零，以防止过拟合。
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在序列化时在内存中是连续的，并在导出时被导出。
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果给定 input_ids，则获取其形状；否则，获取 inputs_embeds 的形状。
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度（即时间步数）。
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从预定义的 position_ids 中选择一段对应于序列长度的位置编码。
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供 token_type_ids，则创建一个与输入形状相同的全零张量。
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则使用 input_ids 从 word_embeddings 中获取嵌入向量。
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 获取 token_type_ids 对应的标记类型嵌入向量。
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将单词嵌入向量和标记类型嵌入向量相加。
        embeddings = inputs_embeds + token_type_embeddings
        # 获取 position_ids 对应的位置嵌入向量。
        position_embeddings = self.position_embeddings(position_ids)
        # 将位置嵌入向量加到之前的结果中。
        embeddings += position_embeddings
        # 应用 LayerNorm 进行归一化处理。
        embeddings = self.LayerNorm(embeddings)
        # 应用 Dropout 进行随机丢弃部分输入。
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入向量。
        return embeddings


# 从 transformers.models.bert.modeling_bert.BertPooler 复制并修改为 RemBertPooler 类。
class RemBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，输入和输出大小均为 hidden_size，用于池化模型隐藏状态。
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数 tanh，用于非线性变换。
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 简单地通过获取第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 经过全连接层处理。
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数。
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出。
        return pooled_output
# 定义一个名为 RemBertSelfAttention 的类，继承自 nn.Module
class RemBertSelfAttention(nn.Module):
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏层大小不是注意力头数的整数倍，且 config 对象没有 embedding_size 属性，则抛出 ValueError 异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 将注意力头数和每个注意力头的大小设置为类的属性
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义用于生成查询、键和值的线性层，并作为类的属性
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义用于在注意力计算过程中进行 dropout 的层，并作为类的属性
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 判断是否为解码器，设置为类的属性
        self.is_decoder = config.is_decoder

    # 定义一个方法用于调整输入张量的形状，以适应多头注意力的计算
    def transpose_for_scores(self, x):
        # 计算新的形状，将最后一维分解为注意力头数和每个头的大小，并对输入张量进行相应的变形
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # 将维度进行置换，以便后续计算
        return x.permute(0, 2, 1, 3)

    # 定义前向传播方法，接收隐藏状态、注意力掩码等作为输入，并进行注意力计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Tuple[Tuple[torch.FloatTensor]] = None,
        output_attentions: bool = False,
    ):
        # 方法体中的实现会在下面补充



# 定义一个名为 RemBertSelfOutput 的类，继承自 nn.Module
class RemBertSelfOutput(nn.Module):
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义用于变换隐藏状态维度的线性层、LayerNorm 层和 dropout 层，并作为类的属性
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 定义前向传播方法，接收隐藏状态和输入张量作为输入，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 层对处理后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 对处理后的隐藏状态进行 LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states



# 定义一个名为 RemBertAttention 的类，继承自 nn.Module
class RemBertAttention(nn.Module):
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义自注意力和自注意力输出层，并作为类的属性
        self.self = RemBertSelfAttention(config)
        self.output = RemBertSelfOutput(config)
        # 初始化一个空集合用于存储被修剪的注意力头
        self.pruned_heads = set()

    # 方法体中的实现会在下面补充
    # 剪枝注意力头部
    def prune_heads(self, heads):
        # 如果头部列表为空，则直接返回
        if len(heads) == 0:
            return
        
        # 调用函数查找可剪枝的注意力头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层：对自注意力模块中的query、key、value以及输出dense层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 从transformers.models.bert.modeling_bert.BertAttention.forward中复制而来
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用self模块的forward方法，传递参数并接收返回的self_outputs
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        
        # 使用self_outputs[0]和hidden_states调用output模块，得到attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        
        # 如果需要输出attentions，则在outputs中加入attention信息
        outputs = (attention_output,) + self_outputs[1:]  # 如果有attentions，将其加入outputs
        return outputs
# 基于修改后的RemBert的中间层实现，继承自nn.Module类
class RemBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入特征大小为config.hidden_size转换为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数，可能是预定义的激活函数映射表ACT2FN中的函数，或者是直接指定的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，输入hidden_states是一个torch.Tensor，输出也是一个torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states经过全连接层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果经过激活函数self.intermediate_act_fn进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 基于修改后的RemBert的输出层实现，继承自nn.Module类
class RemBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入特征大小为config.intermediate_size转换为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建LayerNorm层，归一化config.hidden_size维度的张量，eps是归一化的参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，以config.hidden_dropout_prob的概率随机将输入元素置零
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，输入hidden_states和input_tensor都是torch.Tensor，输出也是一个torch.Tensor
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states经过全连接层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果经过Dropout层self.dropout进行随机置零处理
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的结果与input_tensor相加，并经过LayerNorm层self.LayerNorm进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 基于修改后的RemBert的层实现，继承自nn.Module类
class RemBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设定feed forward过程中的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1
        self.seq_len_dim = 1
        # 创建RemBertAttention对象
        self.attention = RemBertAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力但不作为解码器使用，抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建另一个RemBertAttention对象用于交叉注意力
            self.crossattention = RemBertAttention(config)
        # 创建RemBertIntermediate对象
        self.intermediate = RemBertIntermediate(config)
        # 创建RemBertOutput对象
        self.output = RemBertOutput(config)

    # 基于transformers库中BertLayer的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 声明函数的输入和输出类型注解，这里返回一个 torch.Tensor 的元组

        # 如果过去的键/值对不为 None，则提取自注意力的缓存键/值对（单向），位置在索引 1 和 2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # 执行自注意力计算，传入隐藏状态、注意力掩码、头部掩码等参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 否则，将自注意力计算的输出加入到输出中
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
        

        cross_attn_present_key_value = None
        
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未定义交叉注意力层，则抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 提取过去键/值对的交叉注意力缓存，位置在倒数第二和最后的两个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # 执行交叉注意力计算，传入自注意力输出、注意力掩码、头部掩码、编码器隐藏状态等参数
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            
            # 获取交叉注意力计算的输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到当前键/值对的末尾位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用分块处理，传入分块处理函数、分块大小、序列长度维度和注意力输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
        # 将分块处理后的输出添加到结果元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值对作为最后的输出添加到结果中
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回所有的输出
        return outputs

    # 从 transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk 复制过来的函数
    def feed_forward_chunk(self, attention_output):
        # 执行前馈网络的一部分，传入注意力输出
        intermediate_output = self.intermediate(attention_output)
        
        # 应用输出层，并返回最终的层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
class RemBertEncoder(nn.Module):
    # RemBert 编码器模块，继承自 nn.Module
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 输入嵌入层的线性映射，将输入的嵌入大小映射到隐藏大小
        self.embedding_hidden_mapping_in = nn.Linear(config.input_embedding_size, config.hidden_size)
        
        # 创建一个由多个 RemBert 层组成的层列表，层数由配置中的 num_hidden_layers 决定
        self.layer = nn.ModuleList([RemBertLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 是否启用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 前向传播方法，接受多个输入参数并返回一个 tensor
        # 具体操作由每个 RemBert 层来处理
        pass  # 这里应该有实际的前向传播代码，这里只是为了演示注释结构


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->RemBert
class RemBertPredictionHeadTransform(nn.Module):
    # RemBert 预测头变换模块，继承自 nn.Module
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 根据配置中的激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        # LayerNorm 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，对输入的隐藏状态进行线性变换、激活函数变换和归一化处理
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RemBertLMPredictionHead(nn.Module):
    # RemBert 语言模型预测头模块，继承自 nn.Module
    def __init__(self, config):
        super().__init__()
        
        # 全连接层，将隐藏状态映射到输出嵌入大小
        self.dense = nn.Linear(config.hidden_size, config.output_embedding_size)
        
        # 解码器层，将输出嵌入映射到词汇表大小
        self.decoder = nn.Linear(config.output_embedding_size, config.vocab_size)
        
        # 根据配置中的激活函数类型选择对应的激活函数
        self.activation = ACT2FN[config.hidden_act]
        
        # LayerNorm 层，对输出嵌入进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.output_embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，对输入的隐藏状态进行线性变换、激活函数变换、归一化和解码处理
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->RemBert
class RemBertOnlyMLMHead(nn.Module):
    # 仅包含 RemBert 语言模型头模块，继承自 nn.Module
    def __init__(self, config):
        super().__init__()
        
        # RemBert 语言模型预测头模块
        self.predictions = RemBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，接受序列输出并返回预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RemBertPreTrainedModel(PreTrainedModel):
    """
    RemBert 预训练模型基类，继承自 PreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义配置类为 RemBertConfig
    config_class = RemBertConfig
    # 加载 TensorFlow 权重函数为 load_tf_weights_in_rembert
    load_tf_weights = load_tf_weights_in_rembert
    # 基础模型前缀为 "rembert"
    base_model_prefix = "rembert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，标准差为配置中的 initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，标准差为配置中的 initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
# REMBERT_START_DOCSTRING 是一个原始文档字符串，描述了一个 PyTorch 模型类 RemBert 的基本信息和用法建议
REMBERT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RemBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# REMBERT_INPUTS_DOCSTRING 是一个空白的文档字符串，用于描述模型的输入参数和示例，但当前为空
REMBERT_INPUTS_DOCSTRING = r"""
"""
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记在词汇表中的索引

                # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取详情。

                # [什么是输入 ID？](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 遮罩，避免在填充标记索引上进行注意力计算。遮罩的取值范围为 `[0, 1]`：

                # - 1 表示**未遮罩**的标记，
                # - 0 表示**已遮罩**的标记。

                # [什么是注意力遮罩？](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 段标记索引，用于指示输入的第一部分和第二部分。索引取值范围为 `[0, 1]`：

                # - 0 对应**句子 A** 的标记，
                # - 1 对应**句子 B** 的标记。

                # [什么是标记类型 ID？](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 每个输入序列标记在位置嵌入中的位置索引。索引取值范围为 `[0, config.max_position_embeddings - 1]`。

                # [什么是位置 ID？](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 遮罩，用于屏蔽自注意力模块中选定的注意力头部。遮罩的取值范围为 `[0, 1]`：

                # - 1 表示**未遮罩**的头部，
                # - 0 表示**已遮罩**的头部。

            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选项，可以直接传递嵌入表示而不是 `input_ids`。这对于控制如何将 `input_ids` 索引转换为相关向量比模型内部的嵌入查找矩阵更有用。

            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量。更多细节请参见返回的张量中的 `attentions`。

            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态。更多细节请参见返回的张量中的 `hidden_states`。

            return_dict (`bool`, *optional*):
                # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
    """
    @add_start_docstrings(
        "The bare RemBERT Model transformer outputting raw hidden-states without any specific head on top.",
        REMBERT_START_DOCSTRING,
    )
    """
    class RemBertModel(RemBertPreTrainedModel):
        """
        The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
        cross-attention is added between the self-attention layers, following the architecture described in [Attention is
        all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
        Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
        to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
        `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
        """

        def __init__(self, config, add_pooling_layer=True):
            super().__init__(config)
            self.config = config

            # Initialize embeddings based on configuration
            self.embeddings = RemBertEmbeddings(config)
            # Initialize encoder based on configuration
            self.encoder = RemBertEncoder(config)

            # Optionally initialize a pooling layer based on configuration
            self.pooler = RemBertPooler(config) if add_pooling_layer else None

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            # Return the word embeddings from the embeddings layer
            return self.embeddings.word_embeddings

        def set_input_embeddings(self, value):
            # Set new word embeddings for the embeddings layer
            self.embeddings.word_embeddings = value

        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                # Prune specified heads in the attention layers of the encoder
                self.encoder.layer[layer].attention.prune_heads(heads)

        @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint="google/rembert",
            output_type=BaseModelOutputWithPastAndCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ):
            """
            Forward pass for the RemBERT model.

            Args:
                input_ids: Indices of input sequence tokens in the vocabulary.
                attention_mask: Mask to avoid performing attention on padding token indices.
                token_type_ids: Segment token indices to indicate first and second portions of the inputs.
                position_ids: Indices of positions of each input sequence tokens in the position embeddings.
                head_mask: Mask to nullify selected heads of the attention modules.
                inputs_embeds: Overrides the model's base input word embeddings if provided.
                encoder_hidden_states: Hidden states of the encoder to feed into the cross-attention layer.
                encoder_attention_mask: Mask to avoid performing attention on encoder hidden states.
                past_key_values: Cached key-value pairs for fast autoregressive decoding.
                use_cache: Whether or not to use the past key-value caches.
                output_attentions: Whether or not to return attentions weights.
                output_hidden_states: Whether or not to return hidden states.
                return_dict: Whether or not to return a dictionary as output.

            Returns:
                BaseModelOutputWithPastAndCrossAttentions: Model output.

            Notes:
                Args above are based on REMBERT_INPUTS_DOCSTRING for batch size and sequence length.
            """
            # Actual implementation of the forward pass will follow here, specific to RemBERT's architecture and functionality
            pass
# 用装饰器添加文档字符串，描述这是一个在 `language modeling` 模型基础上加上头部的 RemBERT 模型
@add_start_docstrings("""RemBERT Model with a `language modeling` head on top.""", REMBERT_START_DOCSTRING)
# 定义一个 RemBertForMaskedLM 类，继承自 RemBertPreTrainedModel 类
class RemBertForMaskedLM(RemBertPreTrainedModel):
    # 定义一个类变量，指定共享权重的键值
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果 config 设置为 decoder 模式，则发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `RemBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建一个 RemBertModel 对象，不添加池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # 创建一个 RemBertOnlyMLMHead 对象
        self.cls = RemBertOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输出 embeddings 的方法，返回预测解码器的权重
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出 embeddings 的方法，更新预测解码器的权重
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 重写 forward 方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 如果 `return_dict` 参数为 None，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用输入参数调用 `rembert` 方法，获取输出结果
        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 `outputs` 中获取序列输出
        sequence_output = outputs[0]
        
        # 使用分类头部 `cls` 对序列输出进行预测
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 定义交叉熵损失函数，用于计算masked language modeling loss
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 计算预测分数和标签之间的损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果 `return_dict` 为 False，返回结果元组
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 `return_dict` 为 True，返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # 获取输入 `input_ids` 的形状
        input_shape = input_ids.shape
        # 获取有效的批量大小
        effective_batch_size = input_shape[0]

        # 确保 PAD token 已定义用于生成
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        # 将注意力掩码与新生成的零张量连接，以扩展序列长度
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个填充了 PAD token 的虚拟令牌
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 在输入 `input_ids` 的末尾连接虚拟令牌
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回包含输入 `input_ids` 和注意力掩码的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用装饰器添加文档字符串，描述了这个类的作用是在 CLM fine-tuning 上使用 RemBERT 模型，并带有语言建模头部
@add_start_docstrings(
    """RemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", REMBERT_START_DOCSTRING
)
# 定义 RemBertForCausalLM 类，继承自 RemBertPreTrainedModel 类
class RemBertForCausalLM(RemBertPreTrainedModel):
    # 类属性，指定权重共享的键名
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中不是解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `RemBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 RemBERT 模型，不添加池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # 初始化仅包含 MLM 头部的类
        self.cls = RemBertOnlyMLMHead(config)

        # 调用后处理初始化方法
        self.post_init()

    # 返回输出嵌入层的方法
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层的方法，接受新的嵌入层作为参数
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接受多个输入参数，具体参数的作用通过装饰器和替换返回值的方式进行文档化
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 准备生成输入的方法，接受输入 ID，过去键值，注意力掩码等参数
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            # 获取输入张量的形状
            input_shape = input_ids.shape

            # 如果注意力掩码为空，创建一个与输入形状相同的全为 1 的张量
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果传入了过去键值，裁剪输入 ID
            if past_key_values is not None:
                # 获取过去键值的长度
                past_length = past_key_values[0][0].shape[2]

                # 如果输入 ID 的长度大于过去键值的长度，移除前缀长度为过去键值的长度
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 否则，默认只保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                # 更新输入 ID
                input_ids = input_ids[:, remove_prefix_length:]

            # 返回包含输入 ID、注意力掩码和过去键值的字典
            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排序缓存中的过去键值，以匹配新的束搜索索引顺序
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空元组来存储重新排序后的过去状态
        reordered_past = ()
        # 遍历每一层的过去状态
        for layer_past in past_key_values:
            # 对每个层的过去状态的前两项进行重新排序，根据给定的束搜索索引
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                # 将每层的第三项及其后的项保持不变地添加到重新排序后的元组中
                + layer_past[2:],
            )
        # 返回重新排序后的过去状态元组
        return reordered_past
# 使用装饰器添加文档字符串，描述了这个类是基于RemBERT模型的序列分类/回归模型，适用于GLUE任务等应用。
@add_start_docstrings(
    """
    RemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    REMBERT_START_DOCSTRING,
)
class RemBertForSequenceClassification(RemBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置分类器的类别数
        self.num_labels = config.num_labels
        # 初始化RemBERT模型
        self.rembert = RemBertModel(config)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 分类器，线性层，将RemBERT模型的输出映射到类别数量上
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据是否提供 return_dict 参数来确定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RemBERT 模型进行前向传播
        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取经过池化的输出
        pooled_output = outputs[1]

        # 对经过池化的输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 将 dropout 后的结果输入分类器进行分类
        logits = self.classifier(pooled_output)

        # 初始化损失值
        loss = None
        # 如果提供了标签，则计算损失值
        if labels is not None:
            # 根据配置和标签的数据类型确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择合适的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典类型的输出，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 SequenceClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用多项选择任务头部的 RemBERT 模型（在汇总输出上方添加了一个线性层和 softmax），例如适用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    RemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    REMBERT_START_DOCSTRING,
)
class RemBertForMultipleChoice(RemBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数，初始化 RemBERT 模型
        super().__init__(config)

        # 初始化 RemBERT 模型
        self.rembert = RemBertModel(config)
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 添加一个线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为前向传播方法添加文档字符串注释
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 添加代码示例的文档字符串注释
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 `return_dict` 参数确定是否返回字典类型的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选项数量，即每个样本的选择数目
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入数据展平成二维张量，以便适应模型输入要求
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用模型的前向传播方法，获取输出结果
        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取汇聚后的输出表示
        pooled_output = outputs[1]

        # 对汇聚后的输出进行dropout处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对处理后的输出进行分类预测
        logits = self.classifier(pooled_output)
        # 将预测的 logits 重塑成(batch_size, num_choices)形状
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失值为None
        loss = None
        # 如果有提供标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典类型的结果，则将输出整理成元组形式返回
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典类型的结果，则创建MultipleChoiceModelOutput对象返回
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
RemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 继承自RemBertPreTrainedModel的RemBertForTokenClassification类，用于在RemBERT模型上添加一个用于标记分类的头部
class RemBertForTokenClassification(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化RemBERT模型，不添加池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 分类器线性层，将隐藏状态输出映射到标签数量的空间
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多个输入参数，并返回模型的输出或损失
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用RemBERT模型进行前向传播
        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 应用Dropout层
        sequence_output = self.dropout(sequence_output)
        # 使用分类器线性层计算logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # 如果不使用return_dict，则返回元组形式的输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用return_dict，则返回TokenClassifierOutput对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    RemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    REMBERT_START_DOCSTRING,



# 定义 RemBERT 模型，用于抽取式问答任务（如 SQuAD），其在隐藏状态输出之上带有一个用于计算“span start logits”和“span end logits”的线性分类头部。
# REMBERT_START_DOCSTRING 是一个预定义的文档字符串常量，可能包含 RemBERT 模型的详细描述或指导。
# 定义一个继承自 RemBertPreTrainedModel 的问题回答模型 RemBertForQuestionAnswering 类
class RemBertForQuestionAnswering(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化时设置类别数目
        self.num_labels = config.num_labels

        # 使用 RemBertModel 创建一个 RemBert 对象，不添加池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        
        # 使用 nn.Linear 初始化一个线性层，用于生成问题回答的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接受多种输入参数并返回预测结果
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
    
        # Determine if we should return a dictionary based on the provided argument or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Pass inputs through the RoBERTa model
        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # Extract the sequence output from RoBERTa model outputs
        sequence_output = outputs[0]
    
        # Compute logits for start and end positions from the sequence output
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If the start_positions and end_positions tensors have more than one dimension, squeeze them
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # Clamp the start_positions and end_positions to valid ranges within the sequence length
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
    
            # Define the loss function and compute start and end loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            # Compute total loss as the average of start and end loss
            total_loss = (start_loss + end_loss) / 2
    
        # If return_dict is False, return outputs as tuple
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        # If return_dict is True, return structured output using QuestionAnsweringModelOutput class
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```