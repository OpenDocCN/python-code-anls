# `.\marker\marker\postprocessors\t5.py`

```
# 从 transformers 库中导入 T5Config 和 T5PreTrainedModel 类
from transformers import T5Config, T5PreTrainedModel
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 copy 库中导入 deepcopy 函数
from copy import deepcopy
# 从 typing 库中导入 Optional, Tuple, Union, List 类型
from typing import Optional, Tuple, Union, List
# 从 itertools 库中导入 chain 函数
from itertools import chain

# 从 transformers.modeling_outputs 模块中导入 TokenClassifierOutput 类
from transformers.modeling_outputs import TokenClassifierOutput
# 从 transformers.models.t5.modeling_t5 模块中导入 T5Stack 类
from transformers.models.t5.modeling_t5 import T5Stack
# 从 transformers.utils.model_parallel_utils 模块中导入 get_device_map, assert_device_map 函数
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

# 定义一个函数，用于将文本进行字节编码并分词
def byt5_tokenize(text: str, max_length: int, pad_token_id: int = 0):
    # 初始化一个空列表，用于存储字节编码
    byte_codes = []
    # 遍历文本中的每个字符
    for char in text:
        # 将每个字符进行 UTF-8 编码，并加上 3 以考虑特殊标记
        byte_codes.append([byte + 3 for byte in char.encode('utf-8')])

    # 将字节编码展开成一个列表
    tokens = list(chain.from_iterable(byte_codes))
    # 记录每个字符对应的 token 长度
    char_token_lengths = [len(b) for b in byte_codes]

    # 初始化批量 token 和注意力掩码列表
    batched_tokens = []
    attention_mask = []
    # 按照最大长度将 token 进行分批
    for i in range(0, len(tokens), max_length):
        batched_tokens.append(tokens[i:i + max_length])
        attention_mask.append([1] * len(batched_tokens[-1])

    # 对最后一个批次进行填充
    if len(batched_tokens[-1]) < max_length:
        batched_tokens[-1] += [pad_token_id] * (max_length - len(batched_tokens[-1]))
        attention_mask[-1] += [0] * (max_length - len(attention_mask[-1]))

    # 返回包含分词结果的字典
    return {"input_ids": batched_tokens, "attention_mask": attention_mask, "char_token_lengths": char_token_lengths}

# 定义一个 T5ForTokenClassification 类，继承自 T5PreTrainedModel 类
class T5ForTokenClassification(T5PreTrainedModel):
    # 定义一个列表，用于指定加载时忽略的键
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]
    # 初始化函数，接受一个T5Config对象作为参数
    def __init__(self, config: T5Config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置模型维度为配置中的d_model值
        self.model_dim = config.d_model

        # 创建一个共享的嵌入层，词汇表大小为config.vocab_size，维度为config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置对象，用于创建编码器
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        # 创建T5Stack编码器
        self.encoder = T5Stack(encoder_config, self.shared)

        # 设置分类器的dropout值
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.dropout_rate
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层，输入维度为config.d_model，输出维度为config.num_labels
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化
        self.model_parallel = False
        self.device_map = None


    # 并行化函数，接受一个设备映射device_map作为参数
    def parallelize(self, device_map=None):
        # 如果未提供device_map，则根据编码器块的数量和GPU数量生成一个默认的device_map
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射的有效性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 将编码器并行化
        self.encoder.parallelize(self.device_map)
        # 将分类器移动到编码器的第一个设备上
        self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    # 反并行化函数
    def deparallelize(self):
        # 取消编码器的并行化
        self.encoder.deparallelize()
        # 将编码器和分类器移动到CPU上
        self.encoder = self.encoder.to("cpu")
        self.classifier = self.classifier.to("cpu")
        self.model_parallel = False
        self.device_map = None
        # 释放GPU缓存
        torch.cuda.empty_cache()

    # 获取输入嵌入层函数
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层函数，接受一个新的嵌入层new_embeddings作为参数
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 设置编码器的输入嵌入层为新的嵌入层
        self.encoder.set_input_embeddings(new_embeddings)

    # 获取编码器函数
    def get_encoder(self):
        return self.encoder
    # 对模型中的特定头部进行修剪
    def _prune_heads(self, heads_to_prune):
        # 遍历需要修剪的层和头部
        for layer, heads in heads_to_prune.items():
            # 调用 SelfAttention 模块的 prune_heads 方法进行修剪
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], TokenClassifierOutput]:
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用编码器进行前向传播
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout
        sequence_output = self.dropout(sequence_output)
        # 将序列输出传入分类器得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None

        # 如果不使用 return_dict，则返回输出结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 类返回结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
```