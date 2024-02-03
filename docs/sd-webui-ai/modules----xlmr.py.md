# `stable-diffusion-webui\modules\xlmr.py`

```
# 从 transformers 库中导入 BertPreTrainedModel 和 BertConfig 类
from transformers import BertPreTrainedModel, BertConfig
# 从 torch.nn 模块中导入 nn 类
import torch.nn as nn
# 从 torch 模块中导入 torch
import torch
# 从 transformers.models.xlm_roberta 模块中导入 XLMRobertaConfig 类
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
# 从 transformers 库中导入 XLMRobertaModel 和 XLMRobertaTokenizer 类
from transformers import XLMRobertaModel, XLMRobertaTokenizer
# 从 typing 模块中导入 Optional 类
from typing import Optional

# 定义一个名为 BertSeriesConfig 的类，继承自 BertConfig 类
class BertSeriesConfig(BertConfig):
    # 初始化方法，设置各种参数的默认值
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None, project_dim=512, pooler_fn="average", learn_encoder=False, model_type='bert', **kwargs):
        # 调用父类的初始化方法，设置参数的默认值
        super().__init__(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs)
        # 设置额外的参数
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder

# 定义一个名为 RobertaSeriesConfig 的类，继承自 XLMRobertaConfig 类
class RobertaSeriesConfig(XLMRobertaConfig):
    # 初始化方法，设置各种参数的默认值
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, project_dim=512, pooler_fn='cls', learn_encoder=False, **kwargs):
        # 调用父类的初始化方法，设置参数的默认值
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 设置额外的参数
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder

# 定义一个名为 BertSeriesModelWithTransformation 的类，继承自 BertPreTrainedModel 类
class BertSeriesModelWithTransformation(BertPreTrainedModel):
    # 定义一个列表，用于存储在加载模型时要忽略的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 定义一个列表，用于存储在加载模型时缺失的键
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    # 设置 config_class 属性为 BertSeriesConfig 类
    config_class = BertSeriesConfig
    # 初始化函数，用于设置模型的配置参数和加载预训练模型
    def __init__(self, config=None, **kargs):
        # 如果没有传入配置参数，则使用默认配置
        if config is None:
            # 设置 XLMRobertaConfig 的默认参数
            config = XLMRobertaConfig()
            config.attention_probs_dropout_prob= 0.1
            config.bos_token_id=0
            config.eos_token_id=2
            config.hidden_act='gelu'
            config.hidden_dropout_prob=0.1
            config.hidden_size=1024
            config.initializer_range=0.02
            config.intermediate_size=4096
            config.layer_norm_eps=1e-05
            config.max_position_embeddings=514
            config.num_attention_heads=16
            config.num_hidden_layers=24
            config.output_past=True
            config.pad_token_id=1
            config.position_embedding_type= "absolute"
            config.type_vocab_size= 1
            config.use_cache=True
            config.vocab_size= 250002
            config.project_dim = 768
            config.learn_encoder = False
        # 调用父类的初始化函数，传入配置参数
        super().__init__(config)
        # 初始化 XLMRobertaModel 模型
        self.roberta = XLMRobertaModel(config)
        # 初始化线性变换层
        self.transformation = nn.Linear(config.hidden_size,config.project_dim)
        # 初始化 LayerNorm 层
        self.pre_LN=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化分词器
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        # 初始化池化函数
        self.pooler = lambda x: x[:,0]
        # 调用后续初始化函数
        self.post_init()

    # 编码函数，用于对输入文本进行编码
    def encode(self,c):
        # 获取模型参数所在的设备
        device = next(self.parameters()).device
        # 使用分词器对文本进行编码
        text = self.tokenizer(c,
                        truncation=True,
                        max_length=77,
                        return_length=False,
                        return_overflowing_tokens=False,
                        padding="max_length",
                        return_tensors="pt")
        # 将编码后的输入转换为张量，并发送到指定设备
        text["input_ids"] = torch.tensor(text["input_ids"]).to(device)
        text["attention_mask"] = torch.tensor(
            text['attention_mask']).to(device)
        # 使用模型对编码后的输入进行处理，获取特征
        features = self(**text)
        return features['projection_state']
    # 定义一个前向传播函数，接受多个输入参数，包括输入的张量、注意力掩码、token类型ID、位置ID、头掩码、输入嵌入、编码器隐藏状态、编码器注意力掩码、是否输出注意力、是否返回字典、是否输出隐藏状态
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        """

        # 如果return_dict不为None，则将其赋值给return_dict，否则使用self.config.use_return_dict的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用roberta模型的forward方法，传入各种参数
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 对每个模块进行投影
        sequence_output_ln = self.pre_LN(sequence_output)

        # 对pooler进行处理
        pooler_output = self.pooler(sequence_output_ln)
        pooler_output = self.transformation(pooler_output)
        projection_state = self.transformation(outputs.last_hidden_state)

        # 返回一个字典，包含pooler输出、最后隐藏状态、隐藏状态、注意力、投影状态、序列输出
        return {
            'pooler_output':pooler_output,
            'last_hidden_state':outputs.last_hidden_state,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,
            'projection_state':projection_state,
            'sequence_out': sequence_output
        }
# 定义一个新的类，继承自BertSeriesModelWithTransformation类，用于处理Roberta系列模型并进行转换
class RobertaSeriesModelWithTransformation(BertSeriesModelWithTransformation):
    # 设置基础模型前缀为'roberta'
    base_model_prefix = 'roberta'
    # 设置配置类为RobertaSeriesConfig
    config_class= RobertaSeriesConfig
```