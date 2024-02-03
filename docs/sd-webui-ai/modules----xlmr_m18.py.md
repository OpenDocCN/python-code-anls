# `stable-diffusion-webui\modules\xlmr_m18.py`

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
    # 设置配置类为 BertSeriesConfig
    config_class = BertSeriesConfig
    # 初始化函数，用于初始化模型参数
    def __init__(self, config=None, **kargs):
        # 修改初始化以进行自动加载
        if config is None:
            # 如果配置为空，则创建一个默认的XLMRobertaConfig对象
            config = XLMRobertaConfig()
            # 设置一系列默认参数
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
            config.project_dim = 1024
            config.learn_encoder = False
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建XLMRobertaModel对象
        self.roberta = XLMRobertaModel(config)
        # 创建线性变换层
        self.transformation = nn.Linear(config.hidden_size,config.project_dim)
        # 创建XLMRobertaTokenizer对象
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

        # 判断是否有预处理转换
        self.has_pre_transformation = True
        if self.has_pre_transformation:
            # 如果有预处理转换，则创建线性变换层和LayerNorm层
            self.transformation_pre = nn.Linear(config.hidden_size, config.project_dim)
            self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 调用后续初始化函数
        self.post_init()
    # 对输入的文本进行编码
    def encode(self,c):
        # 获取模型参数所在的设备
        device = next(self.parameters()).device
        # 使用分词器对文本进行处理，设置参数并返回处理后的文本
        text = self.tokenizer(c,
                        truncation=True,
                        max_length=77,
                        return_length=False,
                        return_overflowing_tokens=False,
                        padding="max_length",
                        return_tensors="pt")
        # 将处理后的文本转换为张量，并发送到指定设备
        text["input_ids"] = torch.tensor(text["input_ids"]).to(device)
        text["attention_mask"] = torch.tensor(
            text['attention_mask']).to(device)
        # 使用模型对处理后的文本进行前向传播
        features = self(**text)
        # 返回模型输出中的投影状态
        return features['projection_state']

    # 模型的前向传播函数
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
# 定义一个新的类，继承自BertSeriesModelWithTransformation类，用于处理Roberta系列模型并进行转换
class RobertaSeriesModelWithTransformation(BertSeriesModelWithTransformation):
    # 设置基础模型前缀为'roberta'
    base_model_prefix = 'roberta'
    # 设置配置类为RobertaSeriesConfig
    config_class= RobertaSeriesConfig
```