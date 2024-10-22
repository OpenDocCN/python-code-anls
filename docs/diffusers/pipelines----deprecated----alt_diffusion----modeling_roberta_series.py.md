# `.\diffusers\pipelines\deprecated\alt_diffusion\modeling_roberta_series.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型和元组类型
from typing import Optional, Tuple

# 导入 PyTorch 库
import torch
# 从 torch 模块导入神经网络相关的功能
from torch import nn
# 从 transformers 模块导入 Roberta 预训练模型、配置和模型类
from transformers import RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
# 从 transformers.utils 导入模型输出基类
from transformers.utils import ModelOutput

# 定义一个数据类，用于存储转换模型的输出
@dataclass
class TransformationModelOutput(ModelOutput):
    """
    文本模型输出的基类，包含最后隐藏状态的池化。

    参数：
        text_embeds (`torch.Tensor`，形状为 `(batch_size, output_dim)` *可选*，当模型以 `with_projection=True` 初始化时返回)：
            通过对 pooler_output 应用投影层获得的文本嵌入。
        last_hidden_state (`torch.Tensor`，形状为 `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.Tensor)`，*可选*，当 `output_hidden_states=True` 被传递或当 `config.output_hidden_states=True` 时返回)：
            隐藏状态的元组（每层输出一个，若模型有嵌入层则还包括嵌入输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每层输出的隐藏状态，加上可选的初始嵌入输出。
        attentions (`tuple(torch.Tensor)`，*可选*，当 `output_attentions=True` 被传递或当 `config.output_attentions=True` 时返回)：
            每层的注意力权重元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在自注意力头中计算加权平均的注意力权重。
    """

    # 投影状态，默认为 None，类型为可选的 torch.Tensor
    projection_state: Optional[torch.Tensor] = None
    # 最后隐藏状态，类型为 torch.Tensor，默认为 None
    last_hidden_state: torch.Tensor = None
    # 隐藏状态，类型为可选的元组，包含多个 torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    # 注意力权重，类型为可选的元组，包含多个 torch.Tensor
    attentions: Optional[Tuple[torch.Tensor]] = None

# 定义一个配置类，继承自 XLMRobertaConfig
class RobertaSeriesConfig(XLMRobertaConfig):
    # 初始化方法，设置不同的配置参数
    def __init__(
        self,
        pad_token_id=1,  # 填充 token 的 ID，默认为 1
        bos_token_id=0,  # 句子开头 token 的 ID，默认为 0
        eos_token_id=2,  # 句子结尾 token 的 ID，默认为 2
        project_dim=512,  # 投影维度，默认为 512
        pooler_fn="cls",  # 池化函数，默认为 "cls"
        learn_encoder=False,  # 是否学习编码器，默认为 False
        use_attention_mask=True,  # 是否使用注意力掩码，默认为 True
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，设置填充、开头和结尾 token 的 ID
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 设置投影维度
        self.project_dim = project_dim
        # 设置池化函数
        self.pooler_fn = pooler_fn
        # 设置是否学习编码器
        self.learn_encoder = learn_encoder
        # 设置是否使用注意力掩码
        self.use_attention_mask = use_attention_mask

# 定义一个模型类，继承自 RobertaPreTrainedModel
class RobertaSeriesModelWithTransformation(RobertaPreTrainedModel):
    # 加载时忽略的意外键
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"logit_scale"]
    # 加载时忽略的缺失键
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    # 基础模型前缀
    base_model_prefix = "roberta"
    # 配置类
    config_class = RobertaSeriesConfig
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 初始化 XLM-RoBERTa 模型，使用配置参数
        self.roberta = XLMRobertaModel(config)
        # 定义线性变换层，输入维度为隐藏层大小，输出维度为项目维度
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        # 获取配置中的 has_pre_transformation 属性，默认值为 False
        self.has_pre_transformation = getattr(config, "has_pre_transformation", False)
        # 如果启用预变换层
        if self.has_pre_transformation:
            # 定义另一个线性变换层用于预变换
            self.transformation_pre = nn.Linear(config.hidden_size, config.project_dim)
            # 定义层归一化，应用于隐藏层输出
            self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 调用后续初始化方法
        self.post_init()

    # 前向传播方法，接收多种输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入 ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.Tensor] = None,  # 令牌类型 ID
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID
        head_mask: Optional[torch.Tensor] = None,  # 头掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        return_dict: Optional[bool] = None,  # 是否返回字典格式
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
    ):
        r""" """

        # 如果 return_dict 为 None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用基础模型进行前向传播，传入各种输入参数
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            # 根据是否有预变换层决定是否输出隐藏状态
            output_hidden_states=True if self.has_pre_transformation else output_hidden_states,
            return_dict=return_dict,
        )

        # 如果启用了预变换层
        if self.has_pre_transformation:
            # 获取倒数第二个隐藏状态作为序列输出
            sequence_output2 = outputs["hidden_states"][-2]
            # 对序列输出进行层归一化
            sequence_output2 = self.pre_LN(sequence_output2)
            # 应用预变换层得到投影状态
            projection_state2 = self.transformation_pre(sequence_output2)

            # 返回变换模型输出，包含投影状态和其他输出
            return TransformationModelOutput(
                projection_state=projection_state2,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            # 如果没有预变换层，直接对最后的隐藏状态应用变换层
            projection_state = self.transformation(outputs.last_hidden_state)
            # 返回变换模型输出，包含投影状态和其他输出
            return TransformationModelOutput(
                projection_state=projection_state,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
```