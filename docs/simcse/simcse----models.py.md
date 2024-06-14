# `.\simcse\models.py`

```
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数操作模块
import torch.distributed as dist  # 导入PyTorch的分布式模块

import transformers  # 导入transformers库
from transformers import RobertaTokenizer  # 导入RoBERTa的分词器
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead  # 导入RoBERTa相关模型和头部
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead  # 导入BERT相关模型和头部
from transformers.activations import gelu  # 导入gelu激活函数
from transformers.file_utils import (  # 导入transformers库的文件实用工具
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions  # 导入transformers模型输出

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 初始化全连接层
        self.activation = nn.Tanh()  # 初始化tanh激活函数

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 全连接层前向传播
        x = self.activation(x)  # 应用激活函数

        return x  # 返回结果

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp  # 初始化温度参数
        self.cos = nn.CosineSimilarity(dim=-1)  # 初始化余弦相似度计算

    def forward(self, x, y):
        return self.cos(x, y) / self.temp  # 计算输入张量之间的余弦相似度并除以温度参数

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type  # 初始化池化器类型
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type  # 确保池化器类型合法
    # 定义前向传播函数，接收注意力掩码和模型输出作为输入
    def forward(self, attention_mask, outputs):
        # 获取最后一层隐藏状态
        last_hidden = outputs.last_hidden_state
        # 获取池化器输出
        pooler_output = outputs.pooler_output
        # 获取所有隐藏状态
        hidden_states = outputs.hidden_states

        # 如果池化器类型是 'cls_before_pooler' 或 'cls'，则返回最后一层隐藏状态的第一个位置的输出
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        # 如果池化器类型是 'avg'，则计算加权平均值
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        # 如果池化器类型是 'avg_first_last'，则计算首尾两个隐藏状态的平均值
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        # 如果池化器类型是 'avg_top2'，则计算倒数第二和最后一个隐藏状态的平均值
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        # 如果池化器类型不在已知的类型中，则抛出未实现的错误
        else:
            raise NotImplementedError
def cl_init(cls, config):
    """
    Contrastive learning class init function.
    初始化对比学习类的初始化函数。
    """
    # 设置类属性为模型参数中的池化器类型
    cls.pooler_type = cls.model_args.pooler_type
    # 根据指定的池化器类型创建池化器对象
    cls.pooler = Pooler(cls.model_args.pooler_type)
    # 如果池化器类型是 "cls"，则创建一个MLP层对象
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    # 创建相似度计算对象，使用模型参数中的温度参数
    cls.sim = Similarity(temp=cls.model_args.temp)
    # 初始化权重
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    """
    Contrastive learning forward function.
    对比学习的前向传播函数。
    """
    # 如果未指定返回字典，则根据配置使用默认值
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    # 保存原始输入的token ids
    ori_input_ids = input_ids
    # 获取批处理大小
    batch_size = input_ids.size(0)
    # 获取一个实例中的句子数量
    # 2：成对实例；3：带有硬负例的成对实例
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # 将输入展平以便编码
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    # 获取原始嵌入表示
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # 如果存在MLM辅助目标
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # 池化操作
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # 如果使用 "cls" 池化器类型，添加额外的MLP层
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # 分离表示
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # 如果句子数量为3，添加硬负例表示
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # 如果使用分布式训练，收集所有嵌入表示
    # 检查分布是否已初始化并且当前处于训练模式
    if dist.is_initialized() and cls.training:
        # 收集难负样本
        if num_sent >= 3:
            # 为每个进程创建一个用于存储数据的零张量列表
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            # 在所有进程中收集 z3 数据到 z3_list 中
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            # 将当前进程的 z3 数据放回到对应的 z3_list 中
            z3_list[dist.get_rank()] = z3
            # 合并 z3_list 中所有数据，生成完整的 z3 数据
            z3 = torch.cat(z3_list, 0)

        # 为 allgather 创建用于存储数据的虚拟向量
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # 执行 allgather 操作，将数据收集到 z1_list 和 z2_list 中
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # 由于 allgather 的结果没有梯度，因此用原始张量替换当前进程对应的嵌入向量
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # 获取完整批次的嵌入向量：(bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # 计算余弦相似度
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # 对于难负样本
    if num_sent >= 3:
        # 计算 z1 和 z3 之间的余弦相似度
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        # 将 z1_z3_cos 拼接到 cos_sim 中
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    # 创建标签张量
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    # 使用交叉熵损失函数
    loss_fct = nn.CrossEntropyLoss()

    # 计算带有难负样本的损失
    if num_sent == 3:
        # 注意权重实际上是权重的对数(logits)
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    # 计算损失
    loss = loss_fct(cos_sim, labels)

    # 计算 MLM 的损失
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    # 如果不需要返回字典，则输出结果
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    # 否则，返回序列分类器输出
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
# 定义一个类方法 `sentemb_forward`，用于生成句子级别的嵌入表示。
def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    # 如果没有指定返回字典，使用类配置中的默认设置
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # 调用编码器（通常是BERT等模型）处理输入数据，获取编码器的输出
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        # 如果 `cls.pooler_type` 是 ['avg_top2', 'avg_first_last'] 之一，输出额外的隐藏状态
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # 使用 `cls.pooler` 方法生成池化后的输出特征
    pooler_output = cls.pooler(attention_mask, outputs)

    # 如果池化类型是 "cls" 且模型参数 `mlp_only_train` 为 False，则使用 MLP 处理池化输出
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    # 如果不需要以字典形式返回结果，则返回元组形式的输出
    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    # 否则，返回包含池化输出和其他关注信息的定制输出对象
    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


# 定义了一个继承自 `BertPreTrainedModel` 的子类 `BertForCL`
class BertForCL(BertPreTrainedModel):
    # 在加载模型时需要忽略的键列表
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化方法，接收配置和其他模型参数
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        # 从模型参数中获取 `model_args` 并保存
        self.model_args = model_kargs["model_args"]
        # 使用给定的BERT配置初始化BERT模型，不添加池化层
        self.bert = BertModel(config, add_pooling_layer=False)

        # 如果模型参数要求进行MLM（Masked Language Model）预测，初始化MLM头部
        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        # 调用 `cl_init` 函数进行其他自定义初始化
        cl_init(self, config)

    # 前向传播方法，处理输入数据并返回相应的输出
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
):
    ):
        # 如果 sent_emb 为真，则调用 sentemb_forward 函数进行前向传播
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,                      # 输入的 token IDs
                attention_mask=attention_mask,            # 注意力掩码，指示哪些 token 是真实的，哪些是填充的
                token_type_ids=token_type_ids,            # token 类型 IDs，用于区分句子 A 和句子 B
                position_ids=position_ids,                # 位置 IDs，指示每个 token 在序列中的位置
                head_mask=head_mask,                      # 头部掩码，用于屏蔽某些注意力头部
                inputs_embeds=inputs_embeds,              # 输入的嵌入表示，用于 BERT 模型
                labels=labels,                            # 模型的标签，用于监督学习
                output_attentions=output_attentions,      # 是否输出注意力权重
                output_hidden_states=output_hidden_states,# 是否输出隐藏状态
                return_dict=return_dict,                  # 是否返回字典形式的输出
            )
        # 否则调用 cl_forward 函数进行前向传播
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,                      # 输入的 token IDs
                attention_mask=attention_mask,            # 注意力掩码
                token_type_ids=token_type_ids,            # token 类型 IDs
                position_ids=position_ids,                # 位置 IDs
                head_mask=head_mask,                      # 头部掩码
                inputs_embeds=inputs_embeds,              # 输入的嵌入表示
                labels=labels,                            # 模型的标签
                output_attentions=output_attentions,      # 是否输出注意力权重
                output_hidden_states=output_hidden_states,# 是否输出隐藏状态
                return_dict=return_dict,                  # 是否返回字典形式的输出
                mlm_input_ids=mlm_input_ids,              # MLM 的输入 token IDs
                mlm_labels=mlm_labels,                    # MLM 的标签
            )
# 定义一个自定义的 RoBERTa 模型类，继承自 RoBERTaPreTrainedModel
class RobertaForCL(RobertaPreTrainedModel):
    # 在加载模型时忽略的键名列表，用于处理缺失情况
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化函数，接受配置对象和额外的模型参数
    def __init__(self, config, *model_args, **model_kargs):
        # 调用父类的初始化函数
        super().__init__(config)
        # 从模型参数中获取 model_args 并存储在实例变量中
        self.model_args = model_kargs["model_args"]
        # 使用 RoBERTaModel 类创建 RoBERTa 模型，禁用 pooling 层
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # 如果模型需要执行 MLM（Masked Language Modeling），则创建 LM 头部
        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        # 调用 cl_init 函数初始化额外的分类层和相关设置
        cl_init(self, config)

    # 前向传播函数，接受多个输入参数
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        # 如果 sent_emb 为 True，则调用 sentemb_forward 函数进行句子嵌入计算
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # 否则调用 cl_forward 函数执行分类任务的前向传播计算
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
```