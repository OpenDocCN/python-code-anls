# `.\transformers\models\xlm_roberta_xl\convert_xlm_roberta_xl_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8

# 导入需要的模块和库
import argparse  # 导入用于解析命令行参数的模块
import pathlib  # 提供处理文件路径的类和函数

import fairseq  # 导入 fairseq 库
import torch  # 导入 PyTorch 库
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel  # 导入 fairseq 中的 RoBERTa 模型
from fairseq.modules import TransformerSentenceEncoderLayer  # 导入 fairseq 中的 TransformerSentenceEncoderLayer 模块
from packaging import version  # 导入版本信息模块

from transformers import (  # 导入 transformers 库中的一些类和函数
    XLMRobertaConfig,  # 导入 XLMRobertaConfig 类
    XLMRobertaXLForMaskedLM,  # 导入 XLMRobertaXLForMaskedLM 类
    XLMRobertaXLForSequenceClassification,  # 导入 XLMRobertaXLForSequenceClassification 类
)
from transformers.models.bert.modeling_bert import (  # 导入 transformers 库中的一些 BERT 模型相关的类
    BertIntermediate,  # 导入 BertIntermediate 类
    BertLayer,  # 导入 BertLayer 类
    BertOutput,  # 导入 BertOutput 类
    BertSelfAttention,  # 导入 BertSelfAttention 类
    BertSelfOutput,  # 导入 BertSelfOutput 类
)
from transformers.models.roberta.modeling_roberta import RobertaAttention  # 导入 RoBERTaAttention 类
from transformers.utils import logging  # 导入 logging 模块

# 检查 fairseq 版本是否符合要求
if version.parse(fairseq.__version__) < version.parse("1.0.0a"):
    raise Exception("requires fairseq >= 1.0.0a")  # 如果版本不符合要求则抛出异常

# 设置日志级别为 INFO
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义一个示例文本
SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_xlm_roberta_xl_checkpoint_to_pytorch(
    roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    复制/粘贴/调整 RoBERTa 的权重到我们的 BERT 结构。
    """
    # 加载预训练的 RoBERTa 模型
    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    # 设置为评估模式，关闭 dropout
    roberta.eval()
    # 获取 RoBERTa 的句子编码器
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    # 创建一个 XLMRobertaConfig 配置对象
    config = XLMRobertaConfig(
        vocab_size=roberta_sent_encoder.embed_tokens.num_embeddings,  # 词汇表大小
        hidden_size=roberta.cfg.model.encoder_embed_dim,  # 隐藏层大小
        num_hidden_layers=roberta.cfg.model.encoder_layers,  # 隐藏层层数
        num_attention_heads=roberta.cfg.model.encoder_attention_heads,  # 注意力头数
        intermediate_size=roberta.cfg.model.encoder_ffn_embed_dim,  # 中间层大小
        max_position_embeddings=514,  # 最大位置编码
        type_vocab_size=1,  # 类型词汇表大小
        layer_norm_eps=1e-5,  # 层归一化 epsilon 值
    )
    # 如果有分类头，则设置分类标签数
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]

    # 输出我们的 RoBERTa 配置信息
    print("Our RoBERTa config:", config)

    # 根据是否有分类头选择创建 XLMRobertaXLForSequenceClassification 或 XLMRobertaXLForMaskedLM
    model = XLMRobertaXLForSequenceClassification(config) if classification_head else XLMRobertaXLForMaskedLM(config)
    # 设置为评估模式
    model.eval()

    # 开始复制权重

    # 复制词嵌入权重
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    # 复制位置编码权重
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.

    model.roberta.encoder.LayerNorm.weight = roberta_sent_encoder.layer_norm.weight
    model.roberta.encoder.LayerNorm.bias = roberta_sent_encoder.layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]  # 获取当前层的 BERT 层对象
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]  # 获取对应的 RoBERTa 句子编码器层对象

        attention: RobertaAttention = layer.attention  # 获取当前层的 RoBERTa 注意力对象
        attention.self_attn_layer_norm.weight = roberta_layer.self_attn_layer_norm.weight  # 将 RoBERTa 中的自注意力层权重赋值给当前层的注意力的权重
        attention.self_attn_layer_norm.bias = roberta_layer.self_attn_layer_norm.bias  # 将 RoBERTa 中的自注意力层偏置赋值给当前层的注意力的偏置

        # self attention
        self_attn: BertSelfAttention = layer.attention.self  # 获取当前层的自注意力对象
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape  # 断言确保维度相等
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight  # 赋值 RoBERTa 中自注意力层的查询向量权重给当前层的自注意力层的查询向量权重
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias  # 赋值 RoBERTa 中自注意力层的查询向量偏置给当前层的自注意力层的查询向量偏置
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight  # 赋值 RoBERTa 中自注意力层的键向量权重给当前层的自注意力层的键向量权重
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias  # 赋值 RoBERTa 中自注意力层的键向量偏置给当前层的自注意力层的键向量偏置
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight  # 赋值 RoBERTa 中自注意力层的值向量权重给当前层的自注意力层的值向量权重
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias  # 赋值 RoBERTa 中自注意力层的值向量偏置给当前层的自注意力层的值向量偏置

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output  # 获取当前层的自注意力输出对象
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape  # 断言确保维度相等
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight  # 赋值 RoBERTa 中自注意力层的输出投影权重给当前层的自注意力输出的权重
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias  # 赋值 RoBERTa 中自注意力层的输出投影偏置给当前层的自注意力输出的偏置

        # this one is final layer norm
        layer.LayerNorm.weight = roberta_layer.final_layer_norm.weight  # 赋值 RoBERTa 中最终的层归一化权重给当前层的归一化的权重
        layer.LayerNorm.bias = roberta_layer.final_layer_norm.bias  # 赋值 RoBERTa 中最终的层归一化偏置给当前层的归一化的偏置

        # intermediate
        intermediate: BertIntermediate = layer.intermediate  # 获取当前层的中间层对象
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape  # 断言确保维度相等
        intermediate.dense.weight = roberta_layer.fc1.weight  # 赋值 RoBERTa 中全连接层 1 的权重给当前层的中间层的权重
        intermediate.dense.bias = roberta_layer.fc1.bias  # 赋值 RoBERTa 中全连接层 1 的偏置给当前层的中间层的偏置

        # output
        bert_output: BertOutput = layer.output  # 获取当前层的输出对象
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape  # 断言确保维度相等
        bert_output.dense.weight = roberta_layer.fc2.weight  # 赋值 RoBERTa 中全连接层 2 的权重给当前层的输出的权重
        bert_output.dense.bias = roberta_layer.fc2.bias  # 赋值 RoBERTa 中全连接层 2 的偏置给当前层的输出的偏置
        # end of layer
    如果有分类头
    if classification_head:
        将模型的分类器的权重设置为roberta模型的分类头mnli的dense层的权重
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        将模型的分类器的偏置设置为roberta模型的分类头mnli的dense层的偏置
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        将模型的分类器的输出投影的权重设置为roberta模型的分类头mnli的投影层的权重
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        将模型的分类器的输出投影的偏置设置为roberta模型的分类头mnli的投影层的偏置
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    否则
    else:
        # 语言模型头
        将模型的语言模型头的dense层的权重设置为roberta模型的编码器lm_head的dense层的权重
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        将模型的语言模型头的dense层的偏置设置为roberta模型的编码器lm_head的dense层的偏置
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        将模型的语言模型头的layer_norm的权重设置为roberta模型的编码器lm_head的layer_norm的权重
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        将模型的语言模型头的layer_norm的偏置设置为roberta模型的编码器lm_head的layer_norm的偏置
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        将模型的语言模型头的解码层的权重设置为roberta模型的编码器lm_head的权重
        model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        将模型的语言模型头的解码层的偏置设置为roberta模型的编码器lm_head的偏置
        model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # 检查我们是否得到相同的结果。
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # 批次大小为1

    我们的输出 = 模型（输入ids）[0]
    如果有分类头
        他们的输出 = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    否则
        他们的输出 = roberta.model(input_ids)[0]
    打印我们的输出形状和他们的输出形状
    print(our_output.shape, their_output.shape)
    最大绝对差 = torch.max(torch.abs(our_output - their_output)).item()
    打印最大绝对差
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    成功 = torch.allclose(our_output, their_output, atol=1e-3)
    打印"两个模型是否输出相同的张量？"，如果成功则输出🔥，否则输出💩
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    如果不成功
        引发异常
        raise Exception("Something went wRoNg")

    创建路径为pytorch_dump_folder_path的文件夹，如果父文件夹不存在也进行创建
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    打印将模型保存到pytorch_dump_folder_path
    print(f"Saving model to {pytorch_dump_folder_path}")
    将模型保存到pytorch_dump_folder_path
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数：RoBERTa 模型的检查点路径
    parser.add_argument(
        "--roberta_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加必需参数：输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加可选参数：是否转换最终的分类头
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 XLM-RoBERTa XL 检查点转换为 PyTorch 模型
    convert_xlm_roberta_xl_checkpoint_to_pytorch(
        args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
```