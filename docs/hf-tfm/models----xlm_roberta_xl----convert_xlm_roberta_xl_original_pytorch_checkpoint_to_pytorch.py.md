# `.\models\xlm_roberta_xl\convert_xlm_roberta_xl_original_pytorch_checkpoint_to_pytorch.py`

```
# 指定 Python 文件的编码格式为 UTF-8

# 导入必要的库和模块
import argparse  # 解析命令行参数的库
import pathlib   # 处理路径的库

import fairseq   # 引入 fairseq 库
import torch     # 引入 PyTorch 库
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel  # 导入 fairseq 中的 RoBERTa 模型
from fairseq.modules import TransformerSentenceEncoderLayer  # 导入 fairseq 中的 TransformerSentenceEncoderLayer 模块
from packaging import version  # 用于处理版本号的库

# 从 transformers 库中导入相关模块和类
from transformers import XLMRobertaConfig, XLMRobertaXLForMaskedLM, XLMRobertaXLForSequenceClassification
from transformers.models.bert.modeling_bert import (
    BertIntermediate,   # 导入 BERT 模型中的 BertIntermediate 类
    BertLayer,          # 导入 BERT 模型中的 BertLayer 类
    BertOutput,         # 导入 BERT 模型中的 BertOutput 类
    BertSelfAttention,  # 导入 BERT 模型中的 BertSelfAttention 类
    BertSelfOutput,     # 导入 BERT 模型中的 BertSelfOutput 类
)
from transformers.models.roberta.modeling_roberta import RobertaAttention  # 导入 RoBERTa 模型中的 RobertaAttention 类
from transformers.utils import logging  # 导入 transformers 库中的日志记录模块

# 检查 fairseq 版本是否符合要求
if version.parse(fairseq.__version__) < version.parse("1.0.0a"):
    raise Exception("requires fairseq >= 1.0.0a")

# 设置日志记录的详细程度为 info 级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个示例文本
SAMPLE_TEXT = "Hello world! cécé herlolip"

# 定义函数：将 XLM-RoBERTa XL 的检查点转换为 PyTorch 模型
def convert_xlm_roberta_xl_checkpoint_to_pytorch(
    roberta_checkpoint_path: str,  # RoBERTa 检查点文件路径
    pytorch_dump_folder_path: str,  # 转换后的 PyTorch 模型保存路径
    classification_head: bool  # 是否包含分类头
):
    """
    复制/粘贴/调整 RoBERTa 的权重到我们的 BERT 结构。
    """
    # 从预训练的 RoBERTa 模型加载权重
    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    # 将模型设置为评估模式，禁用 dropout
    roberta.eval()
    # 获取 RoBERTa 模型中的句子编码器
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    # 根据 RoBERTa 模型的配置创建 XLM-RoBERTa 的配置
    config = XLMRobertaConfig(
        vocab_size=roberta_sent_encoder.embed_tokens.num_embeddings,  # 词汇表大小
        hidden_size=roberta.cfg.model.encoder_embed_dim,  # 隐藏层大小
        num_hidden_layers=roberta.cfg.model.encoder_layers,  # 编码器层数
        num_attention_heads=roberta.cfg.model.encoder_attention_heads,  # 注意力头数
        intermediate_size=roberta.cfg.model.encoder_ffn_embed_dim,  # 中间层大小
        max_position_embeddings=514,  # 最大位置嵌入
        type_vocab_size=1,  # 类型词汇表大小
        layer_norm_eps=1e-5,  # 层归一化的 epsilon 值，与 fairseq 使用的 PyTorch 默认值相同
    )
    # 如果包含分类头，则设置配置中的标签数目
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]

    # 打印配置信息
    print("Our RoBERTa config:", config)

    # 根据是否包含分类头选择相应的 XLM-RoBERTa 模型
    model = XLMRobertaXLForSequenceClassification(config) if classification_head else XLMRobertaXLForMaskedLM(config)
    # 将模型设置为评估模式
    model.eval()

    # 开始复制所有权重。
    # 复制嵌入层的权重
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # 将 RoBERTa 模型的 token_type_embeddings 权重置零，因为 RoBERTa 不使用它们。

    model.roberta.encoder.LayerNorm.weight = roberta_sent_encoder.layer_norm.weight
    model.roberta.encoder.LayerNorm.bias = roberta_sent_encoder.layer_norm.bias

    for i in range(config.num_hidden_layers):
        # 循环遍历每一层的编码器

        # 获取当前层的 BertLayer 对象和对应的 TransformerSentenceEncoderLayer 对象
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # 设置注意力层的权重和偏置
        attention: RobertaAttention = layer.attention
        attention.self_attn_layer_norm.weight = roberta_layer.self_attn_layer_norm.weight
        attention.self_attn_layer_norm.bias = roberta_layer.self_attn_layer_norm.bias

        # 设置自注意力机制的权重和偏置
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )
        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # 设置自注意力机制输出的权重和偏置
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias

        # 设置最终的层归一化的权重和偏置
        layer.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        layer.LayerNorm.bias = roberta_layer.final_layer_norm.bias

        # 设置中间层的全连接层的权重和偏置
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # 设置输出层的权重和偏置
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        # 层结束
    # 如果有分类头，则复制 RoBERTa 模型的分类头参数到当前模型的分类器中
    if classification_head:
        # 复制权重和偏置
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        # 复制输出投影的权重和偏置
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # 如果没有分类头，则复制 RoBERTa 模型的语言模型头参数到当前模型的语言模型头中
        # 复制权重和偏置
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        # 复制 LayerNorm 的权重和偏置
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        # 复制解码器的权重和偏置
        model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # 检查模型输出是否一致
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # 将输入编码为张量，并增加一个维度作为批处理的大小为1
    our_output = model(input_ids)[0]  # 获取当前模型的输出
    if classification_head:
        their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        their_output = roberta.model(input_ids)[0]  # 获取 RoBERTa 模型的输出
    print(our_output.shape, their_output.shape)  # 打印两个模型输出的形状
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()  # 计算输出之间的最大绝对差异
    print(f"max_absolute_diff = {max_absolute_diff}")  # 打印最大绝对差异，预期约为 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)  # 检查两个模型输出是否在指定误差范围内一致
    print("Do both models output the same tensors?", "🔥" if success else "💩")  # 打印是否两个模型输出相同
    if not success:
        raise Exception("Something went wRoNg")  # 如果输出不一致，则抛出异常

    # 确保路径存在并创建 PyTorch 模型保存文件夹
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")  # 打印模型保存的路径
    model.save_pretrained(pytorch_dump_folder_path)  # 将当前模型保存到指定路径
if __name__ == "__main__":
    # 如果脚本被直接执行而非作为模块导入，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必填参数
    parser.add_argument(
        "--roberta_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加一个必填参数：RoBERTa 模型的检查点路径，必须是字符串类型，用户必须提供，帮助信息指明它是官方 PyTorch dump 的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个必填参数：输出 PyTorch 模型的文件夹路径，必须是字符串类型，用户必须提供，帮助信息指明它是输出 PyTorch 模型的路径

    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 添加一个标志参数：是否转换最终的分类头部，当存在该参数时设置其值为 True，帮助信息说明了这个参数的作用

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，将 XLM-RoBERTa XL 模型的检查点转换为 PyTorch 格式
    convert_xlm_roberta_xl_checkpoint_to_pytorch(
        args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
```