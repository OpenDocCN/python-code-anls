# `.\models\roberta\convert_roberta_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置 Python 文件编码格式为 UTF-8
# 版权声明和许可协议，这里是 Apache License 2.0
# 详细信息可参见 http://www.apache.org/licenses/LICENSE-2.0

# 导入必要的库和模块
import argparse        # 用于解析命令行参数
import pathlib         # 提供处理路径的类和函数

import fairseq         # 导入 fairseq 库
import torch           # 导入 PyTorch 库
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel  # 导入 Fairseq 中的 RoBERTa 模型
from fairseq.modules import TransformerSentenceEncoderLayer  # 导入 Fairseq 中的 TransformerSentenceEncoderLayer 模块
from packaging import version  # 用于处理版本号的库

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification  # 导入 Hugging Face Transformers 中的 RoBERTa 相关类
from transformers.models.bert.modeling_bert import (  # 导入 Transformers BERT 模型的部分组件类
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging  # 导入 Transformers 的日志模块

# 如果 fairseq 的版本小于 0.9.0，则抛出异常
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

# 设置日志输出级别为 INFO
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 示例文本
SAMPLE_TEXT = "Hello world! cécé herlolip"

# 定义函数，将 RoBERTa 模型的检查点转换为 PyTorch 格式
def convert_roberta_checkpoint_to_pytorch(
    roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    复制/粘贴/调整 RoBERTa 的权重以适应我们的 BERT 结构。
    """
    # 从预训练的 RoBERTa 检查点路径加载模型
    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    # 设置为评估模式，禁用 dropout
    roberta.eval()
    # 获取 RoBERTa 的句子编码器
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    # 创建 RoBERTaConfig 对象，用于定义转换后的 BERT 模型配置
    config = RobertaConfig(
        vocab_size=roberta_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.args.encoder_layers,
        num_attention_heads=roberta.args.encoder_attention_heads,
        intermediate_size=roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch 默认值，与 fairseq 保持一致
    )
    # 如果需要分类头部，则设置 num_labels 属性为对应分类头部的输出维度
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    # 输出 BERT 模型的配置信息
    print("Our BERT config:", config)

    # 创建 RoBERTaForSequenceClassification 或 RoBERTaForMaskedLM 模型对象
    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    # 设置为评估模式
    model.eval()

    # 开始复制所有权重
    # 复制词嵌入权重
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    # 复制位置编码权重
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    # 将 token_type_embeddings 的权重数据置零，因为 RoBERTa 不使用 token_type_embeddings
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.
    # 将 RoBERTa 模型的 LayerNorm 权重和偏置设置为 RoBERTa 句子编码器的对应权重和偏置
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias

    # 遍历每个隐藏层进行参数设置
    for i in range(config.num_hidden_layers):
        # 获取当前层的 BertLayer 对象和对应的 TransformerSentenceEncoderLayer 对象
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # 设置自注意力层的权重和偏置
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

        # 设置自注意力层输出的权重和偏置
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        # 设置中间层的权重和偏置
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # 设置输出层的权重和偏置
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        # 本层设置结束

    # 如果有分类头，则设置分类器的权重和偏置为 RoBERTa 模型中指定分类头的对应权重和偏置
    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # 如果不是分类任务，复制 RoBERTa 模型的语言模型头部权重和偏置到当前模型的语言模型头部
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # 检查我们的模型是否产生相同的输出结果。
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # 批大小为1的输入张量

    our_output = model(input_ids)[0]
    if classification_head:
        # 如果有分类头部，使用 RoBERTa 模型的对应分类头部进行推理
        their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        # 否则直接使用 RoBERTa 模型的输出进行推理
        their_output = roberta.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    # 计算输出张量的最大绝对差异
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 约为 1e-7
    # 检查两个模型的输出张量是否足够接近
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    # 创建存储 PyTorch 模型的文件夹路径，如果不存在则创建
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将当前模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本被直接执行而非被导入，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument(
        "--roberta_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加一个命令行参数，用于指定 RoBERTa 模型的检查点路径，必须提供，类型为字符串

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个命令行参数，用于指定输出 PyTorch 模型的文件夹路径，必须提供，类型为字符串

    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 添加一个命令行参数，用于指定是否转换最终的分类头部，采用布尔标志方式

    args = parser.parse_args()
    # 解析命令行参数并返回一个命名空间对象 args，包含了解析后的参数值

    convert_roberta_checkpoint_to_pytorch(
        args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
    # 调用函数 convert_roberta_checkpoint_to_pytorch，传递命令行参数中指定的 RoBERTa 检查点路径、输出路径和分类头部转换标志作为参数
```