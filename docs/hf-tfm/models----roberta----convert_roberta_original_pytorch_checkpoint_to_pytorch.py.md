# `.\transformers\models\roberta\convert_roberta_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为UTF-8
# 版权声明，告知使用者可以在遵守许可证的情况下使用该文件
# 如果需要，可以从 http://www.apache.org/licenses/LICENSE-2.0 获得许可证副本
# 根据许可证，分发的软件基于“原样”分发，没有任何形式的保证或条件，不论是明示的还是隐含的
# 查看许可证，了解特定语言的权限和限制
# 将RoBERTa检查点转换为PyTorch格式

import argparse  # 导入用于解析命令行参数的模块
import pathlib  # 提供了用于处理文件路径的实用功能

import fairseq  # 导入 fairseq 库
import torch  # 导入PyTorch库
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel  # 从 fairseq 模型中导入 RoBERTa 模型
from fairseq.modules import TransformerSentenceEncoderLayer  # 从 fairseq 模块中导入 TransformerSentenceEncoderLayer
from packaging import version  # 从 packaging 模块中导入 version 类

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification  # 从 transformers库中导入相关接口
from transformers.models.bert.modeling_bert import (  # 从 transformers 库中导入相关接口
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging  # 从 transformers 库中导入日志模块

# 检查 fairseq 库的版本，如果小于0.9.0，则抛出异常
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

# 设置logging模块的输出级别
logging.set_verbosity_info()
# 获取logger对象
logger = logging.get_logger(__name__)

# 示例文本
SAMPLE_TEXT = "Hello world! cécé herlolip"

# 将RoBERTa检查点转换为PyTorch格式
def convert_roberta_checkpoint_to_pytorch(
    roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    # 从给定路径加载预训练的RoBERTa模型
    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    # 关闭dropout，令模型处于评估模式
    roberta.eval()
    # 获取句子编码器
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    # 创建RoBERTa配置对象
    config = RobertaConfig(
        vocab_size=roberta_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.args.encoder_layers,
        num_attention_heads=roberta.args.encoder_attention_heads,
        intermediate_size=roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # 使用fairseq中的PyTorch默认值
    )
    # 如果存在分类头，则将分类标签的数量添加到配置中
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    # 打印配置信息
    print("Our BERT config:", config)

    # 创建RoBERTa模型，如果存在分类头，则创建RoBERTa序列分类模型，否则创建RoBERTa遮蔽语言模型
    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    # 关闭dropout，令模型处于评估模式
    model.eval()

    # 复制所有权重
    # 复制嵌入权重
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    # 复制位置嵌入权重
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    # 将token_type嵌入权重置为0，因为RoBERTa不使用它们
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )
    # 将预训练模型 RoBERTa 的 LayerNorm 权重和偏置赋值给当前模型的 RoBERTa 的嵌入层的 LayerNorm 权重和偏置
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias

    # 遍历每个编码层
    for i in range(config.num_hidden_layers):
        # 获取当前层的 BertLayer 对象和对应的 TransformerSentenceEncoderLayer 对象
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # 验证自注意力层参数的形状是否一致
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        # 将 RoBERTa 中自注意力层的权重和偏置赋值给当前模型的自注意力层
        self_attn: BertSelfAttention = layer.attention.self
        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # 将 RoBERTa 中自注意力层输出层的权重和偏置赋值给当前模型的自注意力层输出层
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        # 将 RoBERTa 中中间层的权重和偏置赋值给当前模型的中间层
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # 将 RoBERTa 中输出层的权重和偏置赋值给当前模型的输出层
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        # 编码层结束

    # 如果有分类头部，则将 RoBERTa 中对应的权重和偏置赋值给当前模型的分类器
    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # 复制 RoBERTa 模型的语言模型头部参数到当前模型的语言模型头部
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # 检查是否得到相同的结果。
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # 批量大小为1的输入

    # 使用当前模型处理输入，获取输出
    our_output = model(input_ids)[0]

    if classification_head:
        # 如果存在分类头部，则使用 RoBERTa 模型的特定分类头部处理输入特征
        their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        # 否则，使用 RoBERTa 模型处理输入并获取输出
        their_output = roberta.model(input_ids)[0]

    # 打印当前模型输出和 RoBERTa 模型输出的形状
    print(our_output.shape, their_output.shape)

    # 计算输出之间的最大绝对差异
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 大约 1e-7

    # 检查两个模型的输出是否非常接近
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")

    # 如果输出不接近，则抛出异常
    if not success:
        raise Exception("Something went wRoNg")

    # 创建目录用于保存 PyTorch 模型
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")

    # 将当前模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接运行，而不是作为被导入模块使用，执行以下代码块
if __name__ == "__main__":
    # 创建一个 argparse 解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个必需的参数，用来指定 RoBERTa 模型的检查点文件路径
    parser.add_argument(
        "--roberta_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加一个必需的参数，用来指定转换后的 PyTorch 模型输出文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个开关参数，用来指定是否转换模型的最后一个分类头
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 使用解析器解析命令行参数，并返回一个能够访问解析后参数的命名空间对象
    args = parser.parse_args()
    # 调用函数 convert_roberta_checkpoint_to_pytorch，将 RoBERTa 模型转换为 PyTorch 模型
    # 传入参数为 RoBERTa 模型检查点文件路径、转换后的 PyTorch 模型输出文件夹路径、是否转换模型的最后一个分类头
    convert_roberta_checkpoint_to_pytorch(
        args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
```