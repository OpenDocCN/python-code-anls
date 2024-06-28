# `.\models\data2vec\convert_data2vec_text_original_pytorch_checkpoint_to_pytorch.py`

```py
# 引入所需的模块和库
import argparse  # 用于解析命令行参数
import os  # 用于操作系统相关的功能
import pathlib  # 提供处理文件和目录路径的类

import fairseq  # 引入fairseq库
import torch  # 引入PyTorch库
from fairseq.modules import TransformerSentenceEncoderLayer  # 从fairseq模块中引入TransformerSentenceEncoderLayer
from packaging import version  # 用于版本比较的包

from transformers import (  # 从transformers库中引入多个类和函数
    Data2VecTextConfig,  # 用于配置Data2VecText模型的类
    Data2VecTextForMaskedLM,  # 用于Data2VecText的MLM任务的类
    Data2VecTextForSequenceClassification,  # 用于Data2VecText的序列分类任务的类
    Data2VecTextModel,  # Data2VecText模型的主类
)
from transformers.models.bert.modeling_bert import (  # 从BERT模型中引入多个类
    BertIntermediate,  # BERT中间层的类
    BertLayer,  # BERT层的类
    BertOutput,  # BERT输出层的类
    BertSelfAttention,  # BERT自注意力机制的类
    BertSelfOutput,  # BERT自注意力输出的类
)

# 重要提示：为了运行本脚本，请确保从以下链接下载字典：`dict.txt` https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
# 文件来源于 https://github.com/pytorch/fairseq/blob/main/examples/data2vec/models/data2vec_text.py
from transformers.utils import logging  # 从transformers工具模块中引入日志记录功能


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")  # 如果fairseq版本低于0.9.0，抛出异常

logging.set_verbosity_info()  # 设置日志详细程度为info
logger = logging.get_logger(__name__)  # 获取当前脚本的日志记录器

SAMPLE_TEXT = "Hello world! cécé herlolip"  # 示例文本

def convert_data2vec_checkpoint_to_pytorch(
    data2vec_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    将data2vec的检查点权重复制/粘贴/调整到我们的BERT结构中。
    """
    # 获取data2vec检查点的路径信息
    data2vec_checkpoint_dir, data2vec_checkpoint_file_name = os.path.split(data2vec_checkpoint_path)
    # 从预训练的data2vec模型加载模型
    data2vec = Data2VecTextModel.from_pretrained(
        data2vec_checkpoint_dir, checkpoint_file=data2vec_checkpoint_file_name
    )
    data2vec.eval()  # 设置模型为评估模式，禁用dropout
    data2vec_model = data2vec.models[0]  # 获取data2vec模型的主体部分
    data2vec_sent_encoder = data2vec_model.encoder.sentence_encoder  # 获取data2vec模型的句子编码器
    # 创建Data2VecTextConfig配置对象，用于后续的BERT模型
    config = Data2VecTextConfig(
        vocab_size=data2vec_sent_encoder.embed_tokens.num_embeddings,  # 词汇表大小
        hidden_size=data2vec_model.args.encoder_embed_dim,  # 隐藏层大小
        num_hidden_layers=data2vec_model.args.encoder_layers,  # 隐藏层层数
        num_attention_heads=data2vec_model.args.encoder_attention_heads,  # 注意力头数
        intermediate_size=data2vec_model.args.encoder_ffn_embed_dim,  # 中间层大小
        max_position_embeddings=514,  # 最大位置编码
        type_vocab_size=1,  # 类型词汇表大小
        layer_norm_eps=1e-5,  # 层归一化epsilon值，与fairseq默认相同
    )
    if classification_head:
        config.num_labels = data2vec.model.classification_heads["mnli"].out_proj.weight.shape[0]  # 如果有分类头，设置标签数目
    print("Our BERT config:", config)  # 打印配置信息
    # 根据是否需要分类头选择合适的模型：如果需要分类头，则使用Data2VecTextForSequenceClassification，否则使用Data2VecTextForMaskedLM
    model = Data2VecTextForSequenceClassification(config) if classification_head else Data2VecTextForMaskedLM(config)
    model.eval()

    # 现在让我们复制所有的权重。

    # 复制嵌入层权重
    model.data2vec_text.embeddings.word_embeddings.weight = data2vec_sent_encoder.embed_tokens.weight
    model.data2vec_text.embeddings.position_embeddings.weight = data2vec_sent_encoder.embed_positions.weight
    model.data2vec_text.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.data2vec_text.embeddings.token_type_embeddings.weight
    )  # 将其置零，因为data2vec不使用这些
    model.data2vec_text.embeddings.LayerNorm.weight = data2vec_sent_encoder.layernorm_embedding.weight
    model.data2vec_text.embeddings.LayerNorm.bias = data2vec_sent_encoder.layernorm_embedding.bias

    if classification_head:
        # 如果存在分类头，复制分类器权重
        model.classifier.dense.weight = data2vec.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = data2vec.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = data2vec.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = data2vec.model.classification_heads["mnli"].out_proj.bias
    else:
        # 否则，复制语言模型头权重
        model.lm_head.dense.weight = data2vec_model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = data2vec_model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = data2vec_model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = data2vec_model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = data2vec_model.encoder.lm_head.weight
        model.lm_head.decoder.bias = data2vec_model.encoder.lm_head.bias

    # 检查是否输出相同的结果。

    # 使用data2vec对样本文本编码并添加批次维度
    input_ids: torch.Tensor = data2vec.encode(SAMPLE_TEXT).unsqueeze(0)

    # 计算我们模型的输出
    our_output = model(input_ids)[0]

    if classification_head:
        # 如果使用分类头，计算data2vec模型的输出
        their_output = data2vec.model.classification_heads["mnli"](data2vec.extract_features(input_ids))
    else:
        # 否则，计算data2vec模型的输出
        their_output = data2vec_model(input_ids)[0]

    # 打印两个输出的形状
    print(our_output.shape, their_output.shape)

    # 计算两者之间的最大绝对差
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 大约为1e-7

    # 检查两个模型输出的张量是否几乎相同
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")

    # 如果两者输出不几乎相同，则抛出异常
    if not success:
        raise Exception("Something went wRoNg")

    # 创建目录以保存PyTorch模型
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")

    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果这个脚本是作为主程序运行时执行以下操作

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加一个必选参数，用于指定官方 PyTorch 转储文件的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个必选参数，用于指定输出 PyTorch 模型的文件夹路径

    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 添加一个选项参数，表示是否转换最终的分类头部

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 变量中

    convert_data2vec_checkpoint_to_pytorch(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
    # 调用函数 convert_data2vec_checkpoint_to_pytorch，传入命令行参数中解析的路径和选项


这段代码是一个典型的命令行工具的入口点，它使用 argparse 模块解析命令行参数，并调用一个函数来处理这些参数指定的任务。
```