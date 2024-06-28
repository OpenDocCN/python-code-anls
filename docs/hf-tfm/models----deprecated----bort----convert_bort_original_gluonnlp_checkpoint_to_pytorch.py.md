# `.\models\deprecated\bort\convert_bort_original_gluonnlp_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse  # 解析命令行参数的库
import os  # 提供与操作系统交互的功能

import gluonnlp as nlp  # GluonNLP，一个自然语言处理工具包
import mxnet as mx  # MXNet，一个深度学习框架
import numpy as np  # NumPy，用于处理数组和数值计算的库
import torch  # PyTorch，一个深度学习框架
from gluonnlp.base import get_home_dir  # 获取用户主目录的函数
from gluonnlp.model.bert import BERTEncoder  # GluonNLP中的BERT编码器
from gluonnlp.model.utils import _load_vocab  # 加载词汇表的内部函数
from gluonnlp.vocab import Vocab  # GluonNLP中的词汇表类
from packaging import version  # 版本管理工具
from torch import nn  # PyTorch中的神经网络模块

from transformers import BertConfig, BertForMaskedLM, BertModel, RobertaTokenizer  # Transformers库中的BERT相关模块
from transformers.models.bert.modeling_bert import (  # Transformers中BERT模型的具体实现
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging  # Transformers中的日志模块

# 检查GluonNLP和MXNet的版本是否符合要求，如果不符合则抛出异常
if version.parse(nlp.__version__) != version.parse("0.8.3"):
    raise Exception("requires gluonnlp == 0.8.3")

if version.parse(mx.__version__) != version.parse("1.5.0"):
    raise Exception("requires mxnet == 1.5.0")

# 设置日志输出等级为INFO
logging.set_verbosity_info()
# 获取logger对象用于记录日志
logger = logging.get_logger(__name__)

# 示例文本
SAMPLE_TEXT = "The Nymphenburg Palace is a beautiful palace in Munich!"


def convert_bort_checkpoint_to_pytorch(bort_checkpoint_path: str, pytorch_dump_folder_path: str):
    """
    Convert the original Bort checkpoint (based on MXNET and Gluonnlp) to our BERT structure-
    将原始的基于MXNet和GluonNLP的Bort检查点转换为我们的BERT结构。
    """

    # 原始的Bort配置参数
    bort_4_8_768_1024_hparams = {
        "attention_cell": "multi_head",
        "num_layers": 4,
        "units": 1024,
        "hidden_size": 768,
        "max_length": 512,
        "num_heads": 8,
        "scaled": True,
        "dropout": 0.1,
        "use_residual": True,
        "embed_size": 1024,
        "embed_dropout": 0.1,
        "word_embed": None,
        "layer_norm_eps": 1e-5,
        "token_type_vocab_size": 2,
    }

    # 使用预定义的Bort参数作为初始参数
    predefined_args = bort_4_8_768_1024_hparams

    # 在此处构建原始的Bort模型
    # 参考自官方的BERT实现，详见：
    # https://github.com/alexa/bort/blob/master/bort/bort.py
    # 创建一个 BERTEncoder 对象，用于编码文本
    encoder = BERTEncoder(
        attention_cell=predefined_args["attention_cell"],  # 设置注意力机制类型
        num_layers=predefined_args["num_layers"],  # 设置编码器层数
        units=predefined_args["units"],  # 设置每个编码器层的单元数
        hidden_size=predefined_args["hidden_size"],  # 设置隐藏层大小
        max_length=predefined_args["max_length"],  # 设置最大序列长度
        num_heads=predefined_args["num_heads"],  # 设置注意力头数
        scaled=predefined_args["scaled"],  # 是否进行缩放的注意力机制
        dropout=predefined_args["dropout"],  # 设置丢弃率
        output_attention=False,  # 是否输出注意力分布
        output_all_encodings=False,  # 是否输出所有编码
        use_residual=predefined_args["use_residual"],  # 是否使用残差连接
        activation=predefined_args.get("activation", "gelu"),  # 激活函数类型，默认为GELU
        layer_norm_eps=predefined_args.get("layer_norm_eps", None),  # LayerNorm层的epsilon值
    )

    # 需要先获取词汇信息
    # 使用 RoBERTa 相同的词汇表名称
    vocab_name = "openwebtext_ccnews_stories_books_cased"

    # 指定 Gluonnlp 的词汇下载文件夹
    gluon_cache_dir = os.path.join(get_home_dir(), "models")

    # 加载词汇表，并使用 Vocab 类进行处理
    bort_vocab = _load_vocab(vocab_name, None, gluon_cache_dir, cls=Vocab)

    # 创建原始的 BERT 模型，用于后续的参数加载
    original_bort = nlp.model.BERTModel(
        encoder,
        len(bort_vocab),  # BERT 模型的词汇表大小
        units=predefined_args["units"],  # 设置每个编码器层的单元数
        embed_size=predefined_args["embed_size"],  # 嵌入向量的大小
        embed_dropout=predefined_args["embed_dropout"],  # 嵌入层的丢弃率
        word_embed=predefined_args["word_embed"],  # 是否使用词嵌入
        use_pooler=False,  # 是否使用池化器
        use_token_type_embed=False,  # 是否使用类型嵌入
        token_type_vocab_size=predefined_args["token_type_vocab_size"],  # 类型嵌入的大小
        use_classifier=False,  # 是否使用分类器
        use_decoder=False,  # 是否使用解码器
    )

    # 加载 BERT 模型的预训练参数
    original_bort.load_parameters(bort_checkpoint_path, cast_dtype=True, ignore_extra=True)

    # 收集模型参数的前缀
    params = original_bort._collect_params_with_prefix()

    # 构建适用于 Transformers 的配置对象
    hf_bort_config_json = {
        "architectures": ["BertForMaskedLM"],  # 模型架构类型
        "attention_probs_dropout_prob": predefined_args["dropout"],  # 注意力概率的丢弃率
        "hidden_act": "gelu",  # 隐藏层的激活函数类型
        "hidden_dropout_prob": predefined_args["dropout"],  # 隐藏层的丢弃率
        "hidden_size": predefined_args["embed_size"],  # 隐藏层的大小
        "initializer_range": 0.02,  # 参数初始化范围
        "intermediate_size": predefined_args["hidden_size"],  # 中间层的大小
        "layer_norm_eps": predefined_args["layer_norm_eps"],  # LayerNorm 层的 epsilon 值
        "max_position_embeddings": predefined_args["max_length"],  # 最大位置嵌入的长度
        "model_type": "bort",  # 模型类型
        "num_attention_heads": predefined_args["num_heads"],  # 注意力头数
        "num_hidden_layers": predefined_args["num_layers"],  # 隐藏层的层数
        "pad_token_id": 1,  # 填充 token 的 ID （2 = BERT, 1 = RoBERTa）
        "type_vocab_size": 1,  # 类型词汇表的大小 （2 = BERT, 1 = RoBERTa）
        "vocab_size": len(bort_vocab),  # 词汇表的大小
    }

    # 从 JSON 配置构建 BertConfig 对象
    hf_bort_config = BertConfig.from_dict(hf_bort_config_json)

    # 创建适用于 Transformers 的 BertForMaskedLM 模型
    hf_bort_model = BertForMaskedLM(hf_bort_config)

    # 将模型设置为评估模式
    hf_bort_model.eval()

    # 参数映射表（从 Gluonnlp 到 Transformers）
    # * 表示层索引
    #
    # | Gluon 参数                       | Transformers 参数
    # | ---------------------------------------------------- | ----------------------
    # Helper function to convert MXNET Arrays to PyTorch 的参数数组
    def to_torch(mx_array) -> nn.Parameter:
        # 将 MXNET 的 NDArray 转换为 PyTorch 的 nn.Parameter 类型
        return nn.Parameter(torch.FloatTensor(mx_array.data().asnumpy()))
    
    # Check param shapes and map new HF param back
    # 检查参数形状并将新的 HF 参数映射回去
    # 定义函数，用于检查和映射参数
    def check_and_map_params(hf_param, gluon_param):
        # 获取 HF 参数的形状
        shape_hf = hf_param.shape

        # 将 gluon_param 转换为 PyTorch 张量
        gluon_param = to_torch(params[gluon_param])
        # 获取 gluon_param 的形状
        shape_gluon = gluon_param.shape

        # 断言 HF 参数和 gluon_param 的形状必须一致，否则抛出异常
        assert (
            shape_hf == shape_gluon
        ), f"The gluon parameter {gluon_param} has shape {shape_gluon}, but expects shape {shape_hf} for Transformers"

        # 返回 gluon_param
        return gluon_param

    # 将 HF Bort 模型的 word embeddings 参数映射到 gluon_param
    hf_bort_model.bert.embeddings.word_embeddings.weight = check_and_map_params(
        hf_bort_model.bert.embeddings.word_embeddings.weight, "word_embed.0.weight"
    )

    # 将 HF Bort 模型的 position embeddings 参数映射到 gluon_param
    hf_bort_model.bert.embeddings.position_embeddings.weight = check_and_map_params(
        hf_bort_model.bert.embeddings.position_embeddings.weight, "encoder.position_weight"
    )

    # 将 HF Bort 模型的 LayerNorm bias 参数映射到 gluon_param
    hf_bort_model.bert.embeddings.LayerNorm.bias = check_and_map_params(
        hf_bort_model.bert.embeddings.LayerNorm.bias, "encoder.layer_norm.beta"
    )

    # 将 HF Bort 模型的 LayerNorm weight 参数映射到 gluon_param
    hf_bort_model.bert.embeddings.LayerNorm.weight = check_and_map_params(
        hf_bort_model.bert.embeddings.LayerNorm.weight, "encoder.layer_norm.gamma"
    )

    # 将 HF Bort 模型的 token_type_embeddings 权重置零，受 RoBERTa 转换脚本启发（Bort 不使用这些参数）
    hf_bort_model.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        hf_bort_model.bert.embeddings.token_type_embeddings.weight.data
    )

    # 将 HF Bort 模型转换为半精度模型，节省空间和能耗
    hf_bort_model.half()

    # 比较两个模型的输出
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

    # 对示例文本进行编码
    input_ids = tokenizer.encode_plus(SAMPLE_TEXT)["input_ids"]

    # 获取 gluon 模型的输出
    gluon_input_ids = mx.nd.array([input_ids])
    output_gluon = original_bort(inputs=gluon_input_ids, token_types=[])

    # 保存并重新加载 HF Bort 模型，以获取 Transformer 的输出
    hf_bort_model.save_pretrained(pytorch_dump_folder_path)
    hf_bort_model = BertModel.from_pretrained(pytorch_dump_folder_path)
    hf_bort_model.eval()

    # 对示例文本再次编码，并获取 HF Bort 模型的输出
    input_ids = tokenizer.encode_plus(SAMPLE_TEXT, return_tensors="pt")
    output_hf = hf_bort_model(**input_ids)[0]

    # 将 gluon_layer 和 hf_layer 转换为 numpy 数组
    gluon_layer = output_gluon[0].asnumpy()
    hf_layer = output_hf[0].detach().numpy()

    # 计算两个输出之间的最大绝对差
    max_absolute_diff = np.max(np.abs(hf_layer - gluon_layer)).item()

    # 检查两个输出是否在给定的误差范围内相等
    success = np.allclose(gluon_layer, hf_layer, atol=1e-3)

    # 如果成功，则打印输出相同的消息；否则打印输出不同的消息及绝对差
    if success:
        print("✔️ Both model do output the same tensors")
    else:
        print("❌ Both model do **NOT** output the same tensors")
        print("Absolute difference is:", max_absolute_diff)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument(
        "--bort_checkpoint_path", default=None, type=str, required=True, help="Path the official Bort params file."
    )
    # 添加一个参数选项：--bort_checkpoint_path，类型为字符串，必选参数，用于指定官方Bort参数文件的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个参数选项：--pytorch_dump_folder_path，类型为字符串，必选参数，用于指定输出PyTorch模型的路径

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在args对象中

    convert_bort_checkpoint_to_pytorch(args.bort_checkpoint_path, args.pytorch_dump_folder_path)
    # 调用convert_bort_checkpoint_to_pytorch函数，传入解析后的参数args中的路径信息作为参数
```