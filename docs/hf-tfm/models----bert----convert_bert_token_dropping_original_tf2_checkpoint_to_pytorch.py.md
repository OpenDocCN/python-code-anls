# `.\transformers\models\bert\convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py`

```py
# 导入必要的库
import argparse  # 用于解析命令行参数

import tensorflow as tf  # 导入 TensorFlow 库
import torch  # 导入 PyTorch 库

from transformers import BertConfig, BertForMaskedLM  # 导入 Transformers 库中的 BertConfig 和 BertForMaskedLM 类
from transformers.models.bert.modeling_bert import (  # 从 Transformers 库中的 bert.modeling_bert 模块导入以下类
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertPooler,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging  # 从 Transformers 库中的 utils 模块导入 logging

# 设置日志级别为信息
logging.set_verbosity_info()

# 定义函数，将 TensorFlow 的检查点转换为 PyTorch 兼容的 BERT 模型
def convert_checkpoint_to_pytorch(tf_checkpoint_path: str, config_path: str, pytorch_dump_path: str):
    # 定义函数，获取被屏蔽语言模型（Masked LM）数组
    def get_masked_lm_array(name: str):
        # 构建完整的变量名
        full_name = f"masked_lm/{name}/.ATTRIBUTES/VARIABLE_VALUE"
        # 从 TensorFlow 检查点中加载指定变量的值
        array = tf.train.load_variable(tf_checkpoint_path, full_name)

        # 如果变量名包含“kernel”，则对数组进行转置
        if "kernel" in name:
            array = array.transpose()

        # 将 NumPy 数组转换为 PyTorch 张量并返回
        return torch.from_numpy(array)

    # 定义函数，获取编码器数组
    def get_encoder_array(name: str):
        # 构建完整的变量名
        full_name = f"encoder/{name}/.ATTRIBUTES/VARIABLE_VALUE"
        # 从 TensorFlow 检查点中加载指定变量的值
        array = tf.train.load_variable(tf_checkpoint_path, full_name)

        # 如果变量名包含“kernel”，则对数组进行转置
        if "kernel" in name:
            array = array.transpose()

        # 将 NumPy 数组转换为 PyTorch 张量并返回
        return torch.from_numpy(array)

    # 定义函数，获取编码器层数组
    def get_encoder_layer_array(layer_index: int, name: str):
        # 构建完整的变量名
        full_name = f"encoder/_transformer_layers/{layer_index}/{name}/.ATTRIBUTES/VARIABLE_VALUE"
        # 从 TensorFlow 检查点中加载指定变量的值
        array = tf.train.load_variable(tf_checkpoint_path, full_name)

        # 如果变量名包含“kernel”，则对数组进行转置
        if "kernel" in name:
            array = array.transpose()

        # 将 NumPy 数组转换为 PyTorch 张量并返回
        return torch.from_numpy(array)

    # 定义函数，获取编码器注意力层数组
    def get_encoder_attention_layer_array(layer_index: int, name: str, orginal_shape):
        # 构建完整的变量名
        full_name = f"encoder/_transformer_layers/{layer_index}/_attention_layer/{name}/.ATTRIBUTES/VARIABLE_VALUE"
        # 从 TensorFlow 检查点中加载指定变量的值
        array = tf.train.load_variable(tf_checkpoint_path, full_name)
        # 将数组形状重新调整为原始形状
        array = array.reshape(orginal_shape)

        # 如果变量名包含“kernel”，则对数组进行转置
        if "kernel" in name:
            array = array.transpose()

        # 将 NumPy 数组转换为 PyTorch 张量并返回
        return torch.from_numpy(array)

    # 打印加载模型的信息
    print(f"Loading model based on config from {config_path}...")
    # 从配置文件加载 BertConfig 对象
    config = BertConfig.from_json_file(config_path)
    # 根据配置文件创建 BertForMaskedLM 模型
    model = BertForMaskedLM(config)

    # Layers（层）
    # 遍历模型的所有隐藏层
    for layer_index in range(0, config.num_hidden_layers):
        # 获取当前层的BERT层对象
        layer: BertLayer = model.bert.encoder.layer[layer_index]

        # Self-attention
        # 获取当前层的自注意力机制对象
        self_attn: BertSelfAttention = layer.attention.self

        # 设置查询权重矩阵
        self_attn.query.weight.data = get_encoder_attention_layer_array(
            layer_index, "_query_dense/kernel", self_attn.query.weight.data.shape
        )
        # 设置查询偏置
        self_attn.query.bias.data = get_encoder_attention_layer_array(
            layer_index, "_query_dense/bias", self_attn.query.bias.data.shape
        )
        # 设置键权重矩阵
        self_attn.key.weight.data = get_encoder_attention_layer_array(
            layer_index, "_key_dense/kernel", self_attn.key.weight.data.shape
        )
        # 设置键偏置
        self_attn.key.bias.data = get_encoder_attention_layer_array(
            layer_index, "_key_dense/bias", self_attn.key.bias.data.shape
        )
        # 设置值权重矩阵
        self_attn.value.weight.data = get_encoder_attention_layer_array(
            layer_index, "_value_dense/kernel", self_attn.value.weight.data.shape
        )
        # 设置值偏置
        self_attn.value.bias.data = get_encoder_attention_layer_array(
            layer_index, "_value_dense/bias", self_attn.value.bias.data.shape
        )

        # Self-attention Output
        # 获取自注意力输出对象
        self_output: BertSelfOutput = layer.attention.output

        # 设置自注意力输出的稠密层权重矩阵
        self_output.dense.weight.data = get_encoder_attention_layer_array(
            layer_index, "_output_dense/kernel", self_output.dense.weight.data.shape
        )
        # 设置自注意力输出的稠密层偏置
        self_output.dense.bias.data = get_encoder_attention_layer_array(
            layer_index, "_output_dense/bias", self_output.dense.bias.data.shape
        )

        # 设置自注意力输出的 LayerNorm 层权重
        self_output.LayerNorm.weight.data = get_encoder_layer_array(layer_index, "_attention_layer_norm/gamma")
        # 设置自注意力输出的 LayerNorm 层偏置
        self_output.LayerNorm.bias.data = get_encoder_layer_array(layer_index, "_attention_layer_norm/beta")

        # Intermediate
        # 获取中间层对象
        intermediate: BertIntermediate = layer.intermediate

        # 设置中间层的稠密层权重
        intermediate.dense.weight.data = get_encoder_layer_array(layer_index, "_intermediate_dense/kernel")
        # 设置中间层的稠密层偏置
        intermediate.dense.bias.data = get_encoder_layer_array(layer_index, "_intermediate_dense/bias")

        # Output
        # 获取输出对象
        bert_output: BertOutput = layer.output

        # 设置输出层的稠密层权重
        bert_output.dense.weight.data = get_encoder_layer_array(layer_index, "_output_dense/kernel")
        # 设置输出层的稠密层偏置
        bert_output.dense.bias.data = get_encoder_layer_array(layer_index, "_output_dense/bias")

        # 设置输出层的 LayerNorm 层权重
        bert_output.LayerNorm.weight.data = get_encoder_layer_array(layer_index, "_output_layer_norm/gamma")
        # 设置输出层的 LayerNorm 层偏置
        bert_output.LayerNorm.bias.data = get_encoder_layer_array(layer_index, "_output_layer_norm/beta")

    # Embeddings
    # 设置嵌入层的位置嵌入权重
    model.bert.embeddings.position_embeddings.weight.data = get_encoder_array("_position_embedding_layer/embeddings")
    # 设置嵌入层的标记类型嵌入权重
    model.bert.embeddings.token_type_embeddings.weight.data = get_encoder_array("_type_embedding_layer/embeddings")
    # 设置嵌入层的 LayerNorm 层权重
    model.bert.embeddings.LayerNorm.weight.data = get_encoder_array("_embedding_norm_layer/gamma")
    # 为 BERT 模型的嵌入层归一化参数赋值，使用给定名称获取对应数组
    model.bert.embeddings.LayerNorm.bias.data = get_encoder_array("_embedding_norm_layer/beta")

    # 获取语言模型头部
    lm_head = model.cls.predictions.transform

    # 设置语言模型头部中密集层的权重参数，使用给定名称获取对应数组
    lm_head.dense.weight.data = get_masked_lm_array("dense/kernel")
    # 设置语言模型头部中密集层的偏置参数，使用给定名称获取对应数组
    lm_head.dense.bias.data = get_masked_lm_array("dense/bias")

    # 设置语言模型头部中归一化层的权重参数，使用给定名称获取对应数组
    lm_head.LayerNorm.weight.data = get_masked_lm_array("layer_norm/gamma")
    # 设置语言模型头部中归一化层的偏置参数，使用给定名称获取对应数组
    lm_head.LayerNorm.bias.data = get_masked_lm_array("layer_norm/beta")

    # 设置 BERT 模型的词嵌入层的权重参数，使用给定名称获取对应数组
    model.bert.embeddings.word_embeddings.weight.data = get_masked_lm_array("embedding_table")

    # 创建并设置 BERT 模型的池化层，并赋值权重和偏置参数，使用给定名称获取对应数组
    model.bert.pooler = BertPooler(config=config)
    model.bert.pooler.dense.weight. BertPooler = get_encoder_array("_pooler_layer/kernel")
    model.bert.pooler.dense.bias. BertPooler = get_encoder_array("_pooler_layer/bias")

    # 导出最终的模型
    model.save_pretrained(pytorch_dump_path)

    # 整合测试 - 应该可以加载而不出现任何错误 ;)
    new_model = BertForMaskedLM.from_pretrained(pytorch_dump_path)
    # 打印新模型以进行评估
    print(new_model.eval())

    # 输出成功完成模型转换的消息
    print("Model conversion was done sucessfully!")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定 TensorFlow Token Dropping 检查点路径
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow Token Dropping checkpoint path."
    )
    # 添加命令行参数，指定 BERT 模型对应的配置文件路径
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    # 添加命令行参数，指定输出的 PyTorch 模型路径
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 TensorFlow 检查点转换为 PyTorch 模型
    convert_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
```