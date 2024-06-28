# `.\models\speech_encoder_decoder\convert_speech_to_text_wav2vec2_seq2seq_original_to_pytorch.py`

```py
# coding=utf-8
# 上面是指定代码文件的编码格式为 UTF-8

# 版权声明和许可证信息，告知代码的版权和使用许可
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Wav2Vec2 checkpoint."""

# 导入需要的模块
import argparse  # 用于命令行解析
import json  # 用于处理 JSON 格式的数据
import os  # 用于处理操作系统相关的功能

import fairseq  # 导入 fairseq 库，用于处理序列到序列任务
import torch  # 导入 PyTorch 深度学习库
from torch import nn  # 导入 PyTorch 的神经网络模块

# 导入 transformers 库中的相关模块和类
from transformers import (
    Speech2Text2Config,  # 语音到文本模型的配置类
    Speech2Text2ForCausalLM,  # 语音到文本模型的主类
    Speech2Text2Tokenizer,  # 语音到文本模型的分词器
    SpeechEncoderDecoderConfig,  # 语音编码解码器的配置类
    SpeechEncoderDecoderModel,  # 语音编码解码器的模型类
    Wav2Vec2Config,  # Wav2Vec2 模型的配置类
    Wav2Vec2FeatureExtractor,  # Wav2Vec2 模型的特征提取器
    Wav2Vec2Model,  # Wav2Vec2 模型的主类
    logging,  # 日志记录模块
)

logging.set_verbosity_info()  # 设置日志记录级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 定义一个映射字典，用于将原始模型的参数名映射到转换后的模型参数名
MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}
# 定义顶级键列表，这些键表示模型的顶级组件或属性
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]

# 定义一个递归设置函数，用于将转换后的模型参数值设置到对应的位置
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 获取当前属性的形状信息
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 断言检查当前属性的形状与期望的值的形状是否一致
    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    # 根据权重类型设置对应的值到当前属性中
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        hf_pointer.data = value
    # 记录信息日志，指示某个变量的初始化情况以及其来源
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重到 wav2vec2 模型中
def recursively_load_weights_wav2vec2(fairseq_model, hf_model):
    # 未使用的权重列表
    unused_weights = []
    # 获取 fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 hf_model 的特征提取器
    feature_extractor = hf_model.feature_extractor

    # 如果编码器与解码器维度不同，则使用 proj_weight
    proj_weight = None

    # 遍历 fairseq 模型的状态字典
    for name, value in fairseq_dict.items():
        # 标记是否被使用的标志
        is_used = False
        # 如果名称中包含 "conv_layers"
        if "conv_layers" in name:
            # 调用加载卷积层的函数
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        # 否则，如果名称以 "proj" 开头
        elif name.split(".")[0] == "proj":
            # 获取 proj_weight
            proj_weight = fairseq_model.proj
            is_used = True
        # 否则，遍历 MAPPING 中的键值对
        else:
            for key, mapped_key in MAPPING.items():
                # 如果 key 出现在名称中或者 key 的一部分出现在名称的开头
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果 mapped_key 中包含通配符 "*"
                    if "*" in mapped_key:
                        # 获取层索引
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称中的关键词确定权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        weight_type = "weight"
                    else:
                        weight_type = None
                    # 递归地设置 hf_model 的权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未被使用，则添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重
    logger.warning(f"Unused weights: {unused_weights}")

    # 返回 proj_weight
    return proj_weight


# 加载卷积层参数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    # 拆分名称
    items = name.split(".")
    # 获取层 ID 和类型 ID
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型 ID 为 0
    if type_id == 0:
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 断言值的形状与特征提取器中卷积层的偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将值设置为特征提取器中卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 否则，如果名称中包含 "weight"
        elif "weight" in name:
            # 断言值的形状与特征提取器中卷积层的权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将值设置为特征提取器中卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果 type_id 等于 2 并且不使用 group normalization，或者 type_id 等于 2 并且 layer_id 等于 0 并且使用 group normalization，则执行以下操作
    if (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果变量名中包含 "bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器 conv_layers[layer_id] 的层归一化偏置数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将值赋给特征提取器 conv_layers[layer_id] 的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，指示层归一化权重已从特定名称初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果变量名中包含 "weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器 conv_layers[layer_id] 的层归一化权重数据形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将值赋给特征提取器 conv_layers[layer_id] 的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，指示层归一化权重已从特定名称初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 如果不满足以上条件
    else:
        # 将未使用的权重名称添加到未使用权重列表中
        unused_weights.append(full_name)
# 从预训练的嵌入层创建一个线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇表大小和嵌入维度
    vocab_size, emb_size = emb.weight.shape
    # 创建一个没有偏置的线性层，输入大小为词汇表大小，输出大小为嵌入维度
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重设置为输入嵌入层的权重
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# 创建一个词汇字典并返回
def create_vocab_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        # 读取字典文件中的所有行
        lines = f.readlines()
        # 提取每行的第一个单词作为词汇列表
        words = [line.split(" ")[0] for line in lines]

    num_words = len(words)

    # 预定义的特殊标记和其索引
    vocab_dict = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
    }

    # 更新词汇字典，将从字典文件中读取的单词和索引添加进去
    vocab_dict.update(dict(zip(words, range(4, num_words + 4))))
    return vocab_dict


# 转换Wav2Vec2模型的检查点到Transformers设计
@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    dict_path,
    encoder_config_path,
    decoder_config_path,
    vocab_size,
    num_decoder_layers,
):
    """
    复制/粘贴/调整模型权重到Transformers设计。
    """
    # 加载Wav2Vec2的编码器和解码器配置
    encoder_config = Wav2Vec2Config.from_pretrained(encoder_config_path)
    decoder_config = Speech2Text2Config.from_pretrained(
        decoder_config_path, vocab_size=vocab_size, decoder_layers=num_decoder_layers, do_stable_layer_norm=True
    )

    # 创建Wav2Vec2特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=True,
    )

    # 加载模型的权重并设置为评估模式
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
    )
    model = model[0].eval()

    # 设置Wav2Vec2编码器的权重
    hf_encoder = Wav2Vec2Model(encoder_config)
    projection_layer = recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    # 创建Speech2Text2的解码器模型
    hf_decoder = Speech2Text2ForCausalLM(decoder_config)
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)

    # 设置输出线性层的权重
    unexpected_keys.remove("embed_out")
    hf_decoder.lm_head.weight = nn.Parameter(model.decoder.embed_out.detach())

    # layer norm 初始化为单位矩阵，因此保持它不变是可以的
    logger.warning(f"加载解码器权重时缺少以下键: {missing_keys}")
    logger.warning(f"加载解码器权重时出现以下意外的键: {unexpected_keys}")

    # 创建包含编码器和解码器的模型
    hf_wav2vec = SpeechEncoderDecoderModel(encoder=hf_encoder, decoder=hf_decoder)
    hf_wav2vec.config.tie_word_embeddings = False

    # 添加投影层
    hf_wav2vec.enc_to_dec_proj.weight = nn.Parameter(projection_layer.weight)
    hf_wav2vec.enc_to_dec_proj.bias = nn.Parameter(projection_layer.bias)

    # 创建词汇字典
    vocab_dict = create_vocab_dict(dict_path)

    # 将词汇字典保存为JSON文件
    with open(os.path.join(pytorch_dump_folder_path, "vocab.json"), "w") as fp:
        json.dump(vocab_dict, fp)

    # 保存Tokenizer的预训练文件
    tokenizer = Speech2Text2Tokenizer(os.path.join(pytorch_dump_folder_path, "vocab.json"))
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # 将配置保存为字典格式
    config = hf_wav2vec.config.to_dict()
    # 将tokenizer的pad_token_id赋给config字典中的pad_token_id键
    config["pad_token_id"] = tokenizer.pad_token_id
    # 将tokenizer的bos_token_id赋给config字典中的bos_token_id键
    config["bos_token_id"] = tokenizer.bos_token_id
    # 将tokenizer的eos_token_id赋给config字典中的eos_token_id键
    config["eos_token_id"] = tokenizer.eos_token_id
    # 将字符串"speech_to_text_2"赋给config字典中的tokenizer_class键
    config["tokenizer_class"] = "speech_to_text_2"
    # 将字符串"wav2vec2"赋给config字典中的feature_extractor_type键
    config["feature_extractor_type"] = "wav2vec2"

    # 使用SpeechEncoderDecoderConfig类从config字典创建hf_wav2vec的配置对象
    hf_wav2vec.config = SpeechEncoderDecoderConfig.from_dict(config)

    # 将hf_wav2vec模型保存到指定路径pytorch_dump_folder_path
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    # 将feature_extractor对象保存到指定路径pytorch_dump_folder_path
    feature_extractor.save_pretrained(pytorch_dump_folder_path)
# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定输出的 PyTorch 模型路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数，指定 Fairseq 检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数，指定经过微调的模型词典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加命令行参数，指定 HF 编码器 wav2vec2 检查点配置路径，默认为预设值
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-large-lv60",
        type=str,
        help="Path to hf encoder wav2vec2 checkpoint config",
    )
    # 添加命令行参数，指定 HF 解码器 s2t 检查点配置路径，默认为预设值
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/s2t-small-mustc-en-fr-st",
        type=str,
        help="Path to hf decoder s2t checkpoint config",
    )
    # 添加命令行参数，指定解码器的词汇表大小，默认为预设值
    parser.add_argument("--vocab_size", default=10224, type=int, help="Vocab size of decoder")
    # 添加命令行参数，指定解码器的层数，默认为预设值
    parser.add_argument("--num_decoder_layers", default=7, type=int, help="Number of decoder layers")

    # 解析命令行参数并将其存储到 args 对象中
    args = parser.parse_args()

    # 调用 convert_wav2vec2_checkpoint 函数，传入命令行参数中指定的各个路径和配置
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.dict_path,
        encoder_config_path=args.encoder_config_path,
        decoder_config_path=args.decoder_config_path,
        vocab_size=args.vocab_size,
        num_decoder_layers=args.num_decoder_layers,
    )
```