# `.\transformers\models\speech_encoder_decoder\convert_speech_to_text_wav2vec2_seq2seq_original_to_pytorch.py`

```py
# 设置编码格式为 utf-8
# 声明版权信息
# 导入依赖库
import argparse  # 导入命令行参数解析模块
import json  # 导入JSON模块
import os  # 导入操作系统模块

import fairseq  # 导入fairseq库
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch的神经网络模块

from transformers import (  # 导入transformers库中的指定模块
    Speech2Text2Config,  # 导入语音到文本配置类
    Speech2Text2ForCausalLM,  # 导入用于语音到文本的Causal LM类
    Speech2Text2Tokenizer,  # 导入用于语音到文本的Tokenizer类
    SpeechEncoderDecoderConfig,  # 导入语音编码解码器配置类
    SpeechEncoderDecoderModel,  # 导入语音编码解码器模型类
    Wav2Vec2Config,  # 导入Wav2Vec2配置类
    Wav2Vec2FeatureExtractor,  # 导入用于Wav2Vec2的特征提取器类
    Wav2Vec2Model,  # 导入Wav2Vec2模型类
    logging,  # 导入日志记录信息库
)


logging.set_verbosity_info()  # 设置日志级别为info
logger = logging.get_logger(__name__)  # 获取logger对象

MAPPING = {  # 定义一个映射关系的字典
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
}  # 定义映射关系的字典

TOP_LEVEL_KEYS = [  # 定义顶层键列表
    "lm_head",  # 语言模型头部
    "quantizer.weight_proj",  # 量化器权重项目
    "quantizer.codevectors",  # 量化器码向量
    "project_q",  # 项目Q
    "project_hid",  # 项目隐藏
]

# 递归设置参数值
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):  # 遍历key并分割
        hf_pointer = getattr(hf_pointer, attribute)  # 获取指定属性的值

    if weight_type is not None:  # 如果权重类型不为None
        hf_shape = getattr(hf_pointer, weight_type).shape  # 获取指定属性的形状
    else:
        hf_shape = hf_pointer.shape  # 否则获取hf_pointer的形状

    assert hf_shape == value.shape, (  # 断言hf_shape和value的形状是否相同
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"  # 如果不相同则输出错误信息
    )

    if weight_type == "weight":  # 如果权重类型为"weight"
        hf_pointer.weight.data = value  # 设置hf_pointer的权重数据为value
    elif weight_type == "weight_g":  # 如果权重类型为"weight_g"
        hf_pointer.weight_g.data = value  # 设置hf_pointer的权重g数据为value
    elif weight_type == "weight_v":  # 如果权重类型为"weight_v"
        hf_pointer.weight_v.data = value  # 设置hf_pointer的权重v数据为value
    elif weight_type == "bias":  # 如果权重类型为"bias"
        hf_pointer.bias.data = value  # 设置hf_pointer的偏置数据为value
    else:
        hf_pointer.data = value  # 否则设置hf_pointer的数据为value
    # 使用 logger 模块记录信息，格式化字符串为变量 key、weight_type、full_name 的组合
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
def recursively_load_weights_wav2vec2(fairseq_model, hf_model):
    # 用于存储未使用的权重名称列表
    unused_weights = []
    # 获取 Fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 Hugging Face 模型的特征提取器
    feature_extractor = hf_model.feature_extractor

    # 如果编码器与解码器的维度不同，则使用 proj_weight
    proj_weight = None

    # 遍历 Fairseq 模型的状态字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果名称中包含 "conv_layers"，则加载卷积层
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        # 如果名称以 "proj" 开头，则记录 proj_weight
        elif name.split(".")[0] == "proj":
            proj_weight = fairseq_model.proj
            is_used = True
        else:
            # 遍历映射字典中的键值对
            for key, mapped_key in MAPPING.items():
                # 检查是否有匹配的键名
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 处理映射键名中的通配符和权重类型
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
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
                    # 递归设置 Hugging Face 模型中的权重值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未使用，则将权重名称添加到未使用列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重名称列表
    logger.warning(f"Unused weights: {unused_weights}")

    # 返回 proj_weight
    return proj_weight


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层的名称
    name = full_name.split("conv_layers.")[-1]
    # 解析卷积层的标识符
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 根据类型加载卷积层的权重
    if type_id == 0:
        if "bias" in name:
            # 断言偏置项的形状与特征提取器中对应的卷积层的形状相匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将特征提取器中对应卷积层的偏置项初始化为给定的值
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 断言权重项的形状与特征提取器中对应的卷积层的形状相匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将特征提取器中对应卷积层的权重项初始化为给定的值
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果 type_id 为 2 且不使用组归一化，或者 type_id 为 2 且是第一层且使用组归一化，则执行以下操作
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 检查值的形状是否与特征提取器的卷积层的层归一化偏差数据的形状相匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 设置特征提取器的卷积层的层归一化偏差数据为给定值
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，表示特征提取器的卷积层的层归一化权重已从给定名称初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"
        elif "weight" in name:
            # 检查值的形状是否与特征提取器的卷积层的层归一化权重数据的形状相匹配
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 设置特征提取器的卷积层的层归一化权重数据为给定值
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，表示特征提取器的卷积层的层归一化权重已从给定名称初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 否则
    else:
        # 将未使用的权重名称添加到列表中
        unused_weights.append(full_name)
# 根据输入的嵌入层创建一个线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇大小和嵌入维度
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，输入大小为词汇大小，输出大小为嵌入维度，无偏置
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 设置线性层的权重与嵌入层的权重相同
    lin_layer.weight.data = emb.weight.data
    # 返回线性层
    return lin_layer

# 创建词汇字典
def create_vocab_dict(dict_path):
    # 打开字典文件并读取所有行
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # 提取每行中的单词
        words = [line.split(" ")[0] for line in lines]

    # 获取单词数量
    num_words = len(words)

    # 预定义一些特殊标记的索引
    vocab_dict = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
    }

    # 创建单词到索引的映射
    vocab_dict.update(dict(zip(words, range(4, num_words + 4)))
    # 返回词汇字典
    return vocab_dict

# 转换模型的权重到transformers设计
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
    Copy/paste/tweak model's weights to transformers design.
    """
    # 从预训练配置中创建编码器和解码器配置
    encoder_config = Wav2Vec2Config.from_pretrained(encoder_config_path)
    decoder_config = Speech2Text2Config.from_pretrained(
        decoder_config_path, vocab_size=vocab_size, decoder_layers=num_decoder_layers, do_stable_layer_norm=True
    )

    # 创建特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=True,
    )

    # 加载模型并设为评估模式
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
    )
    model = model[0].eval()

    # 设置权重为wav2vec2编码器
    hf_encoder = Wav2Vec2Model(encoder_config)
    projection_layer = recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    hf_decoder = Speech2Text2ForCausalLM(decoder_config)
    # 加载解码器权重
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)

    # 设置输出线性层
    unexpected_keys.remove("embed_out")
    hf_decoder.lm_head.weight = nn.Parameter(model.decoder.embed_out.detach())

    # layer norm初始化为单位矩阵所以仍然好
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

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

    # 创建并保存tokenizer
    tokenizer = Speech2Text2Tokenizer(os.path.join(pytorch_dump_folder_path, "vocab.json"))
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # 将配置转换为字典
    config = hf_wav2vec.config.to_dict()
    # 将 tokenizer 的 pad token id 存入 config 中
    config["pad_token_id"] = tokenizer.pad_token_id
    # 将 tokenizer 的 bos token id 存入 config 中
    config["bos_token_id"] = tokenizer.bos_token_id
    # 将 tokenizer 的 eos token id 存入 config 中
    config["eos_token_id"] = tokenizer.eos_token_id
    # 设定 tokenizer 类型为 "speech_to_text_2"
    config["tokenizer_class"] = "speech_to_text_2"
    # 设定特征提取器类型为 "wav2vec2"
    config["feature_extractor_type"] = "wav2vec2"

    # 从给定的配置项创建 SpeechEncoderDecoderConfig 对象
    hf_wav2vec.config = SpeechEncoderDecoderConfig.from_dict(config)

    # 将 hf_wav2vec 的预训练模型保存到指定路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    # 将特征提取器保存到指定路径
    feature_extractor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被当作主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加用于接收 PyTorch 模型输出路径的参数
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加用于接收 fairseq 模型检查点路径的参数
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加用于接收微调模型字典路径的参数
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加用于接收 HF 编码器 wav2vec2 检查点配置路径的参数
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-large-lv60",
        type=str,
        help="Path to hf encoder wav2vec2 checkpoint config",
    )
    # 添加用于接收 HF 解码器 s2t 检查点配置路径的参数
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/s2t-small-mustc-en-fr-st",
        type=str,
        help="Path to hf decoder s2t checkpoint config",
    )
    # 添加用于接收解码器词汇表大小的参数
    parser.add_argument("--vocab_size", default=10224, type=int, help="Vocab size of decoder")
    # 添加用于接收解码器层数的参数
    parser.add_argument("--num_decoder_layers", default=7, type=int, help="Number of decoder layers")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_wav2vec2_checkpoint 函数，将 fairseq wav2vec2 检查点转换为 PyTorch 模型
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
```