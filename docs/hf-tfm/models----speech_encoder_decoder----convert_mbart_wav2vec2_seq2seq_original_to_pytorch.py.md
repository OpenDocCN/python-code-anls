# `.\transformers\models\speech_encoder_decoder\convert_mbart_wav2vec2_seq2seq_original_to_pytorch.py`

```
# 设置文件编码格式为 UTF-8

# 版权声明和许可证信息
# 版权所有 2021 年 HuggingFace 公司团队
# 根据 Apache 许可证第 2.0 版 ("许可证") 许可
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以从以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意
# 根据许可证分发的软件是根据"按原样"的基础分发的
# 无论是明示还是暗示的，都没有任何保修或条件类型的保证
# 有关特定语言的权限和限制，请参阅许可证
"""将 Wav2Vec2 检查点转换为 MBart 检查点"""


# 导入所需的模块
import argparse
import fairseq
import torch
from torch import nn
from transformers import (
    MBart50Tokenizer,
    MBartConfig,
    MBartForCausalLM,
    SpeechEncoderDecoderConfig,
    SpeechEncoderDecoderModel,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    logging,
)

# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义映射字典，用于将 Wav2Vec2 模型的参数在 MBart 模型中对应的位置
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
# 顶层参数键列表
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]

# 递归设置参数值的函数
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 根据键的层次结构设置值
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果权重类型不为空，则获取相应权重的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 断言，确保参数形状一致
    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    # 根据权重类型设置参数值
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
    # 记录日志信息，包括初始化的参数和来源信息
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重到 wav2vec2 模型中
def recursively_load_weights_wav2vec2(fairseq_model, hf_model):
    # 用于存储未使用的权重名称
    unused_weights = []
    # 获取 fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 Hugging Face 模型的特征提取器和适配器
    feature_extractor = hf_model.feature_extractor
    adapter = hf_model.adapter

    # 遍历 fairseq 模型的状态字典中的键值对
    for name, value in fairseq_dict.items():
        # 判断权重是否被使用的标志
        is_used = False
        # 如果名称中包含 "conv_layers" 则表示是卷积层权重
        if "conv_layers" in name:
            # 载入卷积层权重
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        # 如果名称中包含 "adaptor" 或者其他 wav2vec2 特定的键，则表示是适配器权重
        elif any(x in name for x in ["adaptor", "w2v_encoder.proj.", "w2v_proj_ln."]):
            # 载入适配器权重
            load_adapter(name, value, adapter, unused_weights)
            is_used = True
        # 否则，可能是其他 wav2vec2 模型的权重
        else:
            # 遍历映射表中的键值对
            for key, mapped_key in MAPPING.items():
                # 如果键出现在名称中，或者经过处理后的名称符合 wav2vec2 模型中的键
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果映射键中包含通配符 "*"，则替换为对应的层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 确定权重类型
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
                    # 递归设置权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                # 继续下一个映射键
                continue
        # 如果权重未被使用，则添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重名称
    logger.warning(f"Unused weights: {unused_weights}")


# 载入卷积层权重
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层的名称
    name = full_name.split("conv_layers.")[-1]
    # 分割名称
    items = name.split(".")
    # 获取层索引和类型索引
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型索引为 0，则表示是卷积层的权重
    if type_id == 0:
        # 如果名称中包含 "bias"，则表示是偏置项权重
        if "bias" in name:
            # 检查权重的形状是否匹配，并赋值给特征提取器的卷积层的偏置项
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"，则表示是卷积核权重
        elif "weight" in name:
            # 检查权重的形状是否匹配，并赋值给特征提取器的卷积层的卷积核权重
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果类型为 2 且不使用组归一化，或者类型为 2 且为第一层且使用组归一化
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器的卷积层的层归一化偏置数据的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将当前值赋给特征提取器的卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，显示哪一层的特征提取器的层归一化权重被从哪里初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器的卷积层的层归一化权重数据的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将当前值赋给特征提取器的卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，显示哪一层的特征提取器的层归一化权重被从哪里初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 如果不满足以上条件
    else:
        # 将未使用的权重名称添加到未使用权重列表中
        unused_weights.append(full_name)
# 定义一个函数，用于加载适配器参数
def load_adapter(full_name, value, adapter, unused_weights):
    # 根据名称获取适配器层的名称
    name = full_name.split("adaptor.")[-1]
    # 将名称分割成列表
    items = name.split(".")

    # 如果第二个元素是数字，则表示是层的 ID
    if items[1].isdigit():
        layer_id = int(items[1])
    else:
        layer_id = None

    # 如果名称中不包含适配器，则执行以下操作
    if "adaptor" not in full_name:
        # 如果名称中包含 proj_ln，则表示是投影层的 LayerNorm
        if "proj_ln" in full_name:
            # 如果名称中包含 bias，则初始化投影层的偏置
            if "bias" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.bias.data.shape} was found."
                adapter.proj_layer_norm.bias.data = value
                logger.info(f"Adapter proj layer norm bias was initialized from {full_name}.")
            # 如果名称中包含 weight，则初始化投影层的权重
            if "weight" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.weight.data.shape} was found."
                adapter.proj_layer_norm.weight.data = value
        # 如果不是 proj_ln，则表示是投影层
        else:
            # 如果名称中包含 bias，则初始化投影层的偏置
            if "bias" in name:
                assert (
                    value.shape == adapter.proj.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.bias.data.shape} was found."
                adapter.proj.bias.data = value
                logger.info(f"Adapter proj layer bias was initialized from {full_name}.")
            # 如果名称中包含 weight，则初始化投影层的权重
            if "weight" in name:
                assert (
                    value.shape == adapter.proj.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.weight.data.shape} was found."
                adapter.proj.weight.data = value
                logger.info(f"Adapter proj layer weight was initialized from {full_name}.")
    # 如果层 ID 是整数类型，则表示是适配器层
    elif isinstance(layer_id, int):
        # 如果名称中包含 bias，则初始化适配器层的偏置
        if "bias" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.bias.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.bias.data.shape} was found."
            adapter.layers[layer_id].conv.bias.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
        # 如果名称中包含 weight，则初始化适配器层的权重
        elif "weight" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.weight.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.weight.data.shape} was found."
            adapter.layers[layer_id].conv.weight.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
    else:
        # 将未使用的权重添加到列表中
        unused_weights.append(full_name)


# 定义一个函数，根据嵌入层创建线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇大小和嵌入维度大小
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，将词汇大小作为输入维度，嵌入维度作为输出维度，无偏置
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将嵌入层的权重复制给线性层
    lin_layer.weight.data = emb.weight.data
    # 返回线性层
    return lin_layer


# 使用无梯度计算的上下文装饰器定义函数，用于转换 wav2vec2 检查点
@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,  # 存储 PyTorch 模型的文件夹路径
    dict_path,  # 字典文件路径
    config_yaml_path,  # 配置文件路径（YAML 格式）
    encoder_config_path,  # 编码器配置文件路径
    decoder_config_path,  # 解码器配置文件路径
    add_adapter,  # 是否增加适配器的标志
    adapter_kernel_size,  # 适配器的卷积核大小
    adapter_stride,  # 适配器的步长大小
    decoder_start_token_id,  # 解码器起始标记的 ID
    encoder_output_dim,  # 编码器输出维度
    # Copy/paste/tweak model's weights to transformers design.

    # load configs
    # 从预训练模型加载编码器配置，设置适配器和输出维度等参数
    encoder_config = Wav2Vec2Config.from_pretrained(
        encoder_config_path,
        add_adapter=True,
        adapter_stride=adapter_stride,
        adapter_kernel_size=adapter_kernel_size,
        token_token=True,
        output_hidden_size=encoder_output_dim,
    )
    # 从预训练模型加载解码器配置
    decoder_config = MBartConfig.from_pretrained(decoder_config_path)

    # load model
    # 加载模型权重
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides={
            "config_yaml": config_yaml_path,
            "data": "/".join(dict_path.split("/")[:-1]),
            "w2v_path": checkpoint_path,
            "load_pretrained_decoder_from": None,
        },
    )
    # 设置模型为评估模式
    model = model[0].eval()

    # load feature extractor
    # 从预训练模型加载特征提取器配置
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_config_path, token_token=True)

    # set weights for wav2vec2 encoder
    # 设置 wav2vec2 编码器的权重
    hf_encoder = Wav2Vec2Model(encoder_config)
    recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    # load decoder weights
    # 加载解码器权重
    hf_decoder = MBartForCausalLM(decoder_config)
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

    # initialize speech encoder-decoder model
    # 初始化语音编码-解码模型
    hf_wav2vec = SpeechEncoderDecoderModel(encoder=hf_encoder, decoder=hf_decoder)
    hf_wav2vec.config.tie_word_embeddings = False

    # initialize tokenizer
    # 初始化分词器
    tokenizer = MBart50Tokenizer(dict_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # configure and save model and feature extractor
    # 配置并保存模型和特征提取器
    config = hf_wav2vec.config.to_dict()
    config["pad_token_id"] = tokenizer.pad_token_id
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["tokenizer_class"] = "mbart50"
    config["feature_extractor_type"] = "wav2vec2"
    config["decoder_start_token_id"] = tokenizer.eos_token_id
    config["forced_bos_token_id"] = 250004
    config["forced_eos_token_id"] = tokenizer.eos_token_id
    hf_wav2vec.config = SpeechEncoderDecoderConfig.from_dict(config)
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

# 主函数入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_yaml_path", default=None, type=str, help="Path to yaml file of fine-tuned model")
    # 添加一个命令行参数，用于指定编码器的配置文件路径，默认为"facebook/wav2vec2-xls-r-1b"，字符串类型，帮助信息为"Path to hf encoder wav2vec2 checkpoint config"
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-xls-r-1b",
        type=str,
        help="Path to hf encoder wav2vec2 checkpoint config",
    )
    # 添加一个命令行参数，用于指定解码器的配置文件路径，默认为"facebook/mbart-large-50-one-to-many-mmt"，字符串类型，帮助信息为"Path to hf decoder checkpoint config"
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/mbart-large-50-one-to-many-mmt",
        type=str,
        help="Path to hf decoder checkpoint config",
    )
    # 添加一个命令行参数，用于指定是否添加模型适配器层，默认为True，布尔类型，帮助信息为"whethere to add model adapter layers"
    parser.add_argument("--add_adapter", default=True, type=bool, help="whethere to add model adapter layers")
    # 添加一个命令行参数，用于指定适配器层的步长，默认为2，整数类型，帮助信息为"stride of adapter layers"
    parser.add_argument("--adapter_stride", default=2, type=int, help="stride of adapter layers")
    # 添加一个命令行参数，用于指定适配器层的卷积核尺寸，默认为3，整数类型，帮助信息为"kernel size of adapter layers"
    parser.add_argument("--adapter_kernel_size", default=3, type=int, help="kernel size of adapter layers")
    # 添加一个命令行参数，用于指定编码器输出维度，默认为1024，整数类型，帮助信息为"encoder output dim"
    parser.add_argument("--encoder_output_dim", default=1024, type=int, help="encoder output dim")
    # 添加一个命令行参数，用于指定解码器开始标记的ID，默认为250004，整数类型，帮助信息为"`decoder_start_token_id` of model config"
    parser.add_argument("--start_token_id", default=250004, type=int, help="`decoder_start_token_id` of model config")

    # 从命令行参数中解析得到参数对象
    args = parser.parse_args()
    # 将参数传递给函数convert_wav2vec2_checkpoint以执行转换wav2vec2检查点的操作
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.dict_path,
        args.config_yaml_path,
        encoder_config_path=args.encoder_config_path,
        decoder_config_path=args.decoder_config_path,
        add_adapter=args.add_adapter,
        adapter_kernel_size=args.adapter_kernel_size,
        adapter_stride=args.adapter_stride,
        decoder_start_token_id=args.start_token_id,
        encoder_output_dim=args.encoder_output_dim,
    )
```