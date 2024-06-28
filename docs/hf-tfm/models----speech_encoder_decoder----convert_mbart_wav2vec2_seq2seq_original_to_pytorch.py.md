# `.\models\speech_encoder_decoder\convert_mbart_wav2vec2_seq2seq_original_to_pytorch.py`

```py
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块

import fairseq  # 导入fairseq库
import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch的神经网络模块

from transformers import (  # 从transformers库中导入以下模块和类
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

logging.set_verbosity_info()  # 设置日志级别为INFO
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义一个字典，用于将Wav2Vec2模型的参数映射到Hugging Face的命名空间
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

# 定义顶层的关键字列表
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]

# 递归设置函数，用于将权重设置到指定的Hugging Face指针中
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 根据键名逐级获取Hugging Face指针
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 检查Hugging Face指针的形状是否与待设置的值相匹配
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 断言确认形状匹配，否则抛出错误
    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    # 根据权重类型设置相应的值到Hugging Face指针中
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
    # 记录信息到日志中，描述初始化操作的详细情况，包括属性名（如果提供了权重类型）和来源的完整名称。
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载 Fairseq 模型的权重到 Hugging Face 模型中
def recursively_load_weights_wav2vec2(fairseq_model, hf_model):
    # 未使用的权重列表
    unused_weights = []
    # 获取 Fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取 Hugging Face 模型的特征提取器和适配器
    feature_extractor = hf_model.feature_extractor
    adapter = hf_model.adapter

    # 遍历 Fairseq 模型状态字典中的每个键值对
    for name, value in fairseq_dict.items():
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
        # 如果名称中包含任何 "adaptor", "w2v_encoder.proj.", "w2v_proj_ln."
        elif any(x in name for x in ["adaptor", "w2v_encoder.proj.", "w2v_proj_ln."]):
            # 调用加载适配器的函数
            load_adapter(name, value, adapter, unused_weights)
            is_used = True
        else:
            # 遍历 MAPPING 字典中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果键存在于名称中或者其去掉 "w2v_model." 后的部分等于名称的第一个部分
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果映射键包含 "*"，则替换为层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称中的关键字确定权重类型
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
                    # 递归设置 Hugging Face 模型的对应权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果没有使用此权重，则添加到未使用列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重列表到日志
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层的名称
    name = full_name.split("conv_layers.")[-1]
    # 拆分名称
    items = name.split(".")
    # 获取层和类型索引
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果类型索引为 0
    if type_id == 0:
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 断言检查值的形状与特征提取器中相应卷积层的偏置数据形状是否一致
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            # 将值赋给特征提取器中对应卷积层的偏置数据
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"
        elif "weight" in name:
            # 断言检查值的形状与特征提取器中相应卷积层的权重数据形状是否一致
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中对应卷积层的权重数据
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果 type_id 等于 2，并且不使用组归一化，或者 type_id 等于 2，且 layer_id 等于 0 并且使用组归一化
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果变量名中包含 "bias"
        if "bias" in name:
            # 断言当前值的形状与特征提取器中卷积层的层归一化偏置数据的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 将值赋给特征提取器中卷积层的层归一化偏置数据
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，指示层归一化权重已从指定变量名初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果变量名中包含 "weight"
        elif "weight" in name:
            # 断言当前值的形状与特征提取器中卷积层的层归一化权重数据的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 将值赋给特征提取器中卷积层的层归一化权重数据
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，指示层归一化权重已从指定变量名初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 将未使用的权重变量名添加到未使用权重列表中
        unused_weights.append(full_name)
# 定义一个函数，用于加载适配器（adapter）的权重信息
def load_adapter(full_name, value, adapter, unused_weights):
    # 从完整的名称中提取适配器的名称
    name = full_name.split("adaptor.")[-1]
    # 将名称按"."分割为列表
    items = name.split(".")

    # 判断第二个元素是否为数字，如果是则转换为整数，否则设为None
    if items[1].isdigit():
        layer_id = int(items[1])
    else:
        layer_id = None

    # 如果完整名称中不包含 "adaptor"
    if "adaptor" not in full_name:
        # 如果包含 "proj_ln"，则是投影层规范化（layer norm）
        if "proj_ln" in full_name:
            # 如果名称中包含 "bias"，则进行断言和赋值操作
            if "bias" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.bias.data.shape} was found."
                adapter.proj_layer_norm.bias.data = value
                logger.info(f"Adapter proj layer norm bias was initialized from {full_name}.")
            # 如果名称中包含 "weight"，则进行断言和赋值操作
            if "weight" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.weight.data.shape} was found."
                adapter.proj_layer_norm.weight.data = value
        else:
            # 否则是投影层
            # 如果名称中包含 "bias"，则进行断言和赋值操作
            if "bias" in name:
                assert (
                    value.shape == adapter.proj.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.bias.data.shape} was found."
                adapter.proj.bias.data = value
                logger.info(f"Adapter proj layer bias was initialized from {full_name}.")
            # 如果名称中包含 "weight"，则进行断言和赋值操作
            if "weight" in name:
                assert (
                    value.shape == adapter.proj.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.weight.data.shape} was found."
                adapter.proj.weight.data = value
    # 如果 layer_id 是整数
    elif isinstance(layer_id, int):
        # 如果名称中包含 "bias"，则进行断言和赋值操作
        if "bias" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.bias.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.bias.data.shape} was found."
            adapter.layers[layer_id].conv.bias.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
        # 如果名称中包含 "weight"，则进行断言和赋值操作
        elif "weight" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.weight.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.weight.data.shape} was found."
            adapter.layers[layer_id].conv.weight.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
    else:
        # 如果既不是 "adaptor" 开头，也没有整数的 layer_id，将 full_name 添加到未使用的权重列表中
        unused_weights.append(full_name)


# 根据嵌入层（emb）创建一个线性层，并将其权重初始化为嵌入层的权重
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇大小和嵌入维度
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，不带偏置
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重数据初始化为嵌入层的权重数据
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# 使用无梯度的上下文装饰器定义一个函数，用于将 wav2vec2 的检查点转换
@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,        # PyTorch模型保存的文件夹路径
    dict_path,                       # 词典文件路径
    config_yaml_path,                # 配置文件（YAML格式）路径
    encoder_config_path,             # 编码器配置文件路径
    decoder_config_path,             # 解码器配置文件路径
    add_adapter,                     # 是否添加适配器（布尔值）
    adapter_kernel_size,             # 适配器的卷积核大小
    adapter_stride,                  # 适配器的步幅大小
    decoder_start_token_id,          # 解码器起始标记ID
    encoder_output_dim,              # 编码器的输出维度
def copy_weights_to_transformers_model(
    pytorch_dump_folder_path,
    checkpoint_path,
    dict_path,
    config_yaml_path,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    # load configs
    # 从预训练配置文件加载 Wav2Vec2Config
    encoder_config = Wav2Vec2Config.from_pretrained(
        encoder_config_path,
        add_adapter=True,
        adapter_stride=adapter_stride,
        adapter_kernel_size=adapter_kernel_size,
        token_token=True,
        output_hidden_size=encoder_output_dim,
    )
    
    # 从预训练配置文件加载 MBartConfig
    decoder_config = MBartConfig.from_pretrained(decoder_config_path)

    # load model
    # 使用 fairseq 提供的函数加载模型集合和任务
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides={
            "config_yaml": config_yaml_path,
            "data": "/".join(dict_path.split("/")[:-1]),
            "w2v_path": checkpoint_path,
            "load_pretrained_decoder_from": None,
        },
    )
    model = model[0].eval()  # 设置模型为评估模式

    # load feature extractor
    # 从预训练配置文件加载 Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_config_path, token_token=True)

    # set weights for wav2vec2 encoder
    # 使用 Wav2Vec2Model 构建 hf_encoder
    hf_encoder = Wav2Vec2Model(encoder_config)

    # 递归加载 wav2vec2 encoder 的权重
    recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    # load decoder weights
    # 使用 MBartForCausalLM 构建 hf_decoder
    hf_decoder = MBartForCausalLM(decoder_config)
    # 加载模型的 decoder 权重，并记录缺失和意外的键
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

    # 构建 SpeechEncoderDecoderModel
    hf_wav2vec = SpeechEncoderDecoderModel(encoder=hf_encoder, decoder=hf_decoder)
    hf_wav2vec.config.tie_word_embeddings = False  # 设置不共享词嵌入权重

    # 初始化 MBart50Tokenizer
    tokenizer = MBart50Tokenizer(dict_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)  # 保存 tokenizer 到指定路径

    # 构建配置字典并设置相关 token id
    config = hf_wav2vec.config.to_dict()
    config["pad_token_id"] = tokenizer.pad_token_id
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["tokenizer_class"] = "mbart50"
    config["feature_extractor_type"] = "wav2vec2"
    config["decoder_start_token_id"] = tokenizer.eos_token_id
    config["forced_bos_token_id"] = 250004
    config["forced_eos_token_id"] = tokenizer.eos_token_id

    # 从配置字典构建 SpeechEncoderDecoderConfig
    hf_wav2vec.config = SpeechEncoderDecoderConfig.from_dict(config)

    # 将模型保存到指定路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    # 保存 feature extractor 到指定路径
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_yaml_path", default=None, type=str, help="Path to yaml file of fine-tuned model")
    # 添加一个接收命令行参数的选项，用于指定编码器配置文件的路径
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-xls-r-1b",
        type=str,
        help="Path to hf encoder wav2vec2 checkpoint config",
    )
    
    # 添加一个接收命令行参数的选项，用于指定解码器配置文件的路径
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/mbart-large-50-one-to-many-mmt",
        type=str,
        help="Path to hf decoder checkpoint config",
    )
    
    # 添加一个接收命令行参数的选项，指定是否添加模型适配器层，默认为 True
    parser.add_argument("--add_adapter", default=True, type=bool, help="whether to add model adapter layers")
    
    # 添加一个接收命令行参数的选项，用于指定模型适配器层的步幅，默认为 2
    parser.add_argument("--adapter_stride", default=2, type=int, help="stride of adapter layers")
    
    # 添加一个接收命令行参数的选项，用于指定模型适配器层的卷积核大小，默认为 3
    parser.add_argument("--adapter_kernel_size", default=3, type=int, help="kernel size of adapter layers")
    
    # 添加一个接收命令行参数的选项，用于指定编码器输出的维度，默认为 1024
    parser.add_argument("--encoder_output_dim", default=1024, type=int, help="encoder output dim")
    
    # 添加一个接收命令行参数的选项，用于指定解码器启动令牌的ID，默认为 250004
    parser.add_argument("--start_token_id", default=250004, type=int, help="`decoder_start_token_id` of model config")
    
    # 解析命令行参数并将其存储在 args 变量中
    args = parser.parse_args()
    
    # 调用函数 convert_wav2vec2_checkpoint，传递命令行参数中的各个配置项作为参数
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