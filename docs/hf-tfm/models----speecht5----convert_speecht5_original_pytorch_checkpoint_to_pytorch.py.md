# `.\transformers\models\speecht5\convert_speecht5_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文档编码为 UTF-8
# 版权声明
#
# 引入命令行参数解析模块
import argparse
# 引入 PyTorch 模块
import torch
# 引入 transformers 里的 SpeechT5 有关模块
from transformers import (
    SpeechT5Config,
    SpeechT5FeatureExtractor,
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    SpeechT5Tokenizer,
    logging,
)
# 引入 transformers 里通用的 tokenization_utils 模块的 AddedToken 类
from transformers.tokenization_utils import AddedToken
# 设置日志打印级别为 info
# 获取日志记录器
# 下面几个字典用于存储暂时不兼容的参数的映射，用于后续的模型参数迁移
# MAPPING_SPEECH_ENCODER_PRENET 字典
# MAPPING_TEXT_ENCODER_PRENET 字典
# MAPPING_SPEECH_DECODER_PRENET 字典
# MAPPING_SPEECH_DECODER_POSTNET 字典
    # 定义键为 "speech_decoder_postnet.postnet.postnet.3.0"，值为 "speech_decoder_postnet.layers.3.conv" 的映射关系
    "speech_decoder_postnet.postnet.postnet.3.0": "speech_decoder_postnet.layers.3.conv",
    # 定义键为 "speech_decoder_postnet.postnet.postnet.3.1"，值为 "speech_decoder_postnet.layers.3.batch_norm" 的映射关系
    "speech_decoder_postnet.postnet.postnet.3.1": "speech_decoder_postnet.layers.3.batch_norm",
    # 定义键为 "speech_decoder_postnet.postnet.postnet.4.0"，值为 "speech_decoder_postnet.layers.4.conv" 的映射关系
    "speech_decoder_postnet.postnet.postnet.4.0": "speech_decoder_postnet.layers.4.conv",
    # 定义键为 "speech_decoder_postnet.postnet.postnet.4.1"，值为 "speech_decoder_postnet.layers.4.batch_norm" 的映射关系
    "speech_decoder_postnet.postnet.postnet.4.1": "speech_decoder_postnet.layers.4.batch_norm",
# 定义映射关系，将文本到语音解码器中的参数转换为语音到文本解码器中的参数
MAPPING_TEXT_DECODER_PRENET = {
    "text_decoder_prenet.embed_tokens": "speecht5.decoder.prenet.embed_tokens",
}

# 定义映射关系，将文本到语音解码器中的参数转换为文本到文本解码器中的参数
MAPPING_TEXT_DECODER_POSTNET = {
    "text_decoder_postnet.output_projection": "text_decoder_postnet.lm_head",
}

# 定义映射关系，将编码器参数转换为文本到语音或语音到文本解码器中的参数
MAPPING_ENCODER = {
    "encoder.layers.*.self_attn.k_proj": "speecht5.encoder.wrapped_encoder.layers.*.attention.k_proj",
    "encoder.layers.*.self_attn.v_proj": "speecht5.encoder.wrapped_encoder.layers.*.attention.v_proj",
    "encoder.layers.*.self_attn.q_proj": "speecht5.encoder.wrapped_encoder.layers.*.attention.q_proj",
    "encoder.layers.*.self_attn.out_proj": "speecht5.encoder.wrapped_encoder.layers.*.attention.out_proj",
    "encoder.layers.*.self_attn_layer_norm": "speecht5.encoder.wrapped_encoder.layers.*.layer_norm",
    "encoder.layers.*.fc1": "speecht5.encoder.wrapped_encoder.layers.*.feed_forward.intermediate_dense",
    "encoder.layers.*.fc2": "speecht5.encoder.wrapped_encoder.layers.*.feed_forward.output_dense",
    "encoder.layers.*.final_layer_norm": "speecht5.encoder.wrapped_encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "speecht5.encoder.wrapped_encoder.layer_norm",
    "encoder.pos_emb.pe_k": "speecht5.encoder.wrapped_encoder.embed_positions.pe_k",
}

# 定义映射关系，将解码器参数转换为文本到语音或语音到文本解码器中的参数
MAPPING_DECODER = {
    "decoder.layers.*.self_attn.k_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.k_proj",
    "decoder.layers.*.self_attn.v_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.v_proj",
    "decoder.layers.*.self_attn.q_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.q_proj",
    "decoder.layers.*.self_attn.out_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.out_proj",
    "decoder.layers.*.self_attn_layer_norm": "speecht5.decoder.wrapped_decoder.layers.*.self_attn_layer_norm",
    "decoder.layers.*.encoder_attn.k_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.k_proj",
    "decoder.layers.*.encoder_attn.v_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.v_proj",
    "decoder.layers.*.encoder_attn.q_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.q_proj",
    "decoder.layers.*.encoder_attn.out_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.out_proj",
    "decoder.layers.*.encoder_attn_layer_norm": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn_layer_norm",
    "decoder.layers.*.fc1": "speecht5.decoder.wrapped_decoder.layers.*.feed_forward.intermediate_dense",
    "decoder.layers.*.fc2": "speecht5.decoder.wrapped_decoder.layers.*.feed_forward.output_dense",
    "decoder.layers.*.final_layer_norm": "speecht5.decoder.wrapped_decoder.layers.*.final_layer_norm",
}

# 定义文本到语音解码器的映射关系，包括文本到语音编码器的参数和文本到语音解码器的参数
MAPPING_S2T = {
    **MAPPING_SPEECH_ENCODER_PRENET,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
    **MAPPING_TEXT_DECODER_PRENET,
    **MAPPING_TEXT_DECODER_POSTNET,
}

# 定义语音到文本解码器的映射关系，包括语音到文本编码器的参数和语音到文本解码器的参数
MAPPING_T2S = {
    **MAPPING_TEXT_ENCODER_PRENET,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
    **MAPPING_SPEECH_DECODER_PRENET,
    **MAPPING_SPEECH_DECODER_POSTNET,
}
# 创建一个包含多个字典的大字典，将MAPPING_SPEECH_ENCODER_PRENET、MAPPING_ENCODER、MAPPING_DECODER、MAPPING_SPEECH_DECODER_PRENET、MAPPING_SPEECH_DECODER_POSTNET合并
MAPPING_S2S = {
    **MAPPING_SPEECH_ENCODER_PRENET,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
    **MAPPING_SPEECH_DECODER_PRENET,
    **MAPPING_SPEECH_DECODER_POSTNET,
}

# 创建一个空列表
TOP_LEVEL_KEYS = []

# 创建一个包含多个字符串的列表，指明要忽略的特定键名
IGNORE_KEYS = [
    "encoder.version",
    "encoder.layers.*.norm_k.weight",
    "encoder.layers.*.norm_k.bias",
    "decoder.version",
    "decoder.layers.*.norm_k.weight",
    "decoder.layers.*.norm_k.bias",
    "decoder.pos_emb.pe_k",
    "speech_encoder_prenet.embed_positions._float_tensor",
    "text_decoder_prenet.embed_positions._float_tensor",
]

# 创建一个包含IGNORE_KEYS的列表，并添加额外的忽略键
IGNORE_KEYS_S2T = IGNORE_KEYS + [
    "encoder.proj",
    "text_encoder_prenet.*",
    "speech_decoder_prenet.*",
    "speech_decoder_postnet.*",
]

# 创建一个包含IGNORE_KEYS的列表，并添加额外的忽略键
IGNORE_KEYS_T2S = IGNORE_KEYS + [
    "encoder.proj",
    "speech_encoder_prenet.*",
    "text_decoder_prenet.*",
    "text_decoder_postnet.*",
]

# 创建一个包含IGNORE_KEYS的列表，并添加额外的忽略键
IGNORE_KEYS_S2S = IGNORE_KEYS + [
    "encoder.proj",
    "text_encoder_prenet.*",
    "text_decoder_prenet.*",
    "text_decoder_postnet.*",
]

# 递归设置权重
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 逐级获取属性
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果权重类型不为空，获取权重的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 检查形状是否与给定值的形状相等，如果不相等，则抛出ValueError
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型设置权重的值
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    else:
        hf_pointer.data = value

    # 记录权重初始化的信息
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")

# 检查是否应忽略指定的键名
def should_ignore(name, ignore_keys):
    for key in ignore_keys:
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False

# 递归加载权重
def recursively_load_weights(fairseq_dict, hf_model, task):
    # 创建一个空列表，用于记录未使用的权重
    unused_weights = []

    # 如果任务为"s2t"，则进行相应设置
    if task == "s2t":
        feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
        MAPPING = MAPPING_S2T
        IGNORE_KEYS = IGNORE_KEYS_S2T
    # 如果任务是文本转语音
    elif task == "t2s":
        # 特征编码器设置为空
        feature_encoder = None
        # 使用文本到语音的映射和忽略的键
        MAPPING = MAPPING_T2S
        IGNORE_KEYS = IGNORE_KEYS_T2S
    # 如果任务是语音到语音
    elif task == "s2s":
        # 设置特征编码器为预训练网络的音频t5编码器
        feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
        # 使用语音到语音的映射和忽略的键
        MAPPING = MAPPING_S2S
        IGNORE_KEYS = IGNORE_KEYS_S2S
    else:
        # 如果任务不是文本转语音或语音到语音，则报错
        raise ValueError(f"Unsupported task: {task}")

    # 遍历fairseq_dict中的每个键值对
    for name, value in fairseq_dict.items():
        # 如果应该忽略该键，则记录日志并且继续下一个键值对
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        # 标记该键是否被使用
        is_used = False
        # 如果键名包含"conv_layers"
        if "conv_layers" in name:
            # 装载卷积层
            load_conv_layer(
                name,
                value,
                feature_encoder,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            # 标记该键被使用
            is_used = True
        else:
            # 遍历MAPPING中的每个键值对
            for key, mapped_key in MAPPING.items():
                # 如果键中包含"*"
                if "*" in key:
                    # 拆分key为前缀和后缀
                    prefix, suffix = key.split(".*.")
                    # 如果名称中包含前缀和后缀
                    if prefix in name and suffix in name:
                        key = suffix
                
                # 如果键在名称中存在
                if key in name:
                    # 标记该键被使用
                    is_used = True
                    # 如果mapped_key中包含"*"
                    if "*" in mapped_key:
                        # 提取层索引并替换"*"
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称设置权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        weight_type = "weight"
                    elif "running_mean" in name:
                        weight_type = "running_mean"
                    elif "running_var" in name:
                        weight_type = "running_var"
                    elif "num_batches_tracked" in name:
                        weight_type = "num_batches_tracked"
                    else:
                        weight_type = None
                    # 递归地设置权重值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果该键未被使用，将其添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重
    logger.warning(f"Unused weights: {unused_weights}")
# 加载卷积层权重的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 提取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    # 将名称拆分成列表
    items = name.split(".")
    # 提取层编号和类型编号
    layer_id = int(items[0])
    type_id = int(items[1])

    # 处理卷积层的偏置项
    if type_id == 0:
        # 如果名称中包含偏置项
        if "bias" in name:
            # 检查值的形状是否与模型中的形状相匹配
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            # 更新模型的偏置项
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 处理卷积层的权重
        elif "weight" in name:
            # 检查值的形状是否与模型中的形状相匹配
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            # 更新模型的权重
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 处理组归一化层的情况
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 处理归一化层的偏置项
        if "bias" in name:
            # 检查值的形状是否与模型中的形状相匹配
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            # 更新模型的偏置项
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 处理归一化层的权重
        elif "weight" in name:
            # 检查值的形状是否与模型中的形状相匹配
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            # 更新模型的权重
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    # 处理其他情况，将未使用的权重添加到列表中
    else:
        unused_weights.append(full_name)


# 禁用梯度计算的装饰器
@torch.no_grad()
def convert_speecht5_checkpoint(
    task,
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    vocab_path=None,
    repo_id=None,
):
    """
    将模型的权重转换到transformers设计中。
    """
    # 如果提供了配置文件路径，则加载配置
    if config_path is not None:
        config = SpeechT5Config.from_pretrained(config_path)
    else:
        # 否则创建一个新的配置对象
        config = SpeechT5Config()
    # 如果任务名称为“s2t”，则将最大长度设置为文本位置的最大长度，并创建一个用于语音到文本的模型对象
    if task == "s2t":
        config.max_length = config.max_text_positions
        model = SpeechT5ForSpeechToText(config)
    # 如果任务名称为“t2s”，则将最大语音位置设置为1876，最大文本位置设置为600，并创建一个用于文本到语音的模型对象
    elif task == "t2s":
        config.max_speech_positions = 1876
        config.max_text_positions = 600
        config.max_length = config.max_speech_positions
        model = SpeechT5ForTextToSpeech(config)
    # 如果任务名称为“s2s”，则将最大语音位置设置为1876，并创建一个用于语音到语音的模型对象
    elif task == "s2s":
        config.max_speech_positions = 1876
        config.max_length = config.max_speech_positions
        model = SpeechT5ForSpeechToSpeech(config)
    # 如果任务名称不符合上述条件，则抛出数值错误并显示任务名称
    else:
        raise ValueError(f"Unknown task name: {task}")

    # 如果存在词汇路径，则创建一个基于词汇路径的tokenizer对象，并设置模型的最大长度
    if vocab_path:
        tokenizer = SpeechT5Tokenizer(vocab_path, model_max_length=config.max_text_positions)
        # 将掩码标记设置为类似于普通单词的行为，即包括它前面的空格
        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
        tokenizer.mask_token = mask_token
        tokenizer.add_special_tokens({"mask_token": mask_token})
        tokenizer.add_tokens(["<ctc_blank>"])

    # 创建一个语音T5特征提取器对象
    feature_extractor = SpeechT5FeatureExtractor()
    # 创建一个处理器对象，设置tokenizer和特征提取器，然后保存到指定的PyTorch转储文件夹路径
    processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    processor.save_pretrained(pytorch_dump_folder_path)

    # 加载Fairseq检查点文件，递归加载权重到模型中的对应任务中
    fairseq_checkpoint = torch.load(checkpoint_path)
    recursively_load_weights(fairseq_checkpoint["model"], model, task)

    # 将模型保存到指定的PyTorch转储文件夹路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 如果存在repo_id，则将处理器和模型推送到hub
    if repo_id:
        print("Pushing to the hub...")
        processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
# 如果脚本作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：任务类型，默认为's2t'，可选值为's2t', 't2s', 's2s'
    parser.add_argument(
        "--task",
        default="s2t",
        type=str,
        help="Type of the SpeechT5 model you'd like to convert. Should be one of 's2t', 't2s', 's2s'.",
    )
    # 添加命令行参数：fairseq模型的checkpoint路径，必须提供
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数：SentencePiece模型的路径
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to SentencePiece model")
    # 添加命令行参数：待转换模型的hf配置文件（JSON格式）的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数：输出PyTorch模型的文件夹路径，必须提供
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加命令行参数：指定是否将转换后的模型上传到🤗 hub
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数convert_speecht5_checkpoint，将fairseq模型转换为PyTorch模型
    convert_speecht5_checkpoint(
        args.task,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.vocab_path,
        args.push_to_hub,
    )
```py  
```