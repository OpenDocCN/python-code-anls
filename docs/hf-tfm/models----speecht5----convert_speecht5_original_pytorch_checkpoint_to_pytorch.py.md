# `.\models\speecht5\convert_speecht5_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可证信息，指定这段代码的版权归属和使用许可
# 导入命令行参数解析模块
import argparse

# 导入 PyTorch 库
import torch

# 导入 Transformers 库中的相关类和函数
from transformers import (
    SpeechT5Config,  # 导入 SpeechT5 模型配置
    SpeechT5FeatureExtractor,  # 导入 SpeechT5 特征提取器
    SpeechT5ForSpeechToSpeech,  # 导入 SpeechT5 语音到语音模型
    SpeechT5ForSpeechToText,  # 导入 SpeechT5 语音到文本模型
    SpeechT5ForTextToSpeech,  # 导入 SpeechT5 文本到语音模型
    SpeechT5Processor,  # 导入 SpeechT5 处理器
    SpeechT5Tokenizer,  # 导入 SpeechT5 分词器
    logging,  # 导入日志记录模块
)
from transformers.tokenization_utils import AddedToken  # 导入特定的分词工具类

# 设置日志的详细程度为 info
logging.set_verbosity_info()

# 获取特定名称的日志记录器
logger = logging.get_logger("transformers.models.speecht5")

# 定义映射字典，将旧版本中的模型参数映射到新版本中的相应位置
MAPPING_SPEECH_ENCODER_PRENET = {
    "speech_encoder_prenet.layer_norm": "speecht5.encoder.prenet.feature_projection.layer_norm",
    "speech_encoder_prenet.post_extract_proj": "speecht5.encoder.prenet.feature_projection.projection",
    "speech_encoder_prenet.pos_conv.0": "speecht5.encoder.prenet.pos_conv_embed.conv",
    "speech_encoder_prenet.mask_emb": "speecht5.encoder.prenet.masked_spec_embed",
}

MAPPING_TEXT_ENCODER_PRENET = {
    "text_encoder_prenet.encoder_prenet.0": "speecht5.encoder.prenet.embed_tokens",
    "text_encoder_prenet.encoder_prenet.1.alpha": "speecht5.encoder.prenet.encode_positions.alpha",
}

MAPPING_SPEECH_DECODER_PRENET = {
    "speech_decoder_prenet.decoder_prenet.0.0.prenet.0.0": "speecht5.decoder.prenet.layers.0",
    "speech_decoder_prenet.decoder_prenet.0.0.prenet.1.0": "speecht5.decoder.prenet.layers.1",
    "speech_decoder_prenet.decoder_prenet.0.1": "speecht5.decoder.prenet.final_layer",
    "speech_decoder_prenet.decoder_prenet.1.alpha": "speecht5.decoder.prenet.encode_positions.alpha",
    "speech_decoder_prenet.spkembs_layer.0": "speecht5.decoder.prenet.speaker_embeds_layer",
}

MAPPING_SPEECH_DECODER_POSTNET = {
    "speech_decoder_postnet.feat_out": "speech_decoder_postnet.feat_out",
    "speech_decoder_postnet.prob_out": "speech_decoder_postnet.prob_out",
    "speech_decoder_postnet.postnet.postnet.0.0": "speech_decoder_postnet.layers.0.conv",
    "speech_decoder_postnet.postnet.postnet.0.1": "speech_decoder_postnet.layers.0.batch_norm",
    "speech_decoder_postnet.postnet.postnet.1.0": "speech_decoder_postnet.layers.1.conv",
    "speech_decoder_postnet.postnet.postnet.1.1": "speech_decoder_postnet.layers.1.batch_norm",
    "speech_decoder_postnet.postnet.postnet.2.0": "speech_decoder_postnet.layers.2.conv",
    "speech_decoder_postnet.postnet.postnet.2.1": "speech_decoder_postnet.layers.2.batch_norm",
}
    # 定义一个字典，将旧的模型层名称映射到新的模型层名称
    "speech_decoder_postnet.postnet.postnet.3.0": "speech_decoder_postnet.layers.3.conv",
    # 继续定义字典映射
    "speech_decoder_postnet.postnet.postnet.3.1": "speech_decoder_postnet.layers.3.batch_norm",
    # 继续定义字典映射
    "speech_decoder_postnet.postnet.postnet.4.0": "speech_decoder_postnet.layers.4.conv",
    # 继续定义字典映射
    "speech_decoder_postnet.postnet.postnet.4.1": "speech_decoder_postnet.layers.4.batch_norm",
}
# 文本到语音模型的映射，用于将文本解码器的预网络映射到SpeechT5解码器的预网络
MAPPING_TEXT_DECODER_PRENET = {
    "text_decoder_prenet.embed_tokens": "speecht5.decoder.prenet.embed_tokens",
}
# 文本到语音模型的映射，用于将文本解码器的后网络映射到文本解码器的语言模型头部
MAPPING_TEXT_DECODER_POSTNET = {
    "text_decoder_postnet.output_projection": "text_decoder_postnet.lm_head",
}
# 编码器的映射，将SpeechT5编码器的各个层映射到包装的编码器的对应层
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
# 解码器的映射，将SpeechT5解码器的各个层映射到包装的解码器的对应层
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
# 从文本到语音的映射，包括文本编码器、解码器和额外的预网络和后网络映射
MAPPING_S2T = {
    **MAPPING_SPEECH_ENCODER_PRENET,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
    **MAPPING_TEXT_DECODER_PRENET,
    **MAPPING_TEXT_DECODER_POSTNET,
}
# 从语音到文本的映射，包括文本编码器、解码器和语音解码器的预网络和后网络映射
MAPPING_T2S = {
    **MAPPING_TEXT_ENCODER_PRENET,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
    **MAPPING_SPEECH_DECODER_PRENET,
    **MAPPING_SPEECH_DECODER_POSTNET,
}
# 将 MAPPING_SPEECH_ENCODER_PRENET, MAPPING_ENCODER, MAPPING_DECODER,
# MAPPING_SPEECH_DECODER_PRENET, MAPPING_SPEECH_DECODER_POSTNET 合并为一个字典
MAPPING_S2S = {
    **MAPPING_SPEECH_ENCODER_PRENET,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
    **MAPPING_SPEECH_DECODER_PRENET,
    **MAPPING_SPEECH_DECODER_POSTNET,
}

# 顶层键的空列表
TOP_LEVEL_KEYS = []

# 忽略的键列表，包括某些具体路径和通配符
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

# S2T 任务特定的忽略键列表，包括通用的 IGNORE_KEYS 和一些额外的键
IGNORE_KEYS_S2T = IGNORE_KEYS + [
    "encoder.proj",
    "text_encoder_prenet.*",
    "speech_decoder_prenet.*",
    "speech_decoder_postnet.*",
]

# T2S 任务特定的忽略键列表，包括通用的 IGNORE_KEYS 和一些额外的键
IGNORE_KEYS_T2S = IGNORE_KEYS + [
    "encoder.proj",
    "speech_encoder_prenet.*",
    "text_decoder_prenet.*",
    "text_decoder_postnet.*",
]

# S2S 任务特定的忽略键列表，包括通用的 IGNORE_KEYS 和一些额外的键
IGNORE_KEYS_S2S = IGNORE_KEYS + [
    "encoder.proj",
    "text_encoder_prenet.*",
    "text_decoder_prenet.*",
    "text_decoder_postnet.*",
]

# 递归设置模型权重的函数
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 根据键字符串逐级访问对象属性，直至最后一级
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 根据 weight_type 获取当前属性的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 如果形状不匹配，则抛出值错误异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据 weight_type 设置属性的数据值
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

    # 记录权重初始化的信息到日志
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")


# 判断给定名称是否应该被忽略的函数
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


# 递归加载权重到模型的函数
def recursively_load_weights(fairseq_dict, hf_model, task):
    unused_weights = []

    # 如果任务是 S2T
    if task == "s2t":
        # 获取特征编码器对象
        feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
        # 使用 S2T 任务相关的映射和忽略键列表
        MAPPING = MAPPING_S2T
        IGNORE_KEYS = IGNORE_KEYS_S2T
    elif task == "t2s":
        feature_encoder = None
        MAPPING = MAPPING_T2S  # 设置映射表为 T2S 的映射表
        IGNORE_KEYS = IGNORE_KEYS_T2S  # 设置忽略列表为 T2S 的忽略列表
    elif task == "s2s":
        feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder  # 获取特征编码器
        MAPPING = MAPPING_S2S  # 设置映射表为 S2S 的映射表
        IGNORE_KEYS = IGNORE_KEYS_S2S  # 设置忽略列表为 S2S 的忽略列表
    else:
        raise ValueError(f"Unsupported task: {task}")  # 抛出异常，任务不支持

    for name, value in fairseq_dict.items():  # 遍历 fairseq 字典的每个条目
        if should_ignore(name, IGNORE_KEYS):  # 判断是否应该忽略当前条目
            logger.info(f"{name} was ignored")  # 记录日志，指出被忽略的条目
            continue

        is_used = False  # 初始化是否使用的标志为 False
        if "conv_layers" in name:  # 如果条目名包含 "conv_layers"
            load_conv_layer(
                name,
                value,
                feature_encoder,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )  # 调用加载卷积层函数，加载当前条目
            is_used = True  # 设置已使用标志为 True，表示当前条目已被使用
        else:
            for key, mapped_key in MAPPING.items():  # 遍历映射表中的每个映射关系
                # mapped_key = "speecht5." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key

                if "*" in key:  # 如果映射键中包含通配符 *
                    prefix, suffix = key.split(".*.")  # 拆分前缀和后缀
                    if prefix in name and suffix in name:  # 如果条目名包含前缀和后缀
                        key = suffix  # 使用后缀作为当前键

                # if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                if key in name:  # 如果当前键存在于条目名中
                    is_used = True  # 设置已使用标志为 True
                    if "*" in mapped_key:  # 如果映射后的键中包含通配符 *
                        layer_index = name.split(key)[0].split(".")[-2]  # 提取层索引
                        mapped_key = mapped_key.replace("*", layer_index)  # 替换映射键中的通配符

                    # 确定权重类型
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

                    set_recursively(hf_model, mapped_key, value, name, weight_type)  # 递归设置模型参数

                continue  # 继续下一个映射关系的处理

        if not is_used:  # 如果当前条目未被使用
            unused_weights.append(name)  # 将当前条目名添加到未使用的权重列表中

    logger.warning(f"Unused weights: {unused_weights}")  # 记录未使用的权重列表到日志中
# 加载卷积层数据到特征提取器中
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 根据点号分割全名获取层和类型
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])  # 提取层的标识号
    type_id = int(items[1])   # 提取类型的标识号

    # 如果类型标识为0，处理偏置项或权重项
    if type_id == 0:
        if "bias" in name:
            # 检查值的形状是否匹配特征提取器中对应卷积层的偏置项形状
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value  # 设置偏置项数据
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")  # 记录日志
        elif "weight" in name:
            # 检查值的形状是否匹配特征提取器中对应卷积层的权重形状
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value  # 设置权重数据
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")  # 记录日志
    # 如果类型标识为2且不使用组归一化，或者类型标识为2且是第一层且使用了组归一化
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            # 检查值的形状是否匹配特征提取器中对应层归一化的偏置项形状
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value  # 设置层归一化偏置项数据
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")  # 记录日志
        elif "weight" in name:
            # 检查值的形状是否匹配特征提取器中对应层归一化的权重形状
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value  # 设置层归一化权重数据
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")  # 记录日志
    else:
        unused_weights.append(full_name)  # 将未使用的权重名称添加到列表中


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
    将模型的权重复制/粘贴/调整到transformers设计中。
    """
    if config_path is not None:
        config = SpeechT5Config.from_pretrained(config_path)  # 从预训练配置文件加载配置
    else:
        config = SpeechT5Config()  # 创建一个默认配置对象
    # 根据任务类型选择配置参数和模型
    if task == "s2t":
        config.max_length = config.max_text_positions
        # 使用给定的配置创建语音到文本任务的模型对象
        model = SpeechT5ForSpeechToText(config)
    elif task == "t2s":
        config.max_speech_positions = 1876
        config.max_text_positions = 600
        config.max_length = config.max_speech_positions
        # 使用给定的配置创建文本到语音任务的模型对象
        model = SpeechT5ForTextToSpeech(config)
    elif task == "s2s":
        config.max_speech_positions = 1876
        config.max_length = config.max_speech_positions
        # 使用给定的配置创建语音到语音任务的模型对象
        model = SpeechT5ForSpeechToSpeech(config)
    else:
        # 如果任务名未知，则抛出值错误异常
        raise ValueError(f"Unknown task name: {task}")

    if vocab_path:
        # 使用给定的词汇表路径和模型最大长度创建语音T5分词器对象
        tokenizer = SpeechT5Tokenizer(vocab_path, model_max_length=config.max_text_positions)

        # 添加一个特殊的掩码标记，表现得像普通词汇，即在其前面包含空格
        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
        tokenizer.mask_token = mask_token
        tokenizer.add_special_tokens({"mask_token": mask_token})  # 添加特殊标记到分词器中
        tokenizer.add_tokens(["<ctc_blank>"])  # 添加特殊标记到分词器中

    # 创建语音T5特征提取器对象
    feature_extractor = SpeechT5FeatureExtractor()
    # 使用分词器和特征提取器创建语音T5处理器对象
    processor = SpeechT5Processor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    # 将处理器对象保存到指定的PyTorch模型转储文件夹路径
    processor.save_pretrained(pytorch_dump_folder_path)

    # 加载Fairseq检查点中的权重到模型对象中
    fairseq_checkpoint = torch.load(checkpoint_path)
    recursively_load_weights(fairseq_checkpoint["model"], model, task)

    # 将模型对象保存到指定的PyTorch模型转储文件夹路径
    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        # 如果存在repo_id，则推送处理器和模型到Hub上
        print("Pushing to the hub...")
        processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument(
        "--task",
        default="s2t",
        type=str,
        help="Type of the SpeechT5 model you'd like to convert. Should be one of 's2t', 't2s', 's2s'.",
    )
    # 添加名为 "--task" 的命令行参数，指定默认值为 "s2t"，类型为字符串，用于指定要转换的模型类型

    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to fairseq checkpoint")
    # 添加名为 "--checkpoint_path" 的必需命令行参数，类型为字符串，用于指定 fairseq 模型的检查点路径

    parser.add_argument("--vocab_path", default=None, type=str, help="Path to SentencePiece model")
    # 添加名为 "--vocab_path" 的可选命令行参数，类型为字符串，用于指定 SentencePiece 模型的路径

    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加名为 "--config_path" 的可选命令行参数，类型为字符串，用于指定要转换模型的 HF (Hugging Face) 配置文件路径

    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    # 添加名为 "--pytorch_dump_folder_path" 的必需命令行参数，类型为字符串，用于指定输出 PyTorch 模型的文件夹路径

    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    # 添加名为 "--push_to_hub" 的可选命令行参数，类型为字符串，用于指定在 🤗 hub 上上传转换后的模型的位置

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 变量中

    convert_speecht5_checkpoint(
        args.task,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.vocab_path,
        args.push_to_hub,
    )
    # 调用函数 convert_speecht5_checkpoint，并传递解析后的命令行参数作为函数的参数
```