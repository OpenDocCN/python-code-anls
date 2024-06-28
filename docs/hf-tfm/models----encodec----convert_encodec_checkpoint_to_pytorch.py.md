# `.\models\encodec\convert_encodec_checkpoint_to_pytorch.py`

```py
# 设置编码方式为 UTF-8
# 版权声明，指出版权属于 2023 年的 HuggingFace Inc. 团队所有
# 根据 Apache 许可证版本 2.0 使用本文件，详细信息可以访问指定网址获取
# 除非法律要求或书面同意，否则不得使用本文件
# 根据 Apache 许可证版本 2.0，本软件基于“原样”分发，不提供任何形式的担保或条件
# 请查看许可证，了解具体语言版本的细节

"""Convert EnCodec checkpoints."""

# 导入必要的库
import argparse  # 用于解析命令行参数

import torch  # PyTorch 库

from transformers import (  # 导入 transformers 库中的相关模块
    EncodecConfig,  # EnCodec 的配置类
    EncodecFeatureExtractor,  # EnCodec 的特征提取器类
    EncodecModel,  # EnCodec 的模型类
    logging,  # 日志记录模块
)

# 设置日志记录的详细程度为 info 级别
logging.set_verbosity_info()
# 获取名为 "transformers.models.encodec" 的日志记录器
logger = logging.get_logger("transformers.models.encodec")

# 定义映射字典，用于重命名量化器（quantizer）中的模型参数
MAPPING_QUANTIZER = {
    "quantizer.vq.layers.*._codebook.inited": "quantizer.layers.*.codebook.inited",
    "quantizer.vq.layers.*._codebook.cluster_size": "quantizer.layers.*.codebook.cluster_size",
    "quantizer.vq.layers.*._codebook.embed": "quantizer.layers.*.codebook.embed",
    "quantizer.vq.layers.*._codebook.embed_avg": "quantizer.layers.*.codebook.embed_avg",
}

# 定义映射字典，用于重命名编码器（encoder）中的模型参数
MAPPING_ENCODER = {
    "encoder.model.0.conv.conv": "encoder.layers.0.conv",
    "encoder.model.1.block.1.conv.conv": "encoder.layers.1.block.1.conv",
    "encoder.model.1.block.3.conv.conv": "encoder.layers.1.block.3.conv",
    "encoder.model.1.shortcut.conv.conv": "encoder.layers.1.shortcut.conv",
    "encoder.model.3.conv.conv": "encoder.layers.3.conv",
    "encoder.model.4.block.1.conv.conv": "encoder.layers.4.block.1.conv",
    "encoder.model.4.block.3.conv.conv": "encoder.layers.4.block.3.conv",
    "encoder.model.4.shortcut.conv.conv": "encoder.layers.4.shortcut.conv",
    "encoder.model.6.conv.conv": "encoder.layers.6.conv",
    "encoder.model.7.block.1.conv.conv": "encoder.layers.7.block.1.conv",
    "encoder.model.7.block.3.conv.conv": "encoder.layers.7.block.3.conv",
    "encoder.model.7.shortcut.conv.conv": "encoder.layers.7.shortcut.conv",
    "encoder.model.9.conv.conv": "encoder.layers.9.conv",
    "encoder.model.10.block.1.conv.conv": "encoder.layers.10.block.1.conv",
    "encoder.model.10.block.3.conv.conv": "encoder.layers.10.block.3.conv",
    "encoder.model.10.shortcut.conv.conv": "encoder.layers.10.shortcut.conv",
    "encoder.model.12.conv.conv": "encoder.layers.12.conv",
    "encoder.model.13.lstm": "encoder.layers.13.lstm",
    "encoder.model.15.conv.conv": "encoder.layers.15.conv",
}

# 定义映射字典，用于重命名 48kHz 编码器（encoder）中的模型参数
MAPPING_ENCODER_48K = {
    "encoder.model.0.conv.norm": "encoder.layers.0.norm",
    # 这里可以继续添加其他的映射关系
}
    # 定义一个字典，映射旧模型中的层标准化层到新模型中对应的标准化层
    {
        "encoder.model.1.block.1.conv.norm": "encoder.layers.1.block.1.norm",
        "encoder.model.1.block.3.conv.norm": "encoder.layers.1.block.3.norm",
        "encoder.model.1.shortcut.conv.norm": "encoder.layers.1.shortcut.norm",
        "encoder.model.3.conv.norm": "encoder.layers.3.norm",
        "encoder.model.4.block.1.conv.norm": "encoder.layers.4.block.1.norm",
        "encoder.model.4.block.3.conv.norm": "encoder.layers.4.block.3.norm",
        "encoder.model.4.shortcut.conv.norm": "encoder.layers.4.shortcut.norm",
        "encoder.model.6.conv.norm": "encoder.layers.6.norm",
        "encoder.model.7.block.1.conv.norm": "encoder.layers.7.block.1.norm",
        "encoder.model.7.block.3.conv.norm": "encoder.layers.7.block.3.norm",
        "encoder.model.7.shortcut.conv.norm": "encoder.layers.7.shortcut.norm",
        "encoder.model.9.conv.norm": "encoder.layers.9.norm",
        "encoder.model.10.block.1.conv.norm": "encoder.layers.10.block.1.norm",
        "encoder.model.10.block.3.conv.norm": "encoder.layers.10.block.3.norm",
        "encoder.model.10.shortcut.conv.norm": "encoder.layers.10.shortcut.norm",
        "encoder.model.12.conv.norm": "encoder.layers.12.norm",
        "encoder.model.15.conv.norm": "encoder.layers.15.norm",
    }
}
# 闭合上一个字典的定义，表示字典定义的结束

MAPPING_DECODER = {
    "decoder.model.0.conv.conv": "decoder.layers.0.conv",
    "decoder.model.1.lstm": "decoder.layers.1.lstm",
    "decoder.model.3.convtr.convtr": "decoder.layers.3.conv",
    "decoder.model.4.block.1.conv.conv": "decoder.layers.4.block.1.conv",
    "decoder.model.4.block.3.conv.conv": "decoder.layers.4.block.3.conv",
    "decoder.model.4.shortcut.conv.conv": "decoder.layers.4.shortcut.conv",
    "decoder.model.6.convtr.convtr": "decoder.layers.6.conv",
    "decoder.model.7.block.1.conv.conv": "decoder.layers.7.block.1.conv",
    "decoder.model.7.block.3.conv.conv": "decoder.layers.7.block.3.conv",
    "decoder.model.7.shortcut.conv.conv": "decoder.layers.7.shortcut.conv",
    "decoder.model.9.convtr.convtr": "decoder.layers.9.conv",
    "decoder.model.10.block.1.conv.conv": "decoder.layers.10.block.1.conv",
    "decoder.model.10.block.3.conv.conv": "decoder.layers.10.block.3.conv",
    "decoder.model.10.shortcut.conv.conv": "decoder.layers.10.shortcut.conv",
    "decoder.model.12.convtr.convtr": "decoder.layers.12.conv",
    "decoder.model.13.block.1.conv.conv": "decoder.layers.13.block.1.conv",
    "decoder.model.13.block.3.conv.conv": "decoder.layers.13.block.3.conv",
    "decoder.model.13.shortcut.conv.conv": "decoder.layers.13.shortcut.conv",
    "decoder.model.15.conv.conv": "decoder.layers.15.conv",
}
# 映射字典，将模型中的编码器层命名映射到解码器层命名，用于对模型进行结构映射

MAPPING_DECODER_48K = {
    "decoder.model.0.conv.norm": "decoder.layers.0.norm",
    "decoder.model.3.convtr.norm": "decoder.layers.3.norm",
    "decoder.model.4.block.1.conv.norm": "decoder.layers.4.block.1.norm",
    "decoder.model.4.block.3.conv.norm": "decoder.layers.4.block.3.norm",
    "decoder.model.4.shortcut.conv.norm": "decoder.layers.4.shortcut.norm",
    "decoder.model.6.convtr.norm": "decoder.layers.6.norm",
    "decoder.model.7.block.1.conv.norm": "decoder.layers.7.block.1.norm",
    "decoder.model.7.block.3.conv.norm": "decoder.layers.7.block.3.norm",
    "decoder.model.7.shortcut.conv.norm": "decoder.layers.7.shortcut.norm",
    "decoder.model.9.convtr.norm": "decoder.layers.9.norm",
    "decoder.model.10.block.1.conv.norm": "decoder.layers.10.block.1.norm",
    "decoder.model.10.block.3.conv.norm": "decoder.layers.10.block.3.norm",
    "decoder.model.10.shortcut.conv.norm": "decoder.layers.10.shortcut.norm",
    "decoder.model.12.convtr.norm": "decoder.layers.12.norm",
    "decoder.model.13.block.1.conv.norm": "decoder.layers.13.block.1.norm",
    "decoder.model.13.block.3.conv.norm": "decoder.layers.13.block.3.norm",
    "decoder.model.13.shortcut.conv.norm": "decoder.layers.13.shortcut.norm",
    "decoder.model.15.conv.norm": "decoder.layers.15.norm",
}
# 映射字典，将模型中的编码器层的归一化命名映射到解码器层的归一化命名

MAPPING_24K = {
    **MAPPING_QUANTIZER,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
}
# 将量化器、编码器和解码器的映射合并到一个字典中，用于24K配置

MAPPING_48K = {
    **MAPPING_QUANTIZER,
    **MAPPING_ENCODER,
    **MAPPING_ENCODER_48K,
    **MAPPING_DECODER,
    **MAPPING_DECODER_48K,
}
# 将量化器、编码器、解码器48K配置的映射合并到一个字典中，用于48K配置

TOP_LEVEL_KEYS = []
# 初始化一个空列表，用于存储顶层键

IGNORE_KEYS = []
# 初始化一个空列表，用于存储需要忽略的键

def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 将 key 按 "." 分割成属性列表，逐级获取 hf_pointer 的属性值
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果指定了 weight_type，则获取 hf_pointer 对应属性的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        # 否则获取 hf_pointer 自身的形状
        hf_shape = hf_pointer.shape

    # 检查获取的形状是否与 value 的形状相匹配，如果不匹配则抛出 ValueError 异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据 weight_type 类型设置 hf_pointer 对应的数据值
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
    elif weight_type == "weight_ih_l0":
        hf_pointer.weight_ih_l0.data = value
    elif weight_type == "weight_hh_l0":
        hf_pointer.weight_hh_l0.data = value
    elif weight_type == "bias_ih_l0":
        hf_pointer.bias_ih_l0.data = value
    elif weight_type == "bias_hh_l0":
        hf_pointer.bias_hh_l0.data = value
    elif weight_type == "weight_ih_l1":
        hf_pointer.weight_ih_l1.data = value
    elif weight_type == "weight_hh_l1":
        hf_pointer.weight_hh_l1.data = value
    elif weight_type == "bias_ih_l1":
        hf_pointer.bias_ih_l1.data = value
    elif weight_type == "bias_hh_l1":
        hf_pointer.bias_hh_l1.data = value
    else:
        # 如果 weight_type 未指定或未匹配到特定类型，直接设置 hf_pointer 的数据值
        hf_pointer.data = value

    # 记录日志，指示成功初始化的属性和其来源
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")
# 判断给定的文件名是否应该被忽略，根据 ignore_keys 中的规则进行匹配
def should_ignore(name, ignore_keys):
    # 遍历 ignore_keys 列表中的每一个关键字
    for key in ignore_keys:
        # 如果关键字以 ".*" 结尾，检查 name 是否以 key[:-1] 开头，如果是则返回 True
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        # 如果关键字包含 ".*."，则将 key 拆分成前缀 prefix 和后缀 suffix，如果 name 同时包含这两部分则返回 True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        # 否则，如果关键字 key 直接在 name 中出现则返回 True
        elif key in name:
            return True
    # 如果都没有匹配成功，则返回 False，表示不忽略该文件名
    return False


# 根据给定的模型名和原始字典 orig_dict，加载对应模型的权重到 hf_model 中，并返回未使用的权重列表
def recursively_load_weights(orig_dict, hf_model, model_name):
    # 初始化未使用的权重列表
    unused_weights = []

    # 根据不同的模型名选择相应的映射关系
    if model_name == "encodec_24khz" or "encodec_32khz":
        MAPPING = MAPPING_24K
    elif model_name == "encodec_48khz":
        MAPPING = MAPPING_48K
    else:
        # 如果模型名不在支持列表中，抛出 ValueError 异常
        raise ValueError(f"Unsupported model: {model_name}")
    # 遍历原始字典的键值对
    for name, value in orig_dict.items():
        # 如果应该忽略该键名，则记录日志并跳过当前循环
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        # 标志：用于检查是否在后续处理中使用了该键名对应的数值
        is_used = False

        # 遍历映射字典中的键值对
        for key, mapped_key in MAPPING.items():
            # 如果当前映射键包含通配符"*"
            if "*" in key:
                # 拆分通配符前缀和后缀
                prefix, suffix = key.split(".*.")
                # 如果键名同时包含前缀和后缀，则使用后缀作为新的键名
                if prefix in name and suffix in name:
                    key = suffix

            # 如果当前映射键在键名中找到匹配
            if key in name:
                # 特定情况下的处理：防止 ".embed_avg" 初始化为 ".embed"
                if key.endswith("embed") and name.endswith("embed_avg"):
                    continue

                # 设置标志表明该键名已被使用
                is_used = True

                # 如果映射值中存在通配符"*"，则根据层索引替换通配符
                if "*" in mapped_key:
                    layer_index = name.split(key)[0].split(".")[-2]
                    mapped_key = mapped_key.replace("*", layer_index)

                # 根据特定的权重类型为权重键赋值
                if "weight_g" in name:
                    weight_type = "weight_g"
                elif "weight_v" in name:
                    weight_type = "weight_v"
                elif "weight_ih_l0" in name:
                    weight_type = "weight_ih_l0"
                elif "weight_hh_l0" in name:
                    weight_type = "weight_hh_l0"
                elif "bias_ih_l0" in name:
                    weight_type = "bias_ih_l0"
                elif "bias_hh_l0" in name:
                    weight_type = "bias_hh_l0"
                elif "weight_ih_l1" in name:
                    weight_type = "weight_ih_l1"
                elif "weight_hh_l1" in name:
                    weight_type = "weight_hh_l1"
                elif "bias_ih_l1" in name:
                    weight_type = "bias_ih_l1"
                elif "bias_hh_l1" in name:
                    weight_type = "bias_hh_l1"
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

                # 递归地设置新模型的映射键对应的值
                set_recursively(hf_model, mapped_key, value, name, weight_type)

            # 继续下一个映射键的处理
            continue
        
        # 如果没有任何映射键被使用，则将该键名添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重列表到警告日志中
    logger.warning(f"Unused weights: {unused_weights}")
# 用装饰器 @torch.no_grad() 标记该函数，禁止在函数内部进行梯度计算
def convert_checkpoint(
    model_name,
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置文件路径，则从预训练模型加载配置
    if config_path is not None:
        config = EncodecConfig.from_pretrained(config_path)
    else:
        # 否则创建一个新的配置对象
        config = EncodecConfig()

    # 根据模型名称设置配置对象的参数
    if model_name == "encodec_24khz":
        pass  # 对于 "encodec_24khz" 模型，配置已经是正确的
    elif model_name == "encodec_32khz":
        # 根据模型名称调整配置对象的参数
        config.upsampling_ratios = [8, 5, 4, 4]
        config.target_bandwidths = [2.2]
        config.num_filters = 64
        config.sampling_rate = 32_000
        config.codebook_size = 2048
        config.use_causal_conv = False
        config.normalize = False
        config.use_conv_shortcut = False
    elif model_name == "encodec_48khz":
        # 根据模型名称调整配置对象的参数
        config.upsampling_ratios = [8, 5, 4, 2]
        config.target_bandwidths = [3.0, 6.0, 12.0, 24.0]
        config.sampling_rate = 48_000
        config.audio_channels = 2
        config.use_causal_conv = False
        config.norm_type = "time_group_norm"
        config.normalize = True
        config.chunk_length_s = 1.0
        config.overlap = 0.01
    else:
        # 如果模型名称不在已知列表中，抛出异常
        raise ValueError(f"Unknown model name: {model_name}")

    # 根据配置对象创建模型
    model = EncodecModel(config)

    # 根据配置对象创建特征提取器
    feature_extractor = EncodecFeatureExtractor(
        feature_size=config.audio_channels,
        sampling_rate=config.sampling_rate,
        chunk_length_s=config.chunk_length_s,
        overlap=config.overlap,
    )

    # 将特征提取器保存到指定路径
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 加载原始 PyTorch 检查点
    original_checkpoint = torch.load(checkpoint_path)
    
    # 如果原始检查点中包含 "best_state" 键，只保留权重信息
    if "best_state" in original_checkpoint:
        original_checkpoint = original_checkpoint["best_state"]

    # 递归加载权重到模型中
    recursively_load_weights(original_checkpoint, model, model_name)

    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 如果提供了 repo_id，将特征提取器和模型推送到指定的 hub
    if repo_id:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="encodec_24khz",
        type=str,
        help="The model to convert. Should be one of 'encodec_24khz', 'encodec_32khz', 'encodec_48khz'.",
    )
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    # 解析参数
    args = parser.parse_args()
    # 调用函数 convert_checkpoint，用于转换模型的检查点文件格式
    convert_checkpoint(
        args.model,                     # 指定模型名称参数
        args.checkpoint_path,           # 指定检查点文件路径参数
        args.pytorch_dump_folder_path,  # 指定转换后的 PyTorch 模型输出文件夹路径参数
        args.config_path,               # 指定模型配置文件路径参数
        args.push_to_hub,               # 指定是否将转换后的模型推送到 Hub 的参数
    )
```