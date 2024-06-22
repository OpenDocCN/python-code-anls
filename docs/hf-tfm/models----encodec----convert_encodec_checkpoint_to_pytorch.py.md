# `.\models\encodec\convert_encodec_checkpoint_to_pytorch.py`

```
# 设置文件编码
# 版权声明
# 许可证说明
"""转换 EnCodec 检查点。"""
# 导入模块
import argparse
# 导入 torch
import torch
# 从 transformers 模块中导入 EncodecConfig, EncodecFeatureExtractor, EncodecModel, logging
from transformers import (
    EncodecConfig,
    EncodecFeatureExtractor,
    EncodecModel,
    logging,
)

# 定义日志记录的详细程度
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger("transformers.models.encodec")

# 映射器 - 量化器
MAPPING_QUANTIZER = {
    "quantizer.vq.layers.*._codebook.inited": "quantizer.layers.*.codebook.inited",
    "quantizer.vq.layers.*._codebook.cluster_size": "quantizer.layers.*.codebook.cluster_size",
    "quantizer.vq.layers.*._codebook.embed": "quantizer.layers.*.codebook.embed",
    "quantizer.vq.layers.*._codebook.embed_avg": "quantizer.layers.*.codebook.embed_avg",
}
# 映射器 - 编码器
MAPPING_ENCODER = {
    "encoder.model.0.conv.conv": "encoder.layers.0.conv",
    "encoder.model.1.block.1.conv.conv": "encoder.layers.1.block.1.conv",
    ...
}
# 映射器 - 48K 编码器
MAPPING_ENCODER_48K = {
    "encoder.model.0.conv.norm": "encoder.layers.0.norm",
    # 定义一个字典，将模型权重中的规范化层名称映射到相应的编码器层规范化层名称
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
# 定义字典，将解码器的层名称映射到相应的解码器层上
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

# 定义字典，将 48K 解码器的层名称映射到相应的解码器层上
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

# 合并量化器、编码器和解码器的层映射字典，形成 24K 模型的映射字典
MAPPING_24K = {
    **MAPPING_QUANTIZER,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
}

# 合并量化器、48K 编码器、48K 解码器的层映射字典，形成 48K 模型的映射字典
MAPPING_48K = {
    **MAPPING_QUANTIZER,
    **MAPPING_ENCODER,
    **MAPPING_ENCODER_48K,
    **MAPPING_DECODER,
    **MAPPING_DECODER_48K,
}

# 定义空列表 TOP_LEVEL_KEYS
TOP_LEVEL_KEYS = []

# 定义空列表 IGNORE_KEYS
IGNORE_KEYS = []

# 定义函数 set_recursively，用于递归设置某个变量
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 将输入的键按点分隔，逐级获取对象属性
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 如果权重类型不为空，则获取相应属性的形状
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        # 否则获取整个对象的形状
        hf_shape = hf_pointer.shape

    # 如果获取的形状与给定值的形状不相等，则引发 ValueError 异常
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型设置对应属性的值
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
        # 如果权重类型为空或不在已知类型中，则直接设置对象的数据
        hf_pointer.data = value

    # 记录初始化信息
    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")
# 判断给定的文件名是否应该被忽略，根据忽略关键字列表
def should_ignore(name, ignore_keys):
    # 遍历忽略关键字列表
    for key in ignore_keys:
        # 如果忽略关键字以".*"结尾
        if key.endswith(".*"):
            # 如果文件名以去掉最后一个字符的忽略关键字开头，说明需要忽略该文件
            if name.startswith(key[:-1]):
                return True
        # 如果忽略关键字中包含".*."
        elif ".*." in key:
            # 将忽略关键字按".*."分割成前缀和后缀
            prefix, suffix = key.split(".*.")
            # 如果文件名中包含前缀和后缀，说明需要忽略该文件
            if prefix in name and suffix in name:
                return True
        # 如果忽略关键字在文件名中出现，说明需要忽略该文件
        elif key in name:
            return True
    # 如果以上条件都不满足，则不需要忽略该文件
    return False

# 根据模型名称和加载的权重进行递归加载权重
def recursively_load_weights(orig_dict, hf_model, model_name):
    # 初始化未使用的权重列表
    unused_weights = []

    # 根据模型名称选择映射表
    if model_name == "encodec_24khz" or "encodec_32khz":
        MAPPING = MAPPING_24K
    elif model_name == "encodec_48khz":
        MAPPING = MAPPING_48K
    else:
        # 如果模型名称不支持，抛出数值错误
        raise ValueError(f"Unsupported model: {model_name}")
    # 遍历原始字典中的键值对
    for name, value in orig_dict.items():
        # 判断是否应该忽略该键，如果是则记录日志并继续下一个键值对
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        # 检查是否该键被使用
        is_used = False
        # 遍历映射字典中的键值对
        for key, mapped_key in MAPPING.items():
            # 如果键中包含通配符"*"，则进行匹配处理
            if "*" in key:
                prefix, suffix = key.split(".*.")
                if prefix in name and suffix in name:
                    key = suffix

            # 如果该键被使用
            if key in name:
                # Hack：避免 .embed 被初始化为 .embed_avg
                if key.endswith("embed") and name.endswith("embed_avg"):
                    continue

                is_used = True
                # 如果映射的键包含通配符"*"，则进行替换处理
                if "*" in mapped_key:
                    layer_index = name.split(key)[0].split(".")[-2]
                    mapped_key = mapped_key.replace("*", layer_index)
                # 根据键值名称判断权重类型
                if "weight_g" in name:
                    weight_type = "weight_g"
                elif "weight_v" in name:
                    weight_type = "weight_v"
                # ... 其他权重类型的判断
                else:
                    weight_type = None
                # 递归设置模型中的参数值
                set_recursively(hf_model, mapped_key, value, name, weight_type)
            continue
        # 如果未使用该键，添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 输出未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")
# 使用torch.no_grad()修饰器，禁止进行梯度计算
@torch.no_grad()
# 将模型的权重复制/粘贴/调整到transformers设计中
def convert_checkpoint(
    model_name,
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    """
    复制/粘贴/调整模型的权重到transformers设计中。
    """
    # 如果配置路径不为空，则从预训练配置路径中加载配置信息
    if config_path is not None:
        config = EncodecConfig.from_pretrained(config_path)
    else:
        # 否则创建一个新的配置对象
        config = EncodecConfig()

    # 根据模型名称进行条件判断
    if model_name == "encodec_24khz":
        # 如果模型名称是"encodec_24khz"，则不做任何改变
        pass  # 配置已经是正确的
    elif model_name == "encodec_32khz":
        # 如果模型名称是"encodec_32khz"，则设置特定的配置参数
        config.upsampling_ratios = [8, 5, 4, 4]
        config.target_bandwidths = [2.2]
        config.num_filters = 64
        config.sampling_rate = 32_000
        config.codebook_size = 2048
        config.use_causal_conv = False
        config.normalize = False
        config.use_conv_shortcut = False
    elif model_name == "encodec_48khz":
        # 如果模型名称是"encodec_48khz"，则设置特定的配置参数
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
        # 模型名称未知则抛出数值错误
        raise ValueError(f"Unknown model name: {model_name}")

    # 根据配置创建模型
    model = EncodecModel(config)

    # 创建特征提取器
    feature_extractor = EncodecFeatureExtractor(
        feature_size=config.audio_channels,
        sampling_rate=config.sampling_rate,
        chunk_length_s=config.chunk_length_s,
        overlap=config.overlap,
    )
    # 将特征提取器保存到指定路径
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    # 加载原始检查点
    original_checkpoint = torch.load(checkpoint_path)
    if "best_state" in original_checkpoint:
        # 如果原始检查点中包含"best_state"，则可能有保存的训练状态，在这种情况下丢弃yaml结果，仅保留权重
        original_checkpoint = original_checkpoint["best_state"]
    # 递归加载权重
    recursively_load_weights(original_checkpoint, model, model_name)
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 如果存在repo_id，则将特征提取器和模型推送到hub
    if repo_id:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="encodec_24khz",
        type=str,
        help="要转换的模型。应为'encodec_24khz'、'encodec_32khz'、'encodec_48khz'之一。"
    )
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="原始检查点的路径")
    parser.add_argument("--config_path", default=None, type=str, help="要转换的模型的hf config.json的路径")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="输出PyTorch模型的路径。"
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="将转换后的模型上传到🤗 hub的位置。"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_checkpoint 函数，传入参数：模型名称、检查点路径、PyTorch转储文件夹路径、配置文件路径、是否推送到Hub
    convert_checkpoint(
        args.model,  # 模型名称
        args.checkpoint_path,  # 检查点路径
        args.pytorch_dump_folder_path,  # PyTorch转储文件夹路径
        args.config_path,  # 配置文件路径
        args.push_to_hub,  # 是否推送到Hub
    )
```