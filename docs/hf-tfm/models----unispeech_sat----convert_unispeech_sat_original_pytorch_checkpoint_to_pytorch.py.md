# `.\transformers\models\unispeech_sat\convert_unispeech_sat_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明
# 在 Apache 许可证 2.0 版的许可下，除非符合许可的要求，否则不得使用此文件。
# 你可以在以下网址获得许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 如果没有适用的法律要求或书面同意，本软件是基于"按原样"的基础分发的，没有任何种类的担保或条件，无论是明示的还是暗示的。
# 查看特定语言下的特定许可证，以了解权限和限制。
# 转换 UniSpeechSat 检查点。
# 导入 argparse 模块
# 导入 fairseq 模块
# 导入 torch 模块
# 从 transformers 模块中导入 UniSpeechSatConfig、UniSpeechSatForCTC、UniSpeechSatForPreTraining 和 logging
# 设置记录日志的详细程度为信息级别
# 获取记录器实例
# 将 fairseq 模型的键和值映射到 transformers 模型的键和值
# 定义顶层键列表
# 递归设置 hf_pointer 中的键对应的值
    # 如果权重类型为 "bias"，则将其数据设置为给定的值
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    # 否则，将权重数据设置为给定的值
    else:
        hf_pointer.data = value
    
    # 记录一条日志信息，说明该权重已经从指定的源位置初始化
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重
def recursively_load_weights(fairseq_model, hf_model):
    # 未使用的权重列表
    unused_weights = []
    # 获取fairseq模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取hf_model中的特征提取器
    feature_extractor = hf_model.unispeech_sat.feature_extractor

    for name, value in fairseq_dict.items():
        is_used = False
        # 如果名称中包含"conv_layers"
        if "conv_layers" in name:
            # 调用加载卷积层函数
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            # 遍历MAPPING字典
            for key, mapped_key in MAPPING.items():
                # 拼接"unispeech_sat."到mapped_key
                mapped_key = "unispeech_sat." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    if "layer_norm_for_extract" in name and (".".join(name.split(".")[:-1]) != key):
                        # 特殊情况，因为命名非常相似，跳过当前循环
                        continue
                    is_used = True
                    if "*" in mapped_key:
                        # 获取层索引
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
                        # 如果没有匹配项，则为空
                        weight_type = None
                    # 递归设置权重
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未使用，将名称添加到未使用的权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])
    # 检查神经元类型是否为 0
    if type_id == 0:
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 如果值的形状与卷积层的偏置项的形状不相同，抛出数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            # 将值赋给卷积层的偏置项
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            # 记录日志，说明卷积层的偏置项已从指定名称中初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"
        elif "weight" in name:
            # 如果值的形状与卷积层的权重项的形状不相同，抛出数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            # 将值赋给卷积层的权重项
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            # 记录日志，说明卷积层的权重项已从指定名称中初始化
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    # 如果神经元类型为2且未使用组规范，或者神经元类型为2且是第一层且使用组规范
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        # 如果名称中包含 "bias"
        if "bias" in name:
            # 如果值的形状与卷积层的归一化层偏置项的形状不相同，抛出数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor[layer_id].layer_norm.bias.data.shape} was found."
                )
            # 将值赋给卷积层的归一化层偏置项
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，说明卷积层的归一化层偏置项已从指定名称中初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        # 如果名称中包含 "weight"
        elif "weight" in name:
            # 如果值的形状与卷积层的归一化层权重项的形状不相同，抛出数值错误
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
                )
            # 将值赋给卷积层的归一化层权重项
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，说明卷积层的归一化层权重项已从指定名称中初始化
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 将未使用的权重名称添加到列表中
        unused_weights.append(full_name)
# 借助torch.no_grad()上下文管理器，确保在其范围内的操作不会被track到计算图中
@torch.no_grad()
# 将UniSpeech权重转换为transformers设计
def convert_unispeech_sat_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了config_path，则使用预训练的UniSpeechSatConfig
    if config_path is not None:
        config = UniSpeechSatConfig.from_pretrained(config_path)
    # 否则创建一个新的UniSpeechSatConfig
    else:
        config = UniSpeechSatConfig()

    # 重置dict_path
    dict_path = ""

    # 根据is_finetuned的值选择使用UniSpeechSatForCTC或UniSpeechSatForPreTraining模型
    if is_finetuned:
        hf_wav2vec = UniSpeechSatForCTC(config)
    else:
        hf_wav2vec = UniSpeechSatForPreTraining(config)

    # 加载fairseq模型并任务（task） 
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
    )
    model = model[0].eval()  # 设置模型为评估模式

    # 递归地加载权重
    recursively_load_weights(model, hf_wav2vec)

    # 保存转换后的PyTorch模型
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    args = parser.parse_args()
    # 调用转换函数，根据参数配置是否是fine-tuned模型
    convert_unispeech_sat_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```