# `.\transformers\models\wavlm\convert_wavlm_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置代码文件的编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可文件的要求，对代码进行了版权声明
# 你可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发的软件，没有任何保证或条件，无论是明示的还是暗示的。
# 请参阅特定语言规定的许可文件，以及许可下的限制和条件
# 转换 WavLM 检查点
# 导入 argparse 模块
# 导入 torch 模块
# 步骤1：克隆 https://github.com/microsoft/unilm
# 步骤2：git 检出到 https://github.com/microsoft/unilm/commit/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd
# 步骤3：切换到 unilm 目录
# 步骤4：创建符号链接，链接到 wavlm/modules.py 文件
# 从 unilm.wavlm.WavLM 模块中导入 WavLM 和 WavLMConfig 类
# 从 transformers 模块中导入 WavLMConfig, WavLMModel, logging
# 设置日志级别为 info
# 获取记录器实例
# 定义映射关系字典，用于将旧模型的参数映射到新模型的参数
# 定义顶级键列表
# 递归设置值，用于递归设置参数的值
    # 如果权重类型是 "weight"
    if weight_type == "weight":
        # 设置模型指针的权重数据为给定值
        hf_pointer.weight.data = value
    # 如果权重类型是 "weight_g"
    elif weight_type == "weight_g":
        # 设置模型指针的权重梯度数据为给定值
        hf_pointer.weight_g.data = value
    # 如果权重类型是 "weight_v"
    elif weight_type == "weight_v":
        # 设置模型指针的权重变量数据为给定值
        hf_pointer.weight_v.data = value
    # 如果权重类型是 "bias"
    elif weight_type == "bias":
        # 设置模型指针的偏置数据为给定值
        hf_pointer.bias.data = value
    # 如果权重类型不是以上任何类型
    else:
        # 设置模型指针的数据为给定值
        hf_pointer.data = value

    # 记录初始化信息，包括键和权重类型，如果权重类型不为 None，则记录其完整名称
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载权重的函数
def recursively_load_weights(fairseq_model, hf_model):
    # 未使用的权重列表
    unused_weights = []
    # 获取fairseq模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 获取hf_model的特征提取器
    feature_extractor = hf_model.feature_extractor

    # 遍历fairseq模型的状态字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果名称包含"conv_layers"
        if "conv_layers" in name:
            # 加载卷积层
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            # 遍历MAPPING中的键值对
            for key, mapped_key in MAPPING.items():
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果mapped_key包含通配符"*"
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称获取权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name and "relative_attention_bias" not in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: don't match quantizer.weight_proj
                        weight_type = "weight"
                    else:
                        weight_type = None

                    # 递归设定hf_model的属性值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未使用，加入未使用的权重列表
        if not is_used:
            unused_weights.append(name)

    # 打印未使用的权重列表
    logger.warning(f"Unused weights: {unused_weights}")


# 加载卷积层的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    # 获取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 如果是第一个类型
    if type_id == 0:
        # 如果名称包含"bias"
        if "bias" in name:
            # 断言检查值的形状是否匹配，如果匹配，将值赋给特征提取器的卷积层的偏置
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果名称包含"weight"
        elif "weight" in name:
            # 断言检查值的形状是否匹配，如果匹配，将值赋给特征提取器的卷积层的权重
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        # 如果 type_id 等于 2 且不使用组归一化，或者 type_id 等于 2 且 layer_id 等于 0 且使用组归一化
        if "bias" in name:
            # 如果名字中包含"bias"，则判断值的形状是否与特征提取器(conv_layers)中对应层的layer_norm.bias.data的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            # 如果断言通过，将值赋给特征提取器对应层的layer_norm.bias.data
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            # 记录日志，表示成功从 full_name 初始化了 layer_id 层的特征提取器(layer norm)的权重
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            # 如果名字中包含"weight"，则判断值的形状是否与特征提取器(conv_layers)中对应层的layer_norm.weight.data的形状相同
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            # 如果断言通过，将值赋给特征提取器对应层的layer_norm.weight.data
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            # 记录日志，表示成功从 full_name 初始化了 layer_id 层的特征提取器(layer norm)的权重
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        # 如果不满足上述条件，则将 full_name 加入 unused_weights 列表中
        unused_weights.append(full_name)
# 导入必要的库
import torch
import argparse
from transformers import WavLMConfig, WavLMModel

# 禁用梯度计算
@torch.no_grad()
# 定义函数：将一个fairseq模型的checkpoint转换为PyTorch模型
def convert_wavlm_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):

    # 加载预训练的checkpoint
    checkpoint = torch.load(checkpoint_path)
    # 解析模型配置
    cfg = WavLMConfigOrig(checkpoint["cfg"])
    # 创建WavLM模型对象
    model = WavLMOrig(cfg)
    # 加载模型参数
    model.load_state_dict(checkpoint["model"])
    # 设置模型为评估模式
    model.eval()

    if config_path is not None:
        # 从预训练模型的配置文件加载配置信息
        config = WavLMConfig.from_pretrained(config_path)
    else:
        # 创建默认配置
        config = WavLMConfig()

    # 创建HF WavLM模型对象
    hf_wavlm = WavLMModel(config)

    # 递归地加载权重
    recursively_load_weights(model, hf_wavlm)

    # 保存转换后的模型
    hf_wavlm.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将fairseq模型的checkpoint转换为PyTorch模型
    convert_wavlm_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```