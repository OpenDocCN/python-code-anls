# `.\transformers\models\univnet\convert_univnet.py`

```
# 引入 argparse 库，用于处理命令行参数
import argparse

# 引入 torch 库
import torch

# 从 transformers 库中引入 UnivNetConfig 和 UnivNetModel
from transformers import UnivNetConfig, UnivNetModel, logging

# 设置 logging 的详细程度为 info
logging.set_verbosity_info()

# 获取名为 "transformers.models.univnet" 的 logger
logger = logging.get_logger("transformers.models.univnet")


# 根据给定的 config 和前缀，返回旧键和新键之间的映射关系
def get_kernel_predictor_key_mapping(config: UnivNetConfig, old_prefix: str = "", new_prefix: str = ""):
    mapping = {}
    # 初始卷积层
    mapping[f"{old_prefix}.input_conv.0.weight_g"] = f"{new_prefix}.input_conv.weight_g"
    mapping[f"{old_prefix}.input_conv.0.weight_v"] = f"{new_prefix}.input_conv.weight_v"
    mapping[f"{old_prefix}.input_conv.0.bias"] = f"{new_prefix}.input_conv.bias"

    # 核心预测器的 ResNet 块
    for i in range(config.kernel_predictor_num_blocks):
        mapping[f"{old_prefix}.residual_convs.{i}.1.weight_g"] = f"{new_prefix}.resblocks.{i}.conv1.weight_g"
        mapping[f"{old_prefix}.residual_convs.{i}.1.weight_v"] = f"{new_prefix}.resblocks.{i}.conv1.weight_v"
        mapping[f"{old_prefix}.residual_convs.{i}.1.bias"] = f"{new_prefix}.resblocks.{i}.conv1.bias"

        mapping[f"{old_prefix}.residual_convs.{i}.3.weight_g"] = f"{new_prefix}.resblocks.{i}.conv2.weight_g"
        mapping[f"{old_prefix}.residual_convs.{i}.3.weight_v"] = f"{new_prefix}.resblocks.{i}.conv2.weight_v"
        mapping[f"{old_prefix}.residual_convs.{i}.3.bias"] = f"{new_prefix}.resblocks.{i}.conv2.bias"

    # 核心输出卷积
    mapping[f"{old_prefix}.kernel_conv.weight_g"] = f"{new_prefix}.kernel_conv.weight_g"
    mapping[f"{old_prefix}.kernel_conv.weight_v"] = f"{new_prefix}.kernel_conv.weight_v"
    mapping[f"{old_prefix}.kernel_conv.bias"] = f"{new_prefix}.kernel_conv.bias"

    # 偏置输出卷积
    mapping[f"{old_prefix}.bias_conv.weight_g"] = f"{new_prefix}.bias_conv.weight_g"
    mapping[f"{old_prefix}.bias_conv.weight_v"] = f"{new_prefix}.bias_conv.weight_v"
    mapping[f"{old_prefix}.bias_conv.bias"] = f"{new_prefix}.bias_conv.bias"

    return mapping


# 根据给定的 config 返回键映射关系
def get_key_mapping(config: UnivNetConfig):
    mapping = {}

    # 注意：初始卷积层的键是相同的

    # LVC 剩余块
    # 遍历每个 resblock_stride_sizes 中的值
    for i in range(len(config.resblock_stride_sizes)):
        # LVCBlock 的初始 convt 层参数映射
        mapping[f"res_stack.{i}.convt_pre.1.weight_g"] = f"resblocks.{i}.convt_pre.weight_g"
        mapping[f"res_stack.{i}.convt_pre.1.weight_v"] = f"resblocks.{i}.convt_pre.weight_v"
        mapping[f"res_stack.{i}.convt_pre.1.bias"] = f"resblocks.{i}.convt_pre.bias"

        # 获取 Kernel predictor 的参数映射
        kernel_predictor_mapping = get_kernel_predictor_key_mapping(
            config, old_prefix=f"res_stack.{i}.kernel_predictor", new_prefix=f"resblocks.{i}.kernel_predictor"
        )
        # 更新参数映射字典
        mapping.update(kernel_predictor_mapping)

        # 遍历每个 LVC Residual block
        for j in range(len(config.resblock_dilation_sizes[i])):
            # 进行 conv_blocks 的参数映射
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.weight_g"] = f"resblocks.{i}.resblocks.{j}.conv.weight_g"
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.weight_v"] = f"resblocks.{i}.resblocks.{j}.conv.weight_v"
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.bias"] = f"resblocks.{i}.resblocks.{j}.conv.bias"

    # 输出层参数的映射
    mapping["conv_post.1.weight_g"] = "conv_post.weight_g"
    mapping["conv_post.1.weight_v"] = "conv_post.weight_v"
    mapping["conv_post.1.bias"] = "conv_post.bias"

    # 返回参数映射字典
    return mapping
# 重命名状态字典的键并移除指定的键
def rename_state_dict(state_dict, keys_to_modify, keys_to_remove):
    model_state_dict = {}  # 创建一个空字典，用于存储处理后的状态字典
    for key, value in state_dict.items():  # 遍历原状态字典的键值对
        if key in keys_to_remove:  # 如果键在要移除的键列表中则跳过当前循环
            continue

        if key in keys_to_modify:  # 如果键在要修改的键列表中
            new_key = keys_to_modify[key]  # 获取新的键名
            model_state_dict[new_key] = value  # 将新键名和对应值添加到新状态字典中
        else:
            model_state_dict[key] = value  # 否则将原键名和对应值添加到新状态字典中
    return model_state_dict  # 返回处理后的状态字典


def convert_univnet_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
    safe_serialization=False,
):
    model_state_dict_base = torch.load(checkpoint_path, map_location="cpu")  # 加载原始检查点的状态字典
    state_dict = model_state_dict_base["model_g"]  # 获取 generator 的状态字典

    if config_path is not None:  # 如果提供了配置文件路径
        config = UnivNetConfig.from_pretrained(config_path)  # 从预训练模型中加载配置
    else:
        config = UnivNetConfig()  # 否则使用默认配置

    keys_to_modify = get_key_mapping(config)  # 获取键映射
    keys_to_remove = set()  # 创建一个空的要移除的键的集合
    hf_state_dict = rename_state_dict(state_dict, keys_to_modify, keys_to_remove)  # 重命名状态字典的键并移除指定的键

    model = UnivNetModel(config)  # 创建模型对象
    model.apply_weight_norm()  # 应用权重标准化，因为原始检查点已经应用了权重标准化
    model.load_state_dict(hf_state_dict)  # 加载处理后的状态字典
    model.remove_weight_norm()  # 移除权重标准化，为进行推理做准备

    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)  # 将转换后的模型保存到指定路径，并可选择是否使用安全序列化

    if repo_id:  # 如果提供了 repo_id
        print("Pushing to the hub...")  # 打印推送到 hub 的消息
        model.push_to_hub(repo_id)  # 将模型推送到 hub


def main():
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")  # 添加原始检查点路径的命令行参数
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")  # 添加配置文件路径的命令行参数
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )  # 添加输出 PyTorch 模型路径的命令行参数
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )  # 添加在 🤗 hub 上上传转换后的模型的命令行参数
    parser.add_argument(
        "--safe_serialization", action="store_true", help="Whether to save the model using `safetensors`."
    )  # 添加是否使用 'safetensors' 保存模型的命令行参数

    args = parser.parse_args()  # 解析命令行参数

    convert_univnet_checkpoint(  # 调用转换 UnivNet 检查点的函数
        args.checkpoint_path,  # 原始检查点路径
        args.pytorch_dump_folder_path,  # 输出 PyTorch 模型路径
        args.config_path,  # 配置文件路径
        args.push_to_hub,  # 在 hub 上上传的位置
        args.safe_serialization,  # 是否使用安全序列化
    )


if __name__ == "__main__":
    main()  # 执行 main 函数
```