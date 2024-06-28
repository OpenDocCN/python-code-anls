# `.\models\univnet\convert_univnet.py`

```py
# 引入命令行参数解析模块
import argparse

# 引入 PyTorch 模块
import torch

# 从 transformers 库中引入 UnivNetConfig、UnivNetModel 和 logging 模块
from transformers import UnivNetConfig, UnivNetModel, logging

# 设置 logging 模块的详细信息级别
logging.set_verbosity_info()

# 获取名为 "transformers.models.univnet" 的日志记录器
logger = logging.get_logger("transformers.models.univnet")


# 定义函数：获取内核预测器键映射
def get_kernel_predictor_key_mapping(config: UnivNetConfig, old_prefix: str = "", new_prefix: str = ""):
    # 创建空字典 mapping 用于存储键映射关系
    mapping = {}

    # 初始卷积层映射
    mapping[f"{old_prefix}.input_conv.0.weight_g"] = f"{new_prefix}.input_conv.weight_g"
    mapping[f"{old_prefix}.input_conv.0.weight_v"] = f"{new_prefix}.input_conv.weight_v"
    mapping[f"{old_prefix}.input_conv.0.bias"] = f"{new_prefix}.input_conv.bias"

    # 遍历核预测器的残差块
    for i in range(config.kernel_predictor_num_blocks):
        # 第一个卷积层映射
        mapping[f"{old_prefix}.residual_convs.{i}.1.weight_g"] = f"{new_prefix}.resblocks.{i}.conv1.weight_g"
        mapping[f"{old_prefix}.residual_convs.{i}.1.weight_v"] = f"{new_prefix}.resblocks.{i}.conv1.weight_v"
        mapping[f"{old_prefix}.residual_convs.{i}.1.bias"] = f"{new_prefix}.resblocks.{i}.conv1.bias"

        # 第二个卷积层映射
        mapping[f"{old_prefix}.residual_convs.{i}.3.weight_g"] = f"{new_prefix}.resblocks.{i}.conv2.weight_g"
        mapping[f"{old_prefix}.residual_convs.{i}.3.weight_v"] = f"{new_prefix}.resblocks.{i}.conv2.weight_v"
        mapping[f"{old_prefix}.residual_convs.{i}.3.bias"] = f"{new_prefix}.resblocks.{i}.conv2.bias"

    # 核输出卷积层映射
    mapping[f"{old_prefix}.kernel_conv.weight_g"] = f"{new_prefix}.kernel_conv.weight_g"
    mapping[f"{old_prefix}.kernel_conv.weight_v"] = f"{new_prefix}.kernel_conv.weight_v"
    mapping[f"{old_prefix}.kernel_conv.bias"] = f"{new_prefix}.kernel_conv.bias"

    # 偏置输出卷积层映射
    mapping[f"{old_prefix}.bias_conv.weight_g"] = f"{new_prefix}.bias_conv.weight_g"
    mapping[f"{old_prefix}.bias_conv.weight_v"] = f"{new_prefix}.bias_conv.weight_v"
    mapping[f"{old_prefix}.bias_conv.bias"] = f"{new_prefix}.bias_conv.bias"

    # 返回映射字典
    return mapping


# 定义函数：获取键映射
def get_key_mapping(config: UnivNetConfig):
    # 创建空字典 mapping 用于存储键映射关系
    mapping = {}

    # 注意：初始卷积层键保持不变

    # LVC 残差块（未完成的注释）
    # 遍历配置中的残差块步幅大小列表的长度
    for i in range(len(config.resblock_stride_sizes)):
        # 设置 LVCBlock 的初始卷积层权重和偏置的映射关系
        mapping[f"res_stack.{i}.convt_pre.1.weight_g"] = f"resblocks.{i}.convt_pre.weight_g"
        mapping[f"res_stack.{i}.convt_pre.1.weight_v"] = f"resblocks.{i}.convt_pre.weight_v"
        mapping[f"res_stack.{i}.convt_pre.1.bias"] = f"resblocks.{i}.convt_pre.bias"

        # 获取并更新核预测器的映射关系
        kernel_predictor_mapping = get_kernel_predictor_key_mapping(
            config, old_prefix=f"res_stack.{i}.kernel_predictor", new_prefix=f"resblocks.{i}.kernel_predictor"
        )
        mapping.update(kernel_predictor_mapping)

        # 遍历当前残差块的扩张大小列表的长度
        for j in range(len(config.resblock_dilation_sizes[i])):
            # 设置 LVC 残差块内部卷积层权重和偏置的映射关系
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.weight_g"] = f"resblocks.{i}.resblocks.{j}.conv.weight_g"
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.weight_v"] = f"resblocks.{i}.resblocks.{j}.conv.weight_v"
            mapping[f"res_stack.{i}.conv_blocks.{j}.1.bias"] = f"resblocks.{i}.resblocks.{j}.conv.bias"

    # 设置输出卷积层权重和偏置的映射关系
    mapping["conv_post.1.weight_g"] = "conv_post.weight_g"
    mapping["conv_post.1.weight_v"] = "conv_post.weight_v"
    mapping["conv_post.1.bias"] = "conv_post.bias"

    # 返回映射字典
    return mapping
# 定义函数，用于修改状态字典的键，并且可以移除指定的键
def rename_state_dict(state_dict, keys_to_modify, keys_to_remove):
    # 初始化一个空的模型状态字典
    model_state_dict = {}
    # 遍历原始状态字典中的每个键值对
    for key, value in state_dict.items():
        # 如果当前键在要移除的键集合中，则跳过处理
        if key in keys_to_remove:
            continue
        
        # 如果当前键在要修改的键映射中
        if key in keys_to_modify:
            # 使用映射中的新键名替换当前键，并将对应的值存入模型状态字典
            new_key = keys_to_modify[key]
            model_state_dict[new_key] = value
        else:
            # 否则直接将当前键值对存入模型状态字典
            model_state_dict[key] = value
    
    # 返回修改后的模型状态字典
    return model_state_dict


def convert_univnet_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
    safe_serialization=False,
):
    # 使用 torch 加载模型的状态字典，指定在 CPU 上加载
    model_state_dict_base = torch.load(checkpoint_path, map_location="cpu")
    # 获取生成器的状态字典
    state_dict = model_state_dict_base["model_g"]

    # 如果提供了配置文件路径，则从预训练配置文件中加载配置，否则使用默认配置
    if config_path is not None:
        config = UnivNetConfig.from_pretrained(config_path)
    else:
        config = UnivNetConfig()

    # 获取需要修改的键映射
    keys_to_modify = get_key_mapping(config)
    # 初始化要移除的键集合为空
    keys_to_remove = set()
    # 使用定义的函数重命名状态字典中的键，并且应用修改后的映射
    hf_state_dict = rename_state_dict(state_dict, keys_to_modify, keys_to_remove)

    # 创建 UnivNetModel 的实例
    model = UnivNetModel(config)
    # 应用权重规范化，因为原始检查点已应用权重规范化
    model.apply_weight_norm()
    # 加载经过重命名的状态字典
    model.load_state_dict(hf_state_dict)
    # 移除权重规范化，为推断准备
    model.remove_weight_norm()

    # 将模型保存到指定路径，支持安全序列化选项
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)

    # 如果提供了 repo_id，则推送模型到 hub
    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数选项
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", help="Whether to save the model using `safetensors`."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数，传入命令行参数解析得到的参数
    convert_univnet_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
        args.safe_serialization,
    )


if __name__ == "__main__":
    main()
```