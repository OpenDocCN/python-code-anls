# `.\models\clap\convert_clap_original_pytorch_to_hf.py`

```
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 re 模块，用于正则表达式操作
import re

# 从 laion_clap 模块中导入 CLAP_Module 类
from laion_clap import CLAP_Module

# 从 transformers 库中导入 AutoFeatureExtractor、ClapConfig 和 ClapModel 类
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel

# 定义一个字典，用于将旧模型中的键名映射到新模型中对应的键名
KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

# 使用 laion/clap-htsat-unfused 模型来初始化自动特征提取器
processor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused", truncation="rand_trunc")


# 初始化 CLAP 模型的函数，接受检查点路径、模型类型和是否启用融合作为参数
def init_clap(checkpoint_path, model_type, enable_fusion=False):
    model = CLAP_Module(
        amodel=model_type,
        enable_fusion=enable_fusion,
    )
    # 加载模型检查点
    model.load_ckpt(checkpoint_path)
    return model


# 从原始 CLAP 模型中获取配置信息的函数，返回包含音频和文本配置的 ClapConfig 对象
def get_config_from_original(clap_model):
    # 从 CLAP 模型中提取音频配置信息
    audio_config = {
        "patch_embeds_hidden_size": clap_model.model.audio_branch.embed_dim,
        "depths": clap_model.model.audio_branch.depths,
        "hidden_size": clap_model.model.audio_projection[0].in_features,
    }

    # 从 CLAP 模型中提取文本配置信息
    text_config = {"hidden_size": clap_model.model.text_branch.pooler.dense.in_features}

    return ClapConfig(audio_config=audio_config, text_config=text_config)


# 重命名状态字典中键名的函数
def rename_state_dict(state_dict):
    model_state_dict = {}

    # 正则表达式模式，用于匹配包含 "sequential" 的层次结构的键名
    sequential_layers_pattern = r".*sequential.(\d+).*"

    # 正则表达式模式，用于匹配包含 "_projection" 的文本投影层级结构的键名
    text_projection_pattern = r".*_projection.(\d+).*"
    for key, value in state_dict.items():
        # 检查是否有需要修改的键名
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                # 替换需要修改的键名
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # 匹配顺序层模式，并进行替换
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            # 匹配文本投影模式，确定投影层编号
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # 根据 CLAP 中的使用情况，确定 Transformers 投影层编号
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # 将 qkv 分割为查询、键和值
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            # 将分割后的查询、键和值存入模型状态字典
            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            # 将未处理的键值对存入模型状态字典
            model_state_dict[key] = value

    # 返回最终的模型状态字典
    return model_state_dict
# 定义一个函数，用于将 CLAP 模型的检查点转换为 PyTorch 模型
def convert_clap_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, model_type, enable_fusion=False):
    # 初始化 CLAP 模型，使用给定的检查点路径、模型类型和是否启用融合功能
    clap_model = init_clap(checkpoint_path, model_type, enable_fusion=enable_fusion)

    # 将 CLAP 模型设置为评估模式
    clap_model.eval()

    # 获取 CLAP 模型的状态字典
    state_dict = clap_model.model.state_dict()

    # 重命名状态字典的键名，确保适配 PyTorch 的命名规则
    state_dict = rename_state_dict(state_dict)

    # 从 CLAP 模型中获取原始的 Transformers 配置
    transformers_config = get_config_from_original(clap_model)

    # 根据配置创建一个新的 CLAP 模型
    transformers_config.audio_config.enable_fusion = enable_fusion
    model = ClapModel(transformers_config)

    # 加载模型的状态字典，忽略掉声谱图嵌入层（如果有的话）
    model.load_state_dict(state_dict, strict=False)

    # 将转换后的模型保存为 PyTorch 模型
    model.save_pretrained(pytorch_dump_folder_path)

    # 保存 Transformers 配置文件到指定路径
    transformers_config.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定输出的 PyTorch 模型路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数，用于指定 Fairseq 检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数，用于指定模型配置文件的路径（例如 hf config.json）
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数，用于指示是否启用融合功能
    parser.add_argument("--enable_fusion", action="store_true", help="Whether to enable fusion or not")
    # 添加命令行参数，用于指定模型类型，默认为 "HTSAT-tiny"
    parser.add_argument("--model_type", default="HTSAT-tiny", type=str, help="Whether to enable fusion or not")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数，将 CLAP 模型的检查点转换为 PyTorch 模型
    convert_clap_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.model_type, args.enable_fusion
    )
```