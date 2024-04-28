# `.\transformers\models\clap\convert_clap_original_pytorch_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，授权使用此文件
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
import argparse
import re
from laion_clap import CLAP_Module
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel

# 定义需要修改的键值对映射关系
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

# 从预训练模型中加载特征提取器
processor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused", truncation="rand_trunc")

# 初始化 CLAP 模型
def init_clap(checkpoint_path, model_type, enable_fusion=False):
    model = CLAP_Module(
        amodel=model_type,
        enable_fusion=enable_fusion,
    )
    model.load_ckpt(checkpoint_path)
    return model

# 从原始 CLAP 模型中获取配置信息
def get_config_from_original(clap_model):
    audio_config = {
        "patch_embeds_hidden_size": clap_model.model.audio_branch.embed_dim,
        "depths": clap_model.model.audio_branch.depths,
        "hidden_size": clap_model.model.audio_projection[0].in_features,
    }

    text_config = {"hidden_size": clap_model.model.text_branch.pooler.dense.in_features}

    return ClapConfig(audio_config=audio_config, text_config=text_config)

# 重命名状态字典
def rename_state_dict(state_dict):
    model_state_dict = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"
    # 遍历状态字典中的键值对
    for key, value in state_dict.items():
        # 检查是否有需要修改的键
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                # 替换需要修改的键
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # 替换顺序层为列表
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # 因为在 CLAP 中他们使用 `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # 将 qkv 拆分为查询、键和值
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            model_state_dict[key] = value

    return model_state_dict
# 将 CLAP 模型从 fairseq 转换为 PyTorch 模型，并保存在指定路径下
def convert_clap_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path, model_type, enable_fusion=False):
    # 初始化 CLAP 模型
    clap_model = init_clap(checkpoint_path, model_type, enable_fusion=enable_fusion)

    # 将模型设置为评估模式
    clap_model.eval()
    # 获取模型的状态字典
    state_dict = clap_model.model.state_dict()
    # 重命名状态字典的键名
    state_dict = rename_state_dict(state_dict)

    # 从 CLAP 模型获取 Transformers 配置
    transformers_config = get_config_from_original(clap_model)
    # 设置是否启用融合
    transformers_config.audio_config.enable_fusion = enable_fusion
    # 创建 CLAP 模型对象
    model = ClapModel(transformers_config)

    # 忽略声谱图嵌入层的加载
    model.load_state_dict(state_dict, strict=False)

    # 保存 PyTorch 模型和配置
    model.save_pretrained(pytorch_dump_folder_path)
    transformers_config.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--enable_fusion", action="store_true", help="Whether to enable fusion or not")
    parser.add_argument("--model_type", default="HTSAT-tiny", type=str, help="Whether to enable fusion or not")
    args = parser.parse_args()

    # 调用转换函数，将 CLAP 检查点转换为 PyTorch 模型
    convert_clap_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.model_type, args.enable_fusion
    )
```