# `.\transformers\models\blenderbot\convert_blenderbot_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 许可证信息
"""转换 Blenderbot 检查点。"""

# 导入必要的库
import argparse

import torch

from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义要替换的模式列表
PATTERNS = [
    ["attention", "attn"],
    ["encoder_attention", "encoder_attn"],
    ["q_lin", "q_proj"],
    ["k_lin", "k_proj"],
    ["v_lin", "v_proj"],
    ["out_lin", "out_proj"],
    ["norm_embeddings", "layernorm_embedding"],
    ["position_embeddings", "embed_positions"],
    ["embeddings", "embed_tokens"],
    ["ffn.lin", "fc"],
]

# 定义函数，用于重命名 state_dict 键
def rename_state_dict_key(k):
    # 如果键为 "embeddings.weight"，则替换为 "shared.weight"
    if k == "embeddings.weight":
        return "shared.weight"

    # 使用 PATTERNS 列表中的替换规则对键进行替换
    for parlai_name, hf_name in PATTERNS:
        k = k.replace(parlai_name, hf_name)

    # 根据键的前缀进行进一步的替换
    if k.startswith("encoder"):
        k = k.replace(".attn", ".self_attn")
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "final_layer_norm")
    elif k.startswith("decoder"):
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "encoder_attn_layer_norm")
        k = k.replace("norm3", "final_layer_norm")
    return k

# 定义函数，用于重命名 layernorm 键
def rename_layernorm_keys(sd):
    keys = [
        "model.encoder.layernorm_embedding.weight",
        "model.encoder.layernorm_embedding.bias",
        "model.decoder.layernorm_embedding.weight",
        "model.decoder.layernorm_embedding.bias",
    ]
    for k in keys:
        v = sd.pop(k)
        new_k = k.replace("layernorm_embedding", "layer_norm")
        assert new_k not in sd
        sd[new_k] = v

# 定义要忽略的键列表
IGNORE_KEYS = ["START"]

# 定义转换 Parlai 检查点的函数
@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # 加载模型
    model = torch.load(checkpoint_path, map_location="cpu")
    # 获取模型参数
    sd = model["model"]
    # 加载 Blenderbot 的配置文件
    cfg = BlenderbotConfig.from_json_file(config_json_path)
    # 创建 Blenderbot 模型
    m = BlenderbotForConditionalGeneration(cfg)
    # 获取有效的模型键
    valid_keys = m.model.state_dict().keys()
    # 用于存储转换失败的键
    failures = []
    # 用于存储键的映射关系
    mapping = {}
    # 遍历模型参数
    for k, v in sd.items():
        # 如果键在忽略列表中，则跳过
        if k in IGNORE_KEYS:
            continue

        # 对键进行重命名
        new_k = rename_state_dict_key(k)
        # 如果重命名后的键不在有效的模型键中，则添加到失败列表中
        if new_k not in valid_keys:
            failures.append([k, new_k])
        else:
            # 否则将键和值添加到映射中
            mapping[new_k] = v
    # 如果配置中指定在加载之前进行归一化，则对应用于Blenderbot-3B检查点的操作，将layernorm_embedding重命名为layer_norm
    if cfg.normalize_before:  
        rename_layernorm_keys(sd)
    # 加载模型的状态字典
    m.model.load_state_dict(mapping, strict=True)
    # 将模型转换为半精度浮点数
    m.half()
    # 保存预训练模型到指定的PyTorch转储文件夹路径
    m.save_pretrained(pytorch_dump_folder_path)
# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数：源文件路径，类型为字符串，用于指定要转换的模型文件路径
    parser.add_argument("--src_path", type=str, help="like blenderbot-model.bin")
    # 添加可选参数：保存目录，默认为"hf_blenderbot"，类型为字符串，用于指定转换后模型的保存目录
    parser.add_argument("--save_dir", default="hf_blenderbot", type=str, help="Where to save converted model.")
    # 添加可选参数：Hugging Face 配置文件路径，默认为"blenderbot-3b-config.json"，类型为字符串，用于指定转换后模型的配置文件路径
    parser.add_argument(
        "--hf_config_json", default="blenderbot-3b-config.json", type=str, help="Path to config to use"
    )
    # 解析命令行参数，并将它们存储在args对象中
    args = parser.parse_args()
    # 调用convert_parlai_checkpoint函数，传入源文件路径、保存目录和Hugging Face配置文件路径作为参数
    convert_parlai_checkpoint(args.src_path, args.save_dir, args.hf_config_json)
```