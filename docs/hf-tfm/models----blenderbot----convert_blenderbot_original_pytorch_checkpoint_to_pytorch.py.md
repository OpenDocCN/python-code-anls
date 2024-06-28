# `.\models\blenderbot\convert_blenderbot_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明与许可证信息
# 该脚本受 Apache License, Version 2.0 许可证保护，详见链接
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律另有要求或书面同意，本软件是基于"原样"提供的，不提供任何担保或条件，无论是明示的还是暗示的。
# 有关许可证的详细信息，请参阅许可证。
"""Convert Blenderbot checkpoint."""

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import torch  # 导入 PyTorch 库

# 从 transformers 库中导入 BlenderbotConfig 和 BlenderbotForConditionalGeneration
from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration
# 从 transformers.utils 中导入 logging 模块
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义一组模式，用于重命名状态字典的键
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

# 函数：根据指定规则重命名状态字典的键
def rename_state_dict_key(k):
    # 特殊情况：如果键为 "embeddings.weight"，则重命名为 "shared.weight"
    if k == "embeddings.weight":
        return "shared.weight"

    # 遍历预定义的模式列表，逐一替换匹配的键名
    for parlai_name, hf_name in PATTERNS:
        k = k.replace(parlai_name, hf_name)

    # 根据模型结构进一步重命名 encoder 和 decoder 的特定键
    if k.startswith("encoder"):
        k = k.replace(".attn", ".self_attn")
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "final_layer_norm")
    elif k.startswith("decoder"):
        k = k.replace("norm1", "self_attn_layer_norm")
        k = k.replace("norm2", "encoder_attn_layer_norm")
        k = k.replace("norm3", "final_layer_norm")
    return k

# 函数：根据指定规则重命名 Layernorm 层的键
def rename_layernorm_keys(sd):
    # 定义需要重命名的 Layernorm 层的键列表
    keys = [
        "model.encoder.layernorm_embedding.weight",
        "model.encoder.layernorm_embedding.bias",
        "model.decoder.layernorm_embedding.weight",
        "model.decoder.layernorm_embedding.bias",
    ]
    # 遍历每个键，将 Layernorm 替换为 layer_norm，并进行键值对的映射更新
    for k in keys:
        v = sd.pop(k)  # 弹出旧键对应的值
        new_k = k.replace("layernorm_embedding", "layer_norm")  # 构造新的键名
        assert new_k not in sd  # 断言新键名不在原始字典中
        sd[new_k] = v  # 更新字典中的键值对

# 定义需要忽略的键的列表
IGNORE_KEYS = ["START"]

# 函数：将 Parlai 模型的检查点转换为适合 Blenderbot 结构的检查点
@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # 使用 map_location="cpu" 加载模型的检查点数据
    model = torch.load(checkpoint_path, map_location="cpu")
    sd = model["model"]  # 获取模型的状态字典
    cfg = BlenderbotConfig.from_json_file(config_json_path)  # 从 JSON 文件中加载配置信息
    m = BlenderbotForConditionalGeneration(cfg)  # 根据配置创建 Blenderbot 模型实例
    valid_keys = m.model.state_dict().keys()  # 获取 Blenderbot 模型的有效键集合
    failures = []  # 初始化失败列表，用于记录转换过程中的失败情况
    mapping = {}  # 初始化映射字典，用于记录成功映射的键值对
    for k, v in sd.items():
        if k in IGNORE_KEYS:  # 如果键在忽略列表中，则跳过处理
            continue

        new_k = rename_state_dict_key(k)  # 根据预定义规则重命名键名
        if new_k not in valid_keys:  # 如果重命名后的键名不在有效键集合中，记录到失败列表中
            failures.append([k, new_k])
        else:
            mapping[new_k] = v  # 否则，将映射后的键值对添加到映射字典中
    # 如果 cfg.normalize_before 为真，则说明使用 Blenderbot-3B 的检查点。需要将 layernorm_embedding 重命名为 layer_norm
    if cfg.normalize_before:
        # 调用函数 rename_layernorm_keys(sd)，对模型状态字典进行重命名操作
        rename_layernorm_keys(sd)
    
    # 载入模型的状态字典，使用 mapping 进行映射，确保严格匹配
    m.model.load_state_dict(mapping, strict=True)
    
    # 将模型转换为半精度（half precision）
    m.half()
    
    # 将模型保存到指定的 PyTorch dump 文件夹路径中
    m.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument("--src_path", type=str, help="like blenderbot-model.bin")
    # 添加必需的参数：--src_path，类型为字符串，用法说明为"like blenderbot-model.bin"

    parser.add_argument("--save_dir", default="hf_blenderbot", type=str, help="Where to save converted model.")
    # 添加参数：--save_dir，默认值为"hf_blenderbot"，类型为字符串，用法说明为"Where to save converted model."

    parser.add_argument(
        "--hf_config_json", default="blenderbot-3b-config.json", type=str, help="Path to config to use"
    )
    # 添加参数：--hf_config_json，默认值为"blenderbot-3b-config.json"，类型为字符串，用法说明为"Path to config to use"

    args = parser.parse_args()
    # 解析命令行参数并返回一个命名空间对象 args

    convert_parlai_checkpoint(args.src_path, args.save_dir, args.hf_config_json)
    # 调用函数 convert_parlai_checkpoint，传递解析后的参数 args 中的 src_path、save_dir 和 hf_config_json
```