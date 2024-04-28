# `.\transformers\models\bigbird_pegasus\convert_bigbird_pegasus_tf_to_pytorch.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
from typing import Dict  # 用于类型提示

import tensorflow as tf  # 导入 TensorFlow 库
import torch  # 导入 PyTorch 库
from tqdm import tqdm  # 导入 tqdm 进度条库

from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration  # 从transformers库中导入BigBirdPegasusConfig和BigBirdPegasusForConditionalGeneration类

# 定义初始化时需要替换的模式，用于将 TensorFlow 的命名转换为 Hugging Face 的命名
INIT_COMMON = [
    # tf -> hf
    ("/", "."),
    ("layer_", "layers."),
    ("kernel", "weight"),
    ("beta", "bias"),
    ("gamma", "weight"),
    ("pegasus", "model"),
]
# 定义结束时需要替换的模式
END_COMMON = [
    (".output.dense", ".fc2"),
    ("intermediate.LayerNorm", "final_layer_norm"),
    ("intermediate.dense", "fc1"),
]

# 定义用于解码器部分的替换模式
DECODER_PATTERNS = (
    INIT_COMMON
    + [
        ("attention.self.LayerNorm", "self_attn_layer_norm"),
        ("attention.output.dense", "self_attn.out_proj"),
        ("attention.self", "self_attn"),
        ("attention.encdec.LayerNorm", "encoder_attn_layer_norm"),
        ("attention.encdec_output.dense", "encoder_attn.out_proj"),
        ("attention.encdec", "encoder_attn"),
        ("key", "k_proj"),
        ("value", "v_proj"),
        ("query", "q_proj"),
        ("decoder.LayerNorm", "decoder.layernorm_embedding"),
    ]
    + END_COMMON
)

# 定义剩余部分的替换模式
REMAINING_PATTERNS = (
    INIT_COMMON
    + [
        ("embeddings.word_embeddings", "shared.weight"),
        ("embeddings.position_embeddings", "embed_positions.weight"),
        ("attention.self.LayerNorm", "self_attn_layer_norm"),
        ("attention.output.dense", "self_attn.output"),
        ("attention.self", "self_attn.self"),
        ("encoder.LayerNorm", "encoder.layernorm_embedding"),
    ]
    + END_COMMON
)

# 定义需要忽略的键列表
KEYS_TO_IGNORE = [
    "encdec/key/bias",
    "encdec/query/bias",
    "encdec/value/bias",
    "self/key/bias",
    "self/query/bias",
    "self/value/bias",
    "encdec_output/dense/bias",
    "attention/output/dense/bias",
]

# 定义函数，用于将 TensorFlow 的权重转换为 Hugging Face 的权重
def rename_state_dict_key(k, patterns):
    for tf_name, hf_name in patterns:
        k = k.replace(tf_name, hf_name)
    return k

# 定义函数，用于将 BigBirdPegasus 模型从 TensorFlow 转换为 PyTorch
def convert_bigbird_pegasus(tf_weights: dict, config_update: dict) -> BigBirdPegasusForConditionalGeneration:
    # 使用给定的配置更新创建 BigBirdPegasusConfig 对象
    cfg = BigBirdPegasusConfig(**config_update)
    # 使用配置创建 BigBirdPegasusForConditionalGeneration 模型
    torch_model = BigBirdPegasusForConditionalGeneration(cfg)
    # 获取 PyTorch 模型的状态字典
    state_dict = torch_model.state_dict()
    # 初始化一个空的映射字典
    mapping = {}

    # 将解码器权重和剩余权重分开
    decoder_weights = {k: tf_weights[k] for k in tf_weights if k.startswith("pegasus/decoder")}
    remaining_weights = {k: tf_weights[k] for k in tf_weights if not k.startswith("pegasus/decoder")}
    # 遍历 decoder_weights 字典的键值对，并在进度条中显示“tf -> hf conversion”
    for k, v in tqdm(decoder_weights.items(), "tf -> hf conversion"):
        # 判断是否有后缀在 KEYS_TO_IGNORE 中，若有则跳过当前键值对
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        # 使用 DECODER_PATTERNS 对键进行模式匹配重命名
        patterns = DECODER_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        # 如果新键不在 state_dict 中，则抛出异常
        if new_k not in state_dict:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        # 若键中包含 "dense", "query", "key", "value" 中任意一个，则对应的值进行转置
        if any(True if i in k else False for i in ["dense", "query", "key", "value"]):
            v = v.T
        # 将转换后的键值对添加到 mapping 字典中，并转换为 Torch 张量
        mapping[new_k] = torch.from_numpy(v)
        # 断言转换后的值与 state_dict 中对应键的形状相同，否则抛出异常
        assert v.shape == state_dict[new_k].shape, f"{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}"

    # 遍历 remaining_weights 字典的键值对，并在进度条中显示“tf -> hf conversion”
    for k, v in tqdm(remaining_weights.items(), "tf -> hf conversion"):
        # 判断是否有后缀在 KEYS_TO_IGNORE 中，若有则跳过当前键值对
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        # 使用 REMAINING_PATTERNS 对键进行模式匹配重命名
        patterns = REMAINING_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        # 如果新键不在 state_dict 中且键不为 "pegasus/embeddings/position_embeddings"，则抛出异常
        if new_k not in state_dict and k != "pegasus/embeddings/position_embeddings":
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        # 若键中包含 "dense", "query", "key", "value" 中任意一个，则对应的值进行转置
        if any(True if i in k else False for i in ["dense", "query", "key", "value"]):
            v = v.T
        # 将转换后的键值对添加到 mapping 字典中，并转换为 Torch 张量
        mapping[new_k] = torch.from_numpy(v)
        # 如果键不为 "pegasus/embeddings/position_embeddings"，则断言转换后的值与 state_dict 中对应键的形状相同，否则抛出异常
        if k != "pegasus/embeddings/position_embeddings":
            assert v.shape == state_dict[new_k].shape, f"{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}"

    # 将 mapping 字典中的键 "model.embed_positions.weight" 的值赋给 "model.encoder.embed_positions.weight"
    mapping["model.encoder.embed_positions.weight"] = mapping["model.embed_positions.weight"]
    # 弹出 mapping 字典中的键 "model.embed_positions.weight" 并赋值给 "model.decoder.embed_positions.weight"
    mapping["model.decoder.embed_positions.weight"] = mapping.pop("model.embed_positions.weight")
    # 加载模型状态字典，并返回缺失的键、多余的键
    missing, extra = torch_model.load_state_dict(mapping, strict=False)
    # 找到不匹配的缺失键列表中不在指定列表中的键
    unexpected_missing = [
        k
        for k in missing
        if k
        not in [
            "final_logits_bias",
            "model.encoder.embed_tokens.weight",
            "model.decoder.embed_tokens.weight",
            "lm_head.weight",
        ]
    ]
    # 断言没有意外缺失的键
    assert unexpected_missing == [], f"no matches found for the following torch keys {unexpected_missing}"
    # 断言没有意外多余的键
    assert extra == [], f"no matches found for the following tf keys {extra}"
    # 返回 Torch 模型
    return torch_model
# 定义一个函数，从 TensorFlow 检查点文件中获取权重并返回为字典形式
def get_tf_weights_as_numpy(path) -> Dict:
    # 获取 TensorFlow 检查点文件中的初始化变量列表
    init_vars = tf.train.list_variables(path)
    # 初始化一个空字典用于保存 TensorFlow 权重
    tf_weights = {}
    # 定义需要忽略的变量名列表
    ignore_name = ["global_step"]
    # 遍历初始化变量列表
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        # 检查变量名是否包含在需要忽略的列表中
        skip_key = any(pat in name for pat in ignore_name)
        # 如果需要忽略则跳过当前变量
        if skip_key:
            continue
        # 加载 TensorFlow 检查点文件中的变量数组
        array = tf.train.load_variable(path, name)
        # 将变量名和对应的数组存入字典
        tf_weights[name] = array
    # 返回 TensorFlow 权重字典
    return tf_weights

# 定义一个函数，将 BigBird-Pegasus 模型的 TensorFlow 检查点文件转换为 PyTorch 模型
def convert_bigbird_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str, config_update: dict):
    # 获取 TensorFlow 权重字典
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    # 使用 TensorFlow 权重字典和配置更新字典转换为 PyTorch 模型
    torch_model = convert_bigbird_pegasus(tf_weights, config_update)
    # 保存 PyTorch 模型到指定目录
    torch_model.save_pretrained(save_dir)

# 程序入口
if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定 TensorFlow 检查点文件路径
    parser.add_argument("--tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    # 添加命令行参数，用于指定输出 PyTorch 模型的目录路径
    parser.add_argument("--save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数
    args = parser.parse_args()
    # 定义配置更新字典
    config_update = {}
    # 调用函数，将 BigBird-Pegasus 模型的 TensorFlow 检查点文件转换为 PyTorch 模型
    convert_bigbird_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir, config_update=config_update)
```