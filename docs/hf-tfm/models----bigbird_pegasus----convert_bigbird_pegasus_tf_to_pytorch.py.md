# `.\models\bigbird_pegasus\convert_bigbird_pegasus_tf_to_pytorch.py`

```
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
from typing import Dict  # 导入类型提示模块 Dict

import tensorflow as tf  # 导入 TensorFlow 库
import torch  # 导入 PyTorch 库
from tqdm import tqdm  # 导入进度条模块 tqdm

from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration  # 导入 Transformers 库中的类

# TensorFlow 到 Hugging Face 模型命名的初始映射列表
INIT_COMMON = [
    ("/", "."),  # 替换 "/" 为 "."
    ("layer_", "layers."),  # 替换 "layer_" 为 "layers."
    ("kernel", "weight"),  # 替换 "kernel" 为 "weight"
    ("beta", "bias"),  # 替换 "beta" 为 "bias"
    ("gamma", "weight"),  # 替换 "gamma" 为 "weight"
    ("pegasus", "model"),  # 替换 "pegasus" 为 "model"
]

# TensorFlow 到 Hugging Face 模型命名的结尾映射列表
END_COMMON = [
    (".output.dense", ".fc2"),  # 替换 ".output.dense" 为 ".fc2"
    ("intermediate.LayerNorm", "final_layer_norm"),  # 替换 "intermediate.LayerNorm" 为 "final_layer_norm"
    ("intermediate.dense", "fc1"),  # 替换 "intermediate.dense" 为 "fc1"
]

# 解码器模型权重命名模式列表，包含初始、中间和结尾映射
DECODER_PATTERNS = (
    INIT_COMMON
    + [
        ("attention.self.LayerNorm", "self_attn_layer_norm"),  # 替换 "attention.self.LayerNorm" 为 "self_attn_layer_norm"
        ("attention.output.dense", "self_attn.out_proj"),  # 替换 "attention.output.dense" 为 "self_attn.out_proj"
        ("attention.self", "self_attn"),  # 替换 "attention.self" 为 "self_attn"
        ("attention.encdec.LayerNorm", "encoder_attn_layer_norm"),  # 替换 "attention.encdec.LayerNorm" 为 "encoder_attn_layer_norm"
        ("attention.encdec_output.dense", "encoder_attn.out_proj"),  # 替换 "attention.encdec_output.dense" 为 "encoder_attn.out_proj"
        ("attention.encdec", "encoder_attn"),  # 替换 "attention.encdec" 为 "encoder_attn"
        ("key", "k_proj"),  # 替换 "key" 为 "k_proj"
        ("value", "v_proj"),  # 替换 "value" 为 "v_proj"
        ("query", "q_proj"),  # 替换 "query" 为 "q_proj"
        ("decoder.LayerNorm", "decoder.layernorm_embedding"),  # 替换 "decoder.LayerNorm" 为 "decoder.layernorm_embedding"
    ]
    + END_COMMON
)

# 剩余模型权重命名模式列表，包含初始、中间和结尾映射
REMAINING_PATTERNS = (
    INIT_COMMON
    + [
        ("embeddings.word_embeddings", "shared.weight"),  # 替换 "embeddings.word_embeddings" 为 "shared.weight"
        ("embeddings.position_embeddings", "embed_positions.weight"),  # 替换 "embeddings.position_embeddings" 为 "embed_positions.weight"
        ("attention.self.LayerNorm", "self_attn_layer_norm"),  # 替换 "attention.self.LayerNorm" 为 "self_attn_layer_norm"
        ("attention.output.dense", "self_attn.output"),  # 替换 "attention.output.dense" 为 "self_attn.output"
        ("attention.self", "self_attn.self"),  # 替换 "attention.self" 为 "self_attn.self"
        ("encoder.LayerNorm", "encoder.layernorm_embedding"),  # 替换 "encoder.LayerNorm" 为 "encoder.layernorm_embedding"
    ]
    + END_COMMON
)

# 需要忽略的键列表，这些键不进行名称转换
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

def rename_state_dict_key(k, patterns):
    # 根据给定的模式列表 patterns，替换给定的键 k 的名称
    for tf_name, hf_name in patterns:
        k = k.replace(tf_name, hf_name)
    return k

def convert_bigbird_pegasus(tf_weights: dict, config_update: dict) -> BigBirdPegasusForConditionalGeneration:
    # 根据 config_update 创建 BigBirdPegasusConfig 对象 cfg
    cfg = BigBirdPegasusConfig(**config_update)
    # 根据 cfg 创建 BigBirdPegasusForConditionalGeneration 对象 torch_model
    torch_model = BigBirdPegasusForConditionalGeneration(cfg)
    # 获取 torch_model 的状态字典 state_dict
    state_dict = torch_model.state_dict()
    # 创建空字典 mapping，用于存储键的映射关系
    mapping = {}

    # 分离解码器权重
    decoder_weights = {k: tf_weights[k] for k in tf_weights if k.startswith("pegasus/decoder")}
    # 分离剩余权重
    remaining_weights = {k: tf_weights[k] for k in tf_weights if not k.startswith("pegasus/decoder")}
    # 遍历 decoder_weights 字典中的键值对，显示进度条为 "tf -> hf conversion"
    for k, v in tqdm(decoder_weights.items(), "tf -> hf conversion"):
        # 检查当前键是否以 KEYS_TO_IGNORE 中任意后缀结尾，如果是则跳过当前循环
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        # 使用预定义的 DECODER_PATTERNS 对键 k 进行重命名，得到 new_k
        patterns = DECODER_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        # 如果 new_k 不在 state_dict 中，抛出异常，指明无法在 state_dict 中找到对应的新键 new_k（从旧键 k 转换而来）
        if new_k not in state_dict:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        # 如果键 k 中包含 "dense", "query", "key", "value" 中任何一个关键字，则对 v 进行转置操作
        if any(True if i in k else False for i in ["dense", "query", "key", "value"]):
            v = v.T
        # 将 torch.Tensor 类型的 v 赋值给 mapping[new_k]
        mapping[new_k] = torch.from_numpy(v)
        # 断言 v 的形状与 state_dict[new_k] 的形状相同，如果不同则抛出异常，指明不匹配的键与形状信息
        assert v.shape == state_dict[new_k].shape, f"{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}"

    # 遍历 remaining_weights 字典中的键值对，显示进度条为 "tf -> hf conversion"
    for k, v in tqdm(remaining_weights.items(), "tf -> hf conversion"):
        # 检查当前键是否以 KEYS_TO_IGNORE 中任意后缀结尾，如果是则跳过当前循环
        conditions = [k.endswith(ending) for ending in KEYS_TO_IGNORE]
        if any(conditions):
            continue
        # 使用预定义的 REMAINING_PATTERNS 对键 k 进行重命名，得到 new_k
        patterns = REMAINING_PATTERNS
        new_k = rename_state_dict_key(k, patterns)
        # 如果 new_k 不在 state_dict 中，并且 k 不等于 "pegasus/embeddings/position_embeddings"，抛出异常
        # 指明无法在 state_dict 中找到对应的新键 new_k（从旧键 k 转换而来）
        if new_k not in state_dict and k != "pegasus/embeddings/position_embeddings":
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")
        # 如果键 k 中包含 "dense", "query", "key", "value" 中任何一个关键字，则对 v 进行转置操作
        if any(True if i in k else False for i in ["dense", "query", "key", "value"]):
            v = v.T
        # 将 torch.Tensor 类型的 v 赋值给 mapping[new_k]
        mapping[new_k] = torch.from_numpy(v)
        # 如果 k 不等于 "pegasus/embeddings/position_embeddings"，断言 v 的形状与 state_dict[new_k] 的形状相同，
        # 如果不同则抛出异常，指明不匹配的键与形状信息
        if k != "pegasus/embeddings/position_embeddings":
            assert v.shape == state_dict[new_k].shape, f"{new_k}, {k}, {v.shape}, {state_dict[new_k].shape}"

    # 将 mapping 中的键 "model.embed_positions.weight" 的值复制给键 "model.encoder.embed_positions.weight"
    mapping["model.encoder.embed_positions.weight"] = mapping["model.embed_positions.weight"]
    # 弹出 mapping 中键为 "model.embed_positions.weight" 的值，并将其赋给键 "model.decoder.embed_positions.weight"
    mapping["model.decoder.embed_positions.weight"] = mapping.pop("model.embed_positions.weight")
    # 载入 mapping 到 torch_model 的状态字典，允许部分键名不严格匹配
    missing, extra = torch_model.load_state_dict(mapping, strict=False)
    # 找出在 state_dict 中缺失的键，并将其列为 missing
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
    # 断言 unexpected_missing 为空列表，如果不是则抛出异常，指明找不到匹配的 torch 键
    assert unexpected_missing == [], f"no matches found for the following torch keys {unexpected_missing}"
    # 断言 extra 为空列表，如果不是则抛出异常，指明找不到匹配的 tf 键
    assert extra == [], f"no matches found for the following tf keys {extra}"
    # 返回加载了 mapping 后的 torch_model
    return torch_model
# 定义函数，用于从 TensorFlow 模型的检查点文件中获取权重并以字典形式返回
def get_tf_weights_as_numpy(path) -> Dict:
    # 使用 TensorFlow 提供的工具函数列出给定路径下的所有变量和它们的形状
    init_vars = tf.train.list_variables(path)
    # 初始化一个空字典，用于存储 TensorFlow 权重
    tf_weights = {}
    # 定义要忽略的变量名列表，例如全局步数变量
    ignore_name = ["global_step"]
    # 遍历初始化变量列表，并显示转换进度描述为“converting tf checkpoint to dict”
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        # 如果变量名中包含在忽略列表中的任何模式，则跳过该变量
        skip_key = any(pat in name for pat in ignore_name)
        if skip_key:
            continue
        # 加载指定路径中的变量数据并存储到数组中
        array = tf.train.load_variable(path, name)
        # 将加载的变量数据存储到字典中，以变量名作为键
        tf_weights[name] = array
    # 返回整理后的 TensorFlow 权重字典
    return tf_weights


# 定义函数，用于将 BigBird-Pegasus 模型的 TensorFlow 检查点转换为 PyTorch 模型
def convert_bigbird_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str, config_update: dict):
    # 获取 TensorFlow 模型的权重字典
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    # 使用给定的配置更新字典和 TensorFlow 权重字典，转换为 PyTorch 模型
    torch_model = convert_bigbird_pegasus(tf_weights, config_update)
    # 将转换后的 PyTorch 模型保存到指定的目录中
    torch_model.save_pretrained(save_dir)


# 程序入口点，用于命令行参数解析和执行转换操作
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数选项，用于指定 TensorFlow 检查点文件的路径
    parser.add_argument("--tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    # 添加命令行参数选项，用于指定输出 PyTorch 模型的保存路径
    parser.add_argument("--save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数，并存储到 args 对象中
    args = parser.parse_args()
    # 初始化一个空的配置更新字典
    config_update = {}
    # 调用函数，执行 BigBird-Pegasus 模型从 TensorFlow 到 PyTorch 的转换过程
    convert_bigbird_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir, config_update=config_update)
```