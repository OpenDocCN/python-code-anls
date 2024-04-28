# `.\transformers\models\pegasus\convert_pegasus_tf_to_pytorch.py`

```py
# 声明文件编码为UTF-8
# 版权声明
# 根据Apache License, Version 2.0许可，你可以在遵守许可下使用该文件
# 你可以在以下网址获得许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可分发的软件均根据“按原样”分发，没有任何形式的保证或条件，无论是明示还是隐含的
# 有关权限和限制的特定语言，请参阅许可
# 引入模块
import argparse  # 用于解析命令行参数和选项
import os  # 用于访问操作系统功能
from pathlib import Path  # 用于处理文件路径
from typing import Dict  # 用于类型注解

import tensorflow as tf  # TensorFlow深度学习框架
import torch  # PyTorch深度学习框架
from tqdm import tqdm  # 显示循环迭代进度条

from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusTokenizer  # 从transformers模块中引入Pegasus相关类
from transformers.models.pegasus.configuration_pegasus import DEFAULTS, task_specific_params  # 从transformers模块中引入Pegasus配置

# 替换模式列表，用来将Pegasus模型的state_dict的键转换为相应的Hugging Face模型的state_dict的键
PATTERNS = [
    ["memory_attention", "encoder_attn"],
    ["attention", "attn"],
    ["/", "."],
    [".LayerNorm.gamma", "_layer_norm.weight"],
    [".LayerNorm.beta", "_layer_norm.bias"],
    ["r.layer_", "r.layers."],
    ["output_proj", "out_proj"],
    ["ffn.dense_1.", "fc2."],
    ["ffn.dense.", "fc1."],
    ["ffn_layer_norm", "final_layer_norm"],
    ["kernel", "weight"],
    ["encoder_layer_norm.", "encoder.layer_norm."],
    ["decoder_layer_norm.", "decoder.layer_norm."],
    ["embeddings.weights", "shared.weight"],
]

# 将state_dict的键转换为相应的Hugging Face模型的state_dict的键
def rename_state_dict_key(k):
    for pegasus_name, hf_name in PATTERNS:
        k = k.replace(pegasus_name, hf_name)
    return k

# 将Pegasus模型权重转换为Hugging Face模型
def convert_pegasus(tf_weights: dict, cfg_updates: dict) -> PegasusForConditionalGeneration:
    # 复制默认Pegasus配置，并更新为指定的配置
    cfg_kwargs = DEFAULTS.copy()
    cfg_kwargs.update(cfg_updates)
    cfg = PegasusConfig(**cfg_kwargs)
    torch_model = PegasusForConditionalGeneration(cfg)  # 使用指定配置创建Pegasus模型
    sd = torch_model.model.state_dict()  # 获取Pegasus模型的state_dict
    mapping = {}  # 创建state_dict键映射字典
    for k, v in tf_weights.items():
        new_k = rename_state_dict_key(k)  # 根据替换模式将state_dict的键转换为相应的Hugging Face模型的state_dict的键
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")  # 如果未找到相应的键，则抛出异常
        if "dense" in k or "proj" in new_k:
            v = v.T  # 如果键中包含'dense'或'proj'，则进行转置
        mapping[new_k] = torch.tensor(v, dtype=sd[new_k].dtype)  # 将转换后的权重添加到映射字典中
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"  # 断言转换后的权重形状和Hugging Face模型的state_dict的形状相同
    # 确保嵌入的padding_idx被尊重
    mapping["shared.weight"][cfg.pad_token_id] = torch.zeros_like(mapping["shared.weight"][cfg.pad_token_id + 1])
    mapping["encoder.embed_tokens.weight"] = mapping["shared.weight"]
    mapping["decoder.embed_tokens.weight"] = mapping["shared.weight"]
    empty_biases = {k: torch.zeros_like(v) for k, v in sd.items() if k.endswith("bias") and k not in mapping}  # 创建空偏置项映射字典
    mapping.update(**empty_biases)  # 更新映射字典
    # 使用给定的映射加载 PyTorch 模型的状态字典，并且允许不严格匹配
    missing, extra = torch_model.model.load_state_dict(mapping, strict=False)
    # 查找不在指定列表中的意外丢失的键
    unexpected_missing = [
        k for k in missing if k not in ["encoder.embed_positions.weight", "decoder.embed_positions.weight"]
    ]
    # 如果有意外丢失的键，则触发断言错误
    assert unexpected_missing == [], f"no matches found for the following torch keys {unexpected_missing}"
    # 如果存在额外的键，则触发断言错误
    assert extra == [], f"no matches found for the following tf keys {extra}"
    # 返回加载完成的 PyTorch 模型
    return torch_model
```  
# 将 TensorFlow 检查点权重转换为 Python 字典
def get_tf_weights_as_numpy(path="./ckpt/aeslc/model.ckpt-32000") -> Dict:
    # 获取 TensorFlow 检查点中的变量列表
    init_vars = tf.train.list_variables(path)
    # 初始化空字典用于存储 TensorFlow 检查点中的权重
    tf_weights = {}
    # 不需要转换的变量名列表
    ignore_name = ["Adafactor", "global_step"]
    # 遍历 TensorFlow 检查点中的变量名和形状
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        # 检查是否跳过当前变量名
        skip_key = any(pat in name for pat in ignore_name)
        if skip_key:
            continue
        # 加载 TensorFlow 检查点中的变量值
        array = tf.train.load_variable(path, name)
        # 将变量名和值存储到字典中
        tf_weights[name] = array
    return tf_weights

# 将 Pegasus 检查点转换为 PyTorch 模型
def convert_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str):
    # 保存分词器的配置
    dataset = Path(ckpt_path).parent.name
    desired_max_model_length = task_specific_params[f"summarization_{dataset}"]["max_position_embeddings"]
    # 初始化 Pegasus 分词器
    tok = PegasusTokenizer.from_pretrained("sshleifer/pegasus", model_max_length=desired_max_model_length)
    assert tok.model_max_length == desired_max_model_length
    # 保存分词器配置
    tok.save_pretrained(save_dir)

    # 转换模型权重
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    cfg_updates = task_specific_params[f"summarization_{dataset}"]
    if dataset == "large":
        cfg_updates["task_specific_params"] = task_specific_params
    # 转换 Pegasus 模型的权重为 PyTorch 模型
    torch_model = convert_pegasus(tf_weights, cfg_updates)
    torch_model.save_pretrained(save_dir)
    sd = torch_model.state_dict()
    sd.pop("model.decoder.embed_positions.weight")
    sd.pop("model.encoder.embed_positions.weight")
    # 保存 PyTorch 模型权重
    torch.save(sd, Path(save_dir) / "pytorch_model.bin")

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必要参数
    parser.add_argument("tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    # 若未提供保存目录，则使用默认目录
    if args.save_dir is None:
        dataset = Path(args.tf_ckpt_path).parent.name
        args.save_dir = os.path.join("pegasus", dataset)
    # 执行 Pegasus 检查点转换为 PyTorch 模型
    convert_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)
```