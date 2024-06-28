# `.\models\pegasus\convert_pegasus_tf_to_pytorch.py`

```
# 定义全局变量，用于将 TensorFlow 模型的状态字典键转换为 PyTorch 模型的对应键
PATTERNS = [
    # 将左侧字符串替换为右侧字符串，以获取与 BART 模型状态字典相同的关键键值对（与 Pegasus 模型相同）
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

# 函数：根据指定模型的状态字典键转换规则，重命名给定键名
def rename_state_dict_key(k):
    for pegasus_name, hf_name in PATTERNS:
        k = k.replace(pegasus_name, hf_name)
    return k


# 函数：将 TensorFlow 权重转换为 Pegasus 模型的 PyTorch 对象
def convert_pegasus(tf_weights: dict, cfg_updates: dict) -> PegasusForConditionalGeneration:
    # 复制默认配置并更新为指定配置
    cfg_kwargs = DEFAULTS.copy()
    cfg_kwargs.update(cfg_updates)
    # 创建 Pegasus 配置对象
    cfg = PegasusConfig(**cfg_kwargs)
    # 创建 PegasusForConditionalGeneration 模型对象
    torch_model = PegasusForConditionalGeneration(cfg)
    # 获取 PyTorch 模型的状态字典
    sd = torch_model.model.state_dict()
    # 存储键名映射关系的空字典
    mapping = {}
    # 遍历 TensorFlow 权重字典中的每个项
    for k, v in tf_weights.items():
        # 根据转换规则重命名键名
        new_k = rename_state_dict_key(k)
        # 如果重命名后的键名不存在于 PyTorch 模型的状态字典中，抛出错误
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k})")

        # 如果键名中包含 "dense" 或 "proj"，则转置权重矩阵
        if "dense" in k or "proj" in new_k:
            v = v.T
        # 将 TensorFlow 权重转换为 PyTorch 张量，并存储到映射字典中
        mapping[new_k] = torch.tensor(v, dtype=sd[new_k].dtype)
        # 断言 TensorFlow 权重形状与 PyTorch 状态字典中对应项的形状相同
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"

    # 确保嵌入层的 padding_idx 被正确设置
    mapping["shared.weight"][cfg.pad_token_id] = torch.zeros_like(mapping["shared.weight"][cfg.pad_token_id + 1])
    # 将映射后的共享权重应用于编码器和解码器的嵌入层
    mapping["encoder.embed_tokens.weight"] = mapping["shared.weight"]
    mapping["decoder.embed_tokens.weight"] = mapping["shared.weight"]
    # 创建空偏置项的字典，并添加到映射字典中
    empty_biases = {k: torch.zeros_like(v) for k, v in sd.items() if k.endswith("bias") and k not in mapping}
    mapping.update(**empty_biases)
    # 使用给定的映射加载模型的状态字典，允许部分匹配
    missing, extra = torch_model.model.load_state_dict(mapping, strict=False)
    
    # 找出在缺失的键中，不属于特定的例外列表的键
    unexpected_missing = [
        k for k in missing if k not in ["encoder.embed_positions.weight", "decoder.embed_positions.weight"]
    ]
    
    # 如果存在不期望的缺失键，引发断言错误，显示未匹配的torch键
    assert unexpected_missing == [], f"no matches found for the following torch keys {unexpected_missing}"
    
    # 如果存在额外的键，引发断言错误，显示未匹配的tf键
    assert extra == [], f"no matches found for the following tf keys {extra}"
    
    # 返回更新后的torch模型
    return torch_model
# 导入必要的库和模块
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import tensorflow as tf
from transformers import PegasusTokenizer
from utils import task_specific_params  # 假设这是从外部导入的任务特定参数
from convert_pegasus import convert_pegasus  # 假设这是从外部导入的模型转换函数

# 从 TensorFlow 检查点中获取权重并以字典形式返回
def get_tf_weights_as_numpy(path="./ckpt/aeslc/model.ckpt-32000") -> Dict:
    # 列出 TensorFlow 检查点中的变量名和形状
    init_vars = tf.train.list_variables(path)
    # 初始化一个空字典用于存储 TensorFlow 的权重
    tf_weights = {}
    # 忽略特定名称的变量
    ignore_name = ["Adafactor", "global_step"]
    # 遍历变量名和形状列表
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        # 如果变量名中包含需要忽略的关键字，则跳过此变量
        skip_key = any(pat in name for pat in ignore_name)
        if skip_key:
            continue
        # 加载 TensorFlow 检查点中的变量值
        array = tf.train.load_variable(path, name)
        # 将变量名和对应的值存入字典中
        tf_weights[name] = array
    # 返回包含 TensorFlow 权重的字典
    return tf_weights


# 将 Pegasus 模型从 TensorFlow 转换为 PyTorch 并保存
def convert_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str):
    # 首先保存分词器 tokenizer
    dataset = Path(ckpt_path).parent.name
    # 获取特定任务的最大模型长度
    desired_max_model_length = task_specific_params[f"summarization_{dataset}"]["max_position_embeddings"]
    # 根据预训练模型和指定的最大长度创建 tokenizer
    tok = PegasusTokenizer.from_pretrained("sshleifer/pegasus", model_max_length=desired_max_model_length)
    # 确认 tokenizer 的最大长度符合预期
    assert tok.model_max_length == desired_max_model_length
    # 将 tokenizer 保存到指定的目录
    tok.save_pretrained(save_dir)

    # 转换模型
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    # 获取特定任务的配置更新
    cfg_updates = task_specific_params[f"summarization_{dataset}"]
    # 如果数据集为 "large"，则添加任务特定参数到配置更新中
    if dataset == "large":
        cfg_updates["task_specific_params"] = task_specific_params
    # 将 TensorFlow 模型转换为 PyTorch 模型
    torch_model = convert_pegasus(tf_weights, cfg_updates)
    # 将 PyTorch 模型保存到指定的目录
    torch_model.save_pretrained(save_dir)
    # 获取 PyTorch 模型的状态字典
    sd = torch_model.state_dict()
    # 从状态字典中删除特定的位置嵌入权重
    sd.pop("model.decoder.embed_positions.weight")
    sd.pop("model.encoder.embed_positions.weight")
    # 将处理后的状态字典保存为二进制文件
    torch.save(sd, Path(save_dir) / "pytorch_model.bin")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # 必需的参数：TensorFlow 检查点路径
    parser.add_argument("tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    # 可选的参数：保存 PyTorch 模型的路径
    parser.add_argument("save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    
    # 如果未指定保存路径，则根据 TensorFlow 检查点路径确定默认路径
    if args.save_dir is None:
        dataset = Path(args.tf_ckpt_path).parent.name
        args.save_dir = os.path.join("pegasus", dataset)
    
    # 调用函数：将 Pegasus 模型从 TensorFlow 转换为 PyTorch 并保存
    convert_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)
```