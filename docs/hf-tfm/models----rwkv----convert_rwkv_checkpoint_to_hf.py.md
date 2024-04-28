# `.\transformers\models\rwkv\convert_rwkv_checkpoint_to_hf.py`

```
# 设置编码格式为 UTF-8
# 版权声明，指明版权归 The HuggingFace Inc. 团队所有，且使用 Apache License 2.0 开源许可
# 根据 Apache License 2.0 开源许可，可以自由使用该代码，但需要遵守相关许可规定
"""将 BlinkDL 格式的 RWKV 检查点转换为 Hugging Face 格式。"""


# 导入必要的库
import argparse  # 用于解析命令行参数
import gc  # 用于手动垃圾回收
import json  # 用于处理 JSON 格式的数据
import os  # 用于与操作系统交互
import re  # 用于正则表达式操作

# 导入 PyTorch 库
import torch  # PyTorch 深度学习库

# 导入 Hugging Face 相关模块和函数
from huggingface_hub import hf_hub_download  # 从 Hugging Face 模型中心下载模型
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig  # 导入 Hugging Face Transformers 库中的模型和配置类
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint  # 导入 Hugging Face Transformers 库中的模型工具函数


# 不同模型大小对应的隐藏层数
NUM_HIDDEN_LAYERS_MAPPING = {
    "169M": 12,
    "430M": 24,
    "1B5": 24,
    "3B": 32,
    "7B": 32,
    "14B": 40,
}

# 不同模型大小对应的隐藏层大小
HIDEN_SIZE_MAPPING = {
    "169M": 768,
    "430M": 1024,
    "1B5": 2048,
    "3B": 2560,
    "7B": 4096,
    "14B": 5120,
}


# 将 RWKV 检查点的状态字典转换为 Hugging Face 模型格式
def convert_state_dict(state_dict):
    # 获取状态字典的键列表
    state_dict_keys = list(state_dict.keys())
    # 遍历状态字典的键
    for name in state_dict_keys:
        # 获取对应键的值（权重）
        weight = state_dict.pop(name)
        # 如果键以 "emb." 开头，将其替换为 "embeddings."
        if name.startswith("emb."):
            name = name.replace("emb.", "embeddings.")
        # 如果键以 "blocks.0.ln0" 开头，将其替换为 "blocks.0.pre_ln"（仅存在于块 0）
        if name.startswith("blocks.0.ln0"):
            name = name.replace("blocks.0.ln0", "blocks.0.pre_ln")
        # 将键中的 "blocks.<数字>.att" 替换为 "blocks.<数字>.attention"
        name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
        # 将键中的 "blocks.<数字>.ffn" 替换为 "blocks.<数字>.feed_forward"
        name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)
        # 将键中的 ".time_mix_k" 替换为 ".time_mix_key" 并调整形状
        if name.endswith(".time_mix_k"):
            name = name.replace(".time_mix_k", ".time_mix_key")
        # 将键中的 ".time_mix_v" 替换为 ".time_mix_value" 并调整形状
        if name.endswith(".time_mix_v"):
            name = name.replace(".time_mix_v", ".time_mix_value")
        # 将键中的 ".time_mix_r" 替换为 ".time_mix_key" 并调整形状
        if name.endswith(".time_mix_r"):
            name = name.replace(".time_mix_r", ".time_mix_receptance")

        # 如果键不是 "head.weight"，则在键前添加 "rwkv."
        if name != "head.weight":
            name = "rwkv." + name

        # 用处理后的键名作为键，原始权重作为值，重新添加到状态字典中
        state_dict[name] = weight
    # 返回转换后的状态字典
    return state_dict


# 将 RMKV 检查点转换为 Hugging Face 模型格式
def convert_rmkv_checkpoint_to_hf_format(
    repo_id, checkpoint_file, output_dir, size=None, tokenizer_file=None, push_to_hub=False, model_name=None
):
    # 如果可能，构建分词器
    if tokenizer_file is None:
        # 如果没有提供 `--tokenizer_file`，则使用默认分词器
        print("No `--tokenizer_file` provided, we will use the default tokenizer.")
        # 默认词汇量大小为 50277
        vocab_size = 50277
        # 使用预训练模型 "EleutherAI/gpt-neox-20b" 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        # 如果不是形式，则使用预训练的快速标记器来初始化 tokenizer
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        # 获取标记器的词汇表大小
        vocab_size = len(tokenizer)
    # 保存 tokenizer 的配置到输出目录
    tokenizer.save_pretrained(output_dir)

    # 2. 构建配置
    possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
    if size is None:
        # 从检查点文件名中尝试推断大小
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            raise ValueError("Could not infer the size, please provide it with the `--size` argument.")
    if size not in possible_sizes:
        raise ValueError(f"`size` should be one of {possible_sizes}, got {size}.")

    # 创建 RwkvConfig 配置对象
    config = RwkvConfig(
        vocab_size=vocab_size,
        num_hidden_layers=NUM_HIDDEN_LAYERS_MAPPING[size],
        hidden_size=HIDEN_SIZE_MAPPING[size],
    )
    # 保存配置到输出目录
    config.save_pretrained(output_dir)

    # 3. 下载模型文件然后转换 state_dict
    model_file = hf_hub_download(repo_id, checkpoint_file)
    # 加载模型文件的 state_dict
    state_dict = torch.load(model_file, map_location="cpu")
    # 转换 state_dict
    state_dict = convert_state_dict(state_dict)

    # 4. 分割成碎片并保存
    shards, index = shard_checkpoint(state_dict)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))

    if index is not None:
        save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
        # 也保存索引
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        # 5. 清理碎片（由于某个原因，PyTorch 保存的文件占用的空间与整个 state_dict 相同）
        print(
            "Cleaning up shards. This may error with an OOM error, it this is the case don't worry you still have converted the model."
        )
        shard_files = list(shards.keys())

        del state_dict
        del shards
        gc.collect()

        for shard_file in shard_files:
            state_dict = torch.load(os.path.join(output_dir, shard_file))
            torch.save({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(output_dir, shard_file))

    del state_dict
    gc.collect()

    if push_to_hub:
        if model_name is None:
            raise ValueError("Please provide a `model_name` to push the model to the Hub.")
        # 从输出目录创建 AutoModelForCausalLM 模型
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        # 推送模型到 Hub
        model.push_to_hub(model_name, max_shard_size="2GB")
        # 推送 tokenizer 到 Hub
        tokenizer.push_to_hub(model_name)
# 如果该脚本被直接运行，而不是作为导入模块，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--repo_id", default=None, type=str, required=True, help="Repo ID from which to pull the checkpoint."
    )
    parser.add_argument(
        "--checkpoint_file", default=None, type=str, required=True, help="Name of the checkpoint file in the repo."
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True, help="Where to save the converted model."
    )
    parser.add_argument(
        "--tokenizer_file",
        default=None,
        type=str,
        help="Path to the tokenizer file to use (if not provided, only the model is converted).",
    )
    parser.add_argument(
        "--size",
        default=None,
        type=str,
        help="Size of the model. Will be inferred from the `checkpoint_file` if not passed.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to the Hub the converted model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the pushed model on the Hub, including the username / organization.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 RMKV 检查点转换为 HF 格式
    convert_rmkv_checkpoint_to_hf_format(
        args.repo_id,
        args.checkpoint_file,
        args.output_dir,
        size=args.size,
        tokenizer_file=args.tokenizer_file,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
    )
```