# `.\models\rwkv\convert_rwkv_checkpoint_to_hf.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert a RWKV checkpoint from BlinkDL to the Hugging Face format."""

import argparse  # 导入处理命令行参数的模块
import gc  # 导入垃圾回收模块
import json  # 导入处理 JSON 格式的模块
import os  # 导入与操作系统交互的模块
import re  # 导入处理正则表达式的模块

import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入从 HF Hub 下载模型的功能

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint

NUM_HIDDEN_LAYERS_MAPPING = {  # 定义模型尺寸与隐藏层映射关系的字典
    "169M": 12,
    "430M": 24,
    "1B5": 24,
    "3B": 32,
    "7B": 32,
    "14B": 40,
}

HIDEN_SIZE_MAPPING = {  # 定义模型尺寸与隐藏单元大小映射关系的字典
    "169M": 768,
    "430M": 1024,
    "1B5": 2048,
    "3B": 2560,
    "7B": 4096,
    "14B": 5120,
}

def convert_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    for name in state_dict_keys:
        weight = state_dict.pop(name)
        # 对模型参数名称进行转换，适配 Hugging Face 模型格式
        # emb -> embedding
        if name.startswith("emb."):
            name = name.replace("emb.", "embeddings.")
        # ln_0 -> pre_ln (only present at block 0)
        if name.startswith("blocks.0.ln0"):
            name = name.replace("blocks.0.ln0", "blocks.0.pre_ln")
        # att -> attention
        name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
        # ffn -> feed_forward
        name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)
        # time_mix_k -> time_mix_key and reshape
        if name.endswith(".time_mix_k"):
            name = name.replace(".time_mix_k", ".time_mix_key")
        # time_mix_v -> time_mix_value and reshape
        if name.endswith(".time_mix_v"):
            name = name.replace(".time_mix_v", ".time_mix_value")
        # time_mix_r -> time_mix_key and reshape
        if name.endswith(".time_mix_r"):
            name = name.replace(".time_mix_r", ".time_mix_receptance")

        if name != "head.weight":
            name = "rwkv." + name  # 添加前缀以标识 RWKV 格式的参数

        state_dict[name] = weight
    return state_dict

def convert_rmkv_checkpoint_to_hf_format(
    repo_id, checkpoint_file, output_dir, size=None, tokenizer_file=None, push_to_hub=False, model_name=None
):
    # 1. If possible, build the tokenizer.
    if tokenizer_file is None:
        print("No `--tokenizer_file` provided, we will use the default tokenizer.")
        vocab_size = 50277
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # 使用默认的分词器模型
    else:
        # 如果没有指定 tokenizer_file，则使用 PreTrainedTokenizerFast 加载默认的分词器
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        # 获取分词器的词汇表大小
        vocab_size = len(tokenizer)
    # 将 tokenizer 保存到输出目录
    tokenizer.save_pretrained(output_dir)

    # 2. 构建配置文件
    # 定义可能的隐藏层大小列表
    possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
    if size is None:
        # 尝试从 checkpoint 文件名推断 size
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            # 如果无法推断出 size，则抛出错误
            raise ValueError("Could not infer the size, please provide it with the `--size` argument.")
    if size not in possible_sizes:
        # 如果 size 不在可能的大小列表中，则抛出错误
        raise ValueError(f"`size` should be one of {possible_sizes}, got {size}.")

    # 创建 RwkvConfig 对象，配置模型的参数
    config = RwkvConfig(
        vocab_size=vocab_size,
        num_hidden_layers=NUM_HIDDEN_LAYERS_MAPPING[size],
        hidden_size=HIDEN_SIZE_MAPPING[size],
    )
    # 将配置保存到输出目录
    config.save_pretrained(output_dir)

    # 3. 下载模型文件并转换 state_dict
    # 从 HF Hub 下载模型文件
    model_file = hf_hub_download(repo_id, checkpoint_file)
    # 加载模型的 state_dict
    state_dict = torch.load(model_file, map_location="cpu")
    # 转换 state_dict
    state_dict = convert_state_dict(state_dict)

    # 4. 分割成片段并保存
    # 将 state_dict 拆分成多个片段
    shards, index = shard_checkpoint(state_dict)
    for shard_file, shard in shards.items():
        # 保存每个片段到输出目录
        torch.save(shard, os.path.join(output_dir, shard_file))

    if index is not None:
        # 如果存在 index，则保存 index 到输出目录
        save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            # 将 index 写入文件
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        # 5. 清理片段（有时 PyTorch 保存的文件会占用与完整 state_dict 相同的空间）
        print(
            "Cleaning up shards. This may error with an OOM error, it this is the case don't worry you still have converted the model."
        )
        # 获取所有片段文件名列表
        shard_files = list(shards.keys())

        # 清理变量以释放内存
        del state_dict
        del shards
        gc.collect()

        # 重新加载每个片段并保存（确保在 CPU 上）
        for shard_file in shard_files:
            state_dict = torch.load(os.path.join(output_dir, shard_file))
            torch.save({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(output_dir, shard_file))

    # 清理 state_dict 变量以释放内存
    del state_dict
    gc.collect()

    # 如果需要推送到 HF Hub
    if push_to_hub:
        if model_name is None:
            # 如果未提供 model_name，则抛出错误
            raise ValueError("Please provide a `model_name` to push the model to the Hub.")
        # 加载模型并推送到 HF Hub
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        model.push_to_hub(model_name, max_shard_size="2GB")
        # 将分词器也推送到 HF Hub
        tokenizer.push_to_hub(model_name)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需参数
    parser.add_argument(
        "--repo_id", default=None, type=str, required=True, help="Repo ID from which to pull the checkpoint."
    )
    # repo_id 参数，从中获取检查点的仓库 ID

    parser.add_argument(
        "--checkpoint_file", default=None, type=str, required=True, help="Name of the checkpoint file in the repo."
    )
    # checkpoint_file 参数，检查点文件在仓库中的名称

    parser.add_argument(
        "--output_dir", default=None, type=str, required=True, help="Where to save the converted model."
    )
    # output_dir 参数，用于保存转换后模型的目录路径

    parser.add_argument(
        "--tokenizer_file",
        default=None,
        type=str,
        help="Path to the tokenizer file to use (if not provided, only the model is converted).",
    )
    # tokenizer_file 参数，用于指定要使用的分词器文件路径（如果未提供，则仅转换模型）

    parser.add_argument(
        "--size",
        default=None,
        type=str,
        help="Size of the model. Will be inferred from the `checkpoint_file` if not passed.",
    )
    # size 参数，指定模型的大小；如果未传入，则将从 checkpoint_file 推断大小

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to the Hub the converted model.",
    )
    # push_to_hub 参数，如果设置，则推送转换后的模型到 Hub 上

    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the pushed model on the Hub, including the username / organization.",
    )
    # model_name 参数，指定推送到 Hub 上的模型名称，包括用户名或组织名

    args = parser.parse_args()
    # 解析命令行参数并返回一个命名空间对象 args

    convert_rmkv_checkpoint_to_hf_format(
        args.repo_id,
        args.checkpoint_file,
        args.output_dir,
        size=args.size,
        tokenizer_file=args.tokenizer_file,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
    )
    # 调用 convert_rmkv_checkpoint_to_hf_format 函数，传递解析后的参数作为函数的输入
```