# `.\models\gemma\convert_gemma_weights_to_hf.py`

```
# 版权声明和信息
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import os  # 用于操作系统相关功能
import warnings  # 用于警告处理

import torch  # PyTorch库，用于深度学习
from accelerate import init_empty_weights  # 加速库，用于加速训练

from transformers import GemmaConfig, GemmaForCausalLM, GemmaTokenizer  # Hugging Face Transformers库，用于自然语言处理模型

# 尝试导入GemmaTokenizerFast，如果失败则给出警告并设置为None
try:
    from transformers import GemmaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    GemmaTokenizerFast = None

"""
示例用法:

python src/transformers/models/gemma/convert_gemma_weights_to_hf.py \
    --input_dir /path/to/downloaded/gemma/weights --model_size 7B --output_dir /output/path
"""

# Gemma模型配置示例
gemma_2b_config = GemmaConfig(
    num_hidden_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    hidden_size=2048,
    intermediate_size=16384,
)

gemma_7b_config = GemmaConfig()  # Gemma 7B模型配置对象

CONFIG_MAPPING = {"2B": gemma_2b_config, "7B": gemma_7b_config}  # 配置映射字典
LAYER_NAME_MAPPING = {"embedder.weight": "model.embed_tokens.weight"}  # 层名称映射字典


def write_model(save_path, input_base_path, config, safe_serialization=True, push_to_hub=False, dtype=torch.float32):
    # 从指定路径获取模型参数
    num_attn_heads = config.num_attention_heads  # 注意力头数目
    hidden_size = config.hidden_size  # 隐藏层大小
    num_kv_heads = config.num_key_value_heads  # 键值头数目
    head_dim = config.head_dim  # 头维度

    print(f"Fetching all parameters from the checkpoint at '{input_base_path}'")  # 输出信息：从指定路径获取所有参数
    model_state_dict = torch.load(input_base_path, map_location="cpu")["model_state_dict"]  # 加载模型状态字典
    model_state_dict.pop("freqs_cis")  # 移除特定键值对应的值

    state_dict = {}  # 初始化状态字典
    # 遍历模型状态字典中的键值对
    for k, v in model_state_dict.items():
        # 检查键名是否包含 "qkv_proj"
        if "qkv_proj" in k:
            # 如果 num_kv_heads 等于 1，则执行以下操作
            if num_kv_heads == 1:
                # 重塑张量 v 的形状，将其分成查询（q_proj）、键（k_proj）、值（v_proj）投影
                v = v.reshape(num_attn_heads + num_kv_heads * 2, head_dim, hidden_size)
                q_proj = v[:num_attn_heads, ...]  # 提取查询投影
                k_proj = v[num_attn_heads : num_attn_heads + num_kv_heads, ...].repeat(num_kv_heads, 1, 1)  # 提取键投影
                v_proj = v[-num_kv_heads:, ...].repeat(num_kv_heads, 1, 1)  # 提取值投影

                # 将投影后的张量存入状态字典中，键名替换 "qkv_proj" 为 "q_proj", "k_proj", "v_proj"
                state_dict[k.replace("qkv_proj", "q_proj")] = q_proj.reshape(
                    num_attn_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "k_proj")] = k_proj.reshape(
                    num_kv_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "v_proj")] = v_proj[0].clone()  # 取第一个值投影
            else:
                # 如果 num_kv_heads 不等于 1，则执行以下操作
                q_proj, k_proj, v_proj = torch.split(v, v.shape[0] // 3, 0)  # 分割 v 为查询、键、值投影
                state_dict[k.replace("qkv_proj", "q_proj")] = q_proj.reshape(
                    num_attn_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "k_proj")] = k_proj.reshape(
                    num_kv_heads * head_dim, hidden_size
                ).clone()
                state_dict[k.replace("qkv_proj", "v_proj")] = v_proj.clone()  # 存储值投影

        # 如果键名为 "embedder.weight"，将其映射到指定的层名称，并同时将 "lm_head.weight" 也设置为该值
        elif k == "embedder.weight":
            state_dict[LAYER_NAME_MAPPING[k]] = v
            state_dict["lm_head.weight"] = v
        else:
            # 对于其他键名，直接复制对应的值到状态字典中
            state_dict[k] = v

    # 设置默认的张量数据类型
    torch.set_default_dtype(dtype)

    # 输出加载 Gemma 模型的消息
    print("Loading the checkpoint in a Gemma model.")
    
    # 使用空权重初始化上下文管理器
    with init_empty_weights():
        # 根据配置创建 GemmaForCausalLM 模型
        model = GemmaForCausalLM(config)
    
    # 使用状态字典加载模型的参数，允许参数赋值但不强制严格匹配
    model.load_state_dict(state_dict, assign=True, strict=False)

    # 设置模型配置中的 Torch 张量数据类型为 float32
    model.config.torch_dtype = torch.float32
    # 删除模型配置中的 _name_or_path 属性
    del model.config._name_or_path
    # 输出保存为 Transformers 格式的消息
    print("Saving in the Transformers format.")

    # 如果需要推送到 Hub
    if push_to_hub:
        # 输出推送模型到指定路径的消息
        print(f"pushing the model to {save_path}")
        # 将模型推送到 Hub，设置为私有模式
        model.push_to_hub(save_path, safe_serialization=safe_serialization, private=True)
    else:
        # 否则，保存模型到指定路径，进行安全序列化
        model.save_pretrained(save_path, safe_serialization=safe_serialization)
# 主函数，程序的入口点
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入的模型检查点的绝对路径，必选参数
    parser.add_argument(
        "--input_checkpoint",
        help="Absolute path to the target Gemma weights.",
        required=True,
    )
    # 添加命令行参数：Gemma tokenizer 模型的位置，可选参数
    parser.add_argument(
        "--tokenizer_checkpoint",
        help="Location of Gemma tokenizer model",
    )
    # 添加命令行参数：模型的尺寸，默认为 "7B"，可选参数
    parser.add_argument(
        "--model_size",
        default="7B",
        choices=["2B", "7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Gemma2 official release. For more details on Gemma2, checkout the original repo: https://huggingface.co/google/gemma-7b",
    )
    # 添加命令行参数：输出目录，默认为 "google/gemma-7b"，用于保存 HF 模型和 tokenizer
    parser.add_argument(
        "--output_dir",
        default="google/gemma-7b",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数：是否使用 `safetensors` 保存数据，默认为 False，可选参数
    parser.add_argument(
        "--pickle_serialization",
        help="Whether or not to save using `safetensors`.",
        action="store_true",
        default=False,
    )
    # 添加命令行参数：是否转换 tokenizer，默认为 False，可选参数
    parser.add_argument(
        "--convert_tokenizer",
        help="Whether or not to convert the tokenizer as well.",
        action="store_true",
        default=False,
    )
    # 添加命令行参数：是否将模型推送到 HF Hub，默认为 False，可选参数
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    # 添加命令行参数：转换后模型的目标数据类型，默认为 "float32"，可选参数
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Target dtype of the converted model",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 如果指定了 --convert_tokenizer 参数
    if args.convert_tokenizer:
        # 如果未提供 --tokenizer_checkpoint 参数，则抛出数值错误异常
        if args.tokenizer_checkpoint is None:
            raise ValueError("Path to the tokenizer is required when passing --convert_tokenizer")

        # 构建完整的 tokenizer 路径
        spm_path = os.path.join(args.tokenizer_checkpoint)
        # 调用 write_tokenizer 函数，保存或推送 tokenizer
        write_tokenizer(spm_path, args.output_dir, args.push_to_hub)

    # 根据模型尺寸选择对应的配置信息
    config = CONFIG_MAPPING[args.model_size]
    # 将 args.dtype 转换为 torch 中的数据类型
    dtype = getattr(torch, args.dtype)
    # 调用 write_model 函数，保存或推送模型
    write_model(
        config=config,
        input_base_path=args.input_checkpoint,
        save_path=args.output_dir,
        safe_serialization=not args.pickle_serialization,
        push_to_hub=args.push_to_hub,
        dtype=dtype,
    )


# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```