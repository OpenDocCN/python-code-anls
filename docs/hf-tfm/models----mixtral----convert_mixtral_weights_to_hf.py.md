# `.\models\mixtral\convert_mixtral_weights_to_hf.py`

```py
# 引入必要的库和模块
import argparse  # 用于处理命令行参数解析
import json  # 用于处理 JSON 格式的数据
import os  # 用于操作系统相关的功能

import torch  # PyTorch 深度学习库

from transformers import (  # 从 transformers 库中导入指定模块和类
    MixtralConfig,  # Mixtral 模型的配置类
    MixtralForCausalLM,  # Mixtral 的条件语言模型类
)

"""
示例用法：


python src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py \
    --input_dir /path/to/downloaded/mixtral/weights --model_size 7B --output_dir /output/path


之后，可以通过以下方式加载模型：


from transformers import MixtralForCausalLM

model = MixtralForCausalLM.from_pretrained("/output/path")


重要说明：你需要能够将整个模型加载到内存中以执行此脚本（即使最大版本被分成多个检查点，每个检查点都包含模型权重的一部分，因此我们需要将它们全部加载到内存中）。
"""


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    # 计算中间层的尺寸，确保是指定倍数的整数
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    # 读取 JSON 文件并返回其内容
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    # 将文本内容以 JSON 格式写入到指定路径的文件中
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, safe_serialization=True):
    # 创建模型路径，如果不存在则创建
    os.makedirs(model_path, exist_ok=True)

    # 读取模型参数的 JSON 文件
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = 1

    # 从 params.json 中读取滑动窗口大小（如果有的话）
    sliding_window = int(params["sliding_window"]) if "sliding_window" in params else None
    n_layers = params["num_hidden_layers"]  # 隐藏层的数量
    n_heads = params["num_attention_heads"]  # 注意力头的数量
    n_heads_per_shard = n_heads // num_shards  # 每个分片的注意力头数量
    dim = params["hidden_size"]  # 隐藏层的尺寸
    dims_per_head = dim // n_heads  # 每个注意力头的尺寸
    base = params.get("rope_theta", 10000.0)  # 获取 rope_theta 参数，默认为 10000.0
    max_position_embeddings = 4096 * 8  # 最大位置嵌入的数量
    num_local_experts = params["num_local_experts"]  # 本地专家的数量
    ffn_dim = params["intermediate_size"]  # 中间层的尺寸

    vocab_size = params["vocab_size"]  # 词汇表的大小

    if "num_key_value_heads" in params:
        num_key_value_heads = params["num_key_value_heads"]  # 键值头的数量（适用于 GQA / MQA）
        num_local_key_value_heads = num_key_value_heads // num_shards  # 每个分片的键值头的数量
        key_value_dim = dims_per_head * num_local_key_value_heads  # 键值维度
    else:  # 兼容其他检查点
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # 对于切片旋转，重新排列
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        # 重新排列张量 `w`，以便于后续处理
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # 打印消息，指示正在从指定路径加载所有参数

    # 加载权重文件列表
    loaded = [
        torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pt"), map_location="cpu") for i in range(8)
    ]

    # 初始化合并后的状态字典
    merged_state_dict = {}
    # 合并所有加载的状态字典
    for state_dict in loaded:
        merged_state_dict.update(state_dict)

    # 初始化状态字典
    state_dict = {}

    # 更新状态字典的特定部分，包括模型的权重
    state_dict.update(
        {
            "model.norm.weight": merged_state_dict["norm.weight"],
            "model.embed_tokens.weight": merged_state_dict["tok_embeddings.weight"],
            "lm_head.weight": merged_state_dict["output.weight"],
        }
    )

    # 初始化 Mixtral 模型的配置
    config = MixtralConfig(
        hidden_size=dim,
        intermediate_size=ffn_dim,
        num_attention_heads=params["num_attention_heads"],
        num_hidden_layers=params["num_hidden_layers"],
        rms_norm_eps=params["rms_norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        sliding_window=sliding_window,
        num_local_experts=num_local_experts,
    )

    # 打印消息，指示正在加载 Mixtral 模型的检查点
    print("Loading the checkpoint in a Mixtral model.")
    # 在指定设备上初始化 Mixtral 模型
    with torch.device("meta"):
        model = MixtralForCausalLM(config)
    # 从配置中删除保存的路径信息，以避免泄露
    del model.config._name_or_path
    # 设置模型配置的 Torch 数据类型为 float16
    model.config.torch_dtype = torch.float16
    # 打印消息，指示正在以 Transformers 格式保存模型
    print("Saving in the Transformers format.")

    # 加载模型的状态字典
    model.load_state_dict(state_dict, strict=True, assign=True)

    # 检查所有模型参数，确保没有参数保存在 `meta` 设备上
    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"{n} has not been loaded!"

    # 将模型保存为预训练文件格式到指定路径
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
# 定义程序的主函数入口点
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数 --input_dir，用于指定Mixtral权重的位置，包含tokenizer.model和model文件夹
    parser.add_argument(
        "--input_dir",
        help="Location of Mixtral weights, which contains tokenizer.model and model folders",
        required=True,
    )
    
    # 添加命令行参数 --model_size，用于选择模型大小，默认为"7B"，与Mixtral官方发布版本对应
    parser.add_argument(
        "--model_size",
        choices=["7B"],
        help="'f' models correspond to the finetuned versions, and are specific to the Mixtral official release. For more details on Mixtral, checkout the original repo: https://huggingface.co/mistral-ai",
        default="7B",
    )
    
    # 添加命令行参数 --output_dir，用于指定写入HF模型的位置
    parser.add_argument("--output_dir", help="Location to write HF model", required=True)
    
    # 添加命令行参数 --safe_serialization，用于指定是否使用安全张量进行保存
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用write_model函数，传入命令行参数来写入模型
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        safe_serialization=args.safe_serialization,
    )


# 程序的入口点，如果直接运行当前脚本，则调用main函数
if __name__ == "__main__":
    main()
```