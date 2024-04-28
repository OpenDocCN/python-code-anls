# `.\transformers\models\mixtral\convert_mixtral_weights_to_hf.py`

```py
# 导入所需的模块和库
import argparse  # 用于解析命令行参数
import json  # 用于 JSON 数据的读写操作
import os  # 用于操作文件系统

import torch  # 导入 PyTorch 库

# 导入 Transformers 库中的模型相关类和函数
from transformers import (
    MixtralConfig,  # 导入 Mixtral 模型的配置类
    MixtralForCausalLM,  # 导入 Mixtral 模型的语言模型类
)

"""
示例用法:


python src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py \
    --input_dir /path/to/downloaded/mixtral/weights --model_size 7B --output_dir /output/path


之后，可以通过以下方式加载模型:


from transformers import MixtralForCausalLM

model = MixtralForCausalLM.from_pretrained("/output/path")


重要说明: 为了执行此脚本，您需要能够在内存中托管整个模型（即使最大版本也是如此，
因为它们中的每一个都包含模型的每个权重的一部分，所以我们需要将它们全部加载到内存中）。
"""


# 计算中间层大小的函数，采用 GPT 架构的计算方式
def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


# 读取 JSON 文件并返回其中的内容
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


# 将文本写入 JSON 文件
def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


# 将模型写入指定路径
def write_model(model_path, input_base_path, model_size, safe_serialization=True):
    # 确保输出目录存在
    os.makedirs(model_path, exist_ok=True)

    # 从输入路径中读取模型参数
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = 1

    # 解析模型参数
    sliding_window = int(params["sliding_window"]) if "sliding_window" in params else None
    n_layers = params["num_hidden_layers"]
    n_heads = params["num_attention_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["hidden_size"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    max_position_embeddings = 4096 * 8
    num_local_experts = params["num_local_experts"]
    ffn_dim = params["intermediate_size"]
    vocab_size = params["vocab_size"]

    if "num_key_value_heads" in params:
        num_key_value_heads = params["num_key_value_heads"]  # for GQA / MQA
        num_local_key_value_heads = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_local_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    # 定义一个函数 permute，用于对权重 w 进行变换操作
        def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
            # 将权重 w 根据 n_heads 和维度 dim1、dim2 进行 view 操作
            # 然后进行转置和 reshape 操作，返回变换后的权重
            return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    
        # 输出提示信息，说明正在从检查点中获取所有参数
        print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
        # 从指定路径加载 8 个权重文件，并将它们保存在 loaded 列表中
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pt"), map_location="cpu") for i in range(8)
        ]
    
        # 创建一个空的字典 merged_state_dict，用于合并 loaded 列表中的所有权重
        merged_state_dict = {}
        for state_dict in loaded:
            merged_state_dict.update(state_dict)
    
        # 创建一个空的字典 state_dict，用于存储需要使用的部分权重
        state_dict = {}
    
        # 将 merged_state_dict 中的部分权重更新到 state_dict 中
        state_dict.update(
            {
                "model.norm.weight": merged_state_dict["norm.weight"],
                "model.embed_tokens.weight": merged_state_dict["tok_embeddings.weight"],
                "lm_head.weight": merged_state_dict["output.weight"],
            }
        )
    
        # 根据指定的参数创建 MixtralConfig 对象
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
    
        # 输出提示信息，说明正在将检查点加载到 Mixtral 模型中
        print("Loading the checkpoint in a Mixtral model.")
        # 在 "meta" 设备上创建 MixtralForCausalLM 模型
        with torch.device("meta"):
            model = MixtralForCausalLM(config)
        # 从模型配置中删除 "_name_or_path" 属性
        del model.config._name_or_path
        # 将模型的 torch_dtype 属性设置为 torch.float16
        model.config.torch_dtype = torch.float16
        # 输出提示信息，说明正在以 Transformers 格式保存模型
        print("Saving in the Transformers format.")
    
        # 加载 state_dict 中的权重到模型中
        model.load_state_dict(state_dict, strict=True, assign=True)
    
        # 检查所有参数是否已加载成功
        for n, p in model.named_parameters():
            assert p.device.type != "meta", f"{n} has not been loaded!"
    
        # 将模型保存到指定路径
        model.save_pretrained(model_path, safe_serialization=safe_serialization)
# 主函数，程序入口
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入目录路径，必须提供
    parser.add_argument(
        "--input_dir",
        help="Location of Mixtral weights, which contains tokenizer.model and model folders",
        required=True,
    )
    # 添加命令行参数：模型大小，可选值为"7B"，默认为"7B"
    parser.add_argument(
        "--model_size",
        choices=["7B"],
        help="'f' models correspond to the finetuned versions, and are specific to the Mixtral official release. For more details on Mixtral, checkout the original repo: https://huggingface.co/mistral-ai",
        default="7B",
    )
    # 添加命令行参数：输出目录路径，必须提供
    parser.add_argument("--output_dir", help="Location to write HF model", required=True)
    # 添加命令行参数：是否使用安全序列化，布尔类型
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 write_model 函数，传入参数
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        safe_serialization=args.safe_serialization,
    )

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```