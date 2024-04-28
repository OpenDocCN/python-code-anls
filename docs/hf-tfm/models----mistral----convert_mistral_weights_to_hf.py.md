# `.\transformers\models\mistral\convert_mistral_weights_to_hf.py`

```
# 版权声明和导入模块
# 以上代码是版权声明和导入必要的模块

import argparse  # 导入解析命令行参数的模块
import gc  # 导入垃圾回收模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入文件和目录操作相关的模块
import shutil  # 导入文件和目录操作相关的模块
import warnings  # 导入警告处理模块
import torch  # 导入 PyTorch 深度学习框架

from transformers import (  # 从 transformers 模块中导入以下类
    LlamaTokenizer,  # LlamaTokenizer 类
    MistralConfig,  # MistralConfig 类
    MistralForCausalLM,  # MistralForCausalLM 类
)

# 尝试导入 LlamaTokenizerFast 模块
try:
    from transformers import LlamaTokenizerFast  # 导入 LlamaTokenizerFast 类
    tokenizer_class = LlamaTokenizerFast  # 如果导入成功，则使用 LlamaTokenizerFast 类
except ImportError as e:  # 如果导入失败，则捕获 ImportError 异常
    warnings.warn(e)  # 发出警告，显示导入错误信息
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )  # 发出警告，显示提示信息
    tokenizer_class = LlamaTokenizer  # 使用 LlamaTokenizer 类

"""
示例用法:
...
"""

NUM_SHARDS = {"7B": 1}  # 定义 NUM_SHARDS 字典，存储模型大小和分片数量的对应关系

# 计算中间层的大小
def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

# 读取 JSON 文件
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)  # 从文件中载入 JSON 数据并返回

# 写入 JSON 文件
def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)  # 将 JSON 数据写入文件

# 写入模型
def write_model(model_path, input_base_path, model_size, tokenizer_path=None, safe_serialization=True):
    # 为了向后兼容，如果需要仓库被称为 `my_repo/model_size` 
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    os.makedirs(model_path, exist_ok=True)  # 创建模型路径
    tmp_model_path = os.path.join(model_path, "tmp")  # 创建临时模型路径
    os.makedirs(tmp_model_path, exist_ok=True)  # 创建临时模型路径

    params = read_json(os.path.join(input_base_path, "params.json"))  # 读取参数 JSON 文件
    num_shards = NUM_SHARDS[model_size]  # 获取模型大小对应的分片数量

    # 由于某些原因，params.json 中的滑动窗口是字符串
    sliding_window = int(params["sliding_window"])  # 将滑动窗口转换为整数
    n_layers = params["n_layers"]  # 获取层数
    n_heads = params["n_heads"]  # 获取头数
    # 计算每个 shard 中的头的数量
    n_heads_per_shard = n_heads // num_shards
    # 获取模型参数中的 dim 值
    dim = params["dim"]
    # 计算每个头的维度
    dims_per_head = dim // n_heads
    # 获取 rope_theta 参数的值，如果没有则设为 10000.0
    base = params.get("rope_theta", 10000.0)
    # 计算逆频率向量，用于 Rotary Position Embedding
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # 设置最大位置嵌入数量
    max_position_embeddings = 4096 * 8
    
    # 如果提供了 tokenizer 路径，则加载并保存 tokenizer
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)
    # 设置词表大小
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000
    
    # 如果模型参数中有 n_kv_heads，则设置 key-value 头的数量和维度
    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_local_key_value_heads
    # 否则设置为与其他检查点兼容的值
    else:
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim
    
    # 定义一个用于 sliced rotary 的 permute 函数
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    
    # 从检查点中获取所有参数
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = [
        torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
        for i in range(num_shards)
    ]
    param_count = 0
    index_dict = {"weight_map": {}}
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.norm.weight": loaded[0]["norm.weight"],
        "model.embed_tokens.weight": torch.cat([loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1),
        "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
    }
    
    # 构建 state_dict 并记录参数数量及权重映射
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))
    
    # 写入配置文件
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    config = MistralConfig(
        hidden_size=dim,
        intermediate_size=params["hidden_dim"],
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
        sliding_window=sliding_window,
    )
    config.save_pretrained(tmp_model_path)
    
    # 释放内存以便加载模型
    del state_dict
    del loaded
    gc.collect()
    
    # 从临时路径加载 Mistral 模型
    print("Loading the checkpoint in a Mistral model.")
    model = MistralForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # 删除模型配置中的 _name_or_path 属性
    del model.config._name_or_path
    # 设置模型配置的 torch 数据类型为 float16
    model.config.torch_dtype = torch.float16
    # 打印信息，表示将以 Transformers 格式保存模型
    print("Saving in the Transformers format.")
    # 保存模型到指定路径，使用安全序列化方式
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    # 递归删除临时模型路径
    shutil.rmtree(tmp_model_path)
# 定义一个函数，用于将 tokenizer 保存到指定路径
def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # 打印提示信息，说明正在保存 tokenizer 到指定路径
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    # 根据输入的 tokenizer 路径初始化 tokenizer
    tokenizer = tokenizer_class(input_tokenizer_path)
    # 保存 tokenizer 到指定路径
    tokenizer.save_pretrained(tokenizer_path)

# 主函数
def main():
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加输入目录参数
    parser.add_argument(
        "--input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    # 添加模型大小参数
    parser.add_argument(
        "--model_size",
        choices=["7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Mistral2 official release. For more details on Mistral2, checkout the original repo: https://huggingface.co/meta-mistral",
    )
    # 添加输出目录参数
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加安全序列化参数
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析参数
    args = parser.parse_args()
    # 拼接 tokenizer 路径
    spm_path = os.path.join(args.input_dir, "tokenizer.model")
    # 如果模型大小不是 "tokenizer_only"，则调用 write_model 函数
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            model_size=args.model_size,
            safe_serialization=args.safe_serialization,
            tokenizer_path=spm_path,
        )
    # 否则调用 write_tokenizer 函数
    else:
        write_tokenizer(args.output_dir, spm_path)

# 如果当前脚本被执行，则调用主函数
if __name__ == "__main__":
    main()
```