# `.\models\llama\convert_llama_weights_to_hf.py`

```py
# 导入必要的库
import argparse  # 用于解析命令行参数
import gc  # Python 的垃圾回收模块
import json  # 用于 JSON 文件的读写操作
import os  # 提供了对操作系统的接口，用于文件和目录操作
import shutil  # 提供高级的文件操作功能
import warnings  # 用于处理警告信息

import torch  # 引入 PyTorch 库

# 从 transformers 库中导入所需的类
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

try:
    # 尝试从 tokenizers 库中导入 LlamaTokenizerFast 类
    from transformers import LlamaTokenizerFast
except ImportError as e:
    # 如果导入失败，发出警告并提示使用慢速的 tokenizer
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
样例用法：

python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
"""

# 各模型尺寸对应的分片数目
NUM_SHARDS = {
    "7B": 1,
    "7Bf": 1,
    "13B": 2,
    "13Bf": 2,
    "34B": 4,
    "30B": 4,
    "65B": 8,
    "70B": 8,
    "70Bf": 8,
}

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    # 计算中间层的大小，确保是给定倍数的整数
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

def read_json(path):
    # 从指定路径读取 JSON 文件并返回其内容
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    # 将文本内容写入指定路径的 JSON 文件
    with open(path, "w") as f:
        json.dump(text, f)

def write_model(
    model_path, input_base_path, model_size, tokenizer_path=None, safe_serialization=True, llama_version=1
):
    # 为了向后兼容性，如果之前需要 repo 被称为 `my_repo/model_size`
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    # 创建模型路径和临时模型路径
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # 读取模型参数 JSON 文件
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    # 计算每个分片中的注意力头数
    n_heads_per_shard = n_heads // num_shards

    # 从参数字典中获取维度信息
    dim = params["dim"]

    # 计算每个头部的维度大小
    dims_per_head = dim // n_heads

    # 获取参数中的 "rope_theta"，默认为 10000.0
    base = params.get("rope_theta", 10000.0)

    # 计算逆频率，用于位置编码
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # 根据 base 的大小确定最大位置嵌入的值
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        # 根据 Llama 的版本确定默认的最大位置嵌入
        if llama_version == 1:
            max_position_embeddings = 2048
        elif llama_version == 2:
            max_position_embeddings = 4096
        else:
            # 抛出未实现错误，对于不支持的 Llama 版本
            raise NotImplementedError(
                f"Version {llama_version} of llama is not supported yet. "
                "Current supported versions of llama are [1, 2]."
            )

    # 根据 LlamaTokenizerFast 是否为 None 选择正确的 tokenizer 类
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast

    # 如果提供了 tokenizer_path，则初始化 tokenizer 并保存到 model_path
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)

    # 根据 tokenizer_path 是否为 None 决定词汇表大小
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000

    # 如果参数中提供了 n_kv_heads，则使用其定义的键值头数，否则使用默认值
    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:
        # 兼容性处理，对于其他检查点使用默认值
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # 定义用于分片旋转的置换函数
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # 打印加载检查点参数的信息
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    # 加载权重
    if num_shards == 1:
        # 如果不分片，则加载单个文件
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # 如果分片，则加载所有分片的文件
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]

    # 初始化参数计数器和索引字典
    param_count = 0
    index_dict = {"weight_map": {}}

    # 构建模型文件名
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"

    if num_shards == 1:
        # 如果不分片，则构建状态字典
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        # 如果分片，则合并各分片的权重
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }
    # 遍历状态字典中的键值对，将键（参数名称）映射到文件名
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        # 累加参数张量中元素的数量，计算模型参数总数
        param_count += v.numel()
    
    # 使用PyTorch保存模型参数到文件系统中
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # 写入配置信息到索引字典中
    index_dict["metadata"] = {"total_size": param_count * 2}
    # 将索引字典以JSON格式写入到文件系统中
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    
    # 根据参数中的配置，确定FFN维度的倍增器和倍数
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    
    # 创建Llama模型的配置对象
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
    )
    # 将配置保存到临时模型路径
    config.save_pretrained(tmp_model_path)

    # 释放不再需要的对象，清理内存
    del state_dict
    del loaded
    gc.collect()

    # 打印加载Llama模型检查点的消息
    print("Loading the checkpoint in a Llama model.")
    # 从预训练模型路径加载Llama模型，指定张量数据类型和低CPU内存使用
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    
    # 避免将此项设置保存为配置的一部分
    del model.config._name_or_path
    # 将模型配置的张量数据类型设置为float16
    model.config.torch_dtype = torch.float16
    
    # 打印保存为Transformers格式的消息
    print("Saving in the Transformers format.")
    # 将Llama模型保存到指定的模型路径，进行安全序列化
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    
    # 递归删除临时模型路径及其内容
    shutil.rmtree(tmp_model_path)
# 主函数，程序的入口点
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入目录，包含 LLAMA 权重文件，包括 tokenizer.model 和 model 文件夹
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    # 添加命令行参数：模型大小，可选项为不同大小的 Llama 模型或仅令牌化器
    parser.add_argument(
        "--model_size",
        choices=["7B", "7Bf", "13B", "13Bf", "30B", "34B", "65B", "70B", "70Bf", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Llama2 official release. For more details on Llama2, checkout the original repo: https://huggingface.co/meta-llama",
    )
    # 添加命令行参数：输出目录，用于写入 HF 模型和令牌化器
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数：安全序列化选项，指示是否使用 `safetensors` 进行保存
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 添加命令行参数：LLAMA 版本，选择 1 或 2，用于控制上下文大小
    parser.add_argument(
        "--llama_version",
        choices=[1, 2],
        default=1,
        type=int,
        help="Version of the Llama model to convert. Currently supports Llama1 and Llama2. Controls the context size",
    )
    # 解析命令行参数
    args = parser.parse_args()
    
    # 构造令牌化器模型文件路径
    spm_path = os.path.join(args.input_dir, "tokenizer.model")
    
    # 如果模型大小不是 "tokenizer_only"，则调用写入模型函数
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            model_size=args.model_size,
            safe_serialization=args.safe_serialization,
            tokenizer_path=spm_path,
            llama_version=args.llama_version,
        )
    else:
        # 否则，仅写入令牌化器
        write_tokenizer(args.output_dir, spm_path)


# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```