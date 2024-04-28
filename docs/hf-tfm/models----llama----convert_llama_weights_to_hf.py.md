# `.\transformers\models\llama\convert_llama_weights_to_hf.py`

```
# 导入必要的库和模块
import argparse  # 解析命令行参数的库
import gc  # Python 垃圾回收模块
import json  # JSON 数据的编解码模块
import os  # 操作系统相关的功能模块
import shutil  # 文件操作的高级模块
import warnings  # 警告控制模块

import torch  # PyTorch 深度学习框架

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer  # 导入 Llama 模型相关的类


try:
    from transformers import LlamaTokenizerFast  # 尝试导入 LlamaTokenizerFast 类
except ImportError as e:
    warnings.warn(e)  # 发出警告，提示导入错误
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )  # 发出警告，指示使用慢速的 tokenizer。建议更新 tokenizers 库并重新运行 tokenizer 转换
    LlamaTokenizerFast = None  # 设置 LlamaTokenizerFast 为 None


"""
Sample usage:


python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path


Thereafter, models can be loaded via:


from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")


Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

# 定义不同模型大小对应的分片数量
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

# 计算中间大小的函数
def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

# 读取 JSON 文件的函数
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

# 写入 JSON 文件的函数
def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

# 写入模型的函数
def write_model(model_path, input_base_path, model_size, tokenizer_path=None, safe_serialization=True):
    # 为了向后兼容性，在之前需要将仓库命名为 `my_repo/model_size`
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # 读取参数 JSON 文件
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    # 从参数中获取维度信息
    dim = params["dim"]
    # 计算每个注意力头的维度
    dims_per_head = dim // n_heads
    # 获取或设置默认值为10000的参数
    base = params.get("rope_theta", 10000.0)
    # 计算频率的倒数
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # 根据值设定最大位置嵌入长度
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        max_position_embeddings = 2048

    # 根据情况选择使用哪种分词器类
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    # 如果存在分词路径，创建相应的分词器并保存预训练模型
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)
    # 获取词汇表大小
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000

    # 根据参数中的设置确定键值头数和维度
    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # 兼容使用其他检查点
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # 为切片旋转排列函数定义
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # 打印加载参数的信息
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # 加载权重
    if num_shards == 1:
        # 非分片状态
        # （分片实现也可以工作，但这更简单。）
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # 分片状态
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards == 1:
        # 非分片状态
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    # 遍历并保存状态字典中的内容
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # 写入配置信息
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    # 根据参数设置反馈神经网络维度乘数
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    # 如果参数中包含 "multiple_of"，则将其赋值给 multiple_of，否则设为 256
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    # 创建 LlamaConfig 对象，设置各种参数
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
    # 保存配置到临时模型路径
    config.save_pretrained(tmp_model_path)

    # 释放变量以释放内存
    del state_dict
    del loaded
    gc.collect()

    # 打印信息
    print("Loading the checkpoint in a Llama model.")
    # 从预训练的临时模型路径加载模型
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # 避免将其作为配置的一部分保存
    del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    # 打印信息
    print("Saving in the Transformers format.")
    # 保存模型到指定路径，选择安全的序列化方式
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    # 删除临时模型路径
    shutil.rmtree(tmp_model_path)
# 定义一个函数，用于将tokenizer保存到指定路径
def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # 根据输入的模型路径初始化tokenizer，使用LlamaTokenizerFast如果可用，否则使用LlamaTokenizer
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    # 打印提示信息，说明正在保存tokenizer到指定路径
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    # 初始化并保存tokenizer到指定路径
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)

# 主函数，用于解析命令行参数并执行相应操作
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定LLaMA权重的位置，其中包含tokenizer.model和model文件夹
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    # 添加命令行参数，指定模型的尺寸
    parser.add_argument(
        "--model_size",
        choices=["7B", "7Bf", "13B", "13Bf", "30B", "34B", "65B", "70B", "70Bf", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Llama2 official release. For more details on Llama2, checkout the original repo: https://huggingface.co/meta-llama",
    )
    # 添加命令行参数，指定输出目录的位置
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数，指定是否使用'safetensors'进行安全序列化
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析命令行参数
    args = parser.parse_args()
    # 拼接tokenizer.model的完整路径
    spm_path = os.path.join(args.input_dir, "tokenizer.model")
    # 如果模型尺寸不是"tokenizer_only"，则调用write_model函数，否则调用write_tokenizer函数
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            model_size=args.model_size,
            safe_serialization=args.safe_serialization,
            tokenizer_path=spm_path,
        )
    else:
        write_tokenizer(args.output_dir, spm_path)

# 如果该脚本被直接运行，则执行main函数
if __name__ == "__main__":
    main()
```