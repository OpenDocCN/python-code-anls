# `.\models\mistral\convert_mistral_weights_to_hf.py`

```
# 导入必要的库和模块
import argparse  # 解析命令行参数的库
import gc  # Python 的垃圾回收模块
import json  # 处理 JSON 格式数据的库
import os  # 提供与操作系统交互的功能
import shutil  # 提供高级文件操作功能
import warnings  # 发出警告的模块

import torch  # 引入 PyTorch 深度学习库

# 从transformers库中导入所需的类和函数
from transformers import (
    LlamaTokenizer,  # LlamaTokenizer 分词器
    MistralConfig,  # Mistral模型的配置类
    MistralForCausalLM,  # 用于生成文本的Mistral模型
)

try:
    from transformers import LlamaTokenizerFast  # 尝试导入快速版LlamaTokenizer

    tokenizer_class = LlamaTokenizerFast  # 如果导入成功，使用快速版分词器
except ImportError as e:
    warnings.warn(e)  # 输出导入错误的警告
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    tokenizer_class = LlamaTokenizer  # 如果导入失败，使用慢速版分词器

"""
示例用法:

python src/transformers/models/mistral/convert_mistral_weights_to_hf.py \
    --input_dir /path/to/downloaded/mistral/weights --model_size 7B --output_dir /output/path
"""

# 将不同模型大小映射到对应的分片数量
NUM_SHARDS = {"7B": 1}


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    # 计算中间层的尺寸
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    # 读取指定路径下的JSON文件内容并返回解析后的Python对象
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    # 将Python对象text写入到指定路径的JSON文件中
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, tokenizer_path=None, safe_serialization=True):
    # 为了向后兼容，检查参数文件是否位于指定路径，若不是，则修改输入基础路径
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    # 创建存储模型的目录和临时目录
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # 读取参数文件中的参数信息
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]

    # 将参数中的滑动窗口大小转换为整数
    sliding_window = int(params["sliding_window"])
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    # 计算每个分片中的注意力头数量
    n_heads_per_shard = n_heads // num_shards
    # 从参数字典中获取维度信息
    dim = params["dim"]
    # 计算每个注意力头的维度
    dims_per_head = dim // n_heads
    # 获取参数中的 "rope_theta"，默认为 10000.0
    base = params.get("rope_theta", 10000.0)
    # 计算正弦频率的倒数，用于位置编码
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # 设置最大位置编码长度
    max_position_embeddings = 4096 * 8

    # 如果指定了 tokenizer_path，则初始化并保存 tokenizer
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)
    # 获取词汇表大小，如果未指定 tokenizer_path 则默认为 32000
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000

    # 如果参数中包含 "n_kv_heads"，则设置键值头的数量
    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        # 计算每个本地键值头的数量
        num_local_key_value_heads = num_key_value_heads // num_shards
        # 计算键值维度
        key_value_dim = dims_per_head * num_local_key_value_heads
    else:  # 兼容其他检查点
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # 定义用于切片旋转的排列函数
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # 打印加载检查点的消息
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # 加载权重
    loaded = [
        torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
        for i in range(num_shards)
    ]
    # 初始化参数计数器和索引字典
    param_count = 0
    index_dict = {"weight_map": {}}
    # 设置模型文件名
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    # 构建状态字典
    state_dict = {
        "model.norm.weight": loaded[0]["norm.weight"],
        "model.embed_tokens.weight": torch.cat([loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1),
        "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
    }

    # 将状态字典的键值对保存到索引字典中，并统计参数数量
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    # 将状态字典保存到临时模型路径
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # 写入配置信息到索引字典
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    # 创建 Mistral 模型配置
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
    # 将模型配置保存到临时模型路径
    config.save_pretrained(tmp_model_path)

    # 释放不再需要的变量，进行内存回收
    del state_dict
    del loaded
    gc.collect()

    # 打印加载模型检查点的消息
    print("Loading the checkpoint in a Mistral model.")
    # 从预训练模型路径加载 Mistral 模型
    model = MistralForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # 移除模型配置中的 _name_or_path 属性，避免保存到配置中
    del model.config._name_or_path
    # 设置模型配置中的 Torch 数据类型为 float16
    model.config.torch_dtype = torch.float16
    # 打印保存模型为 Transformers 格式的消息
    print("Saving in the Transformers format.")
    # 使用安全序列化选项保存模型到指定路径
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    # 递归删除临时模型路径下的所有文件和文件夹
    shutil.rmtree(tmp_model_path)
# 定义一个函数用于保存 tokenizer
def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # 打印保存 tokenizer 的信息，包括 tokenizer 类型和保存路径
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    # 根据输入的 tokenizer 路径初始化 tokenizer 对象
    tokenizer = tokenizer_class(input_tokenizer_path)
    # 调用预训练模型的方法保存 tokenizer 到指定路径
    tokenizer.save_pretrained(tokenizer_path)


# 定义主函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入目录，用于存放 Mistral 权重，包含 tokenizer.model 和 model 文件夹
    parser.add_argument(
        "--input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    # 添加命令行参数：模型大小，可以选择 "7B" 或 "tokenizer_only"
    parser.add_argument(
        "--model_size",
        choices=["7B", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Mistral2 official release. For more details on Mistral2, checkout the original repo: https://huggingface.co/meta-mistral",
    )
    # 添加命令行参数：输出目录，用于存放 HF 模型和 tokenizer
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数：安全序列化选项，是否使用 `safetensors` 进行保存
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析命令行参数
    args = parser.parse_args()
    # 构建 tokenizer 的路径，拼接输入目录和 tokenizer 文件名
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
    else:
        # 否则，调用 write_tokenizer 函数保存 tokenizer
        write_tokenizer(args.output_dir, spm_path)


# 如果当前脚本作为主程序运行，则执行主函数 main()
if __name__ == "__main__":
    main()
```