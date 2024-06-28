# `.\models\megatron_gpt2\checkpoint_reshaping_and_interoperability.py`

```
# 导入必要的库和模块
import argparse  # 导入处理命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入处理操作系统相关功能的模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关的功能模块
import types  # 导入 types 模块，用于操作类型信息

import torch  # 导入 PyTorch 深度学习库

# 导入 transformers 相关模块和类
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_checkpointing_args(parser):
    # 添加命令行参数：Megatron 代码库的基本目录
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    # 添加命令行参数：是否进行 Megatron 到 Transformers 的检查点转换
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加命令行参数：待转换的检查点路径
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    # 添加命令行参数：转换后保存的检查点路径
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    # 添加命令行参数：是否打印检查点结构
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    # 添加命令行参数：转换后的张量模型并行大小
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加命令行参数：转换后的管道模型并行大小
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加命令行参数：转换后的数据并行大小
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加命令行参数：转换后的参数数据类型
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加命令行参数：使得词汇表大小可被此值整除
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficiency reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加命令行参数：使用分布式优化器
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 返回配置好命令行参数的解析器对象
    return parser
def add_transformers_checkpoint_args(parser):
    """
    添加 Transformers 检查点的参数到解析器中。

    Args:
        parser (ArgumentParser): 解析器对象，用于添加参数

    Returns:
        ArgumentParser: 更新后的解析器对象
    """
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "要保存的预训练分词器的名称。如果不是 None，则会保存分词器。"
            "仅在将 Megatron 检查点转换为 Transformers 检查点时使用。"
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "在分片之前检查点的最大大小。检查点分片将小于此大小。"
            "如果表示为字符串，需由数字后跟单位（如 `5MB`）组成。"
            "仅在将 Megatron 检查点转换为 Transformers 检查点时使用。"
        ),
    )

    return parser


# "automated" rules 名称映射的简单映射。
megatron_to_transformers = {
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
}
# 从 transformers 到 megatron 的反向映射。
transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

tensor_parallel_params = [
    # 在 tp ranks 之间合并的 megatron-lm 层
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # 已弃用
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # 在 tp ranks 之间分割的 transformers 层
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
]


def recursive_print(name, val, spaces=0):
    """
    递归打印检查点的结构。此函数源自 `convert_megatron_gpt2_checkpoint.py`。

    Args:
        name (str): 当前张量参数的名称
        val (Tuple(int)): 当前张量参数的形状
        spaces (int): 输出嵌套结构之前的空格数
    """
    # 格式化消息。
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # 打印并递归（如果需要）。
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    重新排列 param 张量的布局，以便与后续版本兼容为 [num_splits * num_heads * hidden_size, :]。

    Args:
        param: 要重新排列的参数张量
        checkpoint_version: 检查点版本
        num_splits: 分片数
        num_heads: 头数
        hidden_size: 隐藏大小
    """
    # 获取输入张量的形状
    input_shape = param.size()
    
    # 根据不同的检查点版本进行张量重排
    if checkpoint_version == 1.0:
        # 版本 1.0 存储格式为 [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        # 重塑张量形状
        param = param.view(*saved_shape)
        # 转置操作，调整张量维度顺序
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # 其他版本存储格式为 [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        # 重塑张量形状
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    
    # 恢复原始张量形状
    param = param.view(*input_shape)
    
    # 返回处理后的张量
    return param
# 将参数张量的布局重新排列，以适应相应的NVIDIA Megatron-LM检查点版本。输入形状为 [num_splits * num_heads * hidden_size, :]，输出形状根据版本不同分别为 [num_heads * hidden_size * num_splits, :]（版本 1.0及之前）和 [num_heads * num_splits * hidden_size, :]（版本 2.0及之后）。如果参数是自注意力块的权重张量，则在调用此函数之前需要已经进行转置。

def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # 获取输入张量的形状
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # 对于版本 1.0，存储结构为 [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        # 调整张量的形状和顺序以匹配版本 1.0 的要求
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # 对于版本 2.0 及更高，存储结构为 [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        # 调整张量的形状和顺序以匹配版本 2.0 及更高的要求
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    # 恢复原始张量的形状
    param = param.view(*input_shape)
    return param


# 从transformers的分片检查点中合并成一个单一检查点。
def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    # 创建一个空的状态字典用于存储合并后的状态
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        # 构建每个分片检查点的完整路径
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        # 加载当前分片的检查点到内存中
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        # 将当前分片的状态字典更新到总的状态字典中
        state_dict.update(current_chunk)
    return state_dict


# 从NVIDIA Megatron-LM检查点中获取分片状态，基于提供的张量并行大小、管道并行大小和管道并行等级。
def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    # 创建一个空列表来存储张量并行状态字典
    tp_state_dicts = []
    # 遍历指定范围内的整数，生成索引 i，范围从 0 到 tp_size-1
    for i in range(tp_size):
        # 根据进程数 pp_size 的情况生成子目录名
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        
        # 遍历检查点文件名列表，查找存在于文件系统中的第一个检查点文件
        for checkpoint_name in ["model_optim_rng.pt", "model_rng.pt"]:
            # 构建完整的检查点文件路径
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            
            # 如果找到该路径对应的文件存在，则跳出当前循环
            if os.path.isfile(checkpoint_path):
                break
        
        # 使用 CPU 加载检查点文件的状态字典，并存储到列表中
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    
    # 返回包含所有加载状态字典的列表
    return tp_state_dicts
# 根据路径从字典中获取元素。如果元素不存在，则递归添加空字典。
def get_element_from_dict_by_path(d, path):
    # 将路径字符串按 "." 分割为列表
    path = path.split(".")
    # 遍历路径中的每个键
    for k in path:
        # 如果当前键不在字典中，将其添加为一个空字典
        if k not in d:
            d[k] = {}
        # 更新字典为当前键对应的值，以便继续下一级路径的处理
        d = d[k]
    # 返回最终路径对应的值，即字典中指定路径的元素
    return d


# 将 NVIDIA Megatron-LM 的检查点转换为 HuggingFace Transformers 的检查点
def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # 获取 Megatron-LM 检查点目录下的子目录列表
    sub_dirs = os.listdir(args.load_path)
    # 可能的子目录命名约定
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    # 遍历可能的子目录，寻找符合条件的检查点路径
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            # 获取子目录下的第一个文件名作为检查点文件名
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            # 构建完整的检查点路径
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    # 打印加载 Megatron-LM 检查点参数的信息
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    # 使用 torch 加载 Megatron-LM 检查点的状态字典
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    # 从状态字典中获取 Megatron-LM 的参数
    megatron_args = state_dict.get("args", None)
    # 如果未找到 Megatron-LM 参数，则抛出错误
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # 根据 Megatron-LM 的参数创建 Transformers GPT2 的配置
    if megatron_args is not None:
        # 根据 Megatron-LM 的参数选择激活函数
        if megatron_args.bias_gelu_fusion:
            activation_function = "gelu_fast"
        elif megatron_args.openai_gelu:
            activation_function = "gelu_new"
        else:
            activation_function = "gelu"
    else:
        # 如果未提供 Megatron-LM 参数，默认使用 "gelu_new" 作为激活函数
        activation_function = "gelu_new"
    # 确定词汇表大小
    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    # 打印词汇表大小
    print(vocab_size)
    # 使用 GPT2Config 类创建配置对象，设置模型的各种参数
    config = GPT2Config(
        vocab_size=vocab_size,  # 词汇表大小
        n_positions=megatron_args.max_position_embeddings,  # 最大位置编码数
        n_embd=megatron_args.hidden_size,  # 隐藏层大小
        n_layer=megatron_args.num_layers,  # 层数
        n_head=megatron_args.num_attention_heads,  # 注意力头数
        n_inner=megatron_args.ffn_hidden_size,  # FeedForward 层的隐藏大小
        activation_function=activation_function,  # 激活函数类型
        resid_pdrop=0.1,  # 残差连接中的丢弃率
        embd_pdrop=0.1,  # 嵌入层中的丢弃率
        attn_pdrop=0.1,  # 注意力层中的丢弃率
        layer_norm_epsilon=1e-5,  # 层归一化的 epsilon 值
        initializer_range=0.02,  # 初始化范围
        summary_type="cls_index",  # 摘要类型
        summary_use_proj=True,  # 是否使用摘要投影
        summary_activation=None,  # 摘要激活函数类型
        summary_proj_to_labels=True,  # 是否将摘要投影到标签
        summary_first_dropout=0.1,  # 摘要层的第一个丢弃率
        scale_attn_weights=True,  # 是否缩放注意力权重
        use_cache=True,  # 是否使用缓存
        bos_token_id=vocab_size - 1,  # 开始标记的 ID
        eos_token_id=vocab_size - 1,  # 结束标记的 ID
        architectures=["GPT2LMHeadModel"],  # 模型架构列表
    )

    # 初始化空的状态字典
    output_state_dict = {}

    # 从状态字典中获取检查点版本，如果没有则默认为 0.0
    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    
    # 获取 tensor 模型并行的大小和 pipeline 模型并行的大小
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    
    # 设置数据类型为 torch.float32
    dtype = torch.float32
    
    # 编译正则表达式，用于提取层名称
    # 正则表达式用于匹配形式如 layers.(\d+).([a-z0-9_.]+).([a-z]+) 的字符串
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # 输出信息：转换中
    print("Converting")

    # 输出信息：转换嵌入层
    print("Converting embeddings")
    
    # 获取 Megatron 分片状态字典的 tensor 模型并行数据
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # 获取位置嵌入并存储到 output_state_dict 中
    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )
    output_state_dict["transformer.wpe.weight"] = position_embeddings.to(dtype)

    # 获取单词嵌入并存储到 output_state_dict 中
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # 输出信息：转换 transformer 层
    print("Converting transformer layers")
    
    # 获取配置中的头数和每个头的隐藏大小
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head
    n_positions = config.n_positions
    
    # 计算每个 pipeline 模型并行的层数
    num_layers = config.num_hidden_layers // pp_size

    # 如果配置的层数与当前层索引不匹配，则抛出值错误
    if config.n_layer != (layer_idx + 1):
        raise ValueError(f"Expected {config.n_layer} layers but found {layer_idx + 1}")

    # 输出信息：转换最终的 layernorm 层
    print("Converting final layernorm")
    
    # 从 tp_state_dicts 中获取指定路径的参数，并存储到 output_state_dict 中
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["transformer.ln_f.weight"] = params["final_layernorm.weight"].to(dtype)
    output_state_dict["transformer.ln_f.bias"] = params["final_layernorm.bias"].to(dtype)

    # 输出信息：转换语言模型头
    print("Converting LM head")
    # 将 word_embeddings 的权重转换为指定的数据类型，并存入输出状态字典中
    output_state_dict["lm_head.weight"] = word_embeddings.to(dtype)

    # 输出转换完成的信息提示
    print("Conversion from Megatron-LM to Transformers is done!")

    # 如果设置了打印检查点结构的选项，则递归打印输出状态字典的结构
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # 根据参数设置 tokenizer 的名称，若未指定则使用默认名称
    # 创建对应的 AutoTokenizer 对象
    if args.tokenizer_name is None:
        tokenizer_name = "openai-community/gpt2"
    else:
        tokenizer_name = args.tokenizer_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # 获取 tokenizer 类的名称并存入配置对象中
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # 保存配置对象到指定路径
    print("Saving config")
    config.save_pretrained(args.save_path)

    # 根据参数保存 tokenizer 到指定路径
    if args.tokenizer_name is not None:
        print(f"Adding {tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # 将输出状态字典分片并存储到文件中
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # 逐个保存分片后的模型
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    # 若没有分片，则直接输出模型权重文件的保存路径
    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        # 否则保存分片索引到文件中，并输出详细信息
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )
def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    # 如果保存路径不存在，则创建
    os.makedirs(args.save_path, exist_ok=True)
    
    # 将父目录加入系统路径中以便搜索
    # 在当前文件的上级目录中搜索
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    # 如果指定了Megatron路径，则将其作为最高优先级的路径插入系统路径中
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        # 尝试导入Megatron的tokenizer模块中的_vocab_size_with_padding函数
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        # 如果导入失败，则输出错误信息并退出程序
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    # 加载transformers模型的状态字典和配置文件
    sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
    
    # 如果只有一个子目录，直接加载单一的pytorch_model.bin文件
    if len(sub_dirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    else:
        # 如果有多个子目录，调用merge_transformers_sharded_states函数合并多个分片的状态
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    # 从预训练模型路径中加载GPT2Config配置
    config = GPT2Config.from_pretrained(args.load_path)

    # 保存跟踪文件
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        # 写入"release"作为内容
        f.write("release")

    # 创建`release`目录在args.save_path中
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # 设置Megatron的参数
    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "max_position_embeddings": config.n_positions,
        "hidden_size": config.n_embd,
        "num_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "ffn_hidden_size": config.n_inner,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "GPT2BPETokenizer",
    }

    # 根据激活函数类型设置相应的Megatron参数
    if config.activation_function == "gelu":
        megatron_args["bias_gelu_fusion"] = False
        megatron_args["openai_gelu"] = False
    elif config.activation_function == "gelu_fast":
        megatron_args["bias_gelu_fusion"] = True
        megatron_args["openai_gelu"] = False
    elif config.activation_function == "gelu_new":
        megatron_args["bias_gelu_fusion"] = False
        megatron_args["openai_gelu"] = True

    # 使用types模块创建命名空间对象margs，并设置其属性为megatron_args中的键值对
    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)
    # 根据参数设置目标参数的数据类型
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)  # 将数据类型设置为模型参数对象的属性

    # 保存一个虚拟的优化器状态字典
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,  # 优化器的步数
        "param_groups": [
            {
                "lr": 0.0,  # 学习率
                "beta1": 0.0,  # Adam优化器的beta1参数
                "beta2": 0.0,  # Adam优化器的beta2参数
                "eps": 0.0,  # Adam优化器的epsilon参数
                "weight_decay": 0.0,  # 权重衰减参数
                "correct_bias": False,  # 是否校正偏置
                "params": [],  # 参数组
            }
        ],
    }

    # 如果使用分布式优化器
    if args.use_distributed_optimizer:
        for i in range(args.target_pipeline_model_parallel_size):
            for j in range(args.target_tensor_model_parallel_size):
                for k in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size == 1:
                        checkpoint_dir = f"mp_rank_{j:02d}_{k:03d}"
                    else:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}_{k:03d}"
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )

    # 打印信息，开始转换
    print("Converting")

    # 创建一个空列表，用于存储输出的状态字典
    output_state_dict = []

    # 为每个目标张量模型并行大小创建一个空字典
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # 处理嵌入层
    print("converting embedding layer")

    # 将位置嵌入和词嵌入转换为指定的数据类型
    pos_embedding = state_dict["transformer.wpe.weight"].to(dtype)
    word_embedding = state_dict["transformer.wte.weight"].to(dtype)

    orig_vocab_size = config.vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    setattr(margs, "padded_vocab_size", padded_vocab_size)

    # 如果原始词汇表大小大于填充后的大小，则裁剪多余的填充部分
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :]
    # 如果原始词汇表大小小于填充后的大小，则扩展嵌入向量以适应填充后的大小
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
    # 如果原始词汇表大小等于填充后的大小，则直接使用原始词嵌入
    else:
        full_word_embed = word_embedding

    # 将嵌入向量按照目标张量模型并行大小进行分块
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    # 遍历目标张量模型并设置位置嵌入和词嵌入权重
    for i in range(args.target_tensor_model_parallel_size):
        # 获取模型状态字典中位置嵌入的路径并更新其权重为指定的位置嵌入
        pos_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.position_embeddings"
        )
        pos_emb_dict["weight"] = pos_embedding

        # 获取模型状态字典中词嵌入的路径并更新其权重为当前输出的词嵌入的克隆
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i].clone()

    # 转换器层处理
    print("converting transformer layers")

    # 检查注意力头数是否能被目标张量并行大小整除，否则引发数值错误
    if config.num_attention_heads % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )

    # 检查隐藏层数是否能被目标管道并行大小整除，否则引发数值错误
    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )

    # 计算每个管道并行块的转换器层数量
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    # 正则表达式，用于匹配和解析转换器层的名称
    layer_re = re.compile(r"transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Transformer模型中的注意力头数
    heads = config.n_head

    # 每个注意力头的隐藏大小
    hidden_size_per_head = config.n_embd // config.n_head
# 定义主函数入口
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 向参数解析器添加用于检查点的参数
    parser = add_checkpointing_args(parser)
    # 向参数解析器添加用于 Megatron 检查点的参数
    parser = add_megatron_checkpoint_args(parser)
    # 向参数解析器添加用于 Transformers 检查点的参数
    parser = add_transformers_checkpoint_args(parser)
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果命令行参数中包含转换 Megatron 到 Transformers 的选项
    if args.convert_checkpoint_from_megatron_to_transformers:
        # 执行 Megatron 到 Transformers 的检查点转换
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        # 否则执行 Transformers 到 Megatron 的检查点转换
        convert_checkpoint_from_transformers_to_megatron(args)

# 如果该脚本作为主程序运行，则执行 main() 函数
if __name__ == "__main__":
    main()
```