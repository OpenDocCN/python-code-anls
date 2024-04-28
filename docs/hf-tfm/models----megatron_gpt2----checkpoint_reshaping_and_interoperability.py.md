# `.\transformers\models\megatron_gpt2\checkpoint_reshaping_and_interoperability.py`

```py
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入所需的库
import argparse
import json
import os
import re
import sys
import types
import torch

# 从 transformers 库中导入 AutoTokenizer 和 GPT2Config 类
from transformers import AutoTokenizer, GPT2Config
# 从 transformers.modeling_utils 模块中导入 WEIGHTS_INDEX_NAME, WEIGHTS_NAME 和 shard_checkpoint 函数

def add_checkpointing_args(parser):
    # 添加用于检查点的参数
    parser.add_argument("--megatron-path", type=str, default=None, help="Megatron 仓库的基本目录")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "如果为 True，则将 Megatron 检查点转换为 Transformers 检查点。"
            "如果为 False，则将 Transformers 检查点转换为 Megatron 检查点。"
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="要转换的检查点的路径。",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="转换后的检查点的路径。",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser

def add_megatron_checkpoint_args(parser):
    # 添加用于 Megatron 检查点的参数
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "转换后检查点的张量模型并行大小。"
            "仅在将 Transformers 检查点转换为 Megatron 检查点时使用。"
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "转换后检查点的管道模型并行大小。"
            "仅在将 Transformers 检查点转换为 Megatron 检查点时使用。"
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "转换后检查点的数据并��大小。"
            "仅在将 Transformers 检查点转换为 Megatron 检查点时使用。"
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "转换后检查点的数据类型。"
            "仅在将 Transformers 检查点转换为 Megatron 检查点时使用。"
        ),
    )
    # 添加一个命令行参数，用于指定词汇表大小的填充值
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 添加一个命令行参数，用于指定是否使用分布式优化器
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    # 返回解析器对象
    return parser
# 为解析器添加转换器检查点参数
def add_transformers_checkpoint_args(parser):
    # 添加参数：tokenizer_name，类型为字符串，默认为None，用于保存预训练分词器的名称
    # 仅在将Megatron检查点转换为Transformers检查点时使用
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    # 添加参数：max_shard_size，类型为字符串，默认为"10GB"，用于指定检查点在分片之前的最大大小
    # 仅在将Megatron检查点转换为Transformers检查点时使用
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# "automated"规则的简单名称映射
megatron_to_transformers = {
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
}
# 反向映射
transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

# 张量并行参数列表
tensor_parallel_params = [
    # 要合并的megatron-lm层
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
    # 要在tp ranks之间分割的transformers层
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
]


def recursive_print(name, val, spaces=0):
    """
    递归打印检查点的结构。此函数取自`convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): 当前张量参数的名称
        val (Tuple(int)): 当前张量参数的形状
        spaces (int): 输出嵌套结构之前要打印的空格数
    """
    # 格式化消息
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # 打印并递归（如果需要）
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
    将param张量的布局排列为[num_splits * num_heads * hidden_size, :]，以便与后续版本兼容
    def permute_param_for_gpt2(param, checkpoint_version, num_splits, num_heads, hidden_size):
        """
        Rearranges the weight tensor to be compatible with HuggingFace GPT2 from NVIDIA Megatron-LM.
        The inverse operation is performed inside Megatron-LM to read checkpoints:
        https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
        self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
        This function is taken from `convert_megatron_gpt2_checkpoint.py`
    
        Args:
            param (torch.Tensor): the tensor to permute
            checkpoint_version (int): the version of the checkpoint.
            num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
            num_heads (int): the number of attention heads
            hidden_size (int): the hidden size per head
        """
    
        # Get the shape of the input tensor
        input_shape = param.size()
        
        # Check if the checkpoint version is 1.0
        if checkpoint_version == 1.0:
            # Version 1.0 stores [num_heads * hidden_size * num_splits, :]
            saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
            # Reshape the parameter tensor according to the saved shape
            param = param.view(*saved_shape)
            # Transpose the dimensions to match GPT2's format
            param = param.transpose(0, 2)
            param = param.transpose(1, 2).contiguous()
        # If the checkpoint version is greater than or equal to 2.0
        elif checkpoint_version >= 2.0:
            # Other versions store [num_heads * num_splits * hidden_size, :]
            saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
            # Reshape the parameter tensor according to the saved shape
            param = param.view(*saved_shape)
            param = param.transpose(0, 1).contiguous()
        
        # Reshape the parameter tensor back to its original shape
        param = param.view(*input_shape)
        return param
# 转换 Transformers 的参数张量布局，使其与相应的 NVIDIA Megatron-LM 校准点版本兼容。输入为 [num_splits * num_heads * hidden_size, :]，输出为 [num_heads * hidden_size * num_splits, :]（版本1.0）和 [num_heads * num_splits * hidden_size, :]（版本2.0及之后）。
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

    # 获取输入形状
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # 版本1.0存储 [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # 其他版本存储 [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    # 恢复原始形状
    param = param.view(*input_shape)
    return param


# 合并 Transformers 的分片状态成为单个校准点
def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


# 根据给定的张量并行尺寸、管道并行尺寸和管道并行秩从 NVIDIA Megatron-LM 检查点获取分片的状态
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
    tp_state_dicts = []
    # 遍历 tp_size 次数，生成子目录名称
    for i in range(tp_size):
        # 根据 pp_size 的值，生成子目录名称
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        # 遍历两个检查点文件名
        for checkpoint_name in ["model_optim_rng.pt", "model_rng.pt"]:
            # 生成检查点文件的路径
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            # 如果检查点文件存在，则跳出循环
            if os.path.isfile(checkpoint_path):
                break
        # 使用 torch.load() 加载检查点文件中的状态字典，指定在 CPU 上加载
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # 将加载的状态字典添加到列表中
        tp_state_dicts.append(state_dict)
    # 返回加载的状态字典列表
    return tp_state_dicts
# 通过路径从字典中获取元素。如果元素不存在，递归添加空字典。
def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    # 将路径按"."分割成列表
    path = path.split(".")
    # 遍历路径中的键，若键不在字典中则添加空字典
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    # 返回获取到的元素
    return d


# 将 NVIDIA Megatron-LM 检查点转换为 HuggingFace Transformers 检查点
def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # 从状态字典中加载 Megatron-LM 检查点参数
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    # 从 rank0_checkpoint_path 加载状态字典
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # 从 Megatron-LM 参数创建 Transformers GPT2 配置
    if megatron_args is not None:
        if megatron_args.bias_gelu_fusion:
            activation_function = "gelu_fast"
        elif megatron_args.openai_gelu:
            activation_function = "gelu_new"
        else:
            activation_function = "gelu"
    else:
        # 在早期阶段，这个函数用于 "gelu_new"
        activation_function = "gelu_new"
    # 如果没有原始词汇大小，则使用填充后的词汇大小
    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print(vocab_size)
    # 根据给定的参数配置创建 GPT2 模型的配置对象
    config = GPT2Config(
        vocab_size=vocab_size,  # 词汇表的大小
        n_positions=megatron_args.max_position_embeddings,  # 最大位置编码数
        n_embd=megatron_args.hidden_size,  # 隐藏层大小
        n_layer=megatron_args.num_layers,  # 层数
        n_head=megatron_args.num_attention_heads,  # 头数
        n_inner=megatron_args.ffn_hidden_size,  # 内部层大小
        activation_function=activation_function,  # 激活函数
        resid_pdrop=0.1,  # dropout 比例
        embd_pdrop=0.1,  # dropout 比例
        attn_pdrop=0.1,  # dropout 比例
        layer_norm_epsilon=1e-5,  # LN 的 epsilon 值
        initializer_range=0.02,  # 初始化范围
        summary_type="cls_index",  # 摘要类型
        summary_use_proj=True,  # 是否使用投影
        summary_activation=None,  # 摘要激活函数
        summary_proj_to_labels=True,  # 摘要是否应用到标签
        summary_first_dropout=0.1,  # 摘要的首次 dropout 比例
        scale_attn_weights=True,  # 是否缩放注意力权重
        use_cache=True,  # 是否使用缓存
        bos_token_id=vocab_size - 1,  # 句子开始符号的 ID
        eos_token_id=vocab_size - 1,  # 句子结束符号的 ID
        architectures=["GPT2LMHeadModel"],  # 支持的架构类型
    )

    # 初始化输出状态词典
    output_state_dict = {}

    # 获取 state_dict 的版本号
    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    # 获取 tensor_model_parallel_size
    tp_size = megatron_args.tensor_model_parallel_size
    # 获取 pipeline_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    # 设置 dtype 为 torch.float32
    dtype = torch.float32
    # 预编译的正则表达式用于提取层名
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # 开始转换
    print("Converting")

    # 转换嵌入层
    print("Converting embeddings")
    # 获取 tensors 的字典列表
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # 转换并保存位置嵌入
    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )
    output_state_dict["transformer.wpe.weight"] = position_embeddings.to(dtype)

    # 转换并保存单词嵌入
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

    # 转换 Transformer 层
    print("Converting transformer layers")
    # 头数
    heads = config.n_head
    # 每个头的隐藏层大小
    hidden_size_per_head = config.n_embd // config.n_head
    # 位置编码数
    n_positions = config.n_positions
    # 每个 pipeline 阶段的 Transformer 层数
    num_layers = config.num_hidden_layers // pp_size

    if config.n_layer != (layer_idx + 1):
        # 验证期望的层数与实际层数是否一致
        raise ValueError(f"Expected {config.n_layer} layers but found {layer_idx + 1}")

    # 转换最终的 LayerNorm 层
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["transformer.ln_f.weight"] = params["final_layernorm.weight"].to(dtype)
    output_state_dict["transformer.ln_f.bias"] = params["final_layernorm.bias"].to(dtype)

    # 为语言模型头转换权重矩阵
    print("Converting LM head")
    # 将LM头中的权重转换为指定数据类型
    output_state_dict["lm_head.weight"] = word_embeddings.to(dtype)

    # 输出提示信息，表示从Megatron-LM转换到Transformers已完成
    print("Conversion from Megatron-LM to Transformers is done!")

    # 打印转换后状态字典的结构
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # 将分词器类信息添加到配置文件中
    # 参考 https://github.com/huggingface/transformers/issues/13906
    if args.tokenizer_name is None:
        tokenizer_name = "gpt2"
    else:
        tokenizer_name = args.tokenizer_name

    # 利用给定的分词器名实例化自动分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # 获取分词器类名
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # 保存配置到文件
    print("Saving config")
    config.save_pretrained(args.save_path)

    # 根据参数保存分词器
    if args.tokenizer_name is not None:
        print(f"Adding {tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # 将状态字典存储为文件
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # 保存模型
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # 保存索引文件
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
    创建目录 args.save_path（如果不存在的话）
    os.makedirs(args.save_path, exist_ok=True)

    # 在上面的目录中搜索
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    如果 args.megatron_path 不为空，则将 megatron 路径添加到 sys.path 中
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    尝试导入 Megatron 包中的 _vocab_size_with_padding
    try:
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        打印错误信息并退出
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    # 加载 transformers 模型状态字典和配置
    通过检查目录 args.load_path 中以 'pytorch_model' 开头的子目录来获取所有子目录名
    sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
    如果只有一个子目录，则使用"pytorch_model.bin"作为checkpoint名加载模型状态字典
    if len(sub_dirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    否则，合并多个分片的 transformers 状态字典
    else:
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    从 args.load_path 中加载 GPT2 模型配置
    config = GPT2Config.from_pretrained(args.load_path)

    # 保存追踪文件
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        写入内容为"release"
        f.write("release")

    # 在 args.load_path 中创建 "release" 目录
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # Megatron 参数
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

    根据激活函数设置 Megatron 参数中的相关值
    如果激活函数为 "gelu"，则设置相关参数值
    elif config.activation_function == "gelu":
        megatron_args["bias_gelu_fusion"] = False
        megatron_args["openai_gelu"] = False
    如果激活函数为 "gelu_fast"，则设置相关参数值
    elif config.activation_function == "gelu_fast":
        megatron_args["bias_gelu_fusion"] = True
        megatron_args["openai_gelu"] = False
    如果激活函数为 "gelu_new"，则设置相关参数值
    elif config.activation_function == "gelu_new":
        megatron_args["bias_gelu_fusion"] = False
        megatron_args["openai_gelu"] = True

    使用 types.SimpleNamespace 创建一个新的命名空间 margs，并将 Megatron 参数添加为命名空间属性
    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        设置命名空间属性
        setattr(margs, k, v)
    # 根据参数类型设置dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)  # 设置参数类型为dtype

    # 保存虚拟优化器状态字典
    dummy_optim_state_dict = {}  # 创建空的虚拟优化器状态字典
    dummy_optim_state_dict["optimizer"] = {  # 初始化优化器参数
        "step": 0,  # 步数
        "param_groups": [  # 参数组
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],  # 参数为空
            }
        ],
    }
    if args.use_distributed_optimizer:  # 如果使用分布式优化器
        for i in range(args.target_pipeline_model_parallel_size):  # 根据模型并行大小遍历
            for j in range(args.target_tensor_model_parallel_size):  # 根据张量并行大小遍历
                for k in range(args.target_data_parallel_size):  # 根据数据并行大小遍历
                    if args.target_pipeline_model_parallel_size == 1:  # 如果模型并行大小为1
                        checkpoint_dir = f"mp_rank_{j:02d}_{k:03d}"  # 设置检查点目录名称
                    else:
                        checkpoint_dir = f"mp_rank_{j:02d}_{i:03d}_{k:03d}"  # 设置检查点目录名称
                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)  # 添加到发布目录下
                    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建检查点目录
                    torch.save(  # 保存虚拟优化器状态字典到文件
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )

    # 转换
    print("Converting")  # 打印"转换"
    output_state_dict = []  # 创建空的输出状态字典列表
    for i in range(args.target_tensor_model_parallel_size):  # 根据张量并行大小遍历
        output_state_dict.append({})  # 添加空字典到输出状态字典列表中

    # 嵌入层
    print("converting embedding layer")  # 打印"转换嵌入层"
    pos_embedding = state_dict["transformer.wpe.weight"].to(dtype)  # 获取位置嵌入并设置类型为dtype
    word_embedding = state_dict["transformer.wte.weight"].to(dtype)  # 获取词嵌入并设置类型为dtype
    orig_vocab_size = config.vocab_size  # 获取原始词汇大小
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)  # 获取填充后的词汇大小
    setattr(margs, "padded_vocab_size", padded_vocab_size)  # 设置填充后的词汇大小到参数中
    # 剪切不需要的额外填充
    if orig_vocab_size > padded_vocab_size:  # 如果原始词汇大小大于填充后的词汇大小
        full_word_embed = word_embedding[0:padded_vocab_size, :]  # 获取词嵌入并截取到填充后的大小
    # 通过复制最后一个条目扩展嵌入到更大的大小
    elif orig_vocab_size < padded_vocab_size:  # 如果原始词汇大小小于填充后的词汇大小
        padding_size = padded_vocab_size - orig_vocab_size  # 计算填充大小
        full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))  # 扩展嵌入到更大的大小
    # 大小相同！
    else:  # 如果大小相同
        full_word_embed = word_embedding  # 设置词嵌入为嵌入
    # 拆分成新的张量模型并行大小
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)  # 根据张量并行大小拆分词嵌入
    # 对目标张量模型并行大小进行循环迭代
    for i in range(args.target_tensor_model_parallel_size):
        # 从输出状态字典中获取位置嵌入字典
        pos_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.position_embeddings"
        )
        # 将位置嵌入字典的权重设置为给定的位置嵌入
        pos_emb_dict["weight"] = pos_embedding

        # 从输出状态字典中获取词嵌入字典
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        # 将词嵌入字典的权重设置为输出的词嵌入的克隆
        word_emb_dict["weight"] = out_word_embed[i].clone()

    # 转换 Transformer 层
    print("converting transformer layers")
    # 检查注意力头数是否能被张量模型并行数整除
    if config.num_attention_heads % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )

    # 检查隐藏层数是否能被流水线模型并行数整除
    if config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of pipeline parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )

    # 计算每个流水线模型包含的层数
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    # 正则表达式，用于匹配 Transformer 层的命名模式
    layer_re = re.compile(r"transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # 注意力头数
    heads = config.n_head
    # 每个注意力头的隐藏大小
    hidden_size_per_head = config.n_embd // config.n_head
# 定义主函数
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加检查点参数到解析器
    parser = add_checkpointing_args(parser)
    # 添加 Megatron 检查点参数到解析器
    parser = add_megatron_checkpoint_args(parser)
    # 添加 Transformers 检查点参数到解析器
    parser = add_transformers_checkpoint_args(parser)
    # 解析命令行参数
    args = parser.parse_args()
    # 如果需要从 Megatron 转换检查点到 Transformers
    if args.convert_checkpoint_from_megatron_to_transformers:
        # 执行此转换操作
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        # 否则执行从 Transformers 转换检查点到 Megatron
        convert_checkpoint_from_transformers_to_megatron(args)

# 如果当前脚本是作为主程序执行
if __name__ == "__main__":
    # 调用主函数
    main()
```