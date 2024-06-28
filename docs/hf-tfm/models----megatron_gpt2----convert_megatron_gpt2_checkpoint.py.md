# `.\models\megatron_gpt2\convert_megatron_gpt2_checkpoint.py`

```
# 定义递归打印函数，用于打印参数名及其对应的值
def recursive_print(name, val, spaces=0):
    # 格式化消息，根据参数名生成相应格式的字符串，控制输出的对齐
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # 打印消息并递归处理（如果需要的话）
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        # 如果值是 torch.Tensor 类型，则打印参数名、值的尺寸
        print(msg, ":", val.size())
    else:
        # 否则，只打印参数名
        print(msg, ":", val)


# 对参数张量进行重新排序，以适应后续版本的 NVIDIA Megatron-LM
def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # 参数张量的输入形状
    input_shape = param.size()
    # 将布局排列为 [num_splits * num_heads * hidden_size, :]，以便与后续版本的 Megatron-LM 兼容
    # 在 Megatron-LM 内部，会执行反向操作来读取检查点
    # 如果 param 是 self-attention 块的权重张量，则返回的张量需要再次转置，以便 HuggingFace GPT2 读取
    # 参考 Megatron-LM 源码中的实现：https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # 如果检查点版本为1.0：
    if checkpoint_version == 1.0:
        # 版本1.0存储的形状是 [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        # 调整张量的形状为 saved_shape
        param = param.view(*saved_shape)
        # 转置操作：交换维度0和2
        param = param.transpose(0, 2)
        # 再次转置操作：交换维度1和2，并确保内存连续性
        param = param.transpose(1, 2).contiguous()
    
    # 如果检查点版本大于等于2.0：
    elif checkpoint_version >= 2.0:
        # 其他版本存储的形状是 [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        # 调整张量的形状为 saved_shape
        param = param.view(*saved_shape)
        # 转置操作：交换维度0和1，并确保内存连续性
        param = param.transpose(0, 1).contiguous()
    
    # 最后将张量的形状调整为 input_shape，并返回处理后的参数张量
    param = param.view(*input_shape)
    return param
####################################################################################################

# 定义函数用于将 Megatron-LM 模型检查点转换为适用于 Transformers 模型的格式
def convert_megatron_checkpoint(args, input_state_dict, config):
    # 初始化输出状态字典
    output_state_dict = {}

    # 旧版本可能未存储训练参数，检查并获取相关参数
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # 如果存在训练参数，则根据这些参数设置配置对象的相关属性
        config.vocab_size = ds_args.padded_vocab_size
        config.n_positions = ds_args.max_position_embeddings
        config.n_embd = ds_args.hidden_size
        config.n_layer = ds_args.num_layers
        config.n_head = ds_args.num_attention_heads
        config.n_inner = ds_args.ffn_hidden_size

    # 获取注意力头的数量
    heads = config.n_head
    # 计算每个注意力头的隐藏层大小
    hidden_size_per_head = config.n_embd // config.n_head
    # 获取 Megatron-LM 检查点版本信息
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # 获取模型对象
    model = input_state_dict["model"]
    # 获取语言模型
    lm = model["language_model"]
    # 获取嵌入层
    embeddings = lm["embedding"]

    # 获取词嵌入
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # 将词嵌入表截断到指定的词汇量大小
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # 获取位置嵌入
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    # 检查位置嵌入的长度与配置中的位置数是否匹配
    n_positions = pos_embeddings.size(0)
    if n_positions != config.n_positions:
        raise ValueError(
            f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
        )
    # 存储位置嵌入
    output_state_dict["transformer.wpe.weight"] = pos_embeddings

    # 获取变压器层对象，根据是否包含 "transformer" 键来决定
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # 编译用于提取层名称的正则表达式
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Megatron-LM 到 Transformers 的简单映射规则
    megatron_to_transformers = {
        "attention.dense": ".attn.c_proj.",
        "self_attention.dense": ".attn.c_proj.",
        "mlp.dense_h_to_4h": ".mlp.c_fc.",
        "mlp.dense_4h_to_h": ".mlp.c_proj.",
    }

    # 提取层信息的准备工作
    # 遍历transformer.items()中的键值对，其中key为层的名称，val为对应的值（通常是权重或偏置）。
    for key, val in transformer.items():
        # 使用正则表达式匹配层的名称。
        m = layer_re.match(key)

        # 如果匹配结果为None，说明这不是一个层，直接跳出循环。
        if m is None:
            break

        # 提取层的索引。
        layer_idx = int(m.group(1))
        # 提取操作的名称。
        op_name = m.group(2)
        # 判断是权重还是偏置。
        weight_or_bias = m.group(3)

        # 构造层的名称。
        layer_name = f"transformer.h.{layer_idx}"

        # 对于layernorm，直接存储layernorm的值。
        if op_name.endswith("layernorm"):
            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # 转置QKV矩阵。
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # 插入一个1x1xDxD的偏置张量。
            causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
                1, 1, n_positions, n_positions
            )
            output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # 插入一个"虚拟"张量作为masked_bias。
            masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            # 调整QKV矩阵的顺序。
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Megatron存储的是(3*D) x D，但transformers-GPT2需要的是D x (3*D)。
            out_val = out_val.transpose(0, 1).contiguous()
            # 存储。
            output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val

        # 转置偏置。
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # 存储。形状无变化。
            output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val

        # 转置权重。
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # 复制偏置。
        elif weight_or_bias == "bias":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

    # 调试断言，确保层数与config.n_layer相符。
    assert config.n_layer == layer_idx + 1

    # 最终的layernorm。
    output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # 对于LM头，transformers需要权重矩阵来加权嵌入。
    output_state_dict["lm_head.weight"] = word_embeddings

    # 完成任务！
    # 返回函数中的状态字典作为输出
    return output_state_dict
# 定义主函数，程序的入口点
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加布尔型参数 --print-checkpoint-structure，用于指定是否打印检查点结构
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    # 添加位置参数 path_to_checkpoint，表示检查点文件的路径（可以是 .zip 文件或直接的 .pt 文件）
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    # 添加可选参数 --config_file，表示可选的配置 JSON 文件，描述预训练模型
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 提取基本文件名
    basename = os.path.dirname(args.path_to_checkpoint)

    # 加载模型
    # 如果检查点路径以 .zip 结尾，则假设其为压缩文件
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        # 使用 zipfile 库打开 .zip 文件
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            # 打开压缩包中的指定文件 release/mp_rank_00/model_optim_rng.pt
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                # 使用 torch.load 加载 PyTorch 的状态字典
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        # 直接加载 .pt 文件
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    # 从输入状态字典中获取参数 args
    ds_args = input_state_dict.get("args", None)

    # 读取配置文件，或者默认使用 NVIDIA 发布的模型配置
    if args.config_file == "":
        if ds_args is not None:
            if ds_args.bias_gelu_fusion:
                activation_function = "gelu_fast"
            elif ds_args.openai_gelu:
                activation_function = "gelu_new"
            else:
                activation_function = "gelu"
        else:
            # 在早期版本中可能使用的激活函数
            activation_function = "gelu_new"

        # 明确指定所有参数，以防默认值发生更改
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function=activation_function,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
        )
    else:
        # 从 JSON 文件中加载配置
        config = GPT2Config.from_json_file(args.config_file)

    # 设置模型架构为 "GPT2LMHeadModel"
    config.architectures = ["GPT2LMHeadModel"]

    # 转换模型
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # 如果指定了 --print-checkpoint-structure 参数，则递归打印转换后状态字典的结构
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)
    # Add tokenizer class info to config
    # 将分词器类信息添加到配置中

    if ds_args is not None:
        # 如果数据集参数不为空，则获取分词器类型
        tokenizer_type = ds_args.tokenizer_type
        
        if tokenizer_type == "GPT2BPETokenizer":
            # 如果分词器类型为"GPT2BPETokenizer"，选择使用特定的模型
            tokenizer_model_name = "openai-community/gpt2"
        elif tokenizer_type == "PretrainedFromHF":
            # 如果分词器类型为"PretrainedFromHF"，使用数据集参数中指定的模型名称或路径
            tokenizer_model_name = ds_args.tokenizer_name_or_path
        else:
            # 如果分词器类型不被识别，则引发值错误异常
            raise ValueError(f"Unrecognized tokenizer_type {tokenizer_type}")
    else:
        # 如果数据集参数为空，默认使用"openai-community/gpt2"作为模型名称
        tokenizer_model_name = "openai-community/gpt2"

    # 根据模型名称加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    # 获取分词器的类名
    tokenizer_class = type(tokenizer).__name__
    # 将分词器类名存储到配置中
    config.tokenizer_class = tokenizer_class

    # 将配置保存到文件中
    print("Saving config")
    config.save_pretrained(basename)

    # 根据参数保存分词器
    print(f"Adding {tokenizer_class} tokenizer files")
    tokenizer.save_pretrained(basename)

    # 将状态字典保存到文件中
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)
# 如果当前脚本作为主程序运行（而不是被导入），则执行 main 函数
if __name__ == "__main__":
    # 调用主函数，程序的入口点
    main()
```