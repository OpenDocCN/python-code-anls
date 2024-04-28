# `.\transformers\models\switch_transformers\convert_switch_transformers_original_flax_checkpoint_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明信息
# 引入必要的模块和库
# 从 transformers 模块中引入 SwitchTransformersConfig 和 SwitchTransformersForConditionalGeneration 类
# 从 transformers.modeling_flax_pytorch_utils 模块中引入 load_flax_weights_in_pytorch_model 函数
# 从 transformers.utils 模块中引入 logging 函数
# 设置日志的输出级别为 info
# 定义一个映射字典 MOE_LAYER_NAME_MAPPING 用于存储模型中各层的名称映射关系
# 定义重命名键名的函数 rename_keys
# 创建一个空列表 keys 用于存储字典 s_dict 的键值
    # 遍历给定的键列表
    for key in keys:
        # 定义用于匹配层级到块级的正则表达式
        layer_to_block_of_layer = r".*/layers_(\d+)"
        # 将新键初始化为当前键
        new_key = key
        # 如果匹配到层级到块级的模式
        if re.match(layer_to_block_of_layer, key):
            # 使用正则表达式替换新键值
            new_key = re.sub(r"layers_(\d+)", r"block/\1/layer", new_key)

        # 重新定义层级到块级的正则表达式
        layer_to_block_of_layer = r"(encoder|decoder)/"

        # 如果匹配到层级到块级的模式
        if re.match(layer_to_block_of_layer, key):
            # 提取匹配组
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            # 如果是编码器
            if groups[0] == "encoder":
                # 根据规则修改新键值
                new_key = re.sub(r"/mlp/", r"/1/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/1/layer_norm/", new_key)
            # 如果是解码器
            elif groups[0] == "decoder":
                # 根据规则修改新键值
                new_key = re.sub(r"/mlp/", r"/2/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/2/layer_norm/", new_key)

        # 2. 转换其他经典的映射
        # 遍历MOE_LAYER_NAME_MAPPING中的键值对
        for old_key, temp_key in MOE_LAYER_NAME_MAPPING.items():
            # 如果旧键在新键中
            if old_key in new_key:
                # 替换键值
                new_key = new_key.replace(old_key, temp_key)

        # 输出修改前后的键值
        print(f"{key} -> {new_key}")
        # 将旧键值对应的值转移到新键值下
        s_dict[new_key] = s_dict.pop(key)

    # 处理编码器和解码器的相对注意力偏差权重
    if "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T

    if "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T

    # 3. 特别处理专家层
    for key in list(s_dict.keys()):
        # 如果键包含"expert"
        if "expert" in key:
            # 获取专家数量和专家权重
            num_experts = s_dict[key].shape[0]
            expert_weights = s_dict[key]
            # 遍历专家权重
            for idx in range(num_experts):
                # 将专家权重按照索引添加到新的键值下
                s_dict[key.replace("expert/", f"experts/expert_{idx}/")] = expert_weights[idx]
                print(f"{key} -> {key.replace('expert/', f'experts/expert_{idx}/')}")

            # 移除原有键值
            s_dict.pop(key)

    # 返回处理后的字典
    return s_dict
# GIN配置参数到SwitchTransformers配置参数的映射
GIN_TO_CONFIG_MAPPING = {
    "NUM_ENCODER_LAYERS": "num_layers",
    "NUM_DECODER_LAYERS": "num_decoder_layers",
    "NUM_HEADS": "num_heads",
    "HEAD_DIM": "d_kv",
    "EMBED_DIM": "d_model",
    "MLP_DIM": "d_ff",
    "NUM_SELECTED_EXPERTS": "num_selected_experts",
    "NUM_ENCODER_SPARSE_LAYERS": "num_sparse_encoder_layers",
    "NUM_DECODER_SPARSE_LAYERS": "num_sparse_decoder_layers",
    "dense.MlpBlock.activations": "feed_forward_proj",
}

# 将GIN配置文件转换为SwitchTransformers配置对象
def convert_gin_to_config(gin_file, num_experts):
    # 导入必要的模块
    import regex as re
    
    # 读取GIN配置文件内容
    with open(gin_file, "r") as f:
        raw_gin = f.read()
    
    # 使用正则表达式匹配配置参数和值
    regex_match = re.findall(r"(.*) = ([0-9.]*)", raw_gin)
    args = {}
    for param, value in regex_match:
        if param in GIN_TO_CONFIG_MAPPING and value != "":
            args[GIN_TO_CONFIG_MAPPING[param]] = float(value) if "." in value else int(value)
    
    # 匹配激活函数并设置到参数中
    activation = re.findall(r"(.*activations) = \(\'(.*)\',\)", raw_gin)[0]
    args[GIN_TO_CONFIG_MAPPING[activation[0]]] = str(activation[1])
    
    # 添加num_experts参数
    args["num_experts"] = num_experts
    # 使用参数创建SwitchTransformersConfig对象
    config = SwitchTransformersConfig(**args)
    return config

# 将Flax模型检查点转换为PyTorch模型的函数
def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, config_file, gin_file=None, pytorch_dump_path="./", num_experts=8):
    # 初始化PyTorch模型
    
    # 打印加载Flax权重的信息
    print(f"Loading flax weights from : {flax_checkpoint_path}")
    flax_params = checkpoints.load_t5x_checkpoint(flax_checkpoint_path)

    # 如果提供了GIN配置文件，则进行配置转换
    if gin_file is not None:
        config = convert_gin_to_config(gin_file, num_experts)
    else:
        config = SwitchTransformersConfig.from_pretrained(config_file)

    # 初始化SwitchTransformersForConditionalGeneration模型
    pt_model = SwitchTransformersForConditionalGeneration(config)

    # 调整Flax参数的格式
    flax_params = flax_params["target"]
    flax_params = flatten_dict(flax_params, sep="/")
    flax_params = rename_keys(flax_params)
    flax_params = unflatten_dict(flax_params, sep="/")

    # 加载Flax参数到PT模型中
    load_flax_weights_in_pytorch_model(pt_model, flax_params)

    # 保存PyTorch模型到指定路径
    print(f"Save PyTorch model to {pytorch_dump_path}")
    pt_model.save_pretrained(pytorch_dump_path)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained SwitchTransformers model. \nThis specifies the"
            " model architecture. If not provided, a `gin_file` has to be provided."
        ),
    )
    parser.add_argument(
        "--gin_file",
        default=None,
        type=str,
        required=False,
        help="Path to the gin config file. If not provided, a `config_file` has to be passed   ",
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=False, help="Config name of SwitchTransformers model."
    )
    # 导入参数解析器模块，用于处理命令行参数
    parser.add_argument(
        # 添加名为"--pytorch_dump_folder_path"的命令行参数，表示输出 PyTorch 模型的路径，参数类型为字符串，默认值为 None，必须提供
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output pytorch model."
    )
    # 添加名为"--num_experts"的命令行参数，表示专家的数量，参数类型为整数，默认值为8，非必须提供
    parser.add_argument("--num_experts", default=8, type=int, required=False, help="Number of experts")
    # 解析命令行参数，将参数值保存在args对象中
    args = parser.parse_args()
    # 调用convert_flax_checkpoint_to_pytorch函数，转换 Flax 检查点到 PyTorch 格式
    convert_flax_checkpoint_to_pytorch(
        # Flax 检查点文件路径
        args.switch_t5x_checkpoint_path,
        # 配置名称
        args.config_name,
        # GIN 文件路径
        args.gin_file,
        # PyTorch 模型输出文件夹路径
        args.pytorch_dump_folder_path,
        # 专家数量
        args.num_experts,
    )
```