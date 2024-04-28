# `.\transformers\models\reformer\convert_reformer_trax_checkpoint_to_pytorch.py`

```
# 设定编码格式为 UTF-8
# 版权声明及许可证信息
# 导入所需库和模块
# 设置日志打印级别为信息级别

# 为单个层设置参数
def set_param(torch_layer, weight, bias=None):
    # 断言确保 torch_layer 的权重形状与给定的权重形状相匹配
    assert torch_layer.weight.shape == weight.shape, f"{torch_layer} 层的权重形状不匹配"
    # 将给定的权重设置为 torch 模型的参数
    torch_layer.weight = nn.Parameter(weight)
    # 如果存在偏置参数，同样设置偏置参数
    if bias is not None:
        # 断言确保 torch_layer 的偏置形状与给定的形状相匹配
        assert torch_layer.bias.shape == bias.shape, f"{torch_layer} 层的偏置形状不匹配"
        # 将给定的偏置设置为 torch 模型的参数
        torch_layer.bias = nn.Parameter(bias)

# 为 Torch LSH 模块设置权重参数
def set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    # 从权重中获取需要的 Numpy 数组
    np_query_key = np.asarray(weights[0])
    np_value = np.asarray(weights[1])
    np_dense = np.asarray(weights[2])

    # 设置 self-attention 测试用例的输入权重参数
    set_param(
        torch_layer.self_attention.query_key,
        torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.self_attention.value,
        torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.output.dense,
        torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
    )

# 为 Torch Local 模块设置权重参数
def set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    # 从权重中获取需要的 Numpy 数组
    np_query = np.asarray(weights[0])
    np_key = np.asarray(weights[1])
    np_value = np.asarray(weights[2])
    np_dense = np.asarray(weights[3])

    # 设置 self-attention 测试用例的输入权重参数
    set_param(
        torch_layer.self_attention.query,
        torch.tensor(np_query).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.self_attention.key,
        torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.self_attention.value,
        torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.output.dense,
        torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
    )

# 为 Torch Block 设置权重参数
def set_block_weights_in_torch(weights, torch_block, hidden_size):
    # 获取 layernorm 1 的权重参数
    layer_norm_1 = weights[0][0][0]
    layer_norm_1_weight = np.asarray(layer_norm_1[0])
    # lsh weights + output
    # 得到注意力权重
    attn_weights = weights[0][1]
    # 如果注意力权重长度小于4，则调用set_layer_weights_in_torch_lsh方法
    if len(attn_weights) < 4:
        set_layer_weights_in_torch_lsh(attn_weights, torch_block.attention, hidden_size)
    # 否则调用set_layer_weights_in_torch_local方法
    else:
        set_layer_weights_in_torch_local(attn_weights, torch_block.attention, hidden_size)
    
    # intermediate weighs
    # 获取中间参数
    intermediate_weights = weights[2][0][1][2]
    
    # Chunked Feed Forward
    # 如果中间参数长度为4，则获取其中的第3个
    if len(intermediate_weights) == 4:
        intermediate_weights = intermediate_weights[2]
    
    # layernorm 2
    # 获取第二个层归一化的权重和偏置
    layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
    layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
    # 设置第二个层归一化的参数
    set_param(
        torch_block.feed_forward.layer_norm,
        torch.tensor(layer_norm_2_weight),
        torch.tensor(layer_norm_2_bias),
    )
    
    # intermediate dense
    # 获取中间隐藏层的权重和偏置
    inter_dense_weight = np.asarray(intermediate_weights[1][0])
    inter_dense_bias = np.asarray(intermediate_weights[1][1])
    # 设置中间隐藏层的参数
    set_param(
        torch_block.feed_forward.dense.dense,
        torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(),
        torch.tensor(inter_dense_bias),
    )
    
    # intermediate out
    # 获取中间层输出的权重和偏置
    out_dense_weight = np.asarray(intermediate_weights[4][0])
    out_dense_bias = np.asarray(intermediate_weights[4][1])
    # 设置中间层输出的参数
    set_param(
        torch_block.feed_forward.output.dense,
        torch.tensor(out_dense_weight).transpose(0, 1).contiguous(),
        torch.tensor(out_dense_bias),
    )
def set_model_weights_in_torch(weights, torch_model, hidden_size):
    # 从给定的权重中设置 PyTorch 模型的权重
    # 获取 PyTorch Reformer 模型的 reformer 属性
    torch_model_reformer = torch_model.reformer

    # 从权重中获取词嵌入层的权重并设置到 PyTorch 模型中
    word_embeddings = np.asarray(weights[1])
    set_param(
        torch_model_reformer.embeddings.word_embeddings,
        torch.tensor(word_embeddings),
    )

    # 如果权重中包含位置编码的元组，则设置位置编码层的权重到 PyTorch 模型中
    if isinstance(weights[3], tuple):
        position_embeddings = torch_model_reformer.embeddings.position_embeddings
        for emb_idx in range(len(position_embeddings.weights)):
            emb_weights = np.asarray(weights[3][emb_idx][0])
            assert (
                position_embeddings.weights[emb_idx].shape == emb_weights.shape
            ), f"{position_embeddings[emb_idx]} emb does not match"
            position_embeddings.weights[emb_idx] = nn.Parameter(torch.tensor(emb_weights))

    # 获取每一层的权重并设置到 PyTorch 模型对应的层中
    trax_layer_weights = weights[5]
    assert len(torch_model_reformer.encoder.layers) * 4 == len(
        trax_layer_weights
    ), "HF and trax model do not have the same number of layers"
    for layer_idx, layer in enumerate(torch_model_reformer.encoder.layers):
        block_weights = trax_layer_weights[4 * layer_idx : 4 * (layer_idx + 1)]
        set_block_weights_in_torch(block_weights, layer, hidden_size)

    # 设置输出层的 LayerNorm 权重到 PyTorch 模型中
    layer_norm_out_weight = np.asarray(weights[7][0])
    layer_norm_out_bias = np.asarray(weights[7][1])
    set_param(
        torch_model_reformer.encoder.layer_norm,
        torch.tensor(layer_norm_out_weight),
        torch.tensor(layer_norm_out_bias),
    )

    # 设置输出嵌入层的权重到 PyTorch 模型输出头部
    output_embed_weights = np.asarray(weights[9][0])
    output_embed_bias = np.asarray(weights[9][1])
    set_param(
        torch_model.lm_head.decoder,
        torch.tensor(output_embed_weights).transpose(0, 1).contiguous(),
        torch.tensor(output_embed_bias),
    )


def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path):
    # 初始化 PyTorch 模型
    config = ReformerConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = ReformerModelWithLMHead(config)

    # 从 Trax 模型的 pkl 文件中加载权重
    with open(trax_model_pkl_path, "rb") as f:
        model_weights = pickle.load(f)["weights"]

    # 将 Trax 模型的权重转换为 PyTorch 模型的权重
    set_model_weights_in_torch(model_weights, model, config.hidden_size)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必要参数
    parser.add_argument(
        "--trax_model_pkl_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained Reformer model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加命令行参数，指定输出 PyTorch 模型的路径，必须提供且为字符串类型
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 Trax 模型转换为 PyTorch 模型，传入 Trax 模型路径、配置文件路径和 PyTorch 模型输出路径
    convert_trax_checkpoint_to_pytorch(args.trax_model_pkl_path, args.config_file, args.pytorch_dump_path)
```