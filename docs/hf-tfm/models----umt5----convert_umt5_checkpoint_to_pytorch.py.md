# `.\transformers\models\umt5\convert_umt5_checkpoint_to_pytorch.py`

```py
# 引入必要的模块和库
import argparse  # 用于解析命令行参数
import collections  # 提供额外的数据类型，如OrderedDict
import numpy as np  # 用于数值计算
import torch  # PyTorch深度学习库
from flax import traverse_util  # Flax库提供的遍历工具
from t5x import checkpoints  # t5x模块中的checkpoints类
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration  # transformers库中的MT5Config、UMT5EncoderModel、UMT5ForConditionalGeneration类
from transformers.utils import logging  # logging工具用于设置日志级别



# 设置全局日志级别信息为INFO
logging.set_verbosity_info()



# 定义函数 t5x_relpos_bias_lookup
# 传入参数 params、i 和 prefix
# 返回对应层的相对位置偏置参数(不进行转置)
def t5x_relpos_bias_lookup(params, i, prefix):
    return params[f"{prefix}/{prefix}/relpos_bias/rel_embedding"][:, i, :]



# 定义函数 t5x_attention_lookup
# 传入参数 params、i、prefix 和 layer_name，默认值为"attention"
# 返回 (self-)attention 的 KOQV 参数(不进行转置)
def t5x_attention_lookup(params, i, prefix, layer_name="attention"):
    # 将参数转变为连续内存的数组
    k_tmp = k_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/key/kernel"][:, i, :, :])
    k = k_tmp.reshape(k_tmp.shape[0], k_tmp.shape[1] * k_tmp.shape[2])
    o_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/out/kernel"][:, i, :, :])
    o = o_tmp.reshape(o_tmp.shape[0] * o_tmp.shape[1], o_tmp.shape[2])
    q_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/query/kernel"][:, i, :, :])
    q = q_tmp.reshape(q_tmp.shape[0], q_tmp.shape[1] * q_tmp.shape[2])
    v_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/value/kernel"][:, i, :, :])
    v = v_tmp.reshape(v_tmp.shape[0], v_tmp.shape[1] * v_tmp.shape[2])
    return k, o, q, v



# 定义函数 t5x_mlp_lookup
# 传入参数 params、i 和 prefix，以及 split_mlp_wi，默认值为False
# 返回对应层的MLP参数(不进行转置)
def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    # 如果 split_mlp_wi 为真，则返回两个MLP权重wi_0和wi_1
    if split_mlp_wi:
        wi_0 = params[f"{prefix}/{prefix}/mlp/wi_0/kernel"][:, i, :]
        wi_1 = params[f"{prefix}/{prefix}/mlp/wi_1/kernel"][:, i, :]
        wi = (wi_0, wi_1)
    # 如果满足了某种条件
    else:
        # 从参数字典中取出名为 {prefix}/{prefix}/mlp/wi/kernel 的张量
        # 其中 i 是某个索引值，取出对应的切片
        wi = params[f"{prefix}/{prefix}/mlp/wi/kernel"][:, i, :]
    
    # 从参数字典中取出名为 {prefix}/{prefix}/mlp/wo/kernel 的张量
    # 其中 i 是某个索引值，取出对应的切片
    wo = params[f"{prefix}/{prefix}/mlp/wo/kernel"][:, i, :]
    
    # 返回 wi 和 wo 两个张量
    return wi, wo
# 从 T5X-Flax 参数中获取指定层的层标准化参数
def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    """Returns the layer norm param of a layer."""
    # 返回指定层的标准化参数
    return params[f"{prefix}/{prefix}/{layer_name}/scale"][:, i]


# 将 T5X-Flax 模型参数转换为 Transformers-PyTorch 格式
def convert_t5x_to_pytorch(
    variables: dict, *, num_layers: int, is_encoder_only: bool, scalable_attention: bool = False
):
    """Converts the parameters from T5X-Flax to Transformers-PyTorch."""
    # 将原始参数字典扁平化处理
    old = traverse_util.flatten_dict(variables["target"])
    old = {"/".join(k): v for k, v in old.items()}

    # 检查是否存在分离的 MLP wi 参数
    split_mlp_wi = "encoder/encoder/mlp/wi_0/kernel" in old
    print("Split MLP:", split_mlp_wi)

    # 创建新的有序字典存储转换后的参数
    new = collections.OrderedDict()

    # 转换共享词嵌入参数
    new["shared.weight"] = old["token_embedder/embedding"]

    # 转换编码器参数
    for i in range(num_layers):
        # 第 i 层, 第 0 个子层（自注意力）
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_attention_layer_norm")
        k, o, q, v = t5x_attention_lookup(old, i, "encoder", "attention")
        new[f"encoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
        new[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

        # 第 i 层, 第 1 个子层（MLP）
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_mlp_layer_norm")
        wi, wo = t5x_mlp_lookup(old, i, "encoder", split_mlp_wi)
        new[f"encoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
        if split_mlp_wi:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = wi[0].T
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = wi[1].T
        else:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"] = wi.T
        new[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = wo.T
        if scalable_attention:
            # 如果需要可伸缩的注意力，转换相对位置编码
            new[f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"] = t5x_relpos_bias_lookup(
                old, i, "encoder"
            ).T

    # 转换编码器最终的标准化参数
    new["encoder.final_layer_norm.weight"] = old["encoder/encoder_norm/scale"]

    if not scalable_attention:
        # 如果不需要可伸缩的注意力，只转换第一层的相对位置编码
        new["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = t5x_relpos_bias_lookup(
            old, 0, "encoder"
        ).T
        new["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = t5x_relpos_bias_lookup(
            old, 0, "decoder"
        ).T
    if not is_encoder_only:
        # 如果不是仅编码器，则执行以下代码（即解码器相关操作）。

        # 解码器。
        for i in range(num_layers):
            # 循环遍历每个层。

            # Block i, layer 0 (Self Attention).
            # 第 i 个块，第 0 层（自注意力）。

            # 获取层归一化参数
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_self_attention_layer_norm")
            # 获取自注意力的参数
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "self_attention")
            # 更新新模型中的权重
            new[f"decoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
            new[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

            # Block i, layer 1 (Cross Attention).
            # 第 i 个块，第 1 层（跨注意力）。

            # 获取层归一化参数
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_cross_attention_layer_norm")
            # 获取跨注意力的参数
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "encoder_decoder_attention")
            # 更新新模型中的权重
            new[f"decoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
            new[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = v.T

            # Block i, layer 2 (MLP).
            # 第 i 个块，第 2 层（多层感知机）。

            # 获取层归一化参数
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_mlp_layer_norm")
            # 获取多层感知机的参数
            wi, wo = t5x_mlp_lookup(old, i, "decoder", split_mlp_wi)
            # 更新新模型中的权重
            new[f"decoder.block.{i}.layer.2.layer_norm.weight"] = layer_norm
            if split_mlp_wi:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = wi[0].T
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = wi[1].T
            else:
                new[f"encoder.block.{i}.layer.2.DenseReluDense.wi.weight"] = wi.T
            new[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = wo.T

            if scalable_attention:
                # 如果可扩展注意力，则执行以下代码。

                # 转换每层的相对嵌入
                new[
                    f"decoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
                ] = t5x_relpos_bias_lookup(old, i, "decoder").T

        # 更新最终层归一化参数
        new["decoder.final_layer_norm.weight"] = old["decoder/decoder_norm/scale"]

        # 语言模型头部（仅在 v1.1 检查点中，v1.0 中使用嵌入）
        if "decoder/logits_dense/kernel" in old:
            # 如果存在 logits_dense/kernel 参数，则执行以下代码。

            # 更新语言模型头部的权重
            new["lm_head.weight"] = old["decoder/logits_dense/kernel"].T

    # 返回新的模型参数
    return new
# 准备用于 PyTorch 模型的状态字典
def make_state_dict(converted_params, is_encoder_only: bool):
    # 使用 torch 张量创建有序字典的状态字典
    state_dict = collections.OrderedDict([(k, torch.from_numpy(v.copy())) for (k, v) in converted_params.items()])

    # 添加缺失的部分
    if "encoder.embed_tokens.weight" not in state_dict:
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

    if not is_encoder_only:
        if "decoder.embed_tokens.weight" not in state_dict:
            state_dict["decoder.embed_tokens.weight"] = state_dict["shared.weight"]

        # 对于旧的 1.0 模型
        if "lm_head.weight" not in state_dict:
            print("Using shared word embeddings as lm_head.")
            state_dict["lm_head.weight"] = state_dict["shared.weight"]

    return state_dict


# 加载 T5X 转换后的参数到 T5 模型中
def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only, scalable_attention):
    # 使用检查点载入 T5X 转换后的参数
    variables = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    converted = convert_t5x_to_pytorch(
        variables, num_layers=config.num_layers, is_encoder_only=is_encoder_only, scalable_attention=scalable_attention
    )
    state_dict = make_state_dict(converted, is_encoder_only)
    model.load_state_dict(state_dict, strict=True)


def convert_t5x_checkpoint_to_pytorch(
    t5x_checkpoint_path,
    config_file,
    pytorch_dump_path,
    is_encoder_only: bool = False,
    scalable_attention: bool = False,
):
    # 加载配置和模型，转换 T5X 检查点，并保存 PyTorch 检查点
    config = MT5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 对于所有情况，都可以使用 T5Model，旧的 v1.0 检查点将仅有一个 LM 头作为单词嵌入
    if is_encoder_only:
        model = UMT5EncoderModel(config)
    else:
        model = UMT5ForConditionalGeneration(config)

    # 从 tf 检查点加载权重
    load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only, scalable_attention)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 验证是否可以加载检查点
    model.from_pretrained(pytorch_dump_path)
    print("Done")

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a native T5X checkpoint into a PyTorch checkpoint.")
    # 必要参数
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path to the T5X checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained T5 model.\nThis specifies the model architecture.",
    )
    # 添加命令行参数 --pytorch_dump_path，表示输出 PyTorch 模型的路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加命令行参数 --is_encoder_only，表示检查模型是否为编码器-解码器模型
    parser.add_argument(
        "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
    )
    # 添加命令行参数 --scalable_attention，表示模型是否使用可扩展注意力 (umt5 模型)
    parser.add_argument(
        "--scalable_attention",
        action="store_true",
        help="Whether the model uses scaled attention (umt5 model)",
        default=False,
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数convert_t5x_checkpoint_to_pytorch，并传入相应的参数
    convert_t5x_checkpoint_to_pytorch(
        args.t5x_checkpoint_path,
        args.config_file,
        args.pytorch_dump_path,
        args.is_encoder_only,
        args.scalable_attention,
    )
```