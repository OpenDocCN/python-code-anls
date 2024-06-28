# `.\models\umt5\convert_umt5_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数的模块
import collections  # 提供额外的数据结构，如Counter，OrderedDict等

import numpy as np  # 处理数值计算的库
import torch  # PyTorch深度学习框架
from flax import traverse_util  # Flax库，用于遍历和操作参数

from t5x import checkpoints  # 导入T5X库中的checkpoints模块

from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration  # 导入transformers中的模型配置和模型类
from transformers.utils import logging  # 导入transformers中的日志功能模块


logging.set_verbosity_info()  # 设置日志的详细级别为INFO


def t5x_relpos_bias_lookup(params, i, prefix):
    """返回一个层的相对位置偏置参数。不进行转置。"""
    return params[f"{prefix}/{prefix}/relpos_bias/rel_embedding"][:, i, :]


def t5x_attention_lookup(params, i, prefix, layer_name="attention"):
    """返回（自）注意力的KOQV参数。不进行转置。"""
    k_tmp = k_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/key/kernel"][:, i, :, :])
    k = k_tmp.reshape(k_tmp.shape[0], k_tmp.shape[1] * k_tmp.shape[2])
    o_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/out/kernel"][:, i, :, :])
    o = o_tmp.reshape(o_tmp.shape[0] * o_tmp.shape[1], o_tmp.shape[2])
    q_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/query/kernel"][:, i, :, :])
    q = q_tmp.reshape(q_tmp.shape[0], q_tmp.shape[1] * q_tmp.shape[2])
    v_tmp = np.ascontiguousarray(params[f"{prefix}/{prefix}/{layer_name}/value/kernel"][:, i, :, :])
    v = v_tmp.reshape(v_tmp.shape[0], v_tmp.shape[1] * v_tmp.shape[2])
    return k, o, q, v


def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    """返回一个层的MLP参数。不进行转置。"""
    if split_mlp_wi:
        wi_0 = params[f"{prefix}/{prefix}/mlp/wi_0/kernel"][:, i, :]
        wi_1 = params[f"{prefix}/{prefix}/mlp/wi_1/kernel"][:, i, :]
        wi = (wi_0, wi_1)
    # 如果条件不满足，则取出指定路径下的参数，并选取特定索引对应的数据
    else:
        wi = params[f"{prefix}/{prefix}/mlp/wi/kernel"][:, i, :]

    # 取出指定路径下的参数，并选取特定索引对应的数据
    wo = params[f"{prefix}/{prefix}/mlp/wo/kernel"][:, i, :]
    # 返回获取的参数数据 wi 和 wo
    return wi, wo
# 返回指定层的层归一化参数
def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    """Returns the layer norm param of a layer."""
    return params[f"{prefix}/{prefix}/{layer_name}/scale"][:, i]

# 将 T5X-Flax 模型参数转换为 Transformers-PyTorch 的格式
def convert_t5x_to_pytorch(
    variables: dict, *, num_layers: int, is_encoder_only: bool, scalable_attention: bool = False
):
    """Converts the parameters from T5X-Flax to Transformers-PyTorch."""
    # 将目标参数展开为扁平结构
    old = traverse_util.flatten_dict(variables["target"])
    # 使用斜杠连接键，构建新的字典
    old = {"/".join(k): v for k, v in old.items()}

    # v1.1 模型中的 MLP 使用 wi_0 和 wi_1 替代 wi
    split_mlp_wi = "encoder/encoder/mlp/wi_0/kernel" in old
    # 打印是否分割了 MLP
    print("Split MLP:", split_mlp_wi)

    # 新的参数字典，使用有序字典保持顺序
    new = collections.OrderedDict()

    # 共享的嵌入层权重
    new["shared.weight"] = old["token_embedder/embedding"]

    # 编码器部分
    for i in range(num_layers):
        # Block i, layer 0 (Self Attention).
        # 获取层归一化参数
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_attention_layer_norm")
        new[f"encoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
        # 获取注意力机制中的参数
        k, o, q, v = t5x_attention_lookup(old, i, "encoder", "attention")
        new[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

        # Block i, layer 1 (MLP).
        # 获取层归一化参数
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_mlp_layer_norm")
        new[f"encoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
        # 获取 MLP 的参数
        wi, wo = t5x_mlp_lookup(old, i, "encoder", split_mlp_wi)
        if split_mlp_wi:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = wi[0].T
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = wi[1].T
        else:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"] = wi.T
        new[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = wo.T
        # 如果启用可扩展注意力机制，转换每层的相对位置编码
        if scalable_attention:
            new[f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"] = t5x_relpos_bias_lookup(
                old, i, "encoder"
            ).T

    # 最终编码器层的归一化参数
    new["encoder.final_layer_norm.weight"] = old["encoder/encoder_norm/scale"]

    # 如果不使用可扩展注意力机制，转换第一个编码器和解码器块的相对注意力偏置
    if not scalable_attention:
        new["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = t5x_relpos_bias_lookup(
            old, 0, "encoder"
        ).T
        new["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = t5x_relpos_bias_lookup(
            old, 0, "decoder"
        ).T
    # 如果不是仅编码器模式，则执行解码器相关操作
    if not is_encoder_only:
        # 解码器部分的循环，遍历每个层次
        for i in range(num_layers):
            # 第 i 块，第 0 层 (自注意力层)
            # 获取预自注意力层规范化权重
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_self_attention_layer_norm")
            # 获取自注意力层的 k, o, q, v 权重
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "self_attention")
            # 更新新模型的权重：层规范化权重、k、o、q、v
            new[f"decoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
            new[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

            # 第 i 块，第 1 层 (跨注意力层)
            # 获取预交叉注意力层规范化权重
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_cross_attention_layer_norm")
            # 获取跨注意力层的 k, o, q, v 权重
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "encoder_decoder_attention")
            # 更新新模型的权重：层规范化权重、k、o、q、v
            new[f"decoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
            new[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = v.T

            # 第 i 块，第 2 层 (MLP 层)
            # 获取预MLP层规范化权重
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_mlp_layer_norm")
            # 获取MLP层的权重 wi 和 wo
            wi, wo = t5x_mlp_lookup(old, i, "decoder", split_mlp_wi)
            # 更新新模型的权重：层规范化权重、wi、wo
            new[f"decoder.block.{i}.layer.2.layer_norm.weight"] = layer_norm
            if split_mlp_wi:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = wi[0].T
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = wi[1].T
            else:
                new[f"encoder.block.{i}.layer.2.DenseReluDense.wi.weight"] = wi.T
            new[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = wo.T

            # 如果可扩展注意力为真，更新相对注意力偏置权重
            if scalable_attention:
                new[
                    f"decoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
                ] = t5x_relpos_bias_lookup(old, i, "decoder").T

        # 更新最终解码器层规范化权重
        new["decoder.final_layer_norm.weight"] = old["decoder/decoder_norm/scale"]

        # 语言模型头部 (仅在 v1.1 版本的检查点中存在，在 v1.0 版本中使用嵌入)
        if "decoder/logits_dense/kernel" in old:
            # 更新新模型的语言模型头部权重
            new["lm_head.weight"] = old["decoder/logits_dense/kernel"].T

    # 返回更新后的新模型参数
    return new
# 准备一个 PyTorch 模型的状态字典
def make_state_dict(converted_params, is_encoder_only: bool):
    # 使用 torch 张量创建一个有序字典状态字典
    state_dict = collections.OrderedDict([(k, torch.from_numpy(v.copy())) for (k, v) in converted_params.items()])

    # 添加缺失的部分
    if "encoder.embed_tokens.weight" not in state_dict:
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

    if not is_encoder_only:
        if "decoder.embed_tokens.weight" not in state_dict:
            state_dict["decoder.embed_tokens.weight"] = state_dict["shared.weight"]

        if "lm_head.weight" not in state_dict:  # 对于旧的 1.0 版本的模型
            print("Using shared word embeddings as lm_head.")
            state_dict["lm_head.weight"] = state_dict["shared.weight"]

    return state_dict


# 用 T5X 转换的参数替换模型的参数
def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only, scalable_attention):
    variables = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    converted = convert_t5x_to_pytorch(
        variables, num_layers=config.num_layers, is_encoder_only=is_encoder_only, scalable_attention=scalable_attention
    )
    state_dict = make_state_dict(converted, is_encoder_only)
    model.load_state_dict(state_dict, strict=True)


# 将 T5X 检查点转换为 PyTorch 检查点
def convert_t5x_checkpoint_to_pytorch(
    t5x_checkpoint_path,
    config_file,
    pytorch_dump_path,
    is_encoder_only: bool = False,
    scalable_attention: bool = False,
):
    # 加载配置和模型，转换 T5X 检查点，保存 PyTorch 检查点
    config = MT5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 非 v1.1 检查点也可以使用 T5Model，但这对所有版本都有效
    # V1.0 检查点将简单地有一个LM头部，即词嵌入
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a native T5X checkpoint into a PyTorch checkpoint.")
    # 必填参数
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
    # 添加一个参数：用于指定输出的 PyTorch 模型的路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个标志参数：检查模型是否只有编码器部分（encoder-decoder 模型）
    parser.add_argument(
        "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
    )
    # 添加一个标志参数：指示模型是否使用了缩放的注意力机制（例如 umt5 模型）
    parser.add_argument(
        "--scalable_attention",
        action="store_true",
        help="Whether the model uses scaled attention (umt5 model)",
        default=False,
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 T5X 模型的检查点转换为 PyTorch 模型
    convert_t5x_checkpoint_to_pytorch(
        args.t5x_checkpoint_path,
        args.config_file,
        args.pytorch_dump_path,
        args.is_encoder_only,
        args.scalable_attention,
    )
```