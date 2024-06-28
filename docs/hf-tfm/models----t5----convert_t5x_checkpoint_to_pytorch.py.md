# `.\models\t5\convert_t5x_checkpoint_to_pytorch.py`

```
# 指定文件编码为 UTF-8
# 版权声明，版权归谷歌有限责任公司和HuggingFace公司所有
#
# 根据Apache许可证2.0版进行许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何明示或暗示的担保或条件。
# 有关许可证的详细信息，请参阅许可证。
"""
将T5X检查点转换为PyTorch格式

步骤：
- 根据https://cloud.google.com/storage/docs/gsutil_install 安装gsutil
- 在https://github.com/google-research/t5x/blob/main/docs/models.md#t5-11-checkpoints获取T5X检查点 示例：
    `gsutil -m cp -r gs://t5-data/pretrained_models/t5x/t5_1_1_small $HOME/`
- 创建或下载相应模型的配置。例如，对于T5 v1.1 small，您可以使用
    https://huggingface.co/google/t5-v1_1-small/blob/main/config.json
- 转换:
    ```
    python3 convert_t5x_checkpoint_to_pytorch.py --t5x_checkpoint_path=$HOME/t5_1_1_small --config_file=config.json\
      --pytorch_dump_path=$HOME/t5_1_1_small_pt
    ```
"""

import argparse  # 导入命令行参数解析模块
import collections  # 导入collections模块

import torch  # 导入PyTorch库
from flax import traverse_util  # 导入flax库的traverse_util模块
from t5x import checkpoints  # 从t5x库导入checkpoints模块

from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration  # 从transformers库导入必要的类
from transformers.utils import logging  # 从transformers库导入logging模块用于日志记录


logging.set_verbosity_info()  # 设置日志级别为信息级别


def t5x_attention_lookup(params, i, prefix, layer_name="attention"):
    """返回(self-)attention的KOQV参数，不进行转置。"""
    k = params[f"{prefix}/layers_{i}/{layer_name}/key/kernel"]
    o = params[f"{prefix}/layers_{i}/{layer_name}/out/kernel"]
    q = params[f"{prefix}/layers_{i}/{layer_name}/query/kernel"]
    v = params[f"{prefix}/layers_{i}/{layer_name}/value/kernel"]
    return k, o, q, v


def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    """返回层的MLP参数，不进行转置。"""
    if split_mlp_wi:
        wi_0 = params[f"{prefix}/layers_{i}/mlp/wi_0/kernel"]
        wi_1 = params[f"{prefix}/layers_{i}/mlp/wi_1/kernel"]
        wi = (wi_0, wi_1)
    else:
        wi = params[f"{prefix}/layers_{i}/mlp/wi/kernel"]

    wo = params[f"{prefix}/layers_{i}/mlp/wo/kernel"]
    return wi, wo


def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    """返回层的层归一化参数。"""
    return params[f"{prefix}/layers_{i}/{layer_name}/scale"]


def convert_t5x_to_pytorch(variables: dict, *, num_layers: int, num_decoder_layers: int, is_encoder_only: bool):
    """将T5X-Flax的参数转换为Transformers-PyTorch格式。"""
    old = traverse_util.flatten_dict(variables["target"])
    old = {"/".join(k): v for k, v in old.items()}

    # v1.1模型具有具有wi_0和wi_1而不是wi的门控GeLU
    # 检查旧模型中是否存在指定路径，判断是否要分离 MLP 的权重
    split_mlp_wi = "encoder/layers_0/mlp/wi_0/kernel" in old
    # 打印是否分离 MLP 的信息
    print("Split MLP:", split_mlp_wi)

    # 创建一个新的有序字典用于存储转换后的模型参数
    new = collections.OrderedDict()

    # 共享的嵌入层权重
    new["shared.weight"] = old["token_embedder/embedding"]

    # 编码器部分的参数转换
    for i in range(num_layers):
        # 第 i 个块，第 0 层（自注意力层）
        # 获取自注意力层前的层归一化权重
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_attention_layer_norm")
        # 获取自注意力层中的注意力权重（k, o, q, v）
        k, o, q, v = t5x_attention_lookup(old, i, "encoder", "attention")
        new[f"encoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
        new[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

        # 第 i 个块，第 1 层（MLP 层）
        # 获取 MLP 层前的层归一化权重
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_mlp_layer_norm")
        # 获取 MLP 层中的权重（wi, wo）
        wi, wo = t5x_mlp_lookup(old, i, "encoder", split_mlp_wi)
        new[f"encoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
        if split_mlp_wi:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = wi[0].T
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = wi[1].T
        else:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"] = wi.T
        new[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = wo.T

    # 编码器的第一个块的自注意力层的相对注意力偏置权重
    new["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = old[
        "encoder/relpos_bias/rel_embedding"
    ].T
    # 编码器最终层的归一化权重
    new["encoder.final_layer_norm.weight"] = old["encoder/encoder_norm/scale"]
    if not is_encoder_only:
        # 如果不是仅编码器模式，则执行解码器部分

        # 解码器部分
        for i in range(num_decoder_layers):
            # 对于每个解码器层 i：

            # Block i, layer 0 (Self Attention).
            # 第 i 块，第 0 层 (自注意力)
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_self_attention_layer_norm")
            # 获取旧模型中解码器第 i 块的预自注意力层规范化参数
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "self_attention")
            # 获取旧模型中解码器第 i 块的自注意力参数 k, o, q, v
            new[f"decoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
            # 设置新模型中解码器第 i 块的第 0 层的层规范化权重
            new[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
            # 设置新模型中解码器第 i 块的第 0 层自注意力的 k 权重
            new[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
            # 设置新模型中解码器第 i 块的第 0 层自注意力的 o 权重
            new[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
            # 设置新模型中解码器第 i 块的第 0 层自注意力的 q 权重
            new[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T
            # 设置新模型中解码器第 i 块的第 0 层自注意力的 v 权重

            # Block i, layer 1 (Cross Attention).
            # 第 i 块，第 1 层 (交叉注意力)
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_cross_attention_layer_norm")
            # 获取旧模型中解码器第 i 块的预交叉注意力层规范化参数
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "encoder_decoder_attention")
            # 获取旧模型中解码器第 i 块的编码器-解码器注意力参数 k, o, q, v
            new[f"decoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
            # 设置新模型中解码器第 i 块的第 1 层的层规范化权重
            new[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = k.T
            # 设置新模型中解码器第 i 块的第 1 层交叉注意力的 k 权重
            new[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = o.T
            # 设置新模型中解码器第 i 块的第 1 层交叉注意力的 o 权重
            new[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = q.T
            # 设置新模型中解码器第 i 块的第 1 层交叉注意力的 q 权重
            new[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = v.T
            # 设置新模型中解码器第 i 块的第 1 层交叉注意力的 v 权重

            # Block i, layer 2 (MLP).
            # 第 i 块，第 2 层 (多层感知机)
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_mlp_layer_norm")
            # 获取旧模型中解码器第 i 块的预多层感知机层规范化参数
            wi, wo = t5x_mlp_lookup(old, i, "decoder", split_mlp_wi)
            # 获取旧模型中解码器第 i 块的多层感知机参数 wi, wo
            new[f"decoder.block.{i}.layer.2.layer_norm.weight"] = layer_norm
            # 设置新模型中解码器第 i 块的第 2 层的层规范化权重
            if split_mlp_wi:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = wi[0].T
                # 设置新模型中解码器第 i 块的第 2 层多层感知机的 wi_0 权重
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = wi[1].T
                # 设置新模型中解码器第 i 块的第 2 层多层感知机的 wi_1 权重
            else:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi.weight"] = wi.T
                # 设置新模型中解码器第 i 块的第 2 层多层感知机的 wi 权重
            new[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = wo.T
            # 设置新模型中解码器第 i 块的第 2 层多层感知机的 wo 权重

        # 解码器最终层规范化权重
        new["decoder.final_layer_norm.weight"] = old["decoder/decoder_norm/scale"]
        # 设置新模型中解码器的最终层规范化权重
        new["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = old[
            "decoder/relpos_bias/rel_embedding"
        ].T
        # 设置新模型中解码器第 0 块第 0 层自注意力的相对注意力偏置权重

        # LM Head (only in v1.1 checkpoints, in v1.0 embeddings are used instead)
        # 语言模型头部（仅在 v1.1 版本的检查点中，v1.0 版本使用嵌入代替）
        if "decoder/logits_dense/kernel" in old:
            # 如果旧模型中存在 "decoder/logits_dense/kernel"
            new["lm_head.weight"] = old["decoder/logits_dense/kernel"].T
            # 设置新模型中的 lm_head 权重为旧模型中 logits_dense/kernel 的转置

    # 返回新模型
    return new
def make_state_dict(converted_params, is_encoder_only: bool):
    """Prepares a state dict for the PyTorch model."""
    # 创建一个有序字典的状态字典，使用 torch.from_numpy 将每个参数的副本转换为张量
    state_dict = collections.OrderedDict([(k, torch.from_numpy(v.copy())) for (k, v) in converted_params.items()])

    # 如果缺少 "encoder.embed_tokens.weight"，则用 "shared.weight" 补充
    if "encoder.embed_tokens.weight" not in state_dict:
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

    # 如果不仅仅是编码器，还需处理解码器和语言模型头部的参数
    if not is_encoder_only:
        # 如果缺少 "decoder.embed_tokens.weight"，则用 "shared.weight" 补充
        if "decoder.embed_tokens.weight" not in state_dict:
            state_dict["decoder.embed_tokens.weight"] = state_dict["shared.weight"]

        # 对于旧版本 1.0 的模型，如果缺少 "lm_head.weight"，则打印警告并用 "shared.weight" 补充
        if "lm_head.weight" not in state_dict:  # For old 1.0 models.
            print("Using shared word embeddings as lm_head.")
            state_dict["lm_head.weight"] = state_dict["shared.weight"]

    return state_dict


def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only):
    """Replaces the params in model with the T5X converted params."""
    # 加载 T5X checkpoint 中的变量并进行转换为 PyTorch 格式
    variables = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    converted = convert_t5x_to_pytorch(
        variables,
        num_layers=config.num_layers,
        num_decoder_layers=config.num_decoder_layers,
        is_encoder_only=is_encoder_only,
    )
    # 生成状态字典并加载到模型中
    state_dict = make_state_dict(converted, is_encoder_only)
    model.load_state_dict(state_dict, strict=True)


def convert_t5x_checkpoint_to_pytorch(
    t5x_checkpoint_path, config_file, pytorch_dump_path, is_encoder_only: bool = False
):
    """Loads the config and model, converts the T5X checkpoint, and saves a PyTorch checkpoint."""
    # 从配置文件加载配置并初始化 PyTorch 模型
    config = T5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    
    # 根据是否仅为编码器，选择初始化 T5EncoderModel 或 T5ForConditionalGeneration 模型
    if is_encoder_only:
        model = T5EncoderModel(config)
    else:
        model = T5ForConditionalGeneration(config)

    # 从 TensorFlow checkpoint 加载权重到 PyTorch 模型
    load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only)

    # 保存转换后的 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 验证是否成功加载保存的检查点
    model.from_pretrained(pytorch_dump_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a native T5X checkpoint into a PyTorch checkpoint.")
    # 必需参数
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
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
    )
    args = parser.parse_args()
    convert_t5x_checkpoint_to_pytorch(
        args.t5x_checkpoint_path, args.config_file, args.pytorch_dump_path, args.is_encoder_only
    )



# 添加命令行参数 `--pytorch_dump_path`，指定输出的 PyTorch 模型路径，该参数是必须的
parser.add_argument(
    "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
)
# 添加命令行参数 `--is_encoder_only`，表示是否为编码器-解码器模型的标志，该参数为布尔类型，默认为 False
parser.add_argument(
    "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
)
# 解析命令行参数，并将结果保存在 args 变量中
args = parser.parse_args()
# 调用函数 convert_t5x_checkpoint_to_pytorch，将给定的 T5X 模型转换为 PyTorch 模型
convert_t5x_checkpoint_to_pytorch(
    args.t5x_checkpoint_path, args.config_file, args.pytorch_dump_path, args.is_encoder_only
)


这段代码片段用于解析命令行参数，并调用一个函数来执行 T5X 模型到 PyTorch 模型的转换。
```