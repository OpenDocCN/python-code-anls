# `.\transformers\models\t5\convert_t5x_checkpoint_to_pytorch.py`

```
# 设定文件编码格式为UTF-8
# 版权声明
# 2022年由Google LLC和HuggingFace Inc.团队所有
# 根据 Apache 许可，您可以使用此文件。您不能使用此文件，除非符合许可。您可以在以下网址获得许可副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或以书面形式同意，否则根据许可分发的软件以 "现状" 基础分发。
# 没有任何种类的明示或暗示的保证或条件。有关权限和限制的特定语言，请参阅许可
"""
将 T5X checkpoint 转换为 PyTorch

步骤：
- 根据https://cloud.google.com/storage/docs/gsutil_install中的说明安装 gsutil
- 在https://github.com/google-research/t5x/blob/main/docs/models.md#t5-11-checkpoints 获取 T5X checkpoint。例如：
    `gsutil -m cp -r gs://t5-data/pretrained_models/t5x/t5_1_1_small $HOME/`
- 创建或下载与下载的模型对应的配置。例如，对于 T5 v1.1 small，您可以使用
    https://huggingface.co/google/t5-v1_1-small/blob/main/config.json
- 转换：
    ```
    python3 convert_t5x_checkpoint_to_pytorch.py --t5x_checkpoint_path=$HOME/t5_1_1_small --config_file=config.json\
      --pytorch_dump_path=$HOME/t5_1_1_small_pt
    ```
"""

# 引入所需模块
import argparse
import collections
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration
from transformers.utils import logging

# 设置日志记录的详细程度为 info
logging.set_verbosity_info()

# 定义函数 t5x_attention_lookup
def t5x_attention_lookup(params, i, prefix, layer_name="attention"):
    """Returns the KOQV parameters of (self-)attention. Does not transpose."""
    k = params[f"{prefix}/layers_{i}/{layer_name}/key/kernel"]
    o = params[f"{prefix}/layers_{i}/{layer_name}/out/kernel"]
    q = params[f"{prefix}/layers_{i}/{layer_name}/query/kernel"]
    v = params[f"{prefix}/layers_{i}/{layer_name}/value/kernel"]
    return k, o, q, v

# 定义函数 t5x_mlp_lookup
def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    """Returns the MLP parameters of a layer. Does not transpose."""
    if split_mlp_wi:
        wi_0 = params[f"{prefix}/layers_{i}/mlp/wi_0/kernel"]
        wi_1 = params[f"{prefix}/layers_{i}/mlp/wi_1/kernel"]
        wi = (wi_0, wi_1)
    else:
        wi = params[f"{prefix}/layers_{i}/mlp/wi/kernel"]

    wo = params[f"{prefix}/layers_{i}/mlp/wo/kernel"]
    return wi, wo

# 定义函数 t5x_layer_norm_lookup
def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    """Returns the layer norm param of a layer."""
    return params[f"{prefix}/layers_{i}/{layer_name}/scale"]

# 定义函数 convert_t5x_to_pytorch
def convert_t5x_to_pytorch(variables: dict, *, num_layers: int, num_decoder_layers: int, is_encoder_only: bool):
    """Converts the parameters from T5X-Flax to Transformers-PyTorch."""
    old = traverse_util.flatten_dict(variables["target"])
    old = {"/".join(k): v for k, v in old.items()}

    # v1.1 models have a gated GeLU with wi_0 and wi_1 instead of wi
    # 检查 "encoder/layers_0/mlp/wi_0/kernel" 是否在旧字典中
    split_mlp_wi = "encoder/layers_0/mlp/wi_0/kernel" in old
    # 输出 Split MLP 的结果
    print("Split MLP:", split_mlp_wi)
    
    # 创建一个有序字典用于存储新的映射关系
    new = collections.OrderedDict()
    
    # 将共享的嵌入映射到新字典中
    new["shared.weight"] = old["token_embedder/embedding"]
    
    # 处理编码器
    for i in range(num_layers):
        # 处理第 i 层块
        # 第 0 层 (自注意力)
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_attention_layer_norm")
        k, o, q, v = t5x_attention_lookup(old, i, "encoder", "attention")
        # 添加新的权重映射
        new[f"encoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
        new[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T
    
        # 第 1 层 (MLP)
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_mlp_layer_norm")
        wi, wo = t5x_mlp_lookup(old, i, "encoder", split_mlp_wi)
        new[f"encoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
        # 根据 split_mlp_wi 是否为真来添加相应的权重映射
        if split_mlp_wi:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = wi[0].T
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = wi[1].T
        else:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"] = wi.T
        new[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = wo.T
    
    # 添加额外权重映射
    new["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = old["encoder/relpos_bias/rel_embedding"].T
    new["encoder.final_layer_norm.weight"] = old["encoder/encoder_norm/scale"]
    if not is_encoder_only:
        # 如果不是仅编码器，则执行以下步骤

        # 解码器。
        for i in range(num_decoder_layers):
            # 针对第 i 个块，第 0 层（自注意力）进行操作。
            
            # 获取旧模型的 layer_norm 参数，并存储到新模型中
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_self_attention_layer_norm")
            new[f"decoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm

            # 获取旧模型的自注意力（k、o、q、v）参数，并进行转置后存储到新模型中
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "self_attention")
            new[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

            # 针对第 i 个块，第 1 层（交叉注意力）进行操作。

            # 获取旧模型的 layer_norm 参数，并存储到新模型中
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_cross_attention_layer_norm")
            new[f"decoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm

            # 获取旧模型的交叉注意力（k、o、q、v）参数，并进行转置后存储到新模型中
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "encoder_decoder_attention")
            new[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = v.T

            # 针对第 i 个块，第 2 层（多层感知机）进行操作。

            # 获取旧模型的 layer_norm 参数，并存储到新模型中
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_mlp_layer_norm")
            new[f"decoder.block.{i}.layer.2.layer_norm.weight"] = layer_norm

            # 获取旧模型的多层感知机（wi、wo）参数，并进行转置后存储到新模型中
            wi, wo = t5x_mlp_lookup(old, i, "decoder", split_mlp_wi)
            if split_mlp_wi:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = wi[0].T
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = wi[1].T
            else:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi.weight"] = wi.T
            new[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = wo.T

        # 存储旧模型的 decoder_norm/scale 参数到新模型中
        new["decoder.final_layer_norm.weight"] = old["decoder/decoder_norm/scale"]

        # 存储旧模型的 relpos_bias/rel_embedding 参数，并进行转置后存储到新模型中
        new["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = old[
            "decoder/relpos_bias/rel_embedding"
        ].T

        # 语言模型头部（仅在 v1.1 的检查点中存在，在 v1.0 中使用嵌入）
        if "decoder/logits_dense/kernel" in old:
            # 存储旧模型的 logits_dense/kernel 参数，并进行转置后存储到新模型中
            new["lm_head.weight"] = old["decoder/logits_dense/kernel"].T

    # 返回新模型参数
    return new
# 导入所需的库
import collections
import argparse
import torch
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration, checkpoints

# 定义一个函数，根据给定的字典转换为包含 Torch 张量的有序字典
def make_state_dict(converted_params, is_encoder_only: bool):
    # 使用 Torch 提供的 from_numpy 函数将给定字典中的值转换为对应的张量，并创建一个有序字典
    state_dict = collections.OrderedDict([(k, torch.from_numpy(v.copy())) for (k, v) in converted_params.items()])

    # 检查并补充缺失的键值对
    # 如果 state_dict 中不存在 encoder.embed_tokens.weight 键，则添加一个键值对，键为 encoder.embed_tokens.weight，值为 state_dict["shared.weight"]
    if "encoder.embed_tokens.weight" not in state_dict:
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

    # 如果 is_encoder_only 为 False，则执行以下操作
    if not is_encoder_only:
        # 如果 state_dict 中不存在 decoder.embed_tokens.weight 键，则添加一个键值对，键为 decoder.embed_tokens.weight，值为 state_dict["shared.weight"]
        if "decoder.embed_tokens.weight" not in state_dict:
            state_dict["decoder.embed_tokens.weight"] = state_dict["shared.weight"]

        # 如果 state_dict 中不存在 lm_head.weight 键，则添加一个键值对，键为 lm_head.weight，值为 state_dict["shared.weight"]
        if "lm_head.weight" not in state_dict:  # For old 1.0 models.
            print("Using shared word embeddings as lm_head.")
            state_dict["lm_head.weight"] = state_dict["shared.weight"]

    # 返回最终的状态字典
    return state_dict


# 定义一个函数，加载 T5X 转换后的参数，并用它们替换模型中的参数
def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only):
    # 加载 T5X 参数
    variables = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    # 将 T5X 参数转换为 PyTorch 参数
    converted = convert_t5x_to_pytorch(
        variables,
        num_layers=config.num_layers,
        num_decoder_layers=config.num_decoder_layers,
        is_encoder_only=is_encoder_only,
    )
    # 创建模型的状态字典
    state_dict = make_state_dict(converted, is_encoder_only)
    # 使用创建的状态字典替换模型的参数
    model.load_state_dict(state_dict, strict=True)


# 定义一个函数，加载 T5X 检查点，并将其转换为 PyTorch 检查点
def convert_t5x_checkpoint_to_pytorch(
    t5x_checkpoint_path, config_file, pytorch_dump_path, is_encoder_only: bool = False
):
    # 加载配置文件，创建模型
    config = T5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 根据给定的参数初始化模型
    if is_encoder_only:
        model = T5EncoderModel(config)
    else:
        model = T5ForConditionalGeneration(config)

    # 从 T5X 检查点加载权重
    load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 验证检查点是否可加载
    model.from_pretrained(pytorch_dump_path)
    print("Done")


# 程序入口
if __name__ == "__main__":
    # 创建一个命令行参数解析器对象
    parser = argparse.ArgumentParser(description="Converts a native T5X checkpoint into a PyTorch checkpoint.")
    # 添加必需的参数
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
    # 添加命令行参数，用于指定输出的PyTorch模型的路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加命令行参数，用于标记模型是否只包含编码器
    parser.add_argument(
        "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 将T5X检查点文件转换为PyTorch模型
    convert_t5x_checkpoint_to_pytorch(
        args.t5x_checkpoint_path, args.config_file, args.pytorch_dump_path, args.is_encoder_only
    )
```