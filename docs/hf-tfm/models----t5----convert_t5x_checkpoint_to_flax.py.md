# `.\models\t5\convert_t5x_checkpoint_to_flax.py`

```
# 设置脚本的编码格式为 UTF-8
# 版权声明，声明脚本归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件，除非符合许可证要求，否则不得使用该文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”提供的，不提供任何明示或暗示的担保或条件
# 请查阅许可证以获取详细的权利和限制条款
#
"""Convert T5X checkpoints from the original repository to JAX/FLAX model."""

# 导入必要的库和模块
import argparse

# 从 t5x 模块中导入 checkpoints
from t5x import checkpoints

# 从 transformers 库中导入 FlaxT5ForConditionalGeneration 类和 T5Config 类
from transformers import FlaxT5ForConditionalGeneration, T5Config


# 定义函数，将 T5X 检查点转换为 Flax 模型
def convert_t5x_checkpoint_to_flax(t5x_checkpoint_path, config_name, flax_dump_folder_path):
    # 使用指定的配置名称创建 T5Config 对象
    config = T5Config.from_pretrained(config_name)
    # 使用配置对象创建 FlaxT5ForConditionalGeneration 模型
    flax_model = FlaxT5ForConditionalGeneration(config=config)
    # 加载给定路径上的 T5X 检查点，返回 T5X 模型对象
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)

    # 检查是否在 t5x_model 的目标部分的编码器层中存在名为 "wi_0" 的子项
    split_mlp_wi = "wi_0" in t5x_model["target"]["encoder"]["layers_0"]["mlp"]

    # Encoder 部分的转换操作未提供，留待后续补充
    # 遍历配置中指定数量的层
    for layer_index in range(config.num_layers):
        # 构建当前层的名称，格式为 "layers_<层索引>"
        layer_name = f"layers_{str(layer_index)}"

        # 获取当前层的自注意力机制相关参数
        t5x_attention_key = t5x_model["target"]["encoder"][layer_name]["attention"]["key"]["kernel"]
        t5x_attention_out = t5x_model["target"]["encoder"][layer_name]["attention"]["out"]["kernel"]
        t5x_attention_query = t5x_model["target"]["encoder"][layer_name]["attention"]["query"]["kernel"]
        t5x_attention_value = t5x_model["target"]["encoder"][layer_name]["attention"]["value"]["kernel"]

        # 获取当前层的自注意力机制前的归一化参数
        t5x_attention_layer_norm = t5x_model["target"]["encoder"][layer_name]["pre_attention_layer_norm"]["scale"]

        # 根据条件选择当前层的多层感知机的参数
        if split_mlp_wi:
            t5x_mlp_wi_0 = t5x_model["target"]["encoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            t5x_mlp_wi_1 = t5x_model["target"]["encoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            t5x_mlp_wi = t5x_model["target"]["encoder"][layer_name]["mlp"]["wi"]["kernel"]

        # 获取当前层的多层感知机的输出参数
        t5x_mlp_wo = t5x_model["target"]["encoder"][layer_name]["mlp"]["wo"]["kernel"]

        # 获取当前层多层感知机前的归一化参数
        t5x_mlp_layer_norm = t5x_model["target"]["encoder"][layer_name]["pre_mlp_layer_norm"]["scale"]

        # 将 T5X 模型中的参数赋值给 Flax 模型的对应位置
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"][
            "kernel"
        ] = t5x_attention_key
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"][
            "kernel"
        ] = t5x_attention_out
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"][
            "kernel"
        ] = t5x_attention_query
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"][
            "kernel"
        ] = t5x_attention_value

        # 设置 Flax 模型当前层的注意力机制前的归一化参数
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"][
            "weight"
        ] = t5x_attention_layer_norm

        # 根据条件选择并设置 Flax 模型当前层的多层感知机的参数
        if split_mlp_wi:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = t5x_mlp_wi_0
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = t5x_mlp_wi_1
        else:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi"][
                "kernel"
            ] = t5x_mlp_wi

        # 设置 Flax 模型当前层多层感知机的输出参数
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wo"][
            "kernel"
        ] = t5x_mlp_wo

        # 设置 Flax 模型当前层多层感知机前的归一化参数
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"][
            "weight"
        ] = t5x_mlp_layer_norm

    # 仅适用于第一层（layer 0）的操作：获取 T5X 模型的编码器相对位置偏置的嵌入矩阵的转置
    t5x_encoder_rel_embedding = t5x_model["target"]["encoder"]["relpos_bias"]["rel_embedding"].T
    # 将 t5x_encoder_rel_embedding 赋值给 flax_model.params 中的特定路径
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = t5x_encoder_rel_embedding

    # 将 t5x_model 中的特定路径的值赋给 flax_model.params 中的特定路径
    t5x_encoder_norm = t5x_model["target"]["encoder"]["encoder_norm"]["scale"]
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = t5x_encoder_norm

    # 将 t5x_model 中的特定路径的值赋给 flax_model.params 中的特定路径
    # 这是针对解码器的最终归一化层
    tx5_decoder_norm = t5x_model["target"]["decoder"]["decoder_norm"]["scale"]
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = tx5_decoder_norm

    # 将 t5x_model 中的特定路径的值赋给 flax_model.params 中的特定路径
    # 仅对解码器的第一个层的第一个自注意力模块使用相对注意力偏置
    t5x_decoder_rel_embedding = t5x_model["target"]["decoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = t5x_decoder_rel_embedding

    # 将 t5x_model 中的特定路径的值赋给 flax_model.params 中的特定路径
    # 这是共享的令牌嵌入
    tx5_token_embeddings = t5x_model["target"]["token_embedder"]["embedding"]
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # 如果在 t5x_model 的特定路径中存在 "logits_dense"，则将其值赋给 flax_model.params 中的特定路径
    # 这是语言模型头部的内核，仅适用于版本 v1.1 的检查点
    if "logits_dense" in t5x_model["target"]["decoder"]:
        flax_model.params["lm_head"]["kernel"] = t5x_model["target"]["decoder"]["logits_dense"]["kernel"]

    # 将转换后的模型保存到指定路径
    flax_model.save_pretrained(flax_dump_folder_path)
    # 打印成功转换的消息
    print("T5X Model was sucessfully converted!")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    # 添加必需的命令行参数
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path the TX5 checkpoint."
    )
    # t5x_checkpoint_path 参数，指定了 TX5 模型的检查点路径

    parser.add_argument("--config_name", default=None, type=str, required=True, help="Config name of T5 model.")
    # config_name 参数，指定了 T5 模型的配置名称

    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    # flax_dump_folder_path 参数，指定了输出 FLAX 模型的文件夹路径

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_t5x_checkpoint_to_flax，将命令行参数传递给函数
    convert_t5x_checkpoint_to_flax(args.t5x_checkpoint_path, args.config_name, args.flax_dump_folder_path)
```