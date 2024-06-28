# `.\models\longt5\convert_longt5x_checkpoint_to_flax.py`

```py
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块

from t5x import checkpoints  # 导入从原始T5X模型检查点加载模块

from transformers import AutoConfig, FlaxAutoModelForSeq2SeqLM  # 导入自动配置模块和FLAX的序列到序列模型


def convert_t5x_checkpoint_to_flax(t5x_checkpoint_path, config_name, flax_dump_folder_path):
    # 使用给定的配置名创建自动配置对象
    config = AutoConfig.from_pretrained(config_name)
    # 根据配置创建FLAX的序列到序列模型
    flax_model = FlaxAutoModelForSeq2SeqLM.from_config(config=config)
    # 加载T5X模型检查点
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)

    # 检查是否需要分离MLP的权重
    split_mlp_wi = "wi_0" in t5x_model["target"]["encoder"]["layers_0"]["mlp"]

    # 根据配置类型确定编码器注意力机制的名称
    if config.model_type == "t5":
        encoder_attn_name = "SelfAttention"
    if config.model_type == "longt5" and config.encoder_attention_type == "local":
        encoder_attn_name = "LocalSelfAttention"
    elif config.model_type == "longt5" and config.encoder_attention_type == "transient-global":
        encoder_attn_name = "TransientGlobalSelfAttention"
    else:
        # 如果配置不匹配预期的类型和注意力机制，引发错误
        raise ValueError(
            "Given config is expected to have `model_type='t5'`, or `model_type='longt5` with `encoder_attention_type`"
            " attribute with a value from ['local', 'transient-global]."
        )

    # 编码器部分
    # 仅针对第0层处理：
    # 从T5X模型中提取编码器相对位置嵌入，并将其赋值给FLAX模型的相应部分
    t5x_encoder_rel_embedding = t5x_model["target"]["encoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"][encoder_attn_name]["relative_attention_bias"][
        "embedding"
    ] = t5x_encoder_rel_embedding

    # 当模型类型为longt5且编码器注意力机制为transient-global时，处理全局相对位置偏差和层归一化
    if config.model_type == "longt5" and config.encoder_attention_type == "transient-global":
        t5x_encoder_global_rel_embedding = t5x_model["target"]["encoder"]["side_relpos_bias"]["rel_embedding"].T
        flax_model.params["encoder"]["block"]["0"]["layer"]["0"][encoder_attn_name]["global_relative_attention_bias"][
            "embedding"
        ] = t5x_encoder_global_rel_embedding

    # 赋值编码器的最终层归一化参数
    t5x_encoder_norm = t5x_model["target"]["encoder"]["encoder_norm"]["scale"]
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = t5x_encoder_norm

    # 解码器部分
    # 赋值解码器的最终层归一化参数
    tx5_decoder_norm = t5x_model["target"]["decoder"]["decoder_norm"]["scale"]
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = tx5_decoder_norm
    # 只适用于层级 0：

    # 从 T5X 模型中获取目标部分解码器的相对位置偏置的嵌入矩阵，并进行转置
    t5x_decoder_rel_embedding = t5x_model["target"]["decoder"]["relpos_bias"]["rel_embedding"].T
    
    # 将转置后的相对注意力偏置嵌入矩阵赋值给 Flax 模型的对应参数
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = t5x_decoder_rel_embedding

    # Token Embeddings

    # 从 T5X 模型中获取目标部分的 token 嵌入（词嵌入）矩阵
    tx5_token_embeddings = t5x_model["target"]["token_embedder"]["embedding"]
    
    # 将获取的 token 嵌入矩阵赋值给 Flax 模型的共享嵌入层参数
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # LM Head (only in v1.1 and LongT5 checkpoints)

    # 检查 T5X 模型中是否存在 logits_dense 属性，通常出现在 v1.1 和 LongT5 检查点中
    if "logits_dense" in t5x_model["target"]["decoder"]:
        # 将 T5X 模型中的 logits_dense 的核（权重矩阵）赋值给 Flax 模型的语言模型头部参数
        flax_model.params["lm_head"]["kernel"] = t5x_model["target"]["decoder"]["logits_dense"]["kernel"]

    # 将转换后的 Flax 模型保存到指定的文件夹路径
    flax_model.save_pretrained(flax_dump_folder_path)
    
    # 打印转换成功的提示信息
    print("T5X Model was sucessfully converted!")
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则进入主程序入口

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需的参数
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path the T5X checkpoint."
    )
    # 添加命令行参数：T5X 模型的检查点路径，必需，类型为字符串，帮助信息为路径到 T5X 检查点的路径

    parser.add_argument("--config_name", default=None, type=str, required=True, help="Config name of LongT5/T5 model.")
    # 添加命令行参数：LongT5/T5 模型的配置名称，必需，类型为字符串，帮助信息为模型配置的名称

    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    # 添加命令行参数：FLAX 模型的输出文件夹路径，必需，类型为字符串，帮助信息为输出 FLAX 模型的文件夹路径

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_t5x_checkpoint_to_flax(args.t5x_checkpoint_path, args.config_name, args.flax_dump_folder_path)
    # 调用函数 convert_t5x_checkpoint_to_flax，传入命令行参数中的 T5X 模型路径、配置名称和 FLAX 输出文件夹路径作为参数
```