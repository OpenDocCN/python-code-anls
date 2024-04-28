# `.\transformers\models\longt5\convert_longt5x_checkpoint_to_flax.py`

```py
# 导入必要的库
import argparse
# 导入 T5X 模型相关的函数
from t5x import checkpoints
# 导入自动配置和 Flax 版本的 T5 模型
from transformers import AutoConfig, FlaxAutoModelForSeq2SeqLM

# 定义函数，将 T5X 检查点转换为 Flax 模型
def convert_t5x_checkpoint_to_flax(t5x_checkpoint_path, config_name, flax_dump_folder_path):
    # 从预训练模型配置名称创建自动配置对象
    config = AutoConfig.from_pretrained(config_name)
    # 根据配置创建 Flax 版本的 T5 模型
    flax_model = FlaxAutoModelForSeq2SeqLM.from_config(config=config)
    # 加载 T5X 检查点
    t5x_model = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)

    # 检查模型类型以确定编码器的自注意力名称
    if config.model_type == "t5":
        encoder_attn_name = "SelfAttention"
    elif config.model_type == "longt5" and config.encoder_attention_type == "local":
        encoder_attn_name = "LocalSelfAttention"
    elif config.model_type == "longt5" and config.encoder_attention_type == "transient-global":
        encoder_attn_name = "TransientGlobalSelfAttention"
    else:
        raise ValueError(
            "给定的配置应该具有 `model_type='t5'`，或者 `model_type='longt5'` 并且 `encoder_attention_type` 属性应具有值['local', 'transient-global]`."
        )

    # 编码器部分

    # 仅对第 0 层进行操作:
    # 从 T5X 模型中提取编码器相对位置嵌入
    t5x_encoder_rel_embedding = t5x_model["target"]["encoder"]["relpos_bias"]["rel_embedding"].T
    # 将相对注意力偏置嵌入赋值给 Flax 模型的相对注意力偏置嵌入
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"][encoder_attn_name]["relative_attention_bias"][
        "embedding"
    ] = t5x_encoder_rel_embedding

    # 侧/全局相对位置偏置 + 层规范化
    if config.model_type == "longt5" and config.encoder_attention_type == "transient-global":
        # 从 T5X 模型中提取编码器全局相对位置嵌入
        t5x_encoder_global_rel_embedding = t5x_model["target"]["encoder"]["side_relpos_bias"]["rel_embedding"].T
        # 将全局相对注意力偏置嵌入赋值给 Flax 模型的全局相对注意力偏置嵌入
        flax_model.params["encoder"]["block"]["0"]["layer"]["0"][encoder_attn_name]["global_relative_attention_bias"][
            "embedding"
        ] = t5x_encoder_global_rel_embedding

    # 分配编码器最终层规范化的权重
    # 从 T5X 模型中提取编码器的规范化参数
    t5x_encoder_norm = t5x_model["target"]["encoder"]["encoder_norm"]["scale"]
    # 将规范化参数赋值给 Flax 模型的最终层规范化的权重
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = t5x_encoder_norm

    # 解码器部分

    # 解码器规范化
    # 从 T5X 模型中提取解码器的规范化参数
    t5x_decoder_norm = t5x_model["target"]["decoder"]["decoder_norm"]["scale"]
    # 将解码器的规范化参数赋值给 Flax 模型的最终层规范化的权重
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = t5x_decoder_norm
    # 仅对于层级为0的情况：
    # 从T5X模型中提取目标端解码器中的相对位置嵌入矩阵，并进行转置操作
    t5x_decoder_rel_embedding = t5x_model["target"]["decoder"]["relpos_bias"]["rel_embedding"].T
    # 将转置后的相对位置嵌入矩阵赋值给Flax模型的相对注意力偏置的嵌入
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = t5x_decoder_rel_embedding

    # Token嵌入
    # 从T5X模型中提取目标端的token嵌入
    tx5_token_embeddings = t5x_model["target"]["token_embedder"]["embedding"]
    # 将T5X模型中的token嵌入赋值给Flax模型的共享嵌入
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # LM头部（仅适用于v1.1和LongT5检查点）
    # 如果T5X模型中的解码器中存在"logits_dense"关键字
    if "logits_dense" in t5x_model["target"]["decoder"]:
        # 将T5X模型中的"logits_dense"中的核赋值给Flax模型的LM头部的核
        flax_model.params["lm_head"]["kernel"] = t5x_model["target"]["decoder"]["logits_dense"]["kernel"]

    # 将Flax模型保存到指定的文件夹路径中
    flax_model.save_pretrained(flax_dump_folder_path)
    # 打印提示信息，表示T5X模型成功转换
    print("T5X Model was sucessfully converted!")
# 如果当前模块被直接执行，而非被导入到其他模块中
if __name__ == "__main__":
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path the T5X checkpoint."
    )
    parser.add_argument("--config_name", default=None, type=str, required=True, help="Config name of LongT5/T5 model.")
    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入参数
    convert_t5x_checkpoint_to_flax(args.t5x_checkpoint_path, args.config_name, args.flax_dump_folder_path)
```