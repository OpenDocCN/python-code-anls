# `.\transformers\models\pix2struct\convert_pix2struct_original_pytorch_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用该文件
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统模块
import re  # 导入正则表达式模块

import torch  # 导入 PyTorch 模块
from flax.traverse_util import flatten_dict  # 从 flax 库中导入 flatten_dict 函数
from t5x import checkpoints  # 从 t5x 模块中导入 checkpoints 函数

from transformers import (  # 从 transformers 库中导入以下模块
    AutoTokenizer,  # 自动 Tokenizer
    Pix2StructConfig,  # Pix2Struct 模型配置
    Pix2StructForConditionalGeneration,  # Pix2Struct 用于条件生成的模型
    Pix2StructImageProcessor,  # Pix2Struct 图像处理器
    Pix2StructProcessor,  # Pix2Struct 处理器
    Pix2StructTextConfig,  # Pix2Struct 文本配置
    Pix2StructVisionConfig,  # Pix2Struct 视觉配置
)


def get_flax_param(t5x_checkpoint_path):
    # 加载 t5x 检查点
    flax_params = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    # 将参数扁平化
    flax_params = flatten_dict(flax_params)
    return flax_params


def rename_and_convert_flax_params(flax_dict):
    converted_dict = {}

    # 定义参数转换映射
    CONVERSION_MAPPING = {
        "token_embedder": "embeddings",
        "encoder_norm": "layernorm",
        "kernel": "weight",
        ".out": ".output",
        "scale": "weight",
        "embedders_0.pos_embedding": "row_embedder.weight",
        "embedders_1.pos_embedding": "column_embedder.weight",
    }

    DECODER_CONVERSION_MAPPING = {
        "query": "attention.query",
        "key": "attention.key",
        "value": "attention.value",
        "output.dense": "output",
        "encoder_decoder_attention.o": "encoder_decoder_attention.attention.o",
        "pre_self_attention_layer_norm": "self_attention.layer_norm",
        "pre_cross_attention_layer_norm": "encoder_decoder_attention.layer_norm",
        "mlp.": "mlp.DenseReluDense.",
        "pre_mlp_layer_norm": "mlp.layer_norm",
        "self_attention.o": "self_attention.attention.o",
        "decoder.embeddings.embedding": "decoder.embed_tokens.weight",
        "decoder.relpos_bias.rel_embedding": "decoder.layer.0.self_attention.attention.relative_attention_bias.weight",
        "decoder.decoder_norm.weight": "decoder.final_layer_norm.weight",
        "decoder.logits_dense.weight": "decoder.lm_head.weight",
    }
    # 遍历原始字典的键
    for key in flax_dict.keys():
        # 如果键中包含"target"
        if "target" in key:
            # 去掉键的第一个前缀
            new_key = ".".join(key[1:])

            # 重命名键
            for old, new in CONVERSION_MAPPING.items():
                new_key = new_key.replace(old, new)

            # 如果键中包含"decoder"
            if "decoder" in new_key:
                # 根据 DECODER_CONVERSION_MAPPING 替换键中的内容
                for old, new in DECODER_CONVERSION_MAPPING.items():
                    new_key = new_key.replace(old, new)

            # 如果键中包含"layers"且不包含"decoder"
            if "layers" in new_key and "decoder" not in new_key:
                # 使用正则表达式替换层号
                new_key = re.sub(r"layers_(\d+)", r"layer.\1", new_key)
                new_key = new_key.replace("encoder", "encoder.encoder")

            # 如果键中包含"layers"且包含"decoder"
            elif "layers" in new_key and "decoder" in new_key:
                # 使用正则表达式替换层号
                new_key = re.sub(r"layers_(\d+)", r"layer.\1", new_key)

            # 将新键值对加入转换后的字典
            converted_dict[new_key] = flax_dict[key]

    converted_torch_dict = {}
    # 将转换后的字典转换为 torch 格式
    for key in converted_dict.keys():
        # 如果键中不包含"embed_tokens"和"embedder"
        if ("embed_tokens" not in key) and ("embedder" not in key):
            # 将值转换为 torch 张量并转置
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key].T)
        else:
            # 将值转换为 torch 张量
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key])

    # 返回转换后的 torch 字典
    return converted_torch_dict
def convert_pix2struct_original_pytorch_checkpoint_to_hf(
    t5x_checkpoint_path, pytorch_dump_folder_path, use_large=False, is_vqa=False
):
    # 获取 Flax 模型参数
    flax_params = get_flax_param(t5x_checkpoint_path)

    # 根据是否使用大模型选择不同的配置
    if not use_large:
        encoder_config = Pix2StructVisionConfig()
        decoder_config = Pix2StructTextConfig()
    else:
        encoder_config = Pix2StructVisionConfig(
            hidden_size=1536, d_ff=3968, num_attention_heads=24, num_hidden_layers=18
        )
        decoder_config = Pix2StructTextConfig(hidden_size=1536, d_ff=3968, num_heads=24, num_layers=18)
    # 创建 Pix2StructConfig 对象
    config = Pix2StructConfig(
        vision_config=encoder_config.to_dict(), text_config=decoder_config.to_dict(), is_vqa=is_vqa
    )

    # 创建 Pix2StructForConditionalGeneration 模型
    model = Pix2StructForConditionalGeneration(config)

    # 重命名并转换 Flax 参数，加载到模型中
    torch_params = rename_and_convert_flax_params(flax_params)
    model.load_state_dict(torch_params)

    # 加载预训练的分词器
    tok = AutoTokenizer.from_pretrained("ybelkada/test-pix2struct-tokenizer")
    image_processor = Pix2StructImageProcessor()
    processor = Pix2StructProcessor(image_processor=image_processor, tokenizer=tok)

    # 根据是否使用大模型设置最大补丁数和是否为 VQA 模型
    if use_large:
        processor.image_processor.max_patches = 4096
    processor.image_processor.is_vqa = True

    # 如果需要，创建输出文件夹
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # 保存模型和处理器
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

    # 打印保存成功信息
    print("Model saved in {}".format(pytorch_dump_folder_path))


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5x_checkpoint_path", default=None, type=str, help="Path to the original T5x checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--use_large", action="store_true", help="Use large model.")
    parser.add_argument("--is_vqa", action="store_true", help="Use large model.")
    args = parser.parse_args()

    # 调用转换函数
    convert_pix2struct_original_pytorch_checkpoint_to_hf(
        args.t5x_checkpoint_path, args.pytorch_dump_folder_path, args.use_large
    )
```