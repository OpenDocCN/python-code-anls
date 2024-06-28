# `.\models\kosmos2\convert_kosmos2_original_pytorch_checkpoint_to_pytorch.py`

```py
import argparse  # 导入命令行参数解析模块

from fairseq.checkpoint_utils import load_checkpoint_to_cpu  # 从fairseq库中导入加载checkpoint到CPU的函数

from transformers import Kosmos2Config, Kosmos2ForConditionalGeneration  # 从transformers库中导入Kosmos2Config和Kosmos2ForConditionalGeneration类


KEYS_TO_MODIFY_MAPPING = {
    "gpt_model.decoder.output_projection": "text_model.lm_head",  # 将"gpt_model.decoder.output_projection"映射为"text_model.lm_head"
    "gpt_model.decoder": "text_model.model",  # 将"gpt_model.decoder"映射为"text_model.model"
    "img_connector": "image_to_text_projection",  # 将"img_connector"映射为"image_to_text_projection"
    "img_model.visual.class_embedding": "vision_model.model.embeddings.class_embedding",  # 将"img_model.visual.class_embedding"映射为"vision_model.model.embeddings.class_embedding"
    "img_model.visual.positional_embedding": "vision_model.model.embeddings.position_embedding.weight",  # 将"img_model.visual.positional_embedding"映射为"vision_model.model.embeddings.position_embedding.weight"
    "img_model.visual.conv1": "vision_model.model.embeddings.patch_embedding",  # 将"img_model.visual.conv1"映射为"vision_model.model.embeddings.patch_embedding"
    "img_model.visual": "vision_model.model",  # 将"img_model.visual"映射为"vision_model.model"
    "ln_pre": "pre_layrnorm",  # 将"ln_pre"映射为"pre_layrnorm"
    "ln_post": "post_layernorm",  # 将"ln_post"映射为"post_layernorm"
    "transformer.resblocks": "encoder.layers",  # 将"transformer.resblocks"映射为"encoder.layers"
    "ts_attn": "self_attn",  # 将"ts_attn"映射为"self_attn"
    "ln_1": "layer_norm1",  # 将"ln_1"映射为"layer_norm1"
    "ln_2": "layer_norm2",  # 将"ln_2"映射为"layer_norm2"
    "c_fc": "fc1",  # 将"c_fc"映射为"fc1"
    "c_proj": "fc2",  # 将"c_proj"映射为"fc2"
}


KEYS_TO_IGNORE = [
    # 在原始代码中仅用于将权重发送到所需设备的缓冲区
    "gpt_model.decoder.embed_positions._float_tensor",
    # 在原始的KOSMOS-2代码中前向传播中从未使用过的权重
    "gpt_model.decoder.self_attn_sope.scale",
]


def rename_key(key):
    for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
        if key_to_modify in key:
            key = key.replace(key_to_modify, new_key)  # 根据映射表修改键名

    return key


def convert_kosmos2_checkpoint_to_pytorch(checkpoint_path, pytorch_dump_folder_path):
    state = load_checkpoint_to_cpu(checkpoint_path)  # 加载checkpoint到CPU
    state_dict = state["model"]  # 获取模型的state_dict
    state_dict_keys = list(state_dict.keys())  # 获取state_dict中的所有键列表

    config = Kosmos2Config()  # 创建Kosmos2Config实例
    # 为了匹配原始演示给出的结果，设置必要的配置项
    config.text_config.no_repeat_ngram_size = 3
    model = Kosmos2ForConditionalGeneration(config)  # 创建Kosmos2ForConditionalGeneration模型实例

    # 转换（通过重命名键名）
    converted_state_dict = {}
    for key in state_dict_keys:
        if key in KEYS_TO_IGNORE:
            continue  # 跳过需要忽略的键名
        renamed_key = rename_key(key)  # 根据映射重命名键名
        converted_state_dict[renamed_key] = state_dict[key]  # 更新转换后的state_dict

    # 检查权重加载
    model.load_state_dict(converted_state_dict, strict=True)  # 加载转换后的state_dict到模型
    # 保存结果
    model.save_pretrained(pytorch_dump_folder_path)  # 将模型保存为PyTorch格式的文件


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # 必需参数
    parser.add_argument(
        "--kosmos2_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_kosmos2_checkpoint_to_pytorch(args.kosmos2_checkpoint_path, args.pytorch_dump_folder_path)  # 执行转换函数
```