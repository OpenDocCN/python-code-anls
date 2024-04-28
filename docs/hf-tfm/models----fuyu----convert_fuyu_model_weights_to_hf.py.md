# `.\models\fuyu\convert_fuyu_model_weights_to_hf.py`

```
# 引入命令行参数解析库
import argparse
# 引入操作系统交互库
import os
# 引入系统库
import sys
# 引入警告库
import warnings

# 引入将嵌套字典展平的工具库
import flatdict
# 引入PyTorch库
import torch

# 引入转换器相关的库
from transformers import FuyuConfig, FuyuForCausalLM, LlamaTokenizer

# 尝试引入高效的LLama分词器
try:
    from transformers import LlamaTokenizerFast
    # 如果引入成功，则使用快速分词器
    tokenizer_class = LlamaTokenizerFast
# 如果引入失败，则使用慢速分词器，并发出警告
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    tokenizer_class = LlamaTokenizer

# 定义要修改的键与对应新键的映射关系
KEYS_TO_MODIFY_MAPPING = {
    "self_attention": "self_attn",
    "language_model.encoder": "language_model.model",
    "word_embeddings_for_head": "language_model.lm_head",
    "language_model.embedding.word_embeddings": "language_model.model.embed_tokens",
    "vit_encoder.linear_encoder": "vision_embed_tokens",
}

# 定义要移除的键的集合
KEYS_TO_REMOVE = {
    "rotary_emb.inv_freq",
    "image_patch_projection",
    "image_patch_projection.weight",
    "image_patch_projection.bias",
}

# 定义函数，用于重命名状态字典中的键
def rename_state_dict(state_dict):
    # 创建一个新的模型状态字典
    model_state_dict = {}
    # 遍历状态字典中的每个键值对
    for key, value in state_dict.items():
        # 遍历需要修改的键与新键的映射关系
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            # 如果需要修改的键存在于当前键中，则将其替换为新键
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        # 如果当前键在需要移除的键集合中，则跳过不处理
        if key in KEYS_TO_REMOVE:
            continue
        # 将处理后的键值对添加到新的模型状态字典中
        model_state_dict[key] = value
    # 返回一个模型的状态字典
    return model_state_dict
# 定义函数，用于转换 Fuyu 检查点到 PyTorch 格式
def convert_fuyu_checkpoint(pytorch_dump_folder_path, ada_lib_path, pt_model_path, safe_serialization=False):
    # 将 ada_lib_path 添加到系统路径中
    sys.path.insert(0, ada_lib_path)
    # 从 pt_model_path 中加载 PyTorch 模型状态字典
    model_state_dict_base = torch.load(pt_model_path, map_location="cpu")
    # 将模型状态字典扁平化
    state_dict = flatdict.FlatDict(model_state_dict_base["model"], ".")
    # 重命名状态字典中的键
    state_dict = rename_state_dict(state_dict)

     # 初始化 FuyuConfig 对象
    transformers_config = FuyuConfig()
    # 创建 FuyuForCausalLM 模型对象
    model = FuyuForCausalLM(transformers_config).to(torch.bfloat16)
    # 加载模型状态字典
    model.load_state_dict(state_dict)
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    # 保存配置到指定路径
    transformers_config.save_pretrained(pytorch_dump_folder_path)

# 主函数
def main():
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    # 添加输入目录参数
    parser.add_argument(
        "--input_dir",
        help="Location of Fuyu weights, which contains tokenizer.model and model folders",
    )
    # 添加 Fuyu 模型路径参数
    parser.add_argument(
        "--pt_model_path",
        help="Location of Fuyu `model_optim_rng.pt`",
    )
    # 添加输出目录参数
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加 ada 库路径参数
    parser.add_argument(
        "--ada_lib_path",
        help="Location of original source code from adept to deserialize .pt checkpoint",
    )
    # 添加安全序列化参数
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析命令行参数
    args = parser.parse_args()
    # 拼接路径，得到 spm_path
    spm_path = os.path.join(args.input_dir, "adept_vocab.model")

    # 调用转换函数，将 Fuyu 检查点转换为 PyTorch 格式
    convert_fuyu_checkpoint(
        pytorch_dump_folder_path=args.output_dir,
        pt_model_path=args.pt_model_path,
        safe_serialization=args.safe_serialization,
        ada_lib_path=args.ada_lib_path,
    )
    # 创建 tokenizer 对象，加载 spm_path 和设置 bos_token、eos_token
    tokenizer = tokenizer_class(spm_path, bos_token="|ENDOFTEXT|", eos_token="|ENDOFTEXT|")
    # 保存 tokenizer 到输出目录
    tokenizer.save_pretrained(args.output_dir)

# 如果当前脚本作为主程序执行，则调用主函数
if __name__ == "__main__":
    main()
```