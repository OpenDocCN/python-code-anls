# `.\models\fuyu\convert_fuyu_model_weights_to_hf.py`

```
# 引入命令行参数解析库
import argparse
# 引入操作系统相关功能的库
import os
# 引入系统相关的库
import sys
# 引入警告处理的库
import warnings

# 引入用于扁平化字典操作的库
import flatdict
# 引入PyTorch深度学习框架库
import torch

# 从transformers库中引入FuyuConfig、FuyuForCausalLM和LlamaTokenizer
from transformers import FuyuConfig, FuyuForCausalLM, LlamaTokenizer

# 尝试从transformers库中引入LlamaTokenizerFast，如果失败则发出警告并使用慢速的LlamaTokenizer
try:
    from transformers import LlamaTokenizerFast
    tokenizer_class = LlamaTokenizerFast
except ImportError as e:
    # 发出导入错误的警告
    warnings.warn(e)
    # 发出警告，提示使用慢速的tokenizer
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    # 使用慢速的LlamaTokenizer作为tokenizer类
    tokenizer_class = LlamaTokenizer

# 多行注释，提供了代码示例的使用说明和模型加载方法

# 定义需要修改的state_dict键和对应的新键映射关系
KEYS_TO_MODIFY_MAPPING = {
    "self_attention": "self_attn",
    "language_model.encoder": "language_model.model",
    "word_embeddings_for_head": "language_model.lm_head",
    "language_model.embedding.word_embeddings": "language_model.model.embed_tokens",
    "vit_encoder.linear_encoder": "vision_embed_tokens",
}

# 定义需要移除的state_dict键集合
KEYS_TO_REMOVE = {
    "rotary_emb.inv_freq",
    "image_patch_projection",
    "image_patch_projection.weight",
    "image_patch_projection.bias",
}


# 定义一个函数，用于重命名给定state_dict的键
def rename_state_dict(state_dict):
    # 创建空字典，用于存储重命名后的state_dict
    model_state_dict = {}
    # 遍历原始state_dict的键值对
    for key, value in state_dict.items():
        # 遍历需要修改的映射关系
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            # 如果当前键包含需要修改的键
            if key_to_modify in key:
                # 替换当前键为新的键
                key = key.replace(key_to_modify, new_key)
        # 如果当前键在需要移除的集合中，则跳过不处理
        if key in KEYS_TO_REMOVE:
            continue
        # 将更新后的键值对添加到新的model_state_dict中
        model_state_dict[key] = value
    return model_state_dict


    # 返回模型的状态字典
    # 这行代码将函数的执行结果返回给调用者，通常用于将函数内部计算得到的结果传递出去
# 定义一个函数用于将 Fuyu 模型的检查点转换为 PyTorch 格式
def convert_fuyu_checkpoint(pytorch_dump_folder_path, ada_lib_path, pt_model_path, safe_serialization=False):
    # 将 ada_lib_path 添加到系统路径中，以便导入相关库
    sys.path.insert(0, ada_lib_path)
    # 使用 map_location="cpu" 加载 PyTorch 模型的状态字典
    model_state_dict_base = torch.load(pt_model_path, map_location="cpu")
    # 将模型状态字典展开成扁平结构
    state_dict = flatdict.FlatDict(model_state_dict_base["model"], ".")
    # 重命名状态字典中的键
    state_dict = rename_state_dict(state_dict)

    # 创建 FuyuConfig 的实例，用于配置 Transformers 模型
    transformers_config = FuyuConfig()
    # 创建 FuyuForCausalLM 模型的实例，并转换为 torch.bfloat16 数据类型
    model = FuyuForCausalLM(transformers_config).to(torch.bfloat16)
    # 加载转换后的模型状态字典
    model.load_state_dict(state_dict)
    # 将模型保存到指定路径，并可选择安全序列化
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    # 将 Transformers 配置保存到同一路径
    transformers_config.save_pretrained(pytorch_dump_folder_path)


# 主函数，用于解析命令行参数并调用转换函数
def main():
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入目录，包含 tokenizer.model 和 model 文件夹的位置
    parser.add_argument(
        "--input_dir",
        help="Location of Fuyu weights, which contains tokenizer.model and model folders",
    )
    # 添加命令行参数：Fuyu 模型的位置
    parser.add_argument(
        "--pt_model_path",
        help="Location of Fuyu `model_optim_rng.pt`",
    )
    # 添加命令行参数：输出目录，用于存储 HF 模型和 tokenizer
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数：adept 库的位置，用于反序列化 .pt 检查点
    parser.add_argument(
        "--ada_lib_path",
        help="Location of original source code from adept to deserialize .pt checkpoint",
    )
    # 添加命令行参数：是否使用安全张量进行保存
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析命令行参数
    args = parser.parse_args()
    # 构建 spm_path，用于 tokenizer 的路径
    spm_path = os.path.join(args.input_dir, "adept_vocab.model")

    # 调用转换函数，将 Fuyu 模型的检查点转换为 PyTorch 格式
    convert_fuyu_checkpoint(
        pytorch_dump_folder_path=args.output_dir,
        pt_model_path=args.pt_model_path,
        safe_serialization=args.safe_serialization,
        ada_lib_path=args.ada_lib_path,
    )
    # 创建 tokenizer 实例，使用 spm_path 和特定的起始和结束标记
    tokenizer = tokenizer_class(spm_path, bos_token="|ENDOFTEXT|", eos_token="|ENDOFTEXT|")
    # 将 tokenizer 保存到输出目录
    tokenizer.save_pretrained(args.output_dir)


# 如果脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```