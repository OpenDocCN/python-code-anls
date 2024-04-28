# `.\transformers\models\persimmon\convert_persimmon_weights_to_hf.py`

```py
# 导入所需模块
import argparse   # 导入命令行参数解析模块
import os    # 导入操作系统模块
import warnings    # 导入警告模块

import flatdict   # 导入用于扁平化字典的模块
import torch    # 导入PyTorch模块

from transformers import LlamaTokenizer, PersimmonConfig, PersimmonForCausalLM    # 从transformers库中导入LlamaTokenizer、PersimmonConfig和PersimmonForCausalLM类


try:
    from transformers import LlamaTokenizerFast    # 尝试导入transformers库中的LlamaTokenizerFast类

    tokenizer_class = LlamaTokenizerFast    # 将tokenizer_class设置为LlamaTokenizerFast类
except ImportError as e:
    warnings.warn(e)    # 如果导入出错，则发出警告
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )    # 给出警告信息
    tokenizer_class = LlamaTokenizer    # 将tokenizer_class设置为LlamaTokenizer类

"""
Sample usage:


git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path

上述是示例用法。


Thereafter, models can be loaded via:


from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")

之后，可以通过以上方式加载模型。

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
重要提示：执行此脚本需要能够在内存中承载整个模型（即使最大版本拆分为多个检查点, 但它们每个检查点都包含模型的一部分权重，因此我们需要将全部检查点加载到内存中）。

"""


KEYS_TO_MODIFY_MAPPING = {    # 定义需要修改的键的映射关系
    "self_attention": "self_attn",    # 将"self_attention"替换为"self_attn"
    "language_model.encoder": "model",    # 将"language_model.encoder"替换为"model"
    "word_embeddings_for_head": "lm_head",    # 将"word_embeddings_for_head"替换为"lm_head"
    "language_model.embedding.word_embeddings": "model.embed_tokens"    # 将"language_model.embedding.word_embeddings"替换为"model.embed_tokens"
}

KEYS_TO_REMOVE = "rotary_emb.inv_freq"    # 定义需要移除的键


def rename_state_dict(state_dict):
    model_state_dict = {}    # 初始化模型状态字典
    for key, value in state_dict.items():    # 遍历状态字典
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():    # 遍历需要修改的键和新键
            if key_to_modify in key:    # 如果需要修改的键在当前键中
                key = key.replace(key_to_modify, new_key)    # 将当前键中的需要修改的键替换为新键
        if KEYS_TO_REMOVE in key:    # 如果需要移除的键在当前键中
            continue    # 跳过当前循环
        model_state_dict[key] = value    # 将新键和对应的值添加到模型状态字典中
    return model_state_dict    # 返回模型状态字典


def convert_persimmon_checkpoint(pytorch_dump_folder_path, ada_lib_path, pt_model_path, safe_serialization=False):
    import sys

    sys.path.insert(0, ada_lib_path)    # 将ada_lib_path添加到模块搜索路径中
    # 从指定路径加载 PyTorch 模型的状态字典，使用 CPU 进行映射
    model_state_dict_base = torch.load(pt_model_path, map_location="cpu")
    # 将状态字典扁平化处理，使用"."作为分隔符
    state_dict = flatdict.FlatDict(model_state_dict_base["model"], ".")
    # 重命名状态字典中的键
    state_dict = rename_state_dict(state_dict)
    
    # 创建 Persimmon 模型的配置对象
    transformers_config = PersimmonConfig()
    # 使用 Persimmon 模型的配置对象创建 CausalLM 模型，指定 eos_token_id 和 bos_token_id，并将模型放置在 bfloat16 上
    model = PersimmonForCausalLM(transformers_config, eos_token_id=71013, bos_token_id=71013).to(torch.bfloat16)
    # 加载模型的状态字典
    model.load_state_dict(state_dict)
    # 将模型保存到指定的 PyTorch dump 文件夹路径中，使用安全序列化
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    # 将模型的配置保存到指定的 PyTorch dump 文件夹路径中
    transformers_config.save_pretrained(pytorch_dump_folder_path)
# 主函数，用于解析命令行参数并执行相应操作
def main():
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加输入目录参数
    parser.add_argument(
        "--input_dir",
        help="Location of Persimmon weights, which contains tokenizer.model and model folders",
    )
    # 添加模型路径参数
    parser.add_argument(
        "--pt_model_path",
        help="Location of Persimmon `model_optim_rng.pt`",
    )
    # 添加输出目录参数
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加适配库路径参数
    parser.add_argument(
        "--ada_lib_path",
        help="Location to write HF model and tokenizer",
    )
    # 添加安全序列化参数
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # 解析命令行参数
    args = parser.parse_args()
    # 根据输入目录生成词表路径
    spm_path = os.path.join(args.input_dir, "adept_vocab.model")

    # 转换 Persimmon checkpoint
    convert_persimmon_checkpoint(
        pytorch_dump_folder_path=args.output_dir,
        pt_model_path=args.pt_model_path,
        safe_serialization=args.safe_serialization,
        ada_lib_path=args.ada_lib_path,
    )
    # 使用词表路径和特殊起始和结束标记创建分词器对象
    tokenizer = tokenizer_class(spm_path, bos_token="|ENDOFTEXT|", eos_token="|ENDOFTEXT|")
    # 保存分词器预训练模型到输出目录
    tokenizer.save_pretrained(args.output_dir)


# 如果是作为脚本直接执行，则调用主函数
if __name__ == "__main__":
    main()
```