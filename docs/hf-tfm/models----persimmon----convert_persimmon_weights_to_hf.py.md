# `.\models\persimmon\convert_persimmon_weights_to_hf.py`

```
# 导入必要的库和模块
import argparse  # 用于处理命令行参数的库
import os  # 提供与操作系统交互的功能
import warnings  # 用于警告处理的库

import flatdict  # 用于扁平化字典的库
import torch  # PyTorch深度学习库

# 从transformers库中导入所需的类和函数
from transformers import LlamaTokenizer, PersimmonConfig, PersimmonForCausalLM

try:
    from transformers import LlamaTokenizerFast  # 尝试导入快速的LlamaTokenizer
    tokenizer_class = LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)  # 输出导入错误的警告信息
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    tokenizer_class = LlamaTokenizer  # 使用默认的LlamaTokenizer

"""
示例用法:

git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path
"""

# 需要重命名的键值对映射关系
KEYS_TO_MODIFY_MAPPING = {
    "self_attention": "self_attn",  # 将键"self_attention"映射为"self_attn"
    "language_model.encoder": "model",  # 将键"language_model.encoder"映射为"model"
    "word_embeddings_for_head": "lm_head",  # 将键"word_embeddings_for_head"映射为"lm_head"
    "language_model.embedding.word_embeddings": "model.embed_tokens",  # 将键"language_model.embedding.word_embeddings"映射为"model.embed_tokens"
}

KEYS_TO_REMOVE = "rotary_emb.inv_freq"  # 需要从状态字典中移除的键

# 重命名状态字典的函数，根据映射关系修改键名
def rename_state_dict(state_dict):
    model_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        if KEYS_TO_REMOVE in key:
            continue  # 如果键包含需移除的内容，跳过此键
        model_state_dict[key] = value
    return model_state_dict


def convert_persimmon_checkpoint(pytorch_dump_folder_path, ada_lib_path, pt_model_path, safe_serialization=False):
    import sys

    sys.path.insert(0, ada_lib_path)  # 将ada_lib_path插入到系统路径中，用于导入模块
    # 从指定路径加载 PyTorch 模型的状态字典到 model_state_dict_base 变量中，使用 CPU 作为设备
    model_state_dict_base = torch.load(pt_model_path, map_location="cpu")
    
    # 使用 flatdict 库将模型状态字典扁平化，使用 "." 作为分隔符
    state_dict = flatdict.FlatDict(model_state_dict_base["model"], ".")
    
    # 对状态字典进行重命名处理，返回重命名后的状态字典
    state_dict = rename_state_dict(state_dict)
    
    # 创建一个 PersimmonConfig 对象用于配置 Transformers 模型
    transformers_config = PersimmonConfig()
    
    # 使用 PersimmonForCausalLM 类创建一个 Transformers 模型，指定 eos_token_id 和 bos_token_id，并将模型放到 torch.bfloat16 数据类型中
    model = PersimmonForCausalLM(transformers_config, eos_token_id=71013, bos_token_id=71013).to(torch.bfloat16)
    
    # 加载处理后的状态字典到模型中
    model.load_state_dict(state_dict)
    
    # 将模型保存到指定路径 pytorch_dump_folder_path 中，使用安全的序列化方法进行保存
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=safe_serialization)
    
    # 将 Transformers 配置对象保存到指定路径 pytorch_dump_folder_path 中
    transformers_config.save_pretrained(pytorch_dump_folder_path)
# 主程序入口函数
def main():
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：输入目录，用于指定Persimmon权重文件的位置，包括tokenizer.model和model文件夹
    parser.add_argument(
        "--input_dir",
        help="Location of Persimmon weights, which contains tokenizer.model and model folders",
    )
    # 添加命令行参数：模型路径，用于指定Persimmon的`model_optim_rng.pt`文件位置
    parser.add_argument(
        "--pt_model_path",
        help="Location of Persimmon `model_optim_rng.pt`",
    )
    # 添加命令行参数：输出目录，用于指定HF模型和tokenizer的存储位置
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数：ada库路径，用于指定HF模型和tokenizer的存储位置
    parser.add_argument(
        "--ada_lib_path",
        help="Location to write HF model and tokenizer",
    )
    # 添加命令行参数：安全序列化选项，指定是否使用`safetensors`进行保存
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 构建adept_vocab.model文件的完整路径
    spm_path = os.path.join(args.input_dir, "adept_vocab.model")

    # 调用函数：将Persimmon的检查点转换为PyTorch模型
    convert_persimmon_checkpoint(
        pytorch_dump_folder_path=args.output_dir,
        pt_model_path=args.pt_model_path,
        safe_serialization=args.safe_serialization,
        ada_lib_path=args.ada_lib_path,
    )
    
    # 初始化tokenizer对象，使用adept_vocab.model和指定的起始/结束标记
    tokenizer = tokenizer_class(spm_path, bos_token="|ENDOFTEXT|", eos_token="|ENDOFTEXT|")
    # 将tokenizer对象保存到指定的输出目录
    tokenizer.save_pretrained(args.output_dir)

# 如果当前脚本作为主程序运行，则执行main函数
if __name__ == "__main__":
    main()
```