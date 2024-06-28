# `.\models\phi\convert_phi_weights_to_hf.py`

```
# 引入 argparse 模块，用于解析命令行参数
import argparse
# 引入垃圾收集模块 gc，用于手动进行内存管理
import gc
# 引入操作系统相关功能的 os 模块
import os

# 引入 safetensors 库，用于处理安全张量
import safetensors
# 引入 PyTorch 库
import torch
# 从 huggingface_hub 库中导入 hf_hub_download 函数
from huggingface_hub import hf_hub_download

# 从 transformers 库中导入 PhiConfig 和 PhiForCausalLM 类
from transformers import PhiConfig, PhiForCausalLM

# 定义字典 MODELS，映射 Phi 模型名称到其权重文件下载链接的字典
_MODELS = {
    "microsoft/phi-1": ["https://huggingface.co/microsoft/phi-1/blob/main/pytorch_model.bin"],
    "microsoft/phi-1_5": ["https://huggingface.co/microsoft/phi-1_5/blob/main/pytorch_model.bin"],
    "microsoft/phi-2": [
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00001-of-00002.safetensors",
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00002-of-00002.safetensors",
    ],
}

# 定义 PHI_MAPPING 字典，将 Phi 模型中的原始权重键映射到转换后的键
PHI_MAPPING = {
    "transformer.embd.wte.weight": "model.embed_tokens.weight",
    "lm_head.linear": "lm_head",
    "lm_head.ln": "model.final_layernorm",
    "layers": "model.layers",
    "transformer": "model",
    ".h.": ".layers.",
    "ln": "input_layernorm",
    "mixer": "self_attn",
    "Wqkv": "query_key_value",
    "out_proj": "dense",
}

# 定义函数 convert_weights，用于转换原始权重到指定配置的转换后的权重
def convert_weights(original_weights, mapping, config):
    # 初始化一个空字典，用于存储转换后的权重
    converted_weights = {}
    # 获取原始权重的键，并对其进行排序
    original_weights_keys = sorted(original_weights.keys())
    # 遍历原始权重的键列表
    for original_weights_key in original_weights_keys:
        # 创建一个新的键，初始为原始权重的键
        new_key = original_weights_key

        # 如果新键包含"rotary_emb"，跳过当前循环，继续下一个键的处理
        if "rotary_emb" in new_key:
            continue

        # 如果新键包含"Wqkv"，根据后缀"weight"或"bias"分别处理权重和偏置
        if "Wqkv" in new_key:
            if "weight" in new_key:
                # 获取权重数据，并重塑其形状以匹配模型配置
                weight = original_weights[new_key]
                weights_shape = weight.shape
                weight = (
                    weight.view(3, config.num_attention_heads, -1, config.hidden_size)
                    .transpose(0, 1)
                    .reshape(*weights_shape)
                )
                original_weights[new_key] = weight
            elif "bias" in new_key:
                # 获取偏置数据，并重塑其形状以匹配模型配置
                bias = original_weights[new_key]
                bias_shape = bias.shape
                bias = bias.view(3, config.num_attention_heads, -1).transpose(0, 1).reshape(*bias_shape)
                original_weights[new_key] = bias

        # 根据映射字典mapping，替换新键中的字符串
        for k, v in mapping.items():
            if k in new_key:
                new_key = new_key.replace(k, v)

        # 将处理后的原始权重放入转换后的权重字典中，并从原始权重字典中移除原始键
        converted_weights[new_key] = original_weights.pop(original_weights_key)

    # 返回转换后的权重字典
    return converted_weights
# 根据给定的URL生成存储库ID，格式为 "<owner>/<repository>"
repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
# 根据URL提取文件名
filename = f"{url.split('/')[-1]}"
# 使用Hugging Face Hub下载指定的资源
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    force_filename=root,  # 指定下载文件的目标根目录
    local_dir_use_symlinks=False,  # 禁用符号链接，将文件直接复制到目标位置
)


def convert_phi_weights(
    model_name, checkpoint_path, pytorch_dump_folder_path, use_cuda, save_weights_directly, _MODELS
):
    # 如果指定的模型名在_MODELS字典中，则只保留该模型名的条目，否则保持_MODELS不变
    _MODELS = _MODELS if model_name not in _MODELS.keys() else {model_name: _MODELS.get(model_name)}
    # 检测CUDA是否可用，选择相应的设备（GPU或CPU）
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    for model_name, model_url in _MODELS.items():
        converted_checkpoint = {}
        model_checkpoint = {}

        # 针对 phi-2，权重存储在两个不同的 safetensors 文件中，因此需要逐个迭代下载
        for model_each_url in model_url:
            # 构建模型路径，包括模型名称和最后一个文件名部分
            model_path = os.path.join(checkpoint_path, model_name + "_" + model_each_url.split("/")[-1])
            # 如果模型路径不存在，下载模型文件
            if not os.path.exists(model_path):
                print(f"\n{model_name} was not found! Downloading it to {model_path}")
                _download(url=model_each_url, root=model_path)

            # 如果模型路径以 "safetensors" 结尾，使用 safetensors.torch 加载文件到指定设备
            if model_path.endswith("safetensors"):
                loaded_weights = safetensors.torch.load_file(model_path, device=device)
            else:
                # 否则使用 torch 加载模型文件到指定设备
                loaded_weights = torch.load(model_path, map_location=device)
            model_checkpoint.update(**loaded_weights)

        # 解析模型类型，例如 phi-1 或 phi-1_5 或 phi-2
        model_type = model_name.split("/")[1]

        # 初始化 Phi 模型的配置
        config = PhiConfig()
        # 如果是 phi-2 模型，则更新配置参数
        if model_type == "phi-2":
            config.hidden_size = 2560
            config.intermediate_size = 10240
            config.num_hidden_layers = 32
            config.resid_pdrop = 0.1
            config.partial_rotary_factor = 0.4
            config.torch_dtype = "float16"

        # 转换权重
        converted_checkpoint.update(**convert_weights(model_checkpoint, PHI_MAPPING, config))

        # 根据选择保存整个模型权重还是转换后的权重
        if save_weights_directly:
            # 构建保存权重路径并保存权重文件
            save_weights_path = os.path.join(pytorch_dump_folder_path, model_type + "_pytorch_model.bin")
            torch.save(converted_checkpoint, save_weights_path)
            print(f"Model weights saved at {save_weights_path}!")
        else:
            # 创建 PhiForCausalLM 模型实例，并加载转换后的权重
            model = PhiForCausalLM(config).to(device)
            model.load_state_dict(converted_checkpoint, strict=True)
            # 构建保存模型路径并保存模型
            save_model_path = os.path.join(pytorch_dump_folder_path, model_type)
            model.save_pretrained(save_model_path)
            print(f"Model saved at {save_model_path}!")

            # 释放模型相关的 GPU 内存（如果使用了 CUDA）
            del config, model

        # 释放模型检查点和转换后的检查点占用的内存
        del model_checkpoint, converted_checkpoint
        # 如果使用 CUDA，则清空 GPU 缓存
        if use_cuda:
            torch.cuda.empty_cache()
        # 执行 Python 垃圾回收
        gc.collect()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        type=str,
        help="要转换的模型名称。请选择其中之一：phi-1, phi-1_5, phi-2。如果未提供，则转换所有模型。",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="已下载检查点文件夹的路径。（请输入完整路径）"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="PyTorch 模型输出路径。（请输入完整路径）",
    )
    parser.add_argument(
        "--use_cuda",
        default=False,
        type=bool,
        help="在转换过程中是否将权重加载到 GPU 上，默认为 False",
    )
    parser.add_argument(
        "--save_weights_directly",
        default=True,
        type=bool,
        help="是否直接保存转换后的权重，或者将权重加载到 Phi 模型中再保存。默认为 True",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数，传入参数
    convert_phi_weights(
        args.model_name,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.use_cuda,
        args.save_weights_directly,
        _MODELS,  # `_MODELS` 是一个未在提供代码中定义的变量，可能是全局变量或者导入的模块中的变量
    )
```