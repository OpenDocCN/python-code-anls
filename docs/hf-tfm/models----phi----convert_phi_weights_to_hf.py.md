# `.\transformers\models\phi\convert_phi_weights_to_hf.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据 Apache 许可证 2.0 进行许可
# 请在遵守许可证的情况下使用本文件
# 可以在以下链接获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可分发的软件都是基于“原样”分发的，没有任何担保或条件，无论是明示的还是暗示的。
# 查看特定语言版本的特定附带权限或限制的许可证内容
# 详见许可
"""
Weights conversion script for Phi

This script downloads both Phi-1 and Phi-1.5 checkpoints to "checkpoint_path" and then converts the weights to
HugfgingFace model's format and saves them in "pytorch_dump_folder_path".

Example : $python ./convert_phi_weights_to_hf.py --model_name "microsoft/phi-2" --pytorch_dump_folder ./dump_folder/ --checkpoint_path ./ckpt_path/
"""

# 导入必要的库
import argparse
import gc
import os

import safetensors
import torch
from huggingface_hub import hf_hub_download

# 导入 transformers 库中的 PhiConfig 和 PhiForCausalLM 类
from transformers import PhiConfig, PhiForCausalLM

# model的下载链接
_MODELS = {
    "microsoft/phi-1": ["https://huggingface.co/microsoft/phi-1/blob/main/pytorch_model.bin"],
    "microsoft/phi-1_5": ["https://huggingface.co/microsoft/phi-1_5/blob/main/pytorch_model.bin"],
    "microsoft/phi-2": [
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00001-of-00002.safetensors",
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00002-of-00002.safetensors",
    ],
}

# Phi 模型中权重的映射关系
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


# 将权重转换为 HF 模型的格式
def convert_weights(original_weights, mapping, config):
    converted_weights = {}
    # 原始权重的键值按字母顺序排序
    original_weights_keys = sorted(original_weights.keys())
```  
    # 遍历原始权重的所有键
    for original_weights_key in original_weights_keys:
        # 将新键初始化为原始权重的键
        new_key = original_weights_key

        # 如果新键中包含"rotary_emb"，则跳过后续操作
        if "rotary_emb" in new_key:
            continue

        # 如果新键中包含"Wqkv"，则执行以下操作
        if "Wqkv" in new_key:
            # 如果新键中包含"weight"，则对权重进行重塑操作
            if "weight" in new_key:
                # 获取原始权重
                weight = original_weights[new_key]
                # 获取权重形状
                weights_shape = weight.shape
                # 根据一定规则对权重进行重塑
                weight = (
                    weight.view(3, config.num_attention_heads, -1, config.hidden_size)
                    .transpose(0, 1)
                    .reshape(*weights_shape)
                )
                # 更新原始权重
                original_weights[new_key] = weight
            # 如果新键中包含"bias"，则对偏置进行重塑操作
            elif "bias" in new_key:
                # 获取原始偏置
                bias = original_weights[new_key]
                # 获取偏置形状
                bias_shape = bias.shape
                # 根据一定规则对偏置进行重塑
                bias = bias.view(3, config.num_attention_heads, -1).transpose(0, 1).reshape(*bias_shape)
                # 更新原始偏置
                original_weights[new_key] = bias

        # 根据映射关系对键进行替换操作
        for k, v in mapping.items():
            if k in new_key:
                new_key = new_key.replace(k, v)

        # 将处理后的权重添加到转换后的权重字典中，并将其从原始权重字典中移除
        converted_weights[new_key] = original_weights.pop(original_weights_key)

    # 返回转换后的权重字典
    return converted_weights
# 下载函数，从给定的 URL 下载文件到指定的目录
def _download(url: str, root: str):
    # 从 URL 中提取仓库 ID，格式为 "用户名/仓库名"
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    # 提取 URL 中的文件名
    filename = f"{url.split('/')[-1]}"
    # 使用 Hugging Face Hub 的下载函数下载文件
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        # 指定下载的文件名和保存的根目录
        force_filename=root,
        # 禁用本地目录使用符号链接
        local_dir_use_symlinks=False,
    )


# 转换 PHI 权重函数，将模型的权重从给定的格式转换为 PyTorch 格式
def convert_phi_weights(
    model_name, checkpoint_path, pytorch_dump_folder_path, use_cuda, save_weights_directly, _MODELS
):
    # 如果模型名不存在于已知模型字典中，则使用给定的模型名和对应的字典值
    _MODELS = _MODELS if model_name not in _MODELS.keys() else {model_name: _MODELS.get(model_name)}
    # 检测是否有可用的 CUDA 设备，若有则使用 CUDA 运算，否则使用 CPU
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    for model_name, model_url in _MODELS.items():
        # 创建一个空字典，用于存储已转换的检查点
        converted_checkpoint = {}
        # 创建一个空字典，用于存储原始的模型检查点
        model_checkpoint = {}

        # 对于 phi-2，权重分别存储在两个不同的 safetensors 文件中，因此需要迭代下载一个文件
        for model_each_url in model_url:
            # 模型路径由检查点路径和模型名称组成
            model_path = os.path.join(checkpoint_path, model_name + "_" + model_each_url.split("/")[-1])
            # 如果模型文件不存在，则下载模型文件
            if not os.path.exists(model_path):
                print(f"\n{model_name} was not found! Downloading it to {model_path}")
                _download(url=model_each_url, root=model_path)

            # 如果模型路径以 "safetensors" 结尾，则使用 safetensors 加载模型，并将设备设置为指定的设备
            if model_path.endswith("safetensors"):
                loaded_weights = safetensors.torch.load_file(model_path, device=device)
            # 否则使用 torch 加载模型，并将设备设置为指定的设备
            else:
                loaded_weights = torch.load(model_path, map_location=device)
            # 将加载的权重数据更新到原始模型检查点字典中
            model_checkpoint.update(**loaded_weights)

        # 根据模型名称设置模型类型（phi-1、phi-1.5 或 phi-2）
        model_type = model_name.split("/")[1]

        # 初始化 PhiConfig 配置对象用于 phi-1 和 phi-1.5
        config = PhiConfig()
        # 如果是 phi-2，则更新配置对象的属性
        if model_type == "phi-2":
            config.hidden_size = 2560  # 隐藏层大小
            config.intermediate_size = 10240  # 中间层大小
            config.num_hidden_layers = 32  # 隐藏层数
            config.resid_pdrop = 0.1  # 残差连接的概率
            config.partial_rotary_factor = 0.4  # 部分旋转因子
            config.num_hidden_layers = 32  # 隐藏层数
            config.torch_dtype = "float16"  # PyTorch 数据类型

        # 将模型检查点权重转换为指定的格式，并更新到 converted_checkpoint 字典中
        converted_checkpoint.update(**convert_weights(model_checkpoint, PHI_MAPPING, config))

        # 如果 save_weights_directly 为真，则直接保存模型的权重数据
        if save_weights_directly:
            # 保存模型权重数据的路径
            save_weights_path = os.path.join(pytorch_dump_folder_path, model_type + "_pytorch_model.bin")
            # 使用 torch 保存转换后的权重字典到指定路径
            torch.save(converted_checkpoint, save_weights_path)
            print(f"Model weights saved at {save_weights_path}!")
        # 否则，保存整个模型
        else:
            # 创建 PhiForCausalLM 模型，并将配置设置为指定的配置对象，将模型放置到指定的设备上
            model = PhiForCausalLM(config).to(device)
            # 加载转换后的权重数据到模型
            model.load_state_dict(converted_checkpoint, strict=True)
            # 模型保存路径
            save_model_path = os.path.join(pytorch_dump_folder_path, model_type)
            # 保存模型到指定的路径
            model.save_pretrained(save_model_path)
            print(f"Model saved at {save_model_path}!")

            # 释放第二个模型所使用的 GPU 内存，如果使用的是 cuda
            del config, model

        # 释放模型检查点和转换后的检查点所使用的 GPU 内存，如果使用的是 cuda
        del model_checkpoint, converted_checkpoint
        # 如果使用的是 cuda，则释放缓存的 GPU 内存
        if use_cuda:
            torch.cuda.empty_cache()
        # 执行垃圾回收
        gc.collect()
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to convert. (Please enter one of the following: phi-1, phi-1_5, phi-2). If nothing is provided, all models will be converted.",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the folder of downloaded checkpoints. (Please enter full path)"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model. (Please enter full path)",
    )
    parser.add_argument(
        "--use_cuda",
        default=False,
        type=bool,
        help="Whether to load the weights on GPU during conversion or not, False by default",
    )
    parser.add_argument(
        "--save_weights_directly",
        default=True,
        type=bool,
        help="Whether to save the weights directly after conversion or load the weight to the Phi model and then save "
        "the Phi model along with weights. True by default",
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数进行 Phi 模型的权重转换
    convert_phi_weights(
        args.model_name,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.use_cuda,
        args.save_weights_directly,
        _MODELS,
    )
```