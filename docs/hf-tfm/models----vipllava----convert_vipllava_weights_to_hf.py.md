# `.\models\vipllava\convert_vipllava_weights_to_hf.py`

```
# 导入 argparse 库，用于处理命令行参数
import argparse

# 导入 torch 库
import torch
# 从 huggingface_hub 库导入 hf_hub_download 函数
from huggingface_hub import hf_hub_download

# 从 transformers 库导入以下类和函数
from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaProcessor,
    VipLlavaConfig,
    VipLlavaForConditionalGeneration,
)

# 定义一个字典映射，用于修改模型权重的键名
KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "final_linear.0": "linear_1",
    "final_linear.2": "linear_2",
    "multi_modal_projector.clip_layernorm": "multi_modal_projector.projector_layernorm",
}

# 定义函数，将旧版权重字典转换为适合 HF 的新版权重字典
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # 如果键以 ".inv_freq" 结尾，则跳过不处理
        if key.endswith(".inv_freq"):
            continue
        # 遍历键名映射字典，替换对应的键名
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        # 更新新版权重字典
        new_state_dict[key] = value
    return new_state_dict


# 定义函数，将 vipllava_llama 模型转换为适合 HF 的模型
def convert_vipllava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    # 设置默认张量数据类型为 float16
    torch.set_default_dtype(torch.float16)
    
    # 从预训练模型 ID 加载文本配置
    text_config = AutoConfig.from_pretrained(text_model_id)

    # 从预训练模型 ID 加载分词器，并添加特殊标记
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # 从预训练模型 ID 加载图像处理器
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    # 创建 LlavaProcessor 对象，用于处理文本和图像
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # 使用 VipLlavaConfig 配置对象，配置 ViPLlava 模型
    config = VipLlavaConfig(text_config=text_config)
    config.pad_token_id = 32001

    # 在 meta 设备上创建 VipLlavaForConditionalGeneration 模型
    with torch.device("meta"):
        model = VipLlavaForConditionalGeneration(config)

    # 为了提高性能，将输入填充至 64 的形状
    pad_shape = 64

    # 下载并加载旧版权重字典路径
    state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict_7b.bin")
    state_dict = torch.load(state_dict_path, map_location="cpu")  # 在 CPU 上加载权重字典
    state_dict = convert_state_dict_to_hf(state_dict)  # 转换权重字典为适合 HF 的格式
    # 使用给定的状态字典加载模型的状态，严格匹配模型参数，同时允许分配参数
    model.load_state_dict(state_dict, strict=True, assign=True)

    # 获取模型语言模型的词嵌入权重，用于计算均值和协方差
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    # 计算词嵌入的均值向量
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    # 获取词嵌入矩阵的行数，用于计算协方差
    n = pre_expansion_embeddings.size()[0]
    # 计算词嵌入矩阵的协方差矩阵
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    # 创建一个多元正态分布对象，以 mu 为均值，sigma 为协方差矩阵
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # 调整模型的词嵌入层，增加一个特殊的图像标记，以适应扩展后的词汇表大小和填充形状
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    # 为词嵌入矩阵的扩展部分生成样本，并替换模型中的词嵌入权重
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    # 为语言模型头部的权重矩阵的扩展部分生成样本，并替换模型中的权重
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )

    # 将模型推送到指定的 Hub 输出路径
    model.push_to_hub(output_hub_path)
    # 将处理器对象推送到指定的 Hub 输出路径
    processor.push_to_hub(output_hub_path)
# 主程序入口函数
def main():
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数：text_model_id，用于指定文本模型的 Hub 地址
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    
    # 添加命令行参数：vision_model_id，用于指定视觉模型的 Hub 地址
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    
    # 添加命令行参数：output_hub_path，用于指定转换后模型在 Hub 上的位置
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    
    # 添加命令行参数：old_state_dict_id，用于指定原始模型状态字典的 Hub 地址
    # 需要注意文件名应为 `model_state_dict.bin`
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数 convert_vipllava_llama_to_hf，将参数传递给该函数
    convert_vipllava_llama_to_hf(
        args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id
    )


# 如果当前脚本被直接执行，则调用主函数 main()
if __name__ == "__main__":
    main()
```