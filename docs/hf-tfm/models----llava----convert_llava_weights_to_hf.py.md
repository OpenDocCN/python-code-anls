# `.\transformers\models\llava\convert_llava_weights_to_hf.py`

```py
# 版权声明和许可信息

# 导入 argparse 库
import argparse

# 导入 torch 库
import torch
# 从 huggingface_hub 库中导入 hf_hub_download 函数
from huggingface_hub import hf_hub_download
# 从 transformers 库中导入 AutoConfig, AutoTokenizer, CLIPImageProcessor, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor
from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

# 修改的键值对映射
KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}

# 将状态字典转换为 Hugging Face 状态字典
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict

# 将 Llava 或 Llama 模型转换为 Hugging Face 模型
def convert_llava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    # 设置默认数据类型为 float16
    torch.set_default_dtype(torch.float16)
    # 从预训练模型加载文本配置
    text_config = AutoConfig.from_pretrained(text_model_id)

    # 加载文本模型的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    # 添加特殊 token "<image>"
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special=True)
    # 添加特殊 token "<pad>"
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # 从预训练模型加载图像处理器
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    # 创建 LlavaProcessor 对象
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # 创建 Llava 模型配置
    config = LlavaConfig(text_config=text_config)
    config.pad_token_id = 32001

    # 在 meta 设备上创建 LlavaForConditionalGeneration 模型
    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    # 为了提高性能，对齐到 64
    pad_shape = 64

    # 从 Hugging Face Hub 下载旧状态字典文件
    state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict.bin")
    # 加载状态字典并转换为 Hugging Face 状态字典
    state_dict = torch.load(state_dict_path, map_location="cpu")
    state_dict = convert_state_dict_to_hf(state_dict)
    # 加载模型状态字典
    model.load_state_dict(state_dict, strict=True, assign=True)

    # 计算语言模型嵌入的均值和协方差矩阵
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    # 创建一个多变量正态分布对象，传入均值和方差参数
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)
    
    # 添加一个图像标记以便调整模型大小
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    
    # 用多变量正态分布对象生成样本，替换模型的嵌入层权重
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    
    # 用多变量正态分布对象生成样本，替换模型语言模型头部的权重
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )
    
    # 调整模型配置的词汇表大小
    model.config.vocab_size = model.config.vocab_size + pad_shape
    model.config.text_config.vocab_size = model.config.text_config.vocab_size + pad_shape
    
    # 将模型推送到 Hub 上
    model.push_to_hub(output_hub_path)
    
    # 将处理器推送到 Hub 上
    processor.push_to_hub(output_hub_path)
# 主函数，程序的入口
def main():
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定文本模型的位置
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    # 添加命令行参数，用于指定视觉模型的位置
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    # 添加命令行参数，用于指定转换后模型的位置
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    # 添加命令行参数，用于指定原始模型的状态字典文件的位置
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将llava_llama模型转换为hf模型
    convert_llava_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id)


# 如果程序作为脚本直接执行，则调用主函数
if __name__ == "__main__":
    main()
```