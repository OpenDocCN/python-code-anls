# `.\models\llava\convert_llava_weights_to_hf.py`

```py
# 导入必要的模块和库
import argparse  # 导入命令行参数解析模块

import torch  # 导入PyTorch库
from huggingface_hub import hf_hub_download  # 从HuggingFace Hub下载模块

from transformers import (  # 导入transformers库中的多个类和函数
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

# 示例代码的额外说明文本
EPILOG_TXT = """Example:
    python transformers/src/transformers/models/llava/convert_llava_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/llava-v1.5-7b-conv --old_state_dict_id liuhaotian/llava-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/llava-v1.5-7b/model_state_dict.bin")
"""

# 用于修改state_dict中键名的映射关系
KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",  # 将"model.vision_tower."替换为""
    "model.mm_projector": "multi_modal_projector",  # 将"model.mm_projector"替换为"multi_modal_projector"
    "model": "model.model",  # 将"model"替换为"model.model"
    "vision_model.model": "vision_model",  # 将"vision_model.model"替换为"vision_model"
    "lm_head": "language_model.lm_head",  # 将"lm_head"替换为"language_model.lm_head"
    "model.model": "language_model.model",  # 将"model.model"替换为"language_model.model"
    "multi_modal_projector.0": "multi_modal_projector.linear_1",  # 将"multi_modal_projector.0"替换为"multi_modal_projector.linear_1"
    "multi_modal_projector.2": "multi_modal_projector.linear_2",  # 将"multi_modal_projector.2"替换为"multi_modal_projector.linear_2"
}

# 将state_dict转换为适用于Hugging Face模型的格式
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):  # 忽略以".inv_freq"结尾的键
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict

# 将llava_llama模型转换为适用于Hugging Face模型的格式
def convert_llava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)  # 设置PyTorch默认数据类型为float16

    text_config = AutoConfig.from_pretrained(text_model_id)  # 从预训练模型ID加载文本配置
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)  # 从预训练模型ID加载tokenizer
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)  # 添加特殊token"<image>"
    tokenizer.add_special_tokens({"pad_token": "<pad>"})  # 添加特殊token"<pad>"

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)  # 从预训练模型ID加载图像处理器
    # 创建一个 LlavaProcessor 实例，使用给定的 tokenizer 和 image_processor
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # 创建一个 LlavaConfig 实例，设置 text_config，并将 pad_token_id 设置为 32001
    config = LlavaConfig(text_config=text_config)
    config.pad_token_id = 32001

    # 使用 "meta" 设备创建 LlavaForConditionalGeneration 模型实例
    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    # 为了性能考虑，将 pad 形状设为 64
    pad_shape = 64

    # 下载旧的模型状态字典并加载到 state_dict_path
    state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict.bin")

    # 使用 torch 加载状态字典，并将其转换为适合 Hugging Face 模型的格式
    state_dict = torch.load(state_dict_path, map_location="cpu")
    state_dict = convert_state_dict_to_hf(state_dict)

    # 将加载的状态字典加载到模型中，strict=True 表示严格匹配模型参数
    model.load_state_dict(state_dict, strict=True, assign=True)

    # 获取模型的预扩展嵌入
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data

    # 计算嵌入的均值 mu 和标准差 sigma
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n

    # 使用 mu 和 sigma 创建多变量正态分布
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # 增加一个图像标记以重新调整模型的标记嵌入
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)

    # 用从多变量正态分布中采样得到的值来填充模型的标记嵌入权重
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )

    # 用从多变量正态分布中采样得到的值来填充模型语言模型头部的权重
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )

    # 将模型推送到 Hugging Face Hub 上的输出路径
    model.push_to_hub(output_hub_path)

    # 将处理器推送到 Hugging Face Hub 上的输出路径
    processor.push_to_hub(output_hub_path)
# 主程序入口函数，用于处理命令行参数并调用转换函数
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,  # 添加自定义的结尾文本
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 使用原始描述帮助格式
    )
    # 添加命令行参数：--text_model_id，用于指定文本模型的 Hub 地址
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    # 添加命令行参数：--vision_model_id，用于指定视觉模型的 Hub 地址
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    # 添加命令行参数：--output_hub_path，用于指定转换模型输出的 Hub 路径
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    # 添加命令行参数：--old_state_dict_id，用于指定原始模型状态字典的 Hub 地址
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，将 llava_llama 转换为 HF 模型
    convert_llava_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id)


# 如果该脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```