# `.\transformers\models\vipllava\convert_vipllava_weights_to_hf.py`

```
# 导入必要的模块
import argparse  # 解析命令行参数

import torch  # PyTorch 库
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型

from transformers import (  # 从 Transformers 库中导入以下模块和类
    AddedToken,  # 用于添加特殊标记的类
    AutoConfig,  # 自动配置类
    AutoTokenizer,  # 自动标记化器类
    CLIPImageProcessor,  # CLIP 图像处理器类
    LlavaProcessor,  # Llava 处理器类
    VipLlavaConfig,  # VipLlava 配置类
    VipLlavaForConditionalGeneration,  # 用于条件生成的 VipLlava 类
)

# 需要修改的键值对映射，用于转换模型权重中的键名
KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",  # 将 "model.vision_tower." 替换为空字符串
    "model.mm_projector": "multi_modal_projector",  # 将 "model.mm_projector" 替换为 "multi_modal_projector"
    "model": "model.model",  # 将 "model" 替换为 "model.model"
    "vision_model.model": "vision_model",  # 将 "vision_model.model" 替换为 "vision_model"
    "lm_head": "language_model.lm_head",  # 将 "lm_head" 替换为 "language_model.lm_head"
    "model.model": "language_model.model",  # 将 "model.model" 替换为 "language_model.model"
    "multi_modal_projector.0": "multi_modal_projector.linear_1",  # 将 "multi_modal_projector.0" 替换为 "multi_modal_projector.linear_1"
    "multi_modal_projector.2": "multi_modal_projector.linear_2",  # 将 "multi_modal_projector.2" 替换为 "multi_modal_projector.linear_2"
    "final_linear.0": "linear_1",  # 将 "final_linear.0" 替换为 "linear_1"
    "final_linear.2": "linear_2",  # 将 "final_linear.2" 替换为 "linear_2"
    "multi_modal_projector.clip_layernorm": "multi_modal_projector.projector_layernorm",  # 将 "multi_modal_projector.clip_layernorm" 替换为 "multi_modal_projector.projector_layernorm"
}


# 从 llava 转换模型权重到 Hugging Face 格式的函数
def convert_state_dict_to_hf(state_dict):
    # 创建一个新的状态字典
    new_state_dict = {}
    # 遍历原状态字典的键值对
    for key, value in state_dict.items():
        # 遍历需要修改的键值对映射
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            # 如果需要修改的键在当前键中
            if key_to_modify in key:
                # 将当前键名中的需要修改的部分替换为新的键名
                key = key.replace(key_to_modify, new_key)
        # 将修改后的键值对添加到新的状态字典中
        new_state_dict[key] = value
    # 返回转换后的状态字典
    return new_state_dict


# 将 VipLlava Llama 模型转换为 Hugging Face 格式的函数
def convert_vipllava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    # 设置默认的张量数据类型为 float16
    torch.set_default_dtype(torch.float16)
    # 从预训练模型 ID 加载文本配置
    text_config = AutoConfig.from_pretrained(text_model_id)

    # 从预训练模型 ID 加载标记化器
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    # 添加特殊标记 "<image>"
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False))
    # 添加特殊标记 "<pad>"
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # 从预训练模型 ID 加载 CLIP 图像处理器
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)

    # 创建 Llava 处理器
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # 创建 VipLlava 配置
    config = VipLlavaConfig(text_config=text_config)
    # 设置填充标记的 ID
    config.pad_token_id = 32001

    # 在 "meta" 设备上创建 VipLlavaForConditionalGeneration 模型
    with torch.device("meta"):
        model = VipLlavaForConditionalGeneration(config)

    # 为了性能考虑，填充到 64
    pad_shape = 64

    # 下载旧的状态字典
    state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict_7b.bin")
    # 加载旧的状态字典
    state_dict = torch.load(state_dict_path, map_location="cpu")
    # 转换状态字典到 Hugging Face 格式
    state_dict = convert_state_dict_to_hf(state_dict)
    # 加载转换后的状态字典到模型
    model.load_state_dict(state_dict, strict=True, assign=True)
    # 获取模型中的词嵌入权重
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    # 计算词嵌入的平均值
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    # 获取词嵌入的数量
    n = pre_expansion_embeddings.size()[0]
    # 计算词嵌入的协方差矩阵
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    # 创建多元正态分布对象
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # 添加一个图像令牌以调整模型大小
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    # 对词嵌入中的特定索引范围进行采样，并替换为采样结果
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    # 对语言模型头部中的特定索引范围进行采样，并替换为采样结果
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )
    # 更新模型配置的词汇量大小
    model.config.vocab_size = model.config.vocab_size + pad_shape
    # 更新文本配置的词汇量大小
    model.config.text_config.vocab_size = model.config.text_config.vocab_size + pad_shape

    # 将模型推送到 Hub
    model.push_to_hub(output_hub_path)
    # 将处理器推送到 Hub
    processor.push_to_hub(output_hub_path)
# 主函数，用于解析命令行参数并调用转换函数
def main():
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加文本模型的 Hub 位置参数
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    # 添加视觉模型的 Hub 位置参数
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    # 添加转换后模型的 Hub 位置参数
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    # 添加原始模型的状态字典的 Hub 位置参数
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入参数
    convert_vipllava_llama_to_hf(
        args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id
    )

# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```