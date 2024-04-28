# `.\models\donut\convert_donut_to_pytorch.py`

```
# 指定脚本的编码格式为 UTF-8

# 导入必要的模块
import argparse  # 用于解析命令行参数
import torch  # PyTorch 深度学习框架
from datasets import load_dataset  # 从 Hugging Face datasets 库加载数据集
from donut import DonutModel  # 导入 DonutModel 模型
from transformers import (  # 导入 Transformers 库中的模型和配置
    DonutImageProcessor,  # Donut 图像处理器
    DonutProcessor,  # Donut 处理器
    DonutSwinConfig,  # Donut Swin 模型的配置
    DonutSwinModel,  # Donut Swin 模型
    MBartConfig,  # MBart 模型的配置
    MBartForCausalLM,  # MBart 用于因果语言建模
    VisionEncoderDecoderModel,  # 视觉编码器-解码器模型
    XLMRobertaTokenizerFast,  # XLM-Roberta 快速分词器
)


# 定义一个函数，用于从给定模型获取 Donut 和 MBart 的配置
def get_configs(model):
    # 获取原始模型的配置
    original_config = model.config

    # 创建 DonutSwin 模型的配置
    encoder_config = DonutSwinConfig(
        image_size=original_config.input_size,  # 图像尺寸
        patch_size=4,  # 补丁尺寸
        depths=original_config.encoder_layer,  # 编码器层数
        num_heads=[4, 8, 16, 32],  # 注意力头的数量
        window_size=original_config.window_size,  # 窗口大小
        embed_dim=128,  # 嵌入维度
    )
    # 创建 MBart 模型的配置
    decoder_config = MBartConfig(
        is_decoder=True,  # 是否为解码器
        is_encoder_decoder=False,  # 是否为编码器-解码器模型
        add_cross_attention=True,  # 是否添加交叉注意力
        decoder_layers=original_config.decoder_layer,  # 解码器层数
        max_position_embeddings=original_config.max_position_embeddings,  # 最大位置嵌入
        vocab_size=len(
            model.decoder.tokenizer
        ),  # 词汇表大小，注意有一些特殊的令牌被添加到 XLMRobertaTokenizer 的词汇表中
        scale_embedding=True,  # 是否缩放嵌入
        add_final_layer_norm=True,  # 是否添加最终层归一化
    )

    return encoder_config, decoder_config  # 返回编码器和解码器的配置


# 定义一个函数，用于重命名模型中的键名
def rename_key(name):
    # 如果键名中包含 'encoder.model'，则替换为 'encoder'
    if "encoder.model" in name:
        name = name.replace("encoder.model", "encoder")
    # 如果键名中包含 'decoder.model'，则替换为 'decoder'
    if "decoder.model" in name:
        name = name.replace("decoder.model", "decoder")
    # 如果键名中包含 'patch_embed.proj'，则替换为 'embeddings.patch_embeddings.projection'
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # 如果键名中包含 'patch_embed.norm'，则替换为 'embeddings.norm'
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    # 检查文件名是否以"encoder"开头
    if name.startswith("encoder"):
        # 如果文件名中包含"layers"，则添加前缀"encoder."
        if "layers" in name:
            name = "encoder." + name
        # 如果文件名中包含"attn.proj"，则将其替换为"attention.output.dense"
        if "attn.proj" in name:
            name = name.replace("attn.proj", "attention.output.dense")
        # 如果文件名中包含"attn"且不包含"mask"，则将其替换为"attention.self"
        if "attn" in name and "mask" not in name:
            name = name.replace("attn", "attention.self")
        # 如果文件名中包含"norm1"，则将其替换为"layernorm_before"
        if "norm1" in name:
            name = name.replace("norm1", "layernorm_before")
        # 如果文件名中包含"norm2"，则将其替换为"layernorm_after"
        if "norm2" in name:
            name = name.replace("norm2", "layernorm_after")
        # 如果文件名中包含"mlp.fc1"，则将其替换为"intermediate.dense"
        if "mlp.fc1" in name:
            name = name.replace("mlp.fc1", "intermediate.dense")
        # 如果文件名中包含"mlp.fc2"，则将其替换为"output.dense"
        if "mlp.fc2" in name:
            name = name.replace("mlp.fc2", "output.dense")

        # 如果文件名为"encoder.norm.weight"，则将其替换为"encoder.layernorm.weight"
        if name == "encoder.norm.weight":
            name = "encoder.layernorm.weight"
        # 如果文件名为"encoder.norm.bias"，则将其替换为"encoder.layernorm.bias"
        if name == "encoder.norm.bias":
            name = "encoder.layernorm.bias"

    # 返回修改后的文件名
    return name
# 将原始模型的状态字典转换为新模型的状态字典
def convert_state_dict(orig_state_dict, model):
    # 遍历原始状态字典的副本中的所有键
    for key in orig_state_dict.copy().keys():
        # 移除原始状态字典中的键，并获取对应的值
        val = orig_state_dict.pop(key)

        # 检查键中是否包含"qkv"
        if "qkv" in key:
            # 拆分键，并提取层号和块号
            key_split = key.split(".")
            layer_num = int(key_split[3])
            block_num = int(key_split[5])
            dim = model.encoder.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 检查是否为权重参数
            if "weight" in key:
                # 更新新状态字典中的权重参数
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新新状态字典中的偏置参数
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"encoder.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        elif "attn_mask" in key or key in ["encoder.model.norm.weight", "encoder.model.norm.bias"]:
            # HuggingFace 实现不使用 attn_mask 缓冲区
            # 并且模型不使用编码器的最终 LayerNorm
            pass
        else:
            # 更新新状态字典中的其他参数
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# 将 Donut 模型检查点转换为适用于 HuggingFace 的格式
def convert_donut_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    # 加载原始模型
    original_model = DonutModel.from_pretrained(model_name).eval()

    # 加载 HuggingFace 模型
    encoder_config, decoder_config = get_configs(original_model)
    encoder = DonutSwinModel(encoder_config)
    decoder = MBartForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # 获取原始模型的状态字典
    state_dict = original_model.state_dict()
    # 转换原始模型的状态字典
    new_state_dict = convert_state_dict(state_dict, model)
    # 加载新的状态字典到模型中
    model.load_state_dict(new_state_dict)

    # 在扫描文档上验证结果
    dataset = load_dataset("hf-internal-testing/example-documents")
    image = dataset["test"][0]["image"].convert("RGB")

    # 加载模型使用的 tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name, from_slow=True)
    # 创建 Donut 图像处理器
    image_processor = DonutImageProcessor(
        do_align_long_axis=original_model.config.align_long_axis, size=original_model.config.input_size[::-1]
    )
    processor = DonutProcessor(image_processor, tokenizer)
    # 处理图像并获取像素值
    pixel_values = processor(image, return_tensors="pt").pixel_values
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-docvqa"
    if model_name == "naver-clova-ix/donut-base-finetuned-docvqa":
        # 如果是上述模型名称，则设置任务提示为特定格式的字符串，包含用户输入的问题
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        # 设置一个示例问题
        question = "When is the coffee break?"
        # 将用户输入的问题填充到任务提示中
        task_prompt = task_prompt.replace("{user_input}", question)
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-rvlcdip"
    elif model_name == "naver-clova-ix/donut-base-finetuned-rvlcdip":
        # 如果是上述模型名称，则设置任务提示为特定格式的字符串
        task_prompt = "<s_rvlcdip>"
    # 检查模型名称是否在给定的模型列表中
    elif model_name in [
        "naver-clova-ix/donut-base-finetuned-cord-v1",
        "naver-clova-ix/donut-base-finetuned-cord-v1-2560",
    ]:
        # 如果是列表中的某个模型名称，则设置任务提示为特定格式的字符串
        task_prompt = "<s_cord>"
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-cord-v2"
    elif model_name == "naver-clova-ix/donut-base-finetuned-cord-v2":
        # 如果是上述模型名称，则设置任务提示为特定格式的字符串
        task_prompt = "s_cord-v2>"
    # 检查模型名称是否为 "naver-clova-ix/donut-base-finetuned-zhtrainticket"
    elif model_name == "naver-clova-ix/donut-base-finetuned-zhtrainticket":
        # 如果是上述模型名称，则设置任务提示为特定格式的字符串
        task_prompt = "<s_zhtrainticket>"
    # 检查模型名称是否在给定的模型列表中
    elif model_name in ["naver-clova-ix/donut-proto", "naver-clova-ix/donut-base"]:
        # 如果是列表中的某个模型名称，则设置任务提示为固定字符串
        # 这里使用了一个随机的字符串作为示例
        task_prompt = "hello world"
    else:
        # 如果模型名称不在支持的列表中，则抛出异常
        raise ValueError("Model name not supported")

    # 使用原始模型的解码器对任务提示进行编码，返回输入的 token IDs
    prompt_tensors = original_model.decoder.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]

    # 使用原始模型的编码器对像素值进行编码，获取原始的 patch embeddings
    original_patch_embed = original_model.encoder.model.patch_embed(pixel_values)
    # 使用新模型的编码器对像素值进行编码，获取新的 patch embeddings
    patch_embeddings, _ = model.encoder.embeddings(pixel_values)
    # 检查两个 patch embeddings 是否在一定的误差范围内相等
    assert torch.allclose(original_patch_embed, patch_embeddings, atol=1e-3)

    # 验证编码器的隐藏状态是否一致
    # 使用原始模型的编码器获取最后的隐藏状态
    original_last_hidden_state = original_model.encoder(pixel_values)
    # 使用新模型的编码器获取最后的隐藏状态
    last_hidden_state = model.encoder(pixel_values).last_hidden_state
    # 检查两个隐藏状态是否在一定的误差范围内相等
    assert torch.allclose(original_last_hidden_state, last_hidden_state, atol=1e-2)

    # 验证解码器的隐藏状态是否一致
    # 使用原始模型生成原始的 logits
    original_logits = original_model(pixel_values, prompt_tensors, None).logits
    # 使用新模型生成新的 logits
    logits = model(pixel_values, decoder_input_ids=prompt_tensors).logits
    # 检查两个 logits 是否在一定的误差范围内相等
    assert torch.allclose(original_logits, logits, atol=1e-3)
    # 打印验证成功信息
    print("Looks ok!")

    # 如果提供了 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        # 打印保存模型和处理器的信息
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 推送模型到 Hub，使用模型名称的一部分作为路径
        model.push_to_hub("nielsr/" + model_name.split("/")[-1], commit_message="Update model")
        # 推送处理器到 Hub，使用模型名称的一部分作为路径
        processor.push_to_hub("nielsr/" + model_name.split("/")[-1], commit_message="Update model")
# 如果代码被直接运行而不是被引入作为模块，以下内容会被执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加必需的参数：模型名称，默认数值、类型和帮助信息
    parser.add_argument(
        "--model_name",
        default="naver-clova-ix/donut-base-finetuned-docvqa",
        required=False,
        type=str,
        help="Name of the original model you'd like to convert.",
    )
    
    # 添加必需的参数：PyTorch模型输出目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        required=False,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    
    # 添加参数：是否将转换后的模型和处理器推送到🤗 hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数，将Donut模型检查点转换为PyTorch模型，并可选择推送到hub
    convert_donut_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```  
```