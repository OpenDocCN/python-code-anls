# `.\models\data2vec\convert_data2vec_audio_original_pytorch_checkpoint_to_pytorch.py`

```
# 文件编码声明
# 版权声明
# 基于Apache许可的版权声明
# 如果适用的话，根据适用法律或写的约定发布软件
#在按照"原样"基础上分发软件时
#没有任何形式的保证或者条件，无论是明示的或者默示的
#查看特定语言的许可证和
#许可证下的限制
# Wav2Vec2检查点转换

# 导入必要的包
import argparse
import os
from functools import reduce
import fairseq
import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, logging
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig

# 从https://github.com/pytorch/fairseq/blob/main/examples/data2vec/models/data2vec_audio.py中拷贝的
from transformers.models.data2vec.data2vec_audio import Data2VecAudioModel as Dummy  # noqa: F401
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioForCTC, Data2VecAudioModel

# 设置日志级别和获取日志记录器
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 映射关系和顶级键
MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "models.0.layer_norm": "feature_projection.layer_norm",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}
TOP_LEVEL_KEYS = [
    "lm_head",
]

# 递归设置属性值
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 通过拆分键名逐级获取属性指针
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 根据权重类型设置属性值
    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # 校验形状是否匹配
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型设置属性数据
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        hf_pointer.data = value
    # 使用日志记录器输出初始化信息，如果weight_type不为空，则拼接key和weight_type，否则为空字符串
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载 Fairseq 模型的权重到 Hugging Face 模型中
def recursively_load_weights(fairseq_model, hf_model, is_headless):
    # 获取 Fairseq 模型的参数字典
    fairseq_dict = fairseq_model.state_dict()

    # 根据是否使用 headless 模式选择特征提取器和位置卷积嵌入层
    if not is_headless:
        feature_extractor = hf_model.data2vec_audio.feature_extractor
        pos_conv_embedding = hf_model.data2vec_audio.encoder.pos_conv_embed
    else:
        feature_extractor = hf_model.feature_extractor
        pos_conv_embedding = hf_model.encoder.pos_conv_embed

    # 遍历 Fairseq 模型的参数字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果参数名中包含 "conv_layers"，则加载卷积层参数
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
            )
            is_used = True
        # 如果参数名中包含 "pos_conv"，则加载位置卷积层参数
        elif "pos_conv" in name:
            load_pos_conv_layer(
                name,
                value,
                pos_conv_embedding,
                unused_weights,
            )
            is_used = True
        else:
            # 遍历 MAPPING 字典，将 Fairseq 参数名映射到 Hugging Face 参数名
            for key, mapped_key in MAPPING.items():
                if not is_headless:
                    mapped_key = "data2vec_audio." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果映射的 Hugging Face 参数名中包含通配符 "*"，则替换为对应的层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据参数名中的信息确定参数类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: 不要匹配 quantizer.weight_proj
                        weight_type = "weight"
                    else:
                        weight_type = None
                    # 递归设置 Hugging Face 模型的参数值
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果参数未被使用，则将其加入未使用参数列表
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的参数
    logger.warning(f"Unused weights: {unused_weights}")


# 根据字符串路径访问模块的属性
def access_by_string(module, path):
    names = path.split(".")
    return reduce(getattr, names, module)


# 设置权重值
def set_weights(full_name, module, fsq_value, hf_weight_path):
    # 获取 Hugging Face 模型的参数值
    hf_weight = access_by_string(module, hf_weight_path)
    hf_value = hf_weight.data

    # 检查 Fairseq 参数值和 Hugging Face 参数值的形状是否一致
    if fsq_value.shape != hf_value.shape:
        raise ValueError(f"{full_name} has size {fsq_value.shape}, but {hf_value.shape} was found.")
    # 设置 Hugging Face 模型的参数值
    hf_weight.data = fsq_value
    logger.info(f"{full_name} was correctly initialized from {hf_weight_path}.")


# 加载卷积层参数
def load_conv_layer(full_name, value, feature_extractor, unused_weights):
    # 获取卷积层的名称和索引
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 获取参数类型
    weight_type = name.split(".")[-1]
    # 如果 type_id 等于 0，则将 layer_type 设置为 "conv"
    if type_id == 0:
        layer_type = "conv"
    # 如果 type_id 等于 2，则将 layer_type 设置为 "layer_norm"
    elif type_id == 2:
        layer_type = "layer_norm"
    # 如果 type_id 不等于 0 或 2，则将 full_name 添加到未使用权重列表中，然后返回
    else:
        unused_weights.append(full_name)
        return
    
    # 调用 set_weights 函数，设置权重值
    # 参数包括 full_name，feature_extractor，value，以及一个字符串，用于描述权重类型
    set_weights(full_name, feature_extractor, value, f"conv_layers.{layer_id}.{layer_type}.{weight_type}")
def load_pos_conv_layer(full_name, value, pos_conv_embeddings, unused_weights):
    # 从完整名称中获取层名称
    name = full_name.split("pos_conv.")[-1]
    # 将层名称分割为元素列表
    items = name.split(".")
    # 提取层 ID 和类型 ID
    layer_id = int(items[0])
    type_id = int(items[1])

    # 提取权重类型
    weight_type = name.split(".")[-1]
    # 如果类型 ID 不为0，则将完整名称添加到未使用权重列表并返回
    if type_id != 0:
        unused_weights.append(full_name)
        return
    else:
        layer_type = "conv"

    # 使用设置权重函数设置权重
    set_weights(full_name, pos_conv_embeddings, value, f"layers.{layer_id}.{layer_type}.{weight_type}")


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    将模型的权重复制/粘贴/调整到 transformer 设计中。
    """
    # 如果存在配置路径，则从预训练中加载 Data2VecAudioConfig
    if config_path is not None:
        config = Data2VecAudioConfig.from_pretrained(config_path)
    else:
        config = Data2VecAudioConfig()

    # 如果没有进行微调
    if not is_finetuned:
        # 修改 final_proj 层名称
        hf_wav2vec = Data2VecAudioModel(config)
        data2vec_checkpoint_dir = os.path.dirname(checkpoint_path)

        state_dict = torch.load(checkpoint_path)
        state_dict["model"]["final_proj.weight"] = state_dict["model"].pop("final_proj.0.weight")
        state_dict["model"]["final_proj.bias"] = state_dict["model"].pop("final_proj.0.bias")
        converted_ckpt = os.path.join(data2vec_checkpoint_dir, "converted.pt")
        torch.save(state_dict, converted_ckpt)
    else:
        hf_wav2vec = Data2VecAudioForCTC(config)
        converted_ckpt = checkpoint_path

    # 加载 Data2Vec 模型
    def load_data2vec(path):
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        return model[0].eval()

    model = load_data2vec(converted_ckpt)

    # 递归加载权重
    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    # 从预训练中加载 Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")

    # 加载数据集
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    input_audio = [x["array"] for x in ds[:4]["audio"]]

    # 对输入进行预处理
    inputs = processor(input_audio, return_tensors="pt", padding=True)

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask
    
    # 将输入进行编码并获取预测结果
    hf_wav2vec.eval()
    model.eval()
    if is_finetuned:
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)["encoder_out"].transpose(0, 1)
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["logits"]

        pred_ids = torch.argmax(our_output, dim=-1)
        output_string = processor.batch_decode(pred_ids)

        print(f"Expected Output: {ds[:4]['text']}, Pred: {output_string}")
    # 如果不是 finetuned 模型，则执行以下操作
    else:
        # 使用模型进行推理，获取它们的输出
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)[
            "layer_results"
        ][-1][0].transpose(0, 1)
        # 使用 hf_wav2vec 函数获取输出
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["last_hidden_state"]

    # 打印我们的输出和它们的输出的形状
    print(our_output.shape, their_output.shape)
    # 计算输出之间的最大绝对差异
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    # 打印最大绝对差异的值
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    # 检查两个模型的输出是否非常接近
    success = torch.allclose(our_output, their_output, atol=1e-3)
    # 打印是否两个模型的输出完全相同
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    # 如果两个模型的输出不相同，则抛出异常
    if not success:
        raise Exception("Something went wRoNg")

    # 保存 hf_wav2vec 模型到指定路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)

    # 如果是 finetuned 模型，则保存 processor 到指定路径
    if is_finetuned:
        processor.save_pretrained(pytorch_dump_folder_path)
    # 如果不是 finetuned 模型，则保存 feature_extractor 到指定路径
    else:
        processor.feature_extractor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下操作
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数--pytorch_dump_folder_path，指定输出的PyTorch模型路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数--checkpoint_path，指定fairseq检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加命令行参数--dict_path，指定fine-tuned模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加命令行参数--config_path，指定要转换的模型的hf config.json路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加命令行参数--not_finetuned，指定要转换的模型是否是fine-tuned模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用convert_wav2vec2_checkpoint函数，将fairseq检查点转换为PyTorch模型
    convert_wav2vec2_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```