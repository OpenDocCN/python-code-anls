# `.\models\data2vec\convert_data2vec_audio_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8

# 版权声明及许可证信息
# Copyright 2021 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Wav2Vec2 checkpoint."""

# 导入必要的库
import argparse
import os
from functools import reduce

import fairseq  # 导入 fairseq 库
import torch  # 导入 PyTorch 库
from datasets import load_dataset  # 导入 load_dataset 函数

from transformers import Wav2Vec2Processor, logging  # 导入 Wav2Vec2Processor 和 logging
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig  # 导入 Data2VecAudioConfig

# 从 fairseq 库复制了 Data2VecAudioModel 别名为 Dummy，未使用的导入，故标记为 F401
# Copied from https://github.com/pytorch/fairseq/blob/main/examples/data2vec/models/data2vec_audio.py
from transformers.models.data2vec.data2vec_audio import Data2VecAudioModel as Dummy  # noqa: F401
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioForCTC, Data2VecAudioModel  # 导入相关的模型定义


logging.set_verbosity_info()  # 设置 logging 级别为 info
logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象

# 映射字典，用于映射模型中的参数名称
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


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    # 递归设置参数值的函数
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)  # 获取指定属性的值

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape  # 获取指定权重类型的形状信息
    else:
        hf_shape = hf_pointer.shape  # 获取对象的形状信息

    # 检查形状是否匹配，若不匹配则引发 ValueError
    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    # 根据权重类型设置参数值
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
    # 使用日志记录器对象输出信息，格式化字符串包含动态部分
    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")
# 递归加载 Fairseq 模型的权重到 Hugging Face 模型中
def recursively_load_weights(fairseq_model, hf_model, is_headless):
    # 存储未使用的权重列表
    unused_weights = []
    # 获取 Fairseq 模型的状态字典
    fairseq_dict = fairseq_model.state_dict()

    # 根据是否 headless 设置特征提取器和位置卷积嵌入器
    if not is_headless:
        feature_extractor = hf_model.data2vec_audio.feature_extractor
        pos_conv_embedding = hf_model.data2vec_audio.encoder.pos_conv_embed
    else:
        feature_extractor = hf_model.feature_extractor
        pos_conv_embedding = hf_model.encoder.pos_conv_embed

    # 遍历 Fairseq 模型的状态字典
    for name, value in fairseq_dict.items():
        is_used = False
        # 如果名称中包含 "conv_layers"，则加载卷积层权重
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
            )
            is_used = True
        # 如果名称中包含 "pos_conv"，则加载位置卷积层权重
        elif "pos_conv" in name:
            load_pos_conv_layer(
                name,
                value,
                pos_conv_embedding,
                unused_weights,
            )
            is_used = True
        else:
            # 否则，根据映射表 MAPPING 加载对应的权重
            for key, mapped_key in MAPPING.items():
                if not is_headless:
                    # 根据条件修改 mapped_key
                    mapped_key = "data2vec_audio." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    # 如果 mapped_key 中包含 "*", 则替换为层索引
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    # 根据名称确定权重类型
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: 不匹配 quantizer.weight_proj
                        weight_type = "weight"
                    else:
                        weight_type = None
                    # 递归设置权重到 Hugging Face 模型中
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        # 如果未使用，则将名称添加到未使用权重列表中
        if not is_used:
            unused_weights.append(name)

    # 记录未使用的权重
    logger.warning(f"Unused weights: {unused_weights}")


# 根据字符串路径访问模块中的对象
def access_by_string(module, path):
    names = path.split(".")
    return reduce(getattr, names, module)


# 设置权重到指定路径的函数
def set_weights(full_name, module, fsq_value, hf_weight_path):
    # 通过字符串路径获取 Hugging Face 模型中的权重
    hf_weight = access_by_string(module, hf_weight_path)
    hf_value = hf_weight.data

    # 检查 Fairseq 和 Hugging Face 模型的权重形状是否匹配
    if fsq_value.shape != hf_value.shape:
        raise ValueError(f"{full_name} has size {fsq_value.shape}, but {hf_value.shape} was found.")
    # 设置 Fairseq 模型的值到 Hugging Face 模型的权重中
    hf_weight.data = fsq_value
    # 记录权重初始化成功的信息
    logger.info(f"{full_name} was correctly initialized from {hf_weight_path}.")


# 加载卷积层权重的函数
def load_conv_layer(full_name, value, feature_extractor, unused_weights):
    # 获取卷积层名称
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    weight_type = name.split(".")[-1]
    # 如果 type_id 等于 0，则将 layer_type 设置为 "conv"
    if type_id == 0:
        layer_type = "conv"
    # 如果 type_id 等于 2，则将 layer_type 设置为 "layer_norm"
    elif type_id == 2:
        layer_type = "layer_norm"
    # 如果 type_id 不是 0 也不是 2，则将 full_name 添加到 unused_weights 列表中并返回
    else:
        unused_weights.append(full_name)
        return

    # 调用 set_weights 函数来设置权重，使用给定的 full_name、feature_extractor、value 和拼接的路径字符串
    set_weights(full_name, feature_extractor, value, f"conv_layers.{layer_id}.{layer_type}.{weight_type}")
def load_pos_conv_layer(full_name, value, pos_conv_embeddings, unused_weights):
    # 从完整名称中提取出layer_id和type_id
    name = full_name.split("pos_conv.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    # 提取权重类型
    weight_type = name.split(".")[-1]
    
    # 如果type_id不为0，则将full_name加入unused_weights列表并返回
    if type_id != 0:
        unused_weights.append(full_name)
        return
    else:
        layer_type = "conv"

    # 调用set_weights函数，设置权重
    set_weights(full_name, pos_conv_embeddings, value, f"layers.{layer_id}.{layer_type}.{weight_type}")


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    将模型的权重复制/粘贴/调整为transformers设计。
    """
    # 如果提供了config_path，则从预训练模型加载Data2VecAudioConfig
    if config_path is not None:
        config = Data2VecAudioConfig.from_pretrained(config_path)
    else:
        config = Data2VecAudioConfig()

    # 如果不是finetuned状态
    if not is_finetuned:
        # 修改final_proj层的名称
        hf_wav2vec = Data2VecAudioModel(config)
        data2vec_checkpoint_dir = os.path.dirname(checkpoint_path)

        # 加载原始checkpoint的状态字典
        state_dict = torch.load(checkpoint_path)
        # 调整final_proj层权重和偏置的命名
        state_dict["model"]["final_proj.weight"] = state_dict["model"].pop("final_proj.0.weight")
        state_dict["model"]["final_proj.bias"] = state_dict["model"].pop("final_proj.0.bias")
        # 保存转换后的checkpoint
        converted_ckpt = os.path.join(data2vec_checkpoint_dir, "converted.pt")
        torch.save(state_dict, converted_ckpt)
    else:
        # 加载finetuned状态的模型
        hf_wav2vec = Data2VecAudioForCTC(config)
        converted_ckpt = checkpoint_path

    # 定义函数，用于加载fairseq模型
    def load_data2vec(path):
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        return model[0].eval()

    # 加载转换后的模型
    model = load_data2vec(converted_ckpt)

    # 递归加载权重到hf_wav2vec模型中
    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    # 从预训练模型facebook/wav2vec2-large-lv60加载processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")

    # 加载LibriSpeech ASR的验证集数据集
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    input_audio = [x["array"] for x in ds[:4]["audio"]]

    # 使用processor对输入音频进行处理，返回inputs字典
    inputs = processor(input_audio, return_tensors="pt", padding=True)

    # 提取inputs中的input_values和attention_mask
    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    # 打印原始的input_values和attention_mask（已注释）
    # input_values = inputs.input_values[:, :-1]
    # attention_mask = inputs.attention_mask[:, :-1]

    # 设置hf_wav2vec和model为eval模式
    hf_wav2vec.eval()
    model.eval()

    # 如果是finetuned状态
    if is_finetuned:
        # 获取模型预测的输出和hf_wav2vec的输出
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)["encoder_out"].transpose(0, 1)
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["logits"]

        # 计算预测的标签id，并通过processor进行批量解码
        pred_ids = torch.argmax(our_output, dim=-1)
        output_string = processor.batch_decode(pred_ids)

        # 打印预期输出和模型预测的输出字符串
        print(f"Expected Output: {ds[:4]['text']}, Pred: {output_string}")
    # 如果条件为假，执行以下操作
    else:
        # 使用模型进行推理，获取输出张量
        their_output = model(
            source=input_values,  # 输入数据
            padding_mask=(1 - attention_mask),  # 填充掩码
            mask=False,  # 不使用遮罩
            features_only=True  # 仅返回特征结果
        )["layer_results"][-1][0].transpose(0, 1)
        
        # 使用hf_wav2vec模型获取输出张量
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["last_hidden_state"]

    # 打印我们的输出和他们的输出的形状
    print(our_output.shape, their_output.shape)
    
    # 计算两个张量之间的最大绝对差异
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 输出最大绝对差异，预期在1e-7左右
    
    # 检查两个模型的输出张量是否在给定的容差范围内接近
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")  # 打印两个模型是否输出相同的张量
    
    # 如果输出不接近，抛出异常
    if not success:
        raise Exception("Something went wRoNg")

    # 将hf_wav2vec模型保存到指定路径
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)

    # 如果模型已经微调，则保存processor；否则，保存特征提取器
    if is_finetuned:
        processor.save_pretrained(pytorch_dump_folder_path)
    else:
        processor.feature_extractor.save_pretrained(pytorch_dump_folder_path)
# 如果这个脚本被直接执行而不是被导入，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个参数：输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个参数：fairseq 检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    # 添加一个参数：微调模型的字典路径
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    # 添加一个参数：待转换模型的 HF（Hugging Face）配置文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 添加一个标志参数：指示待转换模型是否是经过微调的模型
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数 convert_wav2vec2_checkpoint，传递命令行参数以执行模型转换操作
    convert_wav2vec2_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
```