# `.\transformers\models\vivit\convert_vivit_flax_to_pytorch.py`

```py
# 设置编码格式为 utf-8
# 版权声明，使用 Apache 许可证 2.0 版本
# 创作团队为 HuggingFace Inc.，2023年
# 未经许可不得使用本文件，详情请参阅许可证
# 获取许可证内容的网址
# https://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或经书面同意，本软件按“原样”分发
# 不附带任何担保或条件，无论是明示的还是隐含的
# 详见许可证以获取权限范围和限制
"""转换来自原始存储库的Flax ViViT检查点为PyTorch格式。URL: 
https://github.com/google-research/scenic/tree/main/scenic/projects/vivit
"""
# 导入模块
import argparse
import json
import os.path
from collections import OrderedDict
# 导入第三方模块
import numpy as np
import requests
import torch
from flax.training.checkpoints import restore_checkpoint
from huggingface_hub import hf_hub_download

from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor
from transformers.image_utils import PILImageResampling

# 下载检查点
def download_checkpoint(path):
    url = "https://storage.googleapis.com/scenic-bucket/vivit/kinetics_400/vivit_base_16x2_unfactorized/checkpoint"

    with open(path, "wb") as f:
        with requests.get(url, stream=True) as req:
            for chunk in req.iter_content(chunk_size=2048):
                f.write(chunk)

# 获取 ViViT 的配置
def get_vivit_config() -> VivitConfig:
    # 创建 VivitConfig 对象
    config = VivitConfig()
    # 设置标签数量为 400
    config.num_labels = 400
    # 从 HuggingFace Hub 下载标签文件，获取标签映射
    repo_id = "huggingface/label-files"
    filename = "kinetics400-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 转换标签映射字典的键为整型值
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    # 设置标签到ID的映射
    config.label2id = {v: k for k, v in id2label.items()}
    return config

# 准备视频数据
def prepare_video():
    # 从 HuggingFace Hub 下载视频文件
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
    )
    # 加载视频数据
    video = np.load(file)
    return list(video)

# 转换注意力机制
def transform_attention(current: np.ndarray):
    # 如果是二维数组，转换注意力偏置
    if np.ndim(current) == 2:
        return transform_attention_bias(current)
    # 如果是三维数组，转换注意力核
    elif np.ndim(current) == 3:
        return transform_attention_kernel(current)
    # 其他维数抛出异常
    else:
        raise Exception(f"Invalid number of dimesions: {np.ndim(current)}")

# 转换注意力偏置
def transform_attention_bias(current: np.ndarray):
    return current.flatten()

# 转换注意力核
def transform_attention_kernel(current: np.ndarray):
    return np.reshape(current, (current.shape[0], current.shape[1] * current.shape[2])).T

# 转换注意力输出权重
def transform_attention_output_weight(current: np.ndarray):
# 待填充部分，需要根据代码功能填写
    # 将current数组重新调整形状为(current.shape[0] * current.shape[1], current.shape[2])的转置
        return np.reshape(current, (current.shape[0] * current.shape[1], current.shape[2])).T
# 转换编码器块中的状态字典，根据给定的索引 i 进行处理
def transform_state_encoder_block(state_dict, i):
    # 获取编码器块的状态
    state = state_dict["optimizer"]["target"]["Transformer"][f"encoderblock_{i}"]

    # 设置前缀
    prefix = f"encoder.layer.{i}."
    
    # 构建新状态字典
    new_state = {
        prefix + "intermediate.dense.bias": state["MlpBlock_0"]["Dense_0"]["bias"],
        prefix + "intermediate.dense.weight": np.transpose(state["MlpBlock_0"]["Dense_0"]["kernel"]),
        prefix + "output.dense.bias": state["MlpBlock_0"]["Dense_1"]["bias"],
        prefix + "output.dense.weight": np.transpose(state["MlpBlock_0"]["Dense_1"]["kernel"]),
        prefix + "layernorm_before.bias": state["LayerNorm_0"]["bias"],
        prefix + "layernorm_before.weight": state["LayerNorm_0"]["scale"],
        prefix + "layernorm_after.bias": state["LayerNorm_1"]["bias"],
        prefix + "layernorm_after.weight": state["LayerNorm_1"]["scale"],
        prefix + "attention.attention.query.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["query"]["bias"]
        ),
        prefix + "attention.attention.query.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["query"]["kernel"]
        ),
        prefix + "attention.attention.key.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["key"]["bias"]
        ),
        prefix + "attention.attention.key.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["key"]["kernel"]
        ),
        prefix + "attention.attention.value.bias": transform_attention(
            state["MultiHeadDotProductAttention_0"]["value"]["bias"]
        ),
        prefix + "attention.attention.value.weight": transform_attention(
            state["MultiHeadDotProductAttention_0"]["value"]["kernel"]
        ),
        prefix + "attention.output.dense.bias": state["MultiHeadDotProductAttention_0"]["out"]["bias"],
        prefix + "attention.output.dense.weight": transform_attention_output_weight(
            state["MultiHeadDotProductAttention_0"]["out"]["kernel"]
        ),
    }

    return new_state


# 获取给定状态字典中的编码器块数
def get_n_layers(state_dict):
    return sum([1 if "encoderblock_" in k else 0 for k in state_dict["optimizer"]["target"]["Transformer"].keys()])


# 转换整体状态字典的函数，可以选择是否有分类头
def transform_state(state_dict, classification_head=False):
    # 获取编码器块的总层数
    transformer_layers = get_n_layers(state_dict)

    # 构建新状态字典
    new_state = OrderedDict()

    new_state["layernorm.bias"] = state_dict["optimizer"]["target"]["Transformer"]["encoder_norm"]["bias"]
    new_state["layernorm.weight"] = state_dict["optimizer"]["target"]["Transformer"]["encoder_norm"]["scale"]

    new_state["embeddings.patch_embeddings.projection.weight"] = np.transpose(
        state_dict["optimizer"]["target"]["embedding"]["kernel"], (4, 3, 0, 1, 2)
    )
    new_state["embeddings.patch_embeddings.projection.bias"] = state_dict["optimizer"]["target"]["embedding"]["bias"]

    new_state["embeddings.cls_token"] = state_dict["optimizer"]["target"]["cls"]
    # 将state_dict中的位置嵌入赋值给new_state中的嵌入位置
    new_state["embeddings.position_embeddings"] = state_dict["optimizer"]["target"]["Transformer"]["posembed_input"]["pos_embedding"]

    # 循环遍历transformer层数量
    for i in range(transformer_layers):
        # 调用transform_state_encoder_block函数更新new_state
        new_state.update(transform_state_encoder_block(state_dict, i))

    # 如果包含分类头
    if classification_head:
        # 添加前缀"vivit."，同时重新组织new_state
        new_state = {"vivit." + k: v for k, v in new_state.items()}
        # 将state_dict中的输出投影核转置后赋值给new_state中的分类器权重
        new_state["classifier.weight"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["kernel"])
        # 将state_dict中的输出投影偏置转置后赋值给new_state中的分类器偏置
        new_state["classifier.bias"] = np.transpose(state_dict["optimizer"]["target"]["output_projection"]["bias"])

    # 将new_state中的每个值都转换为torch.tensor后返回
    return {k: torch.tensor(v) for k, v in new_state.items()}
# 检查图像处理器设置是否与原始实现中的相同
# 原始实现: https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/data/video_tfrecord_dataset.py
# 数据集特定配置：
# https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/configs/kinetics400/vivit_base_k400.py
def get_processor() -> VivitImageProcessor:
    # 创建 VivitImageProcessor 实例
    extractor = VivitImageProcessor()

    # 断言是否进行了缩放
    assert extractor.do_resize is True
    # 断言缩放大小为 {"shortest_edge": 256}
    assert extractor.size == {"shortest_edge": 256}
    # 断言是否进行了中心裁剪
    assert extractor.do_center_crop is True
    # 断言裁剪大小为 {"width": 224, "height": 224}
    assert extractor.crop_size == {"width": 224, "height": 224}
    # 断言是否使用了 PILImageResampling.BILINEAR 插值方式
    assert extractor.resample == PILImageResampling.BILINEAR

    # 在这里: https://github.com/deepmind/dmvr/blob/master/dmvr/modalities.py
    # 可以看到 add_image 的默认值为 normalization_mean 和 normalization_std，分别设置为 0 和 1
    # 这实际上意味着没有归一化（而 ViViT 在调用此函数时不会覆盖这些值）
    assert extractor.do_normalize is False
    # 断言是否进行了重新缩放
    assert extractor.do_rescale is True
    # 断言重新缩放因子为 1/255
    assert extractor.rescale_factor == 1 / 255

    # 零中心化在原始实现中设置为 True
    assert extractor.do_zero_centering is True

    # 返回图像处理器实例
    return extractor


def convert(output_path: str):
    flax_model_path = "checkpoint"

    # 如果 flax 模型路径不存在，则下载模型
    if not os.path.exists(flax_model_path):
        download_checkpoint(flax_model_path)

    # 恢复检查点并转换状态
    state_dict = restore_checkpoint(flax_model_path, None)
    new_state = transform_state(state_dict, classification_head=True)

    # 获取 ViViT 的配置
    config = get_vivit_config()

    # 断言图像尺寸为 224
    assert config.image_size == 224
    # 断言帧数为 32
    assert config.num_frames == 32

    # 创建 VivitForVideoClassification 模型实例
    model = VivitForVideoClassification(config)
    model.load_state_dict(new_state)
    model.eval()

    # 获取图像处理器实例
    extractor = get_processor()

    # 准备视频数据
    video = prepare_video()
    # 使用图像处理器对视频进行处理，返回 PyTorch 张量
    inputs = extractor(video, return_tensors="pt")

    # 将输入数据传递给模型，获取输出结果
    outputs = model(**inputs)

    # 预期输出的形状为 [1, 400]
    expected_shape = torch.Size([1, 400])
    # 预期输出的前五个元素
    expected_slice = torch.tensor([-1.0543, 2.0764, -0.2104, 0.4439, -0.9658])

    # 断言输出的形状是否与预期相同
    assert outputs.logits.shape == expected_shape
    # 断言输出的前五个元素是否与预期相近，允许的误差为 1e-4
    assert torch.allclose(outputs.logits[0, :5], expected_slice, atol=1e-4), outputs.logits[0, :5]

    # 保存模型到指定路径
    model.save_pretrained(output_path)
    # 保存图像处理器到指定路径
    extractor.save_pretrained(output_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加输出模型名称参数
    parser.add_argument("--output_model_name", "-o", type=str, help="Output path for the converted HuggingFace model")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert 函数，传入输出模型名称参数
    convert(args.output_model_name)
```