# `.\models\vilt\convert_vilt_original_to_pytorch.py`

```py
# coding=utf-8
# 设置脚本的字符编码为UTF-8

# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，指明代码的版权信息

# Licensed under the Apache License, Version 2.0 (the "License");
# 使用 Apache License, Version 2.0 许可证

# you may not use this file except in compliance with the License.
# 按照许可证要求，除非获得许可，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 无论是明示的还是隐含的，不附带任何形式的担保或条件

# See the License for the specific language governing permissions and
# 请查阅许可证，获取具体的使用权限和

# limitations under the License.
# 限制和条件

"""Convert ViLT checkpoints from the original Github repository."""

# 从原始的 Github 仓库中转换 ViLT 检查点

import argparse
# 导入 argparse 用于解析命令行参数

import json
# 导入 json 模块用于处理 JSON 数据

from pathlib import Path
# 从 pathlib 模块中导入 Path 类，用于处理文件路径

import requests
# 导入 requests 模块，用于发送 HTTP 请求

import torch
# 导入 torch 模块，用于 PyTorch 相关操作

from huggingface_hub import hf_hub_download
# 从 huggingface_hub 库中导入 hf_hub_download 函数，用于从 Hugging Face Hub 下载模型

from PIL import Image
# 从 PIL 库中导入 Image 模块，用于图像处理

from transformers import (
    BertTokenizer,
    ViltConfig,
    ViltForImageAndTextRetrieval,
    ViltForImagesAndTextClassification,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltImageProcessor,
    ViltProcessor,
)
# 从 transformers 库中导入多个类和函数，用于加载和处理 ViLT 模型的不同配置和任务

from transformers.utils import logging
# 从 transformers.utils 中导入 logging 模块，用于设置日志信息

logging.set_verbosity_info()
# 设置日志记录级别为 info

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

# here we list all keys to be renamed (original name on the left, our name on the right)
# 在此处列出需要重命名的所有键（左侧为原始名称，右侧为我们的名称）

def create_rename_keys(config, vqa_model=False, nlvr_model=False, irtr_model=False):
    # 定义一个函数，用于生成重命名键的列表，根据不同的模型类型设置参数

    rename_keys = []
    # 初始化空的重命名键列表

    for i in range(config.num_hidden_layers):
        # 遍历隐藏层的数量，进行重命名操作

        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        # 编码器层：输出投影，2 个前馈神经网络和 2 个层归一化

        rename_keys.append((f"transformer.blocks.{i}.norm1.weight", f"vilt.encoder.layer.{i}.layernorm_before.weight"))
        # 添加归一化层权重的重命名映射

        rename_keys.append((f"transformer.blocks.{i}.norm1.bias", f"vilt.encoder.layer.{i}.layernorm_before.bias"))
        # 添加归一化层偏置的重命名映射

        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.weight", f"vilt.encoder.layer.{i}.attention.output.dense.weight")
        )
        # 添加注意力投影层权重的重命名映射

        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.bias", f"vilt.encoder.layer.{i}.attention.output.dense.bias")
        )
        # 添加注意力投影层偏置的重命名映射

        rename_keys.append((f"transformer.blocks.{i}.norm2.weight", f"vilt.encoder.layer.{i}.layernorm_after.weight"))
        # 添加第二层归一化权重的重命名映射

        rename_keys.append((f"transformer.blocks.{i}.norm2.bias", f"vilt.encoder.layer.{i}.layernorm_after.bias"))
        # 添加第二层归一化偏置的重命名映射

        rename_keys.append(
            (f"transformer.blocks.{i}.mlp.fc1.weight", f"vilt.encoder.layer.{i}.intermediate.dense.weight")
        )
        # 添加 MLP 第一层权重的重命名映射

        rename_keys.append(
            (f"transformer.blocks.{i}.mlp.fc1.bias", f"vilt.encoder.layer.{i}.intermediate.dense.bias")
        )
        # 添加 MLP 第一层偏置的重命名映射

        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.weight", f"vilt.encoder.layer.{i}.output.dense.weight"))
        # 添加 MLP 第二层权重的重命名映射

        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.bias", f"vilt.encoder.layer.{i}.output.dense.bias"))
        # 添加 MLP 第二层偏置的重命名映射

    # embeddings
    # 处理嵌入层的重命名，暂缺省略部分
    # 将下列键值对列表扩展到已有的 rename_keys 列表中，用于重命名模型中的参数路径
    rename_keys.extend(
        [
            # 文本嵌入
            ("text_embeddings.word_embeddings.weight", "vilt.embeddings.text_embeddings.word_embeddings.weight"),
            ("text_embeddings.position_embeddings.weight", "vilt.embeddings.text_embeddings.position_embeddings.weight"),
            ("text_embeddings.position_ids", "vilt.embeddings.text_embeddings.position_ids"),
            ("text_embeddings.token_type_embeddings.weight", "vilt.embeddings.text_embeddings.token_type_embeddings.weight"),
            ("text_embeddings.LayerNorm.weight", "vilt.embeddings.text_embeddings.LayerNorm.weight"),
            ("text_embeddings.LayerNorm.bias", "vilt.embeddings.text_embeddings.LayerNorm.bias"),
            # 补丁嵌入
            ("transformer.cls_token", "vilt.embeddings.cls_token"),
            ("transformer.patch_embed.proj.weight", "vilt.embeddings.patch_embeddings.projection.weight"),
            ("transformer.patch_embed.proj.bias", "vilt.embeddings.patch_embeddings.projection.bias"),
            ("transformer.pos_embed", "vilt.embeddings.position_embeddings"),
            # 标记类型嵌入
            ("token_type_embeddings.weight", "vilt.embeddings.token_type_embeddings.weight"),
        ]
    )
    
    # 最终的 Layernorm 和池化器
    rename_keys.extend(
        [
            ("transformer.norm.weight", "vilt.layernorm.weight"),
            ("transformer.norm.bias", "vilt.layernorm.bias"),
            ("pooler.dense.weight", "vilt.pooler.dense.weight"),
            ("pooler.dense.bias", "vilt.pooler.dense.bias"),
        ]
    )
    
    # 分类器头部
    if vqa_model:
        # 如果是 VQA 模型，添加 VQA 分类器的参数路径映射
        rename_keys.extend(
            [
                ("vqa_classifier.0.weight", "classifier.0.weight"),
                ("vqa_classifier.0.bias", "classifier.0.bias"),
                ("vqa_classifier.1.weight", "classifier.1.weight"),
                ("vqa_classifier.1.bias", "classifier.1.bias"),
                ("vqa_classifier.3.weight", "classifier.3.weight"),
                ("vqa_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    elif nlvr_model:
        # 如果是 NLVR 模型，添加 NLVR2 分类器的参数路径映射
        rename_keys.extend(
            [
                ("nlvr2_classifier.0.weight", "classifier.0.weight"),
                ("nlvr2_classifier.0.bias", "classifier.0.bias"),
                ("nlvr2_classifier.1.weight", "classifier.1.weight"),
                ("nlvr2_classifier.1.bias", "classifier.1.bias"),
                ("nlvr2_classifier.3.weight", "classifier.3.weight"),
                ("nlvr2_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    else:
        pass
    
    # 返回更新后的 rename_keys 列表，其中包含了所有需要重命名的模型参数路径映射
    return rename_keys
# 按照每个编码器层的要求，从状态字典中读取查询（query）、键（key）和值（value）的权重和偏置
def read_in_q_k_v(state_dict, config):
    # 遍历编码器层的数量
    for i in range(config.num_hidden_layers):
        prefix = "vilt."
        # 读取输入投影层的权重和偏置（在timm中，这是一个单独的矩阵加偏置）
        in_proj_weight = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.bias")
        # 将查询（query）、键（key）、值（value）依次添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[config.hidden_size : config.hidden_size * 2, :]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[config.hidden_size : config.hidden_size * 2]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 从状态字典中移除分类头部分的权重和偏置
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命名字典中的键名
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 转换ViLT模型的检查点，将其权重复制/粘贴/调整到我们的ViLT结构中
@torch.no_grad()
def convert_vilt_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重到我们的ViLT结构中。
    """

    # 定义配置并初始化HuggingFace模型
    config = ViltConfig(image_size=384, patch_size=32, tie_word_embeddings=False)
    mlm_model = False
    vqa_model = False
    nlvr_model = False
    irtr_model = False
    
    # 根据checkpoint_url的内容选择初始化不同的模型
    if "vqa" in checkpoint_url:
        vqa_model = True
        config.num_labels = 3129
        repo_id = "huggingface/label-files"
        filename = "vqa2-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        model = ViltForQuestionAnswering(config)
    elif "nlvr" in checkpoint_url:
        nlvr_model = True
        config.num_labels = 2
        config.id2label = {0: "False", 1: "True"}
        config.label2id = {v: k for k, v in config.id2label.items()}
        config.modality_type_vocab_size = 3
        model = ViltForImagesAndTextClassification(config)
    elif "irtr" in checkpoint_url:
        irtr_model = True
        model = ViltForImageAndTextRetrieval(config)
    elif "mlm_itm" in checkpoint_url:
        # 如果 URL 中包含 "mlm_itm"，则设置 mlm_model 为 True，并使用 ViltForMaskedLM 创建模型对象
        mlm_model = True
        model = ViltForMaskedLM(config)
    else:
        # 如果 URL 不包含 "mlm_itm"，则抛出 ValueError，表示未知的模型类型
        raise ValueError("Unknown model type")

    # 加载原始模型的 state_dict，移除和重命名一些键
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]
    rename_keys = create_rename_keys(config, vqa_model, nlvr_model, irtr_model)
    for src, dest in rename_keys:
        # 调用 rename_key 函数，用新的键名重命名 state_dict 中的键
        rename_key(state_dict, src, dest)
    # 处理 state_dict，读入 query、key 和 value 相关信息
    read_in_q_k_v(state_dict, config)
    if mlm_model or irtr_model:
        # 如果是 mlm_model 或 irtr_model，则忽略特定的键
        ignore_keys = ["itm_score.fc.weight", "itm_score.fc.bias"]
        for k in ignore_keys:
            # 从 state_dict 中移除指定的键
            state_dict.pop(k, None)

    # 将 state_dict 加载到 HuggingFace 模型中
    model.eval()
    if mlm_model:
        # 如果是 mlm_model，使用非严格模式加载 state_dict，并验证缺失的键
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert missing_keys == ["mlm_score.decoder.bias"]
    else:
        # 否则，使用严格模式加载 state_dict
        model.load_state_dict(state_dict)

    # 定义处理器对象
    image_processor = ViltImageProcessor(size=384)
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    processor = ViltProcessor(image_processor, tokenizer)

    # 对示例输入进行前向传播（图像 + 文本）
    if nlvr_model:
        # 如果是 nlvr_model，加载两个相同的图像和文本描述，使用 processor 对象编码
        image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        text = (
            "The left image contains twice the number of dogs as the right image, and at least two dogs in total are"
            " standing."
        )
        encoding_1 = processor(image1, text, return_tensors="pt")
        encoding_2 = processor(image2, text, return_tensors="pt")
        # 将编码后的输入传递给模型进行推断
        outputs = model(
            input_ids=encoding_1.input_ids,
            pixel_values=encoding_1.pixel_values,
            pixel_values_2=encoding_2.pixel_values,
        )
    else:
        # 否则，加载单个图像和相应的文本描述，使用 processor 对象编码
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        if mlm_model:
            # 如果是 mlm_model，使用包含 [MASK] 的文本描述
            text = "a bunch of [MASK] laying on a [MASK]."
        else:
            # 否则，使用问句描述
            text = "How many cats are there?"
        encoding = processor(image, text, return_tensors="pt")
        # 将编码后的输入传递给模型进行推断
        outputs = model(**encoding)

    # 验证模型输出
    if mlm_model:
        # 如果是 mlm_model，验证输出的形状和特定位置的数值
        expected_shape = torch.Size([1, 11, 30522])
        expected_slice = torch.tensor([-12.5061, -12.5123, -12.5174])
        assert outputs.logits.shape == expected_shape
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=1e-4)

        # 验证预测的 MASK 标记是否等于 "cats"
        predicted_id = outputs.logits[0, 4, :].argmax(-1).item()
        assert tokenizer.decode([predicted_id]) == "cats"
    # 如果是 VQA 模型，则执行以下操作
    elif vqa_model:
        # 预期的输出形状为 [1, 3129]
        expected_shape = torch.Size([1, 3129])
        # 预期的输出切片为 [-15.9495, -18.1472, -10.3041]
        expected_slice = torch.tensor([-15.9495, -18.1472, -10.3041])
        # 检查模型输出的前三个元素是否与预期切片接近
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
        # 检查模型输出的形状是否与预期形状一致
        assert outputs.logits.shape == expected_shape
        # 再次检查模型输出的前三个元素是否与预期切片接近
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=1e-4)

        # 验证 VQA 模型的预测结果是否等于 "2"
        predicted_idx = outputs.logits.argmax(-1).item()
        assert model.config.id2label[predicted_idx] == "2"
    
    # 如果是 NLVR 模型，则执行以下操作
    elif nlvr_model:
        # 预期的输出形状为 [1, 2]
        expected_shape = torch.Size([1, 2])
        # 预期的输出切片为 [-2.8721, 2.1291]
        expected_slice = torch.tensor([-2.8721, 2.1291])
        # 检查模型输出的前三个元素是否与预期切片接近
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
        # 检查模型输出的形状是否与预期形状一致
        assert outputs.logits.shape == expected_shape

    # 确保目录存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印信息，说明正在保存模型和处理器到指定路径
    print(f"Saving model and processor to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 将处理器保存到指定路径
    processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果脚本作为主程序执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt",
        type=str,
        help="URL of the checkpoint you'd like to convert."
    )
    # 添加必需的命令行参数：checkpoint_url，指定了默认的模型检查点 URL

    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the output PyTorch model directory."
    )
    # 添加命令行参数：pytorch_dump_folder_path，用于指定输出的 PyTorch 模型目录的路径

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 变量中

    convert_vilt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
    # 调用函数 convert_vilt_checkpoint，传入解析得到的参数 checkpoint_url 和 pytorch_dump_folder_path
```