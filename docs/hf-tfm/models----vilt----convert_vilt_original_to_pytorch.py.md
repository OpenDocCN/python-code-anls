# `.\transformers\models\vilt\convert_vilt_original_to_pytorch.py`

```py
# 设置源码文件的编码格式为 utf-8
# 版权声明
# 根据 Apache License 2.0 许可使用本文件
# 获取许可证的链接
# 在软件分发时，根据 "AS IS" 的基础分发，无需附加任何担保或条件，无论明示或默示
# 查看特定语言的授权权限，限制或个别许可的具体语言文档
# 从原始的 Github 存储库中转换 ViLT 检查点

# 导入必要的库
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
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
from transformers.utils import logging

# 设置日志的显示级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 创建需要重命名的键值对列表
def create_rename_keys(config, vqa_model=False, nlvr_model=False, irtr_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # 编码层：输出映射、两个前馈神经网络和两个 layernorm
        rename_keys.append((f"transformer.blocks.{i}.norm1.weight", f"vilt.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm1.bias", f"vilt.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.weight", f"vilt.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.bias", f"vilt.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"transformer.blocks.{i}.norm2.weight", f"vilt.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm2.bias", f"vilt.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.mlp.fc1.weight", f"vilt.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc1.bias", f"vilt.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.weight", f"vilt.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.bias", f"vilt.encoder.layer.{i}.output.dense.bias"))

    # embeddings
    # 向 rename_keys 列表中添加文本嵌入相关的键值对
    rename_keys.extend(
        [
            # 文本嵌入的词嵌入权重
            ("text_embeddings.word_embeddings.weight", "vilt.embeddings.text_embeddings.word_embeddings.weight"),
            # 文本嵌入的位置嵌入权重
            (
                "text_embeddings.position_embeddings.weight",
                "vilt.embeddings.text_embeddings.position_embeddings.weight",
            ),
            # 文本嵌入的位置ID
            ("text_embeddings.position_ids", "vilt.embeddings.text_embeddings.position_ids"),
            # 文本嵌入的标记类型嵌入权重
            (
                "text_embeddings.token_type_embeddings.weight",
                "vilt.embeddings.text_embeddings.token_type_embeddings.weight",
            ),
            # 文本嵌入的 LayerNorm 权重
            ("text_embeddings.LayerNorm.weight", "vilt.embeddings.text_embeddings.LayerNorm.weight"),
            # 文本嵌入的 LayerNorm 偏置
            ("text_embeddings.LayerNorm.bias", "vilt.embeddings.text_embeddings.LayerNorm.bias"),
            # 补丁嵌入相关
            # Transformer 模型的 cls_token
            ("transformer.cls_token", "vilt.embeddings.cls_token"),
            # Transformer 模型的 patch_embed 映射权重
            ("transformer.patch_embed.proj.weight", "vilt.embeddings.patch_embeddings.projection.weight"),
            # Transformer 模型的 patch_embed 映射偏置
            ("transformer.patch_embed.proj.bias", "vilt.embeddings.patch_embeddings.projection.bias"),
            # Transformer 模型的位置嵌入
            ("transformer.pos_embed", "vilt.embeddings.position_embeddings"),
            # 标记类型嵌入权重
            ("token_type_embeddings.weight", "vilt.embeddings.token_type_embeddings.weight"),
        ]
    )

    # 最终的 LayerNorm 和 pooler
    # 向 rename_keys 列表中添加最终的 LayerNorm 和 pooler 相关的键值对
    rename_keys.extend(
        [
            # Transformer 模型的 LayerNorm 权重
            ("transformer.norm.weight", "vilt.layernorm.weight"),
            # Transformer 模型的 LayerNorm 偏置
            ("transformer.norm.bias", "vilt.layernorm.bias"),
            # pooler 模型的 dense 权重
            ("pooler.dense.weight", "vilt.pooler.dense.weight"),
            # pooler 模型的 dense 偏置
            ("pooler.dense.bias", "vilt.pooler.dense.bias"),
        ]
    )

    # 分类器头部
    # 如果是 VQA 模型时，添加 VQA 分类器头部相关的键值对
    if vqa_model:
        # 分类头部
        rename_keys.extend(
            [
                # VQA 分类器的第一层权重
                ("vqa_classifier.0.weight", "classifier.0.weight"),
                # VQA 分类器的第一层偏置
                ("vqa_classifier.0.bias", "classifier.0.bias"),
                # VQA 分类器的第二层权重
                ("vqa_classifier.1.weight", "classifier.1.weight"),
                # VQA 分类器的第二层偏置
                ("vqa_classifier.1.bias", "classifier.1.bias"),
                # VQA 分类器的第三层权重
                ("vqa_classifier.3.weight", "classifier.3.weight"),
                # VQA 分类器的第三层偏置
                ("vqa_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    # 如果是 NLVR 模型时，添加 NLVR 分类器头部相关的键值对
    elif nlvr_model:
        # 分类头部
        rename_keys.extend(
            [
                # NLVR2 分类器的第一层权重
                ("nlvr2_classifier.0.weight", "classifier.0.weight"),
                # NLVR2 分类器的第一层偏置
                ("nlvr2_classifier.0.bias", "classifier.0.bias"),
                # NLVR2 分类器的第二层权重
                ("nlvr2_classifier.1.weight", "classifier.1.weight"),
                # NLVR2 分类器的第二层偏置
                ("nlvr2_classifier.1.bias", "classifier.1.bias"),
                # NLVR2 分类器的第三层权重
                ("nlvr2_classifier.3.weight", "classifier.3.weight"),
                # NLVR2 分类器的第三层偏置
                ("nlvr2_classifier.3.bias", "classifier.3.bias"),
            ]
        )
    else:
        pass
    # 返回更新后的 rename_keys 列表
    return rename_keys
# 将每个编码器层的矩阵分割成查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config):
    # 对每个编码器层进行循环处理
    for i in range(config.num_hidden_layers):
        # 设置前缀为 "vilt."
        prefix = "vilt."
        # 读取输入投影层（在timm中，这是一个单矩阵加偏置）的权重和偏置
        in_proj_weight = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.bias")
        # 将查询（query）、键（key）和值（value）（按顺序）添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]

# 移除分类头部的权重和偏置
def remove_classification_head_(state_dict):
    # 忽略指定的键
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        # 如果存在指定的键，则从状态字典中移除
        state_dict.pop(k, None)

# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值插入到新键下
    dct[new] = val

# 将模型权重转换为我们的 ViLT 结构
@torch.no_grad()
def convert_vilt_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重以适应我们的 ViLT 结构。
    """

    # 定义配置并初始化 HuggingFace 模型
    config = ViltConfig(image_size=384, patch_size=32, tie_word_embeddings=False)
    mlm_model = False
    vqa_model = False
    nlvr_model = False
    irtr_model = False
    if "vqa" in checkpoint_url:
        vqa_model = True
        config.num_labels = 3129
        # 配置 VQA 模型的标签
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
        # 配置 NLVR 模型的标签
        config.id2label = {0: "False", 1: "True"}
        config.label2id = {v: k for k, v in config.id2label.items()}
        config.modality_type_vocab_size = 3
        model = ViltForImagesAndTextClassification(config)
    elif "irtr" in checkpoint_url:
        irtr_model = True
        model = ViltForImageAndTextRetrieval(config)
    elif "mlm_itm" in checkpoint_url:
        # 如果检查点 URL 中包含 "mlm_itm"，则设置 mlm_model 为 True
        mlm_model = True
        # 使用 ViltForMaskedLM 类来创建模型对象
        model = ViltForMaskedLM(config)
    else:
        # 如果模型类型未知，则抛出 ValueError 异常
        raise ValueError("Unknown model type")

    # 从给定的检查点 URL 中加载原始模型的 state_dict，并移除或重命名一些键
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]
    # 创建将要被重命名的键的列表，根据不同的模型类型创建不同的键
    rename_keys = create_rename_keys(config, vqa_model, nlvr_model, irtr_model)
    # 遍历 rename_keys 列表，将 state_dict 中的相应键进行重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 从 state_dict 中读入问题、键、值
    read_in_q_k_v(state_dict, config)
    # 如果是 mlm_model 或者 irtr_model，则移除忽略的键
    if mlm_model or irtr_model:
        ignore_keys = ["itm_score.fc.weight", "itm_score.fc.bias"]
        for k in ignore_keys:
            state_dict.pop(k, None)

    # 将 state_dict 加载到 HuggingFace 模型中
    model.eval()
    if mlm_model:
        # 对于 mlm_model，加载 state_dict，并允许缺失一些键
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # 断言加载到模型中的缺失键应为 ["mlm_score.decoder.bias"]
        assert missing_keys == ["mlm_score.decoder.bias"]
    else:
        # 对于非 mlm_model，加载 state_dict
        model.load_state_dict(state_dict)

    # 定义 ViltProcessor 对象，用于将图像和文本进行处理
    image_processor = ViltImageProcessor(size=384)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processor = ViltProcessor(image_processor, tokenizer)

    # 对示例输入进行前向传播（图像 + 文本）
    if nlvr_model:
        # 如果是 nlvr_model，从 URL 中打开两个图像，并提供相应的文本
        image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        text = (
            "The left image contains twice the number of dogs as the right image, and at least two dogs in total are"
            " standing."
        )
        encoding_1 = processor(image1, text, return_tensors="pt")
        encoding_2 = processor(image2, text, return_tensors="pt")
        # 使用模型对输入进行预测
        outputs = model(
            input_ids=encoding_1.input_ids,
            pixel_values=encoding_1.pixel_values,
            pixel_values_2=encoding_2.pixel_values,
        )
    else:
        # 如果不是 nlvr_model，则从 URL 中打开一个图像，并提供相应的文本
        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        if mlm_model:
            # 如果是 mlm_model，则提供一个含有 [MASK] 的文本
            text = "a bunch of [MASK] laying on a [MASK]."
        else:
            # 如果是其它类型的模型，则提供一个文本询问 "How many cats are there?"
            text = "How many cats are there?"
        encoding = processor(image, text, return_tensors="pt")
        # 使用模型对输入进行预测
        outputs = model(**encoding)

    # 验证输出结果
    if mlm_model:
        expected_shape = torch.Size([1, 11, 30522])
        expected_slice = torch.tensor([-12.5061, -12.5123, -12.5174])
        # 断言输出的 logits 的形状为 [1, 11, 30522]
        assert outputs.logits.shape == expected_shape
        # 断言输出 logits 的第一条数据的前三个值与 expected_slice 的值相近，误差不超过 1e-4
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=1e-4)

        # 验证预测的被遮蔽的标记是否等于 "cats"
        predicted_id = outputs.logits[0, 4, :].argmax(-1).item()
        assert tokenizer.decode([predicted_id]) == "cats"
    # 如果是 VQA 模型
    elif vqa_model:
        # 期望输出的形状是 [1, 3129]
        expected_shape = torch.Size([1, 3129])
        # 期望输出的前三个值
        expected_slice = torch.tensor([-15.9495, -18.1472, -10.3041])
        # 断言前三个预测值与期望值的接近度在 1e-4 的范围内
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
        # 断言输出的形状是否符合预期形状
        assert outputs.logits.shape == expected_shape
        # 断言前三个预测值与期望值的接近度在 1e-4 的范围内
        assert torch.allclose(outputs.logits[0, 0, :3], expected_slice, atol=1e-4)

        # 验证 VQA 预测是否等于 "2"
        predicted_idx = outputs.logits.argmax(-1).item()
        assert model.config.id2label[predicted_idx] == "2"
    # 如果是 NLVR 模型
    elif nlvr_model:
        # 期望输出的形状是 [1, 2]
        expected_shape = torch.Size([1, 2])
        # 期望输出的前两个值
        expected_slice = torch.tensor([-2.8721, 2.1291])
        # 断言前两个预测值与期望值的接近度在 1e-4 的范围内
        assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
        # 断言输出的形状是否符合预期形状
        assert outputs.logits.shape == expected_shape

    # 创建路径，如果路径不存在则创建路径
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型和处理器的信息
    print(f"Saving model and processor to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 保存处理器到指定路径
    processor.save_pretrained(pytorch_dump_folder_path)
# 如果该脚本是直接执行的，而不是被导入到别的脚本中，则执行以下代码
if __name__ == "__main__":
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 添加必要的参数
    parser.add_argument(
        "--checkpoint_url",  # 参数名
        default="https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt",  # 默认值
        type=str,  # 参数类型
        help="URL of the checkpoint you'd like to convert."  # 参数的帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 参数名
        default=None,  # 默认值
        type=str,  # 参数类型
        help="Path to the output PyTorch model directory."  # 参数的帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数convert_vilt_checkpoint，传入解析后的参数checkpoint_url和pytorch_dump_folder_path
    convert_vilt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```