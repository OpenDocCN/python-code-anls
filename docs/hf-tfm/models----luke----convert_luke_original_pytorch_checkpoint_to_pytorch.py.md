# `.\models\luke\convert_luke_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8，确保可以正确处理中文等特殊字符
# 版权声明，此代码版权归 The HuggingFace Inc. 团队所有，基于 Apache License, Version 2.0 发布
# 只有在符合许可证的条件下才能使用该文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"现状"分发本软件，无任何明示或暗示的担保或条件
# 请查阅许可证，了解详细的法律条款和免责声明
"""Convert LUKE checkpoint."""

import argparse  # 导入解析命令行参数的库
import json  # 导入处理 JSON 格式数据的库
import os  # 导入操作系统功能的库

import torch  # 导入 PyTorch 深度学习框架

from transformers import LukeConfig, LukeModel, LukeTokenizer, RobertaTokenizer  # 导入 LUKE 模型相关类
from transformers.tokenization_utils_base import AddedToken  # 导入用于添加特殊 token 的类


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # 从 metadata 文件中加载配置信息
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    # 使用 metadata 中的配置信息创建 LUKE 模型配置对象
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # 从 checkpoint_path 加载模型的权重参数
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 加载实体词汇表文件
    entity_vocab = load_entity_vocab(entity_vocab_path)

    # 根据 metadata 中的配置信息加载 RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # 添加特殊 token 到 tokenizer 的词汇表，用于下游任务
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [entity_token_1, entity_token_2]})
    config.vocab_size += 2  # 更新配置中的词汇表大小

    # 打印信息，保存 tokenizer 到指定路径
    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    # 将实体词汇表保存为 JSON 文件
    with open(os.path.join(pytorch_dump_folder_path, LukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    # 从指定路径加载 LUKETokenizer
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    # 初始化特殊 token 的嵌入向量
    word_emb = state_dict["embeddings.word_embeddings.weight"]
    ent_emb = word_emb[tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    ent2_emb = word_emb[tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])

    # 初始化实体感知自注意力机制的查询层
    # 对每一个编码器层进行循环，范围是从 0 到 config.num_hidden_layers - 1
    for layer_index in range(config.num_hidden_layers):
        # 对于每一个矩阵名称，例如 "query.weight" 和 "query.bias"，进行循环
        for matrix_name in ["query.weight", "query.bias"]:
            # 构建前缀，形如 "encoder.layer.{layer_index}.attention.self."
            prefix = f"encoder.layer.{layer_index}.attention.self."
            # 复制指定矩阵名称的值到三个不同的状态字典键中
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # 使用"[MASK]"实体的嵌入初始化"[MASK2]"实体的嵌入，用于下游任务
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_emb[entity_vocab["[MASK2]"]] = entity_emb[entity_vocab["[MASK]"]]

    # 实例化Luke模型，并设置为评估模式
    model = LukeModel(config=config).eval()

    # 加载模型的状态字典，忽略丢失的键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 检查是否只丢失了一个键，并且该键是"embeddings.position_ids"
    if not (len(missing_keys) == 1 and missing_keys[0] == "embeddings.position_ids"):
        raise ValueError(f"Missing keys {', '.join(missing_keys)}. Expected only missing embeddings.position_ids")
    # 检查是否所有意外的键都以"entity_predictions"或"lm_head"开头
    if not (all(key.startswith("entity_predictions") or key.startswith("lm_head") for key in unexpected_keys)):
        raise ValueError(
            "Unexpected keys"
            f" {', '.join([key for key in unexpected_keys if not (key.startswith('entity_predictions') or key.startswith('lm_head'))])}"
        )

    # 检查模型输出
    # 根据路径加载Luke的tokenizer，用于实体分类任务
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")

    # 设置用于测试的文本和实体范围
    text = (
        "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the"
        " new world number one avoid a humiliating second- round exit at Wimbledon ."
    )
    span = (39, 42)
    # 使用tokenizer对文本进行编码，指定实体范围和其他参数
    encoding = tokenizer(text, entity_spans=[span], add_prefix_space=True, return_tensors="pt")

    # 将编码传递给模型，获取输出结果
    outputs = model(**encoding)

    # 验证词级别的隐藏状态
    if model_size == "large":
        expected_shape = torch.Size((1, 42, 1024))
        expected_slice = torch.tensor(
            [[0.0133, 0.0865, 0.0095], [0.3093, -0.2576, -0.7418], [-0.1720, -0.2117, -0.2869]]
        )
    else:  # base
        expected_shape = torch.Size((1, 42, 768))
        expected_slice = torch.tensor([[0.0037, 0.1368, -0.0091], [0.1099, 0.3329, -0.1095], [0.0765, 0.5335, 0.1179]])

    # 检查模型输出的最后隐藏状态的形状是否符合预期
    if not (outputs.last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    # 检查模型输出的部分隐藏状态是否与预期接近
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # 验证实体级别的隐藏状态
    if model_size == "large":
        expected_shape = torch.Size((1, 1, 1024))
        expected_slice = torch.tensor([[0.0466, -0.0106, -0.0179]])
    else:  # base
        expected_shape = torch.Size((1, 1, 768))
        expected_slice = torch.tensor([[0.1457, 0.1044, 0.0174]])
    # 检查输出的实体最后隐藏状态的形状是否与预期形状不同，如果是则抛出值错误异常
    if not (outputs.entity_last_hidden_state.shape != expected_shape):
        raise ValueError(
            f"Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is"
            f" {expected_shape}"
        )
    
    # 检查输出的实体最后隐藏状态的一个子集是否与预期的切片在数值上接近，如果不是则抛出值错误异常
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # 最后，将 PyTorch 模型和分词器保存到指定的路径中
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)
# 定义一个函数用于加载实体词汇表
def load_entity_vocab(entity_vocab_path):
    # 创建一个空字典，用于存储实体词汇
    entity_vocab = {}
    # 使用 UTF-8 编码打开实体词汇表文件
    with open(entity_vocab_path, "r", encoding="utf-8") as f:
        # 逐行读取文件内容并枚举行号
        for index, line in enumerate(f):
            # 去除行末尾的换行符并按制表符分割行内容，取得标题部分
            title, _ = line.rstrip().split("\t")
            # 将标题和其对应的行号添加到实体词汇字典中
            entity_vocab[title] = index

    # 返回构建好的实体词汇字典
    return entity_vocab


if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument("--checkpoint_path", type=str, help="Path to a pytorch_model.bin file.")
    parser.add_argument(
        "--metadata_path", default=None, type=str, help="Path to a metadata.json file, defining the configuration."
    )
    parser.add_argument(
        "--entity_vocab_path",
        default=None,
        type=str,
        help="Path to an entity_vocab.tsv file, containing the entity vocabulary.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to where to dump the output PyTorch model."
    )
    parser.add_argument(
        "--model_size", default="base", type=str, choices=["base", "large"], help="Size of the model to be converted."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 Luke 模型的 checkpoint 转换为 PyTorch 格式
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
```