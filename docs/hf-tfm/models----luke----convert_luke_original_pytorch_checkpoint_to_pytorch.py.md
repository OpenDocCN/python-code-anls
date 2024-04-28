# `.\transformers\models\luke\convert_luke_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设定脚本编码格式为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可证规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据本许可证发布的软件将按"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的管理权限，请参阅许可证。
"""转换 LUKE 检查点。"""

import argparse  # 导入解析命令行参数的模块
import json  # 导入 JSON 模块
import os  # 导入操作系统功能的模块

import torch  # 导入 PyTorch 模块

# 从 transformers 模块中导入 LUKE 配置、模型、分词器以及 Roberta 分词器
from transformers import LukeConfig, LukeModel, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken  # 导入添加特殊标记的工具类


@torch.no_grad()  # 禁用梯度计算的装饰器
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # 加载元数据文件中定义的配置
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # 从检查点路径加载权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 加载实体词汇文件
    entity_vocab = load_entity_vocab(entity_vocab_path)

    # 使用元数据中的 BERT 模型名初始化分词器
    tokenizer = RobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # 为下游任务向分词器的标记词汇中添加特殊标记
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [entity_token_1, entity_token_2]})
    config.vocab_size += 2

    # 打印信息，保存分词器到指定路径
    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    with open(os.path.join(pytorch_dump_folder_path, LukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    # 从指定路径加载 LUKE 分词器
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    # 初始化特殊标记的嵌入向量
    word_emb = state_dict["embeddings.word_embeddings.weight"]
    ent_emb = word_emb[tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    ent2_emb = word_emb[tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])

    # 初始化实体感知自注意力机制的查询层
    # 遍历每个编码器层
    for layer_index in range(config.num_hidden_layers):
        # 遍历每个权重矩阵和偏置项
        for matrix_name in ["query.weight", "query.bias"]:
            # 构建前缀以访问编码器层的自注意力子模块
            prefix = f"encoder.layer.{layer_index}.attention.self."
            # 复制权重和偏置项到不同的子模块
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # 使用“[MASK]”实体的嵌入来初始化“[MASK2]”实体的嵌入，用于下游任务
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_emb[entity_vocab["[MASK2]"]] = entity_emb[entity_vocab["[MASK]"]]

    # 初始化模型并设为评估模式
    model = LukeModel(config=config).eval()

    # 加载模型参数，允许缺失键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 如果缺失键不是 embeddings.position_ids，则抛出错误
    if not (len(missing_keys) == 1 and missing_keys[0] == "embeddings.position_ids"):
        raise ValueError(f"Missing keys {', '.join(missing_keys)}. Expected only missing embeddings.position_ids")
    # 如果意外键不是以“entity_predictions”或“lm_head”开头，则抛出错误
    if not (all(key.startswith("entity_predictions") or key.startswith("lm_head") for key in unexpected_keys)):
        raise ValueError(
            "Unexpected keys"
            f" {', '.join([key for key in unexpected_keys if not (key.startswith('entity_predictions') or key.startswith('lm_head'))])}"
        )

    # 检查模型输出
    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")

    # 准备文本和实体跨度，并编码
    text = (
        "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the"
        " new world number one avoid a humiliating second- round exit at Wimbledon ."
    )
    span = (39, 42)
    encoding = tokenizer(text, entity_spans=[span], add_prefix_space=True, return_tensors="pt")

    # 获取模型输出
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

    # 如果输出的最后一个隐藏状态的形状不符合预期，则抛出错误
    if not (outputs.last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    # 如果输出的最后一个隐藏状态的切片不符合预期，则抛出错误
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # 验证实体级别的隐藏状态
    if model_size == "large":
        expected_shape = torch.Size((1, 1, 1024))
        expected_slice = torch.tensor([[0.0466, -0.0106, -0.0179]])
    else:  # base
        expected_shape = torch.Size((1, 1, 768))
        expected_slice = torch.tensor([[0.1457, 0.1044, 0.0174]])
    # 检查输出的实体最后隐藏状态的形状是否符合预期形状
    if not (outputs.entity_last_hidden_state.shape != expected_shape):
        # 如果形状不符合预期，抛出数值错误并显示包含实际形状和预期形状的消息
        raise ValueError(
            f"Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is"
            f" {expected_shape}"
        )
    
    # 检查输出的实体最后隐藏状态的部分值是否与预期的切片值接近
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        # 如果不接近预期的切片值，则抛出数值错误
        raise ValueError

    # 最后，保存我们的 PyTorch 模型和分词器
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)
# 从指定路径加载实体词汇表
def load_entity_vocab(entity_vocab_path):
    # 创建一个空的实体词汇表字典
    entity_vocab = {}
    # 打开实体词汇表文件
    with open(entity_vocab_path, "r", encoding="utf-8") as f:
        # 遍历实体词汇表文件的每一行，获取下标和内容
        for index, line in enumerate(f):
            # 对每一行内容进行处理，根据制表符分隔得到标题，忽略第二个部分
            title, _ = line.rstrip().split("\t")
            # 将标题作为键，下标作为值，加入实体词汇表字典
            entity_vocab[title] = index
    # 返回加载好的实体词汇表字典
    return entity_vocab

# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 必选参数
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
    # 调用函数，将checkpoint转换为PyTorch模型
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
```