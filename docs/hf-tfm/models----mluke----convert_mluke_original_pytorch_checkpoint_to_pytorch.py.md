# `.\transformers\models\mluke\convert_mluke_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# 定义编码规范为 UTF-8
# 版权声明
# 2021 年 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本授权
# 在遵守许可证的情况下可以使用本文件
# 您可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则在“原样”基础上分发的软件
# 没有任何形式的保证或条件，无论是明示的还是暗示的
# 请参阅许可证，了解特定语言下的权限和限制
"""Convert mLUKE checkpoint."""
# 文件功能说明，转换 mLUKE 检查点

import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 文件的模块
import os  # 导入处理文件和目录路径的模块
from collections import OrderedDict  # 导入排序的字典模块

import torch  # 导入 PyTorch 模块

from transformers import LukeConfig, LukeForMaskedLM, MLukeTokenizer, XLMRobertaTokenizer  # 从 transformers 模块中导入相关类
from transformers.tokenization_utils_base import AddedToken  # 从 transformers 模块中导入 AddedToken 类


@torch.no_grad()  # 用装饰器声明下面的函数不需要对计算图进行梯度追踪
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # 转换 mLUKE 检查点的函数，传入参数包括检查点路径、元数据文件路径、实体词汇文件路径、PyTorch 导出文件夹路径和模型大小

    # 从元数据文件中加载定义的配置信息
    with open(metadata_path) as metadata_file:  # 打开元数据文件
        metadata = json.load(metadata_file)  # 加载元数据文件内容
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])  # 使用元数据文件中的模型配置信息创建 LukeConfig 对象

    # 从检查点路径中加载权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")["module"]  # 使用 PyTorch 加载检查点文件中的模型权重，"module" 是模型在训练时采用的 DataParallel 封装导致的keyname

    # 加载实体词汇文件
    entity_vocab = load_original_entity_vocab(entity_vocab_path)  # 调用 load_original_entity_vocab 函数加载原始实体词汇文件
    # 为 [MASK2] 添加一个条目
    entity_vocab["[MASK2]"] = max(entity_vocab.values()) + 1  # 为 [MASK2] 添加新的索引
    config.entity_vocab_size += 1  # 更新实体词汇大小

    tokenizer = XLMRobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])  # 使用预训练的 XLM-Roberta Tokenizer

    # 为下游任务向标记词汇表添加特殊标记
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)  # 添加特殊的实体标记 1
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)  # 添加特殊的实体标记 2
    tokenizer.add_special_tokens({"additional_special_tokens": [entity_token_1, entity_token_2]})  # 向 Tokenizer 添加特殊标记
    config.vocab_size += 2  # 更新词汇表大小

    # 保存 Tokenizer 到指定路径
    print(f"Saving tokenizer to {pytorch_dump_folder_path}")  # 打印保存 Tokenizer 到指定路径的信息
    tokenizer.save_pretrained(pytorch_dump_folder_path)  # 保存 Tokenizer 到指定路径
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "r") as f:  # 打开 Tokenizer 配置文件
        tokenizer_config = json.load(f)  # 读取 Tokenizer 配置文件
    tokenizer_config["tokenizer_class"] = "MLukeTokenizer"  # 更新 Tokenizer 类别为 MLukeTokenizer
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "w") as f:  # 再次打开 Tokenizer 配置文件
        json.dump(tokenizer_config, f)  # 写入更新后的 Tokenizer 配置文件

    with open(os.path.join(pytorch_dump_folder_path, MLukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:  # 打开实体词汇文件
        json.dump(entity_vocab, f)  # 写入实体词汇文件

    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)  # 从预训练的 Tokenizer 文件夹中加载 MLukeTokenizer

    # 初始化特殊标记的嵌入
    ent_init_index = tokenizer.convert_tokens_to_ids(["@"])[0]  # 获取特殊标记 @ 的索引
    ent2_init_index = tokenizer.convert_tokens_to_ids(["#"])[0]  # 获取特殊标记 # 的索引

    word_emb = state_dict["embeddings.word_embeddings.weight"]  # 获取嵌入层的词嵌入权重张量
    ent_emb = word_emb[ent_init_index].unsqueeze(0)  # 获取特殊标记 @ 对应的词嵌入向量
    # 将ent2_emb初始化，并在第0维度上增加一个维度
    ent2_emb = word_emb[ent2_init_index].unsqueeze(0)
    # 更新state_dict中的'embeddings.word_embeddings.weight'值为word_emb, ent_emb, ent2_emb的拼接
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])
    
    # 为'entity_predictions.bias'添加特殊标记
    for bias_name in ["lm_head.decoder.bias", "lm_head.bias"]:
        # 获取对应的decoder_bias
        decoder_bias = state_dict[bias_name]
        # 获取ent_init_index位置的decoder_bias，并在第0维度上增加一个维度
        ent_decoder_bias = decoder_bias[ent_init_index].unsqueeze(0)
        # 获取ent2_init_index位置的decoder_bias，并在第0维度上增加一个维度
        ent2_decoder_bias = decoder_bias[ent2_init_index].unsqueeze(0)
        # 更新state_dict中的bias_name为decoder_bias, ent_decoder_bias, ent2_decoder_bias的拼接
        state_dict[bias_name] = torch.cat([decoder_bias, ent_decoder_bias, ent2_decoder_bias])

    # 初始化实体感知自注意力机制的查询层
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ["query.weight", "query.bias"]:
            # 设置前缀
            prefix = f"encoder.layer.{layer_index}.attention.self."
            # 复制state_dict中的matrix_name值到w2e_, e2w_, e2e_对应的值
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # 使用[MASK]实体的嵌入来初始化[MASK2]实体的嵌入
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_mask_emb = entity_emb[entity_vocab["[MASK]"]].unsqueeze(0)
    # 更新state_dict中的'entity_embeddings.entity_embeddings.weight'值为entity_emb, entity_mask_emb的拼接
    state_dict["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb, entity_mask_emb])
    # 为'entity_predictions.bias'添加[MASK2]
    entity_prediction_bias = state_dict["entity_predictions.bias"]
    entity_mask_bias = entity_prediction_bias[entity_vocab["[MASK]"]].unsqueeze(0)
    # 更新state_dict中的'entity_predictions.bias'值为entity_prediction_bias, entity_mask_bias的拼接
    state_dict["entity_predictions.bias"] = torch.cat([entity_prediction_bias, entity_mask_bias])

    # 初始化LukeForMaskedLM模型并设为评估模式
    model = LukeForMaskedLM(config=config).eval()

    # 移除不需要的键
    state_dict.pop("entity_predictions.decoder.weight")
    state_dict.pop("lm_head.decoder.weight")
    state_dict.pop("lm_head.decoder.bias")
    # 创建一个有序字典state_dict_for_hugging_face,包含符合Hugging Face模型的健值对
    state_dict_for_hugging_face = OrderedDict()
    for key, value in state_dict.items():
        if not (key.startswith("lm_head") or key.startswith("entity_predictions")):
            state_dict_for_hugging_face[f"luke.{key}"] = state_dict[key]
        else:
            state_dict_for_hugging_face[key] = state_dict[key]

    # 加载模型参数，并检查缺失和多余的键，并设为非严格模式
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_for_hugging_face, strict=False)

    # 如果有意外的多余的键，抛出ValueError
    if set(unexpected_keys) != {"luke.embeddings.position_ids"}:
        raise ValueError(f"Unexpected unexpected_keys: {unexpected_keys}")
    # 如果有意外的缺失的键，抛出ValueError
    if set(missing_keys) != {
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
        "entity_predictions.decoder.weight",
    }:
        raise ValueError(f"Unexpected missing_keys: {missing_keys}")

    # 将模型的word_embeddings权重与lm_head的decoder权重进行断言比较
    model.tie_weights()
    assert (model.luke.embeddings.word_embeddings.weight == model.lm_head.decoder.weight).all()
    # 将模型的entity_embeddings权重与entity_predictions的decoder权重进行断言比较
    assert (model.luke.entity_embeddings.entity_embeddings.weight == model.entity_predictions.decoder.weight).all()

    # 检查输出
    # 从预训练模型中加载 MLukeTokenizer 对象
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")

    # 定义待处理的文本
    text = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
    # 定义要标记的实体的起始和结束位置
    span = (0, 9)
    # 对文本进行编码，包括实体标记，并返回张量化的结果
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")

    # 使用编码后的文本向模型进行推理，得到输出
    outputs = model(**encoding)

    # 验证词的隐藏状态
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        # 定义期望的张量形状
        expected_shape = torch.Size((1, 33, 768))
        # 定义预期的张量切片
        expected_slice = torch.tensor([[0.0892, 0.0596, -0.2819], [0.0134, 0.1199, 0.0573], [-0.0169, 0.0927, 0.0644]])

    # 如果模型输出的最后隐藏状态的形状与预期形状不符，则引发值错误
    if not (outputs.last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    # 如果模型输出的最后隐藏状态的切片与预期切片不相似，则引发值错误
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # 验证实体的隐藏状态
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        # 定义期望的张量形状
        expected_shape = torch.Size((1, 1, 768))
        # 定义预期的张量切片
        expected_slice = torch.tensor([[-0.1482, 0.0609, 0.0322]])

    # 如果模型输出的实体最后隐藏状态的形状与预期形状不符，则引发值错误
    if not (outputs.entity_last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is"
            f" {expected_shape}"
        )
    # 如果模型输出的实体最后隐藏状态的切片与预期切片不相似，则引发值错误
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # 验证蒙面词/实体预测
    # 从预训练模型中加载 MLukeTokenizer 对象（此处已重新定义 tokenizer，注意覆盖）
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)
    # 重新定义待处理的文本
    text = "Tokyo is the capital of <mask>."
    span = (24, 30)
    # 对文本进行编码，包括实体标记，并返回张量化的结果
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")

    # 使用编码后的文本向模型进行推理，得到输出
    outputs = model(**encoding)

    # 取出输入的词汇 ID 列表，并将其转换为标准的 Python 列表类型
    input_ids = encoding["input_ids"][0].tolist()
    # 找到标记 "<mask>" 的位置
    mask_position_id = input_ids.index(tokenizer.convert_tokens_to_ids("<mask>"))
    # 获取蒙面词的预测 ID
    predicted_id = outputs.logits[0][mask_position_id].argmax(dim=-1)
    # 断言蒙面词的预测 ID 对应的词汇是 "Japan"
    assert "Japan" == tokenizer.decode(predicted_id)

    # 获取实体的预测 ID
    predicted_entity_id = outputs.entity_logits[0][0].argmax().item()
    # 根据预测的实体 ID 找到对应的实体
    multilingual_predicted_entities = [
        entity for entity, entity_id in tokenizer.entity_vocab.items() if entity_id == predicted_entity_id
    ]
    # 断言返回的实体列表中，以 "en:" 开头的第一个实体是 "en:Japan"
    assert [e for e in multilingual_predicted_entities if e.startswith("en:")][0] == "en:Japan"

    # 最后，保存我们的 PyTorch 模型和 tokenizer
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    # 保存 PyTorch 模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
# 加载实体词汇表
def load_original_entity_vocab(entity_vocab_path):
    # 定义特殊标记
    SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[UNK]"]

    # 从文件中读取数据并转为 JSON 格式
    data = [json.loads(line) for line in open(entity_vocab_path)]

    # 创建新的实体名称到 ID 的映射
    new_mapping = {}
    for entry in data:
        # 获取当前实体的 ID
        entity_id = entry["id"]
        # 遍历当前实体的所有名称和语言
        for entity_name, language in entry["entities"]:
            # 如果实体名称是特殊标记之一，则直接映射到 ID
            if entity_name in SPECIAL_TOKENS:
                new_mapping[entity_name] = entity_id
                break
            # 否则创建一个带语言前缀的新实体名称
            new_entity_name = f"{language}:{entity_name}"
            new_mapping[new_entity_name] = entity_id
    # 返回新的实体名称到 ID 的映射
    return new_mapping


# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 定义必需的参数
    parser.add_argument("--checkpoint_path", type=str, help="Path to a pytorch_model.bin file.")
    parser.add_argument("--metadata_path", default=None, type=str, help="Path to a metadata.json file, defining the configuration.")
    parser.add_argument("--entity_vocab_path", default=None, type=str, help="Path to an entity_vocab.tsv file, containing the entity vocabulary.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to where to dump the output PyTorch model.")
    parser.add_argument("--model_size", default="base", type=str, choices=["base", "large"], help="Size of the model to be converted.")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_luke_checkpoint 函数，传入解析的参数
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
```