# `.\models\mluke\convert_mluke_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
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
"""Convert mLUKE checkpoint."""

import argparse
import json
import os
from collections import OrderedDict

import torch

from transformers import LukeConfig, LukeForMaskedLM, MLukeTokenizer, XLMRobertaTokenizer
from transformers.tokenization_utils_base import AddedToken


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # 从元数据文件中加载配置信息
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    # 根据元数据配置创建 LukeConfig 对象
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # 加载来自 checkpoint_path 的模型权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")["module"]

    # 加载实体词汇表文件
    entity_vocab = load_original_entity_vocab(entity_vocab_path)
    # 添加一个新条目用于 [MASK2]
    entity_vocab["[MASK2]"] = max(entity_vocab.values()) + 1
    config.entity_vocab_size += 1

    # 根据元数据中指定的 BERT 模型名称加载 tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # 为下游任务向 token 词汇表添加特殊 token
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [entity_token_1, entity_token_2]})
    config.vocab_size += 2

    # 打印信息，保存 tokenizer 到指定路径
    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    # 更新 tokenizer 配置文件
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["tokenizer_class"] = "MLukeTokenizer"
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    # 将实体词汇表写入指定路径
    with open(os.path.join(pytorch_dump_folder_path, MLukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    # 从保存路径加载 MLukeTokenizer
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    # 初始化特殊 token 的嵌入向量
    ent_init_index = tokenizer.convert_tokens_to_ids(["@"])[0]
    ent2_init_index = tokenizer.convert_tokens_to_ids(["#"])[0]

    # 获取词嵌入权重
    word_emb = state_dict["embeddings.word_embeddings.weight"]
    # 提取第一个特殊 token 的嵌入向量并扩展维度
    ent_emb = word_emb[ent_init_index].unsqueeze(0)
    # 获取第二个实体的嵌入向量，并添加一个维度使其成为二维张量
    ent2_emb = word_emb[ent2_init_index].unsqueeze(0)
    
    # 将词嵌入、第一个实体嵌入和第二个实体嵌入连接起来，更新到模型的权重中
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])
    
    # 为 'entity_predictions.bias' 添加特殊的标记
    for bias_name in ["lm_head.decoder.bias", "lm_head.bias"]:
        # 获取当前偏置的张量
        decoder_bias = state_dict[bias_name]
        # 获取第一个实体的偏置并添加一个维度使其成为二维张量
        ent_decoder_bias = decoder_bias[ent_init_index].unsqueeze(0)
        # 获取第二个实体的偏置并添加一个维度使其成为二维张量
        ent2_decoder_bias = decoder_bias[ent2_init_index].unsqueeze(0)
        # 将三个偏置连接起来，更新到模型的偏置中
        state_dict[bias_name] = torch.cat([decoder_bias, ent_decoder_bias, ent2_decoder_bias])

    # 初始化实体感知自注意力机制中查询层的权重和偏置
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ["query.weight", "query.bias"]:
            prefix = f"encoder.layer.{layer_index}.attention.self."
            # 复制查询层权重和偏置到不同的实体组合中
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # 使用 '[MASK]' 实体的嵌入来初始化 '[MASK2]' 实体的嵌入，用于下游任务
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_mask_emb = entity_emb[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb, entity_mask_emb])
    
    # 为 'entity_predictions.bias' 添加 '[MASK2]' 实体的偏置
    entity_prediction_bias = state_dict["entity_predictions.bias"]
    entity_mask_bias = entity_prediction_bias[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_predictions.bias"] = torch.cat([entity_prediction_bias, entity_mask_bias])

    # 初始化 Luke 模型作为一个评估模型
    model = LukeForMaskedLM(config=config).eval()

    # 移除不需要的权重
    state_dict.pop("entity_predictions.decoder.weight")
    state_dict.pop("lm_head.decoder.weight")
    state_dict.pop("lm_head.decoder.bias")
    
    # 创建一个有序字典，以适应 Hugging Face 模型的加载要求
    state_dict_for_hugging_face = OrderedDict()
    for key, value in state_dict.items():
        if not (key.startswith("lm_head") or key.startswith("entity_predictions")):
            state_dict_for_hugging_face[f"luke.{key}"] = state_dict[key]
        else:
            state_dict_for_hugging_face[key] = state_dict[key]

    # 使用加载字典更新模型的权重，并忽略严格检查模式
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_for_hugging_face, strict=False)

    # 检查是否存在不期望的键
    if set(unexpected_keys) != {"luke.embeddings.position_ids"}:
        raise ValueError(f"Unexpected unexpected_keys: {unexpected_keys}")
    
    # 检查是否存在缺失的键
    if set(missing_keys) != {
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
        "entity_predictions.decoder.weight",
    }:
        raise ValueError(f"Unexpected missing_keys: {missing_keys}")

    # 对模型的权重进行绑定
    model.tie_weights()
    
    # 断言 Luke 模型的词嵌入与 lm_head 解码器的权重完全相等
    assert (model.luke.embeddings.word_embeddings.weight == model.lm_head.decoder.weight).all()
    
    # 断言 Luke 模型的实体嵌入与 entity_predictions 解码器的权重完全相等
    assert (model.luke.entity_embeddings.entity_embeddings.weight == model.entity_predictions.decoder.weight).all()

    # 检查输出
    # 从预训练模型文件夹路径加载 MLukeTokenizer，用于实体分类任务的标记化
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")
    
    # 定义要输入的文本字符串及其实体范围
    text = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
    span = (0, 9)
    # 使用加载的 tokenizer 对文本进行编码，指定实体范围，并返回 PyTorch 张量
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")
    
    # 使用模型进行推理，传入编码后的文本
    outputs = model(**encoding)
    
    # 验证词级别的隐藏状态
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        expected_shape = torch.Size((1, 33, 768))  # 预期的隐藏状态张量形状
        expected_slice = torch.tensor([[0.0892, 0.0596, -0.2819], [0.0134, 0.1199, 0.0573], [-0.0169, 0.0927, 0.0644]])  # 预期的部分张量切片
    
    # 检查模型输出的最后隐藏状态张量的形状是否符合预期
    if not (outputs.last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    # 检查模型输出的最后隐藏状态张量的部分切片是否与预期的张量切片在指定容差下相似
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError
    
    # 验证实体级别的隐藏状态
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        expected_shape = torch.Size((1, 1, 768))  # 预期的实体级别隐藏状态张量形状
        expected_slice = torch.tensor([[-0.1482, 0.0609, 0.0322]])  # 预期的实体级别隐藏状态张量切片
    
    # 检查模型输出的实体级别隐藏状态张量的形状是否符合预期
    if not (outputs.entity_last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is"
            f" {expected_shape}"
        )
    # 检查模型输出的实体级别隐藏状态张量的部分切片是否与预期的张量切片在指定容差下相似
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError
    
    # 验证掩码词/实体预测
    # 重新加载 tokenizer（可能是为了覆盖先前的实体分类配置）
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)
    text = "Tokyo is the capital of <mask>."
    span = (24, 30)
    # 使用重新加载的 tokenizer 对新文本进行编码，指定实体范围，并返回 PyTorch 张量
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")
    
    # 使用模型进行推理，传入编码后的文本
    outputs = model(**encoding)
    
    # 获取输入的 token_ids 并找到 <mask> 的位置
    input_ids = encoding["input_ids"][0].tolist()
    mask_position_id = input_ids.index(tokenizer.convert_tokens_to_ids("<mask>"))
    # 在模型输出的 logits 中找到预测的 token_id
    predicted_id = outputs.logits[0][mask_position_id].argmax(dim=-1)
    assert "Japan" == tokenizer.decode(predicted_id)  # 断言预测的实体是 "Japan"
    
    # 在实体 logits 中找到预测的实体 ID，并根据 tokenizer 的实体词汇表找到对应的多语言实体
    predicted_entity_id = outputs.entity_logits[0][0].argmax().item()
    multilingual_predicted_entities = [
        entity for entity, entity_id in tokenizer.entity_vocab.items() if entity_id == predicted_entity_id
    ]
    assert [e for e in multilingual_predicted_entities if e.startswith("en:")][0] == "en:Japan"  # 断言多语言实体是 "en:Japan"
    
    # 最后，保存 PyTorch 模型和 tokenizer 到指定路径
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)
# 加载原始实体词汇表的函数
def load_original_entity_vocab(entity_vocab_path):
    # 定义特殊的标记列表
    SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[UNK]"]

    # 打开实体词汇表文件，逐行加载JSON数据
    data = [json.loads(line) for line in open(entity_vocab_path)]

    # 创建一个新的映射字典
    new_mapping = {}
    # 遍历加载的每个实体词汇表条目
    for entry in data:
        # 获取实体的唯一标识符
        entity_id = entry["id"]
        # 遍历每个实体的名称和语言信息
        for entity_name, language in entry["entities"]:
            # 如果实体名称在特殊标记列表中，则将其映射到对应的实体ID
            if entity_name in SPECIAL_TOKENS:
                new_mapping[entity_name] = entity_id
                break
            # 否则，将实体名称和语言组合成新的实体名称
            new_entity_name = f"{language}:{entity_name}"
            # 将新的实体名称映射到对应的实体ID
            new_mapping[new_entity_name] = entity_id

    # 返回创建的新映射字典
    return new_mapping


if __name__ == "__main__":
    # 创建参数解析器
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

    # 调用函数，转换LUKE模型的检查点
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
```