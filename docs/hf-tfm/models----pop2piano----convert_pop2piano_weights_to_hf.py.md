# `.\transformers\models\pop2piano\convert_pop2piano_weights_to_hf.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Inc. 团队所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

""" 用于从官方存储库加载 Pop2Piano 模型权重并展示分词器词汇表是如何构建的文件"""

import json

import torch

from transformers import Pop2PianoConfig, Pop2PianoForConditionalGeneration


########################## MODEL WEIGHTS ##########################

# 这些权重是从官方 pop2piano 存储库下载的
# https://huggingface.co/sweetcocoa/pop2piano/blob/main/model-1999-val_0.67311615.ckpt
official_weights = torch.load("./model-1999-val_0.67311615.ckpt")
state_dict = {}


# 加载配置并初始化模型
cfg = Pop2PianoConfig.from_pretrained("sweetcocoa/pop2piano")
model = Pop2PianoForConditionalGeneration(cfg)


# 加载相对注意力偏置
state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = official_weights["state_dict"][
    "transformer.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
]
state_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = official_weights["state_dict"][
    "transformer.decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
]

# 加载编码器和解码器的嵌入标记和最终层规范化
state_dict["encoder.embed_tokens.weight"] = official_weights["state_dict"]["transformer.encoder.embed_tokens.weight"]
state_dict["decoder.embed_tokens.weight"] = official_weights["state_dict"]["transformer.decoder.embed_tokens.weight"]

state_dict["encoder.final_layer_norm.weight"] = official_weights["state_dict"][
    "transformer.encoder.final_layer_norm.weight"
]
state_dict["decoder.final_layer_norm.weight"] = official_weights["state_dict"][
    "transformer.decoder.final_layer_norm.weight"
]

# 加载 lm_head、mel_conditioner.emb 和 shared
state_dict["lm_head.weight"] = official_weights["state_dict"]["transformer.lm_head.weight"]
state_dict["mel_conditioner.embedding.weight"] = official_weights["state_dict"]["mel_conditioner.embedding.weight"]
state_dict["shared.weight"] = official_weights["state_dict"]["transformer.shared.weight"]

# 加载每��编码器块
for i in range(cfg.num_layers):
    # 第 0 层
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.q.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的自注意力机制的k权重
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.k.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的自注意力机制的v权重
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.v.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的自注意力机制的o权重
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.o.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的自注意力机制的层归一化权重
    state_dict[f"encoder.block.{i}.layer.0.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.layer_norm.weight"
    ]

    # layer 1
    # 更新状态字典中的键，表示编码器块中第i个层的第二层DenseReluDense层的wi_0权重
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的第二层DenseReluDense层的wi_1权重
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的第二层DenseReluDense层的wo权重
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wo.weight"
    ]
    # 更新状态字典中的键，表示编码器块中第i个层的第二层层归一化权重
    state_dict[f"encoder.block.{i}.layer.1.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.layer_norm.weight"
    ]
# 加载每个解码器块的权重

for i in range(6):
    # 第0层
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.q.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.k.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.v.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.o.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.layer_norm.weight"
    ]

    # 第1层
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.q.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.k.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.v.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.o.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.layer_norm.weight"
    ]

    # 第2层
    state_dict[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.DenseReluDense.wo.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.2.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.layer_norm.weight"
    ]

# 使用权重来加载模型的状态字典，严格模式
model.load_state_dict(state_dict, strict=True)

# 保存权重
torch.save(state_dict, "./pytorch_model.bin")

# tokenizer
# tokenize和detokenize方法取自官方实现
# 链接：https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L34
# 根据给定的索引和 token 类型返回 token 值
def tokenize(idx, token_type, n_special=4, n_note=128, n_velocity=2):
    # 如果是 TOKEN_TIME，返回计算后的索引值
    if token_type == "TOKEN_TIME":
        return n_special + n_note + n_velocity + idx
    # 如果是 TOKEN_VELOCITY，返回计算后的索引值
    elif token_type == "TOKEN_VELOCITY":
        return n_special + n_note + idx
    # 如果是 TOKEN_NOTE，返回计算后的索引值
    elif token_type == "TOKEN_NOTE":
        return n_special + idx
    # 如果是 TOKEN_SPECIAL，返回原始索引值
    elif token_type == "TOKEN_SPECIAL":
        return idx
    # 如果未知 token 类型，返回 -1
    else:
        return -1

# 根据索引值和特殊标记数量进行反解析，返回 token 类型和具体值
def detokenize(idx, n_special=4, n_note=128, n_velocity=2, time_idx_offset=0):
    # 如果索引值大于等于 n_special + n_note + n_velocity，返回 TOKEN_TIME 类型和具体值的元组
    if idx >= n_special + n_note + n_velocity:
        return "TOKEN_TIME", (idx - (n_special + n_note + n_velocity)) + time_idx_offset
    # 如果索引值大于等于 n_special + n_note，返回 TOKEN_VELOCITY 类型和具体值的元组
    elif idx >= n_special + n_note:
        return "TOKEN_VELOCITY", idx - (n_special + n_note)
    # 如果索引值大于等于 n_special，返回 TOKEN_NOTE 类型和具体值的元组
    elif idx >= n_special:
        return "TOKEN_NOTE", idx - n_special
    # 否则返回 TOKEN_SPECIAL 类型和具体值的元组
    else:
        return "TOKEN_SPECIAL", idx

# 创建解码器，将索引值和对应的标记值连接成字符串
decoder = {}
for i in range(cfg.vocab_size):
    decoder.update({i: f"{detokenize(i)[1]}_{detokenize(i)[0]}"})

# 创建编码器，将解码器中的键值对进行翻转
encoder = {v: k for k, v in decoder.items()}

# 将编码器保存为 JSON 文件
with open("./vocab.json", "w") as file:
    file.write(json.dumps(encoder))
```