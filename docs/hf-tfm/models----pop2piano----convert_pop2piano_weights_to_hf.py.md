# `.\models\pop2piano\convert_pop2piano_weights_to_hf.py`

```py
# 加载版权声明和许可证信息
# 版权所有 2023 年 HuggingFace Inc. 团队。保留所有权利。
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件将按“原样”分发，
# 无任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。

""" 用于从官方库加载 Pop2Piano 模型权重并展示 tokenizer 词汇构建方法的文件 """

import json  # 导入 JSON 模块
import torch  # 导入 PyTorch

from transformers import Pop2PianoConfig, Pop2PianoForConditionalGeneration  # 导入 Pop2Piano 相关类


########################## 模型权重 ##########################

# 这些权重是从官方 pop2piano 仓库下载的
# https://huggingface.co/sweetcocoa/pop2piano/blob/main/model-1999-val_0.67311615.ckpt
official_weights = torch.load("./model-1999-val_0.67311615.ckpt")
state_dict = {}  # 初始化状态字典


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

# 加载编码器和解码器的嵌入标记和最终层归一化
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

# 加载每个编码器块
for i in range(cfg.num_layers):
    # 第 i 层
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.q.weight"
    ]
    # 设置编码器（encoder）的每个块（block）中的 SelfAttention 模块的权重参数
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.k.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.v.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.o.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.layer_norm.weight"
    ]

    # 设置编码器（encoder）的每个块（block）中的第二层（layer 1）的 DenseReluDense 模块的权重参数
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wo.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.1.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.layer_norm.weight"
    ]
# 加载每个解码器块的权重

# 循环遍历6个解码器块
for i in range(6):
    # 第 0 层
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

    # 第 1 层
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

    # 第 2 层
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

# 使用加载的状态字典更新模型的权重
model.load_state_dict(state_dict, strict=True)

# 将模型的状态字典保存到文件
torch.save(state_dict, "./pytorch_model.bin")

########################## TOKENIZER ##########################

# tokenize 和 detokenize 方法来自官方实现

# 链接: https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L34
# 定义一个函数用于生成特定类型的令牌编号
def tokenize(idx, token_type, n_special=4, n_note=128, n_velocity=2):
    # 如果令牌类型是 TOKEN_TIME，返回对应的编号
    if token_type == "TOKEN_TIME":
        return n_special + n_note + n_velocity + idx
    # 如果令牌类型是 TOKEN_VELOCITY，返回对应的编号
    elif token_type == "TOKEN_VELOCITY":
        return n_special + n_note + idx
    # 如果令牌类型是 TOKEN_NOTE，返回对应的编号
    elif token_type == "TOKEN_NOTE":
        return n_special + idx
    # 如果令牌类型是 TOKEN_SPECIAL，返回对应的编号
    elif token_type == "TOKEN_SPECIAL":
        return idx
    # 如果令牌类型不在已知类型中，返回 -1
    else:
        return -1


# link : https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L48
# 定义一个函数用于将令牌编号反向解析为令牌类型和具体编号
def detokenize(idx, n_special=4, n_note=128, n_velocity=2, time_idx_offset=0):
    # 根据令牌编号判断其属于哪种类型的令牌，并返回对应的令牌类型和具体编号
    if idx >= n_special + n_note + n_velocity:
        return "TOKEN_TIME", (idx - (n_special + n_note + n_velocity)) + time_idx_offset
    elif idx >= n_special + n_note:
        return "TOKEN_VELOCITY", idx - (n_special + n_note)
    elif idx >= n_special:
        return "TOKEN_NOTE", idx - n_special
    else:
        return "TOKEN_SPECIAL", idx


# 创建一个空字典用于存储解析后的令牌编号和对应的字符串表示
decoder = {}
# 遍历令牌的总数，更新 decoder 字典，将每个令牌编号映射为其解析后的字符串表示
for i in range(cfg.vocab_size):
    decoder.update({i: f"{detokenize(i)[1]}_{detokenize(i)[0]}"})

# 创建一个 encoder 字典，将 decoder 中的键值对反转，用于编码时快速查找令牌编号
encoder = {v: k for k, v in decoder.items()}

# 将 encoder 字典保存为 JSON 文件，用于后续使用
with open("./vocab.json", "w") as file:
    file.write(json.dumps(encoder))
```