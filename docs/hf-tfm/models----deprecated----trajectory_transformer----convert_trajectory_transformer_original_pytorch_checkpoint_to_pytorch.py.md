# `.\models\deprecated\trajectory_transformer\convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2022 The Trajectory Transformers paper authors and The HuggingFace Inc. team. All rights reserved.
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
""" TrajectoryTransformer pytorch checkpoint conversion"""

import torch
import trajectory.utils as utils

from transformers import TrajectoryTransformerModel


class Parser(utils.Parser):
    dataset: str = "halfcheetah-medium-expert-v2"  # 设置默认数据集名称为 'halfcheetah-medium-expert-v2'
    config: str = "config.offline"  # 设置默认配置文件名称为 'config.offline'


def convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch(logbase, dataset, loadpath, epoch, device):
    """Converting Sequential blocks to ModuleList"""
    # 调用外部函数加载模型 gpt，并指定使用的设备和加载的 epoch
    gpt, gpt_epoch = utils.load_model(logbase, dataset, loadpath, epoch=epoch, device=device)
    # 根据 gpt 的配置创建 TrajectoryTransformerModel 实例
    trajectory_transformer = TrajectoryTransformerModel(gpt.config)

    # 将 gpt 的 token embedding 的状态字典加载到 trajectory_transformer 中
    trajectory_transformer.tok_emb.load_state_dict(gpt.tok_emb.state_dict())
    # 直接赋值 gpt 的 position embedding 到 trajectory_transformer 的 position embedding
    trajectory_transformer.pos_emb = gpt.pos_emb
    # 将 gpt 的 dropout 层的状态字典加载到 trajectory_transformer 中
    trajectory_transformer.drop.load_state_dict(gpt.drop.state_dict())
    # 将 gpt 的 layer normalization 层的状态字典加载到 trajectory_transformer 中
    trajectory_transformer.ln_f.load_state_dict(gpt.ln_f.state_dict())
    # 将 gpt 的 head 层的状态字典加载到 trajectory_transformer 中
    trajectory_transformer.head.load_state_dict(gpt.head.state_dict())

    # 遍历 gpt 的各个 block
    for i, block in enumerate(gpt.blocks):
        # 将 gpt 的第 i 个 block 的 layer normalization 层的状态字典加载到 trajectory_transformer 的第 i 个 block 中
        trajectory_transformer.blocks[i].ln1.load_state_dict(gpt.blocks[i].ln1.state_dict())
        trajectory_transformer.blocks[i].ln2.load_state_dict(gpt.blocks[i].ln2.state_dict())
        # 将 gpt 的第 i 个 block 的 attention 层的状态字典加载到 trajectory_transformer 的第 i 个 block 中
        trajectory_transformer.blocks[i].attn.load_state_dict(gpt.blocks[i].attn.state_dict())

        # 将 gpt 的第 i 个 block 的 MLP 的各层状态字典加载到 trajectory_transformer 的第 i 个 block 中
        trajectory_transformer.blocks[i].l1.load_state_dict(gpt.blocks[i].mlp[0].state_dict())
        trajectory_transformer.blocks[i].act.load_state_dict(gpt.blocks[i].mlp[1].state_dict())
        trajectory_transformer.blocks[i].l2.load_state_dict(gpt.blocks[i].mlp[2].state_dict())
        trajectory_transformer.blocks[i].drop.load_state_dict(gpt.blocks[i].mlp[3].state_dict())

    # 将转换后的模型的状态字典保存为 pytorch_model.bin 文件
    torch.save(trajectory_transformer.state_dict(), "pytorch_model.bin")


if __name__ == "__main__":
    """
    To run this script you will need to install the original repository to run the original model. You can find it
    here: https://github.com/jannerm/trajectory-transformer From this repository code you can also download the
    original pytorch checkpoints.

    Run with the command:

    ```
    >>> python convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py --dataset <dataset_name>
    ...     --gpt_loadpath <path_to_original_pytorch_checkpoint>
    ```
    """
    # 使用 Parser 类解析命令行参数并设置默认参数值
    args = Parser().parse_args("plan")
    # 调用函数 convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch，转换原始 PyTorch 检查点到 PyTorch 格式
    # 使用参数 args.logbase 作为日志基础路径
    # 使用参数 args.dataset 指定数据集
    # 使用参数 args.gpt_loadpath 指定 GPT 模型加载路径
    # 使用参数 args.gpt_epoch 指定 GPT 模型的 epoch 数
    # 使用参数 args.device 指定设备（如 GPU 或 CPU）
    convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch(
        args.logbase, args.dataset, args.gpt_loadpath, args.gpt_epoch, args.device
    )
```