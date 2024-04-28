# `.\models\deprecated\trajectory_transformer\convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# 版权声明和许可信息
# 版权所有 2022 年 轨迹变换器 论文作者 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（"许可证"）获得许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 无论是明示还是暗示的，都没有任何担保或条件。
# 有关许可证的详细信息，请参阅许可证。
""" 将 TrajectoryTransformer pytorch 检查点转换为 pytorch """

import torch
import trajectory.utils as utils

from transformers import TrajectoryTransformerModel


class Parser(utils.Parser):
    dataset: str = "halfcheetah-medium-expert-v2"
    config: str = "config.offline"


def convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch(logbase, dataset, loadpath, epoch, device):
    """将顺序块转换为模块列表"""

    # 加载模型
    gpt, gpt_epoch = utils.load_model(logbase, dataset, loadpath, epoch=epoch, device=device)
    # 创建 TrajectoryTransformerModel 实例
    trajectory_transformer = TrajectoryTransformerModel(gpt.config)

    # 拷贝参数
    trajectory_transformer.tok_emb.load_state_dict(gpt.tok_emb.state_dict())
    trajectory_transformer.pos_emb = gpt.pos_emb
    trajectory_transformer.drop.load_state_dict(gpt.drop.state_dict())
    trajectory_transformer.ln_f.load_state_dict(gpt.ln_f.state_dict())
    trajectory_transformer.head.load_state_dict(gpt.head.state_dict())

    # 遍历每个块并拷贝参数
    for i, block in enumerate(gpt.blocks):
        trajectory_transformer.blocks[i].ln1.load_state_dict(gpt.blocks[i].ln1.state_dict())
        trajectory_transformer.blocks[i].ln2.load_state_dict(gpt.blocks[i].ln2.state_dict())
        trajectory_transformer.blocks[i].attn.load_state_dict(gpt.blocks[i].attn.state_dict())

        trajectory_transformer.blocks[i].l1.load_state_dict(gpt.blocks[i].mlp[0].state_dict())
        trajectory_transformer.blocks[i].act.load_state_dict(gpt.blocks[i].mlp[1].state_dict())
        trajectory_transformer.blocks[i].l2.load_state_dict(gpt.blocks[i].mlp[2].state_dict())
        trajectory_transformer.blocks[i].drop.load_state_dict(gpt.blocks[i].mlp[3].state_dict())

    # 保存新模型的参数
    torch.save(trajectory_transformer.state_dict(), "pytorch_model.bin")


if __name__ == "__main__":
    """
    要运行此脚本，您需要安装原始存储库以运行原始模型。您可以在这里找到它
    https://github.com/jannerm/trajectory-transformer 从该存储库代码中，您还可以下载
    原始 pytorch 检查点。

    使用以下命令运行：

    ```sh
    >>> python convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py --dataset <dataset_name>
    ...     --gpt_loadpath <path_to_original_pytorch_checkpoint>
    ```
    """

    # 解析命令行参数
    args = Parser().parse_args("plan")
    # 调用函数convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch，将原始的PyTorch检查点转换为PyTorch格式
    (
        args.logbase, args.dataset, args.gpt_loadpath, args.gpt_epoch, args.device
    )
```