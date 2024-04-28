# `.\transformers\models\longformer\convert_longformer_original_pytorch_lightning_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明代码使用 Apache License, Version 2.0
# 初始化 arg 解析器
import argparse
# 引入 pytorch_lightning 库并简写为 pl
import pytorch_lightning as pl
# 引入 torch 库
import torch
# 引入 torch 的 nn 模块
from torch import nn
# 引入 transformers 库中的 LongformerForQuestionAnswering 和 LongformerModel 类
from transformers import LongformerForQuestionAnswering, LongformerModel

# LightningModel 类继承自 pl.LightningModule
class LightningModel(pl.LightningModule):
    # 初始化函数，接受一个模型参数
    def __init__(self, model):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的模型参数赋值给 self.model
        self.model = model
        # 设置 self.num_labels 为 2
        self.num_labels = 2
        # 初始化一个线性层，输入维度为模型隐藏层的大小，输出维度为 self.num_labels
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, self.num_labels)

    # 实现 forward 方法，由于 lightning 要求必须实现此方法，但该方法为空
    def forward(self):
        pass

# 定义一个函数，用于将 Longformer 问题回答模型的检查点转换为 PyTorch 格式
def convert_longformer_qa_checkpoint_to_pytorch(
    longformer_model: str, longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str
):
    # 从模型标识符加载 Longformer 模型
    longformer = LongformerModel.from_pretrained(longformer_model)
    # 初始化 LightningModel 对象，传入 Longformer 模型
    lightning_model = LightningModel(longformer)

    # 加载长模型问题回答模型的检查点
    ckpt = torch.load(longformer_question_answering_ckpt_path, map_location=torch.device("cpu"))
    # 加载检查点的状态字典到 lightning_model 对象中
    lightning_model.load_state_dict(ckpt["state_dict"])

    # 初始化 Longformer 问题回答模型
    longformer_for_qa = LongformerForQuestionAnswering.from_pretrained(longformer_model)

    # 转移权重
    longformer_for_qa.longformer.load_state_dict(lightning_model.model.state_dict())
    longformer_for_qa.qa_outputs.load_state_dict(lightning_model.qa_outputs.state_dict())
    longformer_for_qa.eval()

    # 保存模型
    longformer_for_qa.save_pretrained(pytorch_dump_folder_path)

    # 输出转换成功的消息，以及保存模型的路径
    print(f"Conversion successful. Model saved under {pytorch_dump_folder_path}")


# 如果当前文件被直接执行
if __name__ == "__main__":
    # 创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--longformer_model",
        default=None,
        type=str,
        required=True,
        help="model identifier of longformer. Should be either `longformer-base-4096` or `longformer-large-4096`.",
    )
    parser.add_argument(
        "--longformer_question_answering_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch Lightning Checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析参数
    args = parser.parse_args()
    # 调用函数将 Longformer QA 模型的检查点转换为 PyTorch 格式
    convert_longformer_qa_checkpoint_to_pytorch(
        args.longformer_model, args.longformer_question_answering_ckpt_path, args.pytorch_dump_folder_path
    )
```