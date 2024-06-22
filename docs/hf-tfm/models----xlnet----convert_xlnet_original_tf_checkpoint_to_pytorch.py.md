# `.\transformers\models\xlnet\convert_xlnet_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# 文件编码声明以及版权声明

# 导入必要的库
import argparse
import os
import torch
from transformers import (
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetForSequenceClassification,
    XLNetLMHeadModel,
    load_tf_weights_in_xlnet,
)
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging

# GLUE 任务的标签数量
GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

# 设置日志级别
logging.set_verbosity_info()

# 将 XLNet 检查点转换为 PyTorch 格式
def convert_xlnet_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None
):
    # 初始化 PyTorch 模型
    config = XLNetConfig.from_json_file(bert_config_file)

    # 根据微调任务类型选择对应的模型
    finetuning_task = finetuning_task.lower() if finetuning_task is not None else ""
    if finetuning_task in GLUE_TASKS_NUM_LABELS:
        print(f"Building PyTorch XLNetForSequenceClassification model from configuration: {config}")
        config.finetuning_task = finetuning_task
        config.num_labels = GLUE_TASKS_NUM_LABELS[finetuning_task]
        model = XLNetForSequenceClassification(config)
    elif "squad" in finetuning_task:
        config.finetuning_task = finetuning_task
        model = XLNetForQuestionAnswering(config)
    else:
        model = XLNetLMHeadModel(config)

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
    print(f"Save PyTorch model to {os.path.abspath(pytorch_weights_dump_path)}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"Save configuration file to {os.path.abspath(pytorch_config_dump_path)}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())

# 如果是直接运行本文件，则执行以下代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必要的参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 添加一个命令行参数，用于指定 XLNet 预训练模型的配置文件路径
    parser.add_argument(
        "--xlnet_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained XLNet model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加一个命令行参数，用于指定存储 PyTorch 模型或数据集/词汇表的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to store the PyTorch model or dataset/vocab.",
    )
    # 添加一个命令行参数，用于指定 XLNet TensorFlow 模型微调的任务名称
    parser.add_argument(
        "--finetuning_task",
        default=None,
        type=str,
        help="Name of a task on which the XLNet TensorFlow model was fine-tuned",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 打印解析后的参数
    print(args)

    # 将 XLNet TensorFlow 模型转换成 PyTorch 模型
    convert_xlnet_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.xlnet_config_file, args.pytorch_dump_folder_path, args.finetuning_task
    )
```