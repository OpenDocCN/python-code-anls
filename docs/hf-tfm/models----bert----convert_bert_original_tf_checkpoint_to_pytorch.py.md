# `.\transformers\models\bert\convert_bert_original_tf_checkpoint_to_pytorch.py`

```
# 设置编码格式为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""将 BERT 检查点转换为 PyTorch 格式。"""

# 导入所需的库
import argparse
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()

# 定义函数将 TensorFlow 检查点转换为 PyTorch 格式
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # 从配置文件中加载 BertConfig
    config = BertConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 根据配置文件创建 BertForPreTraining 模型
    model = BertForPreTraining(config)

    # 从 TensorFlow 检查点中加载权重
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)

# 如果作为独立脚本运行，则解析命令行参数并调用转换函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    # 调用转换函数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
```