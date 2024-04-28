# `.\transformers\models\big_bird\convert_bigbird_original_tf_checkpoint_to_pytorch.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""将 BigBird 检查点转换为 PyTorch 模型。"""

# 导入所需库
import argparse
from transformers import BigBirdConfig, BigBirdForPreTraining, BigBirdForQuestionAnswering, load_tf_weights_in_big_bird
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()

# 定义函数，将 TensorFlow 检查点转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, big_bird_config_file, pytorch_dump_path, is_trivia_qa):
    # 从配置文件中加载 BigBird 配置
    config = BigBirdConfig.from_json_file(big_bird_config_file)
    print(f"Building PyTorch model from configuration: {config}")

    # 根据是否为 trivia_qa 模型选择不同的 BigBird 模型
    if is_trivia_qa:
        model = BigBirdForQuestionAnswering(config)
    else:
        model = BigBirdForPreTraining(config)

    # 从 TensorFlow 检查点中加载权重
    load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=is_trivia_qa)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--big_bird_config_file",
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
    parser.add_argument(
        "--is_trivia_qa", action="store_true", help="Whether to convert a model with a trivia_qa head."
    )
    args = parser.parse_args()
    # 调用函数进行转换
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.big_bird_config_file, args.pytorch_dump_path, args.is_trivia_qa
    )
```