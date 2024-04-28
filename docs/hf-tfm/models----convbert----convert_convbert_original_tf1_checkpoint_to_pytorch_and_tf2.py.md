# `.\models\convbert\convert_convbert_original_tf1_checkpoint_to_pytorch_and_tf2.py`

```py
# coding=utf-8
# 版权声明
# 基于Apache License, Version 2.0 (许可证)授权
# 除非符合许可证规定，否则不得使用该文件
# 可以获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律有要求或书面同意，否则软件将基于“原样”分发，无论是明示还是暗示的保证或条件
# 有关特定语言的具体语言，参见许可证
# 在许可证的限制下，实现权限与限制

"""Convert ConvBERT checkpoint."""

# 导入模块
import argparse
# 从transformers库中导入ConvBertConfig, ConvBertModel, TFConvBertModel, load_tf_weights_in_convbert
from transformers import ConvBertConfig, ConvBertModel, TFConvBertModel, load_tf_weights_in_convbert
# 从transformers的utils模块中导入logging
from transformers.utils import logging

# 设置日志的输出级别为info
logging.set_verbosity_info()

# 定义函数：将原始TF1检查点转换为pytorch
def convert_orig_tf1_checkpoint_to_pytorch(tf_checkpoint_path, convbert_config_file, pytorch_dump_path):
    # 从json文件中加载ConvBertConfig，得到配置对象
    conf = ConvBertConfig.from_json_file(convbert_config_file)
    # 根据配置对象创建ConvBertModel模型
    model = ConvBertModel(conf)

    # 加载TF检查点中的权重到ConvBertModel模型中
    model = load_tf_weights_in_convbert(model, conf, tf_checkpoint_path)
    # 将转换后的模型保存为预训练模型
    model.save_pretrained(pytorch_dump_path)

    # 从转换后的pytorch模型创建TFConvBertModel模型
    tf_model = TFConvBertModel.from_pretrained(pytorch_dump_path, from_pt=True)
    # 将TFConvBertModel模型保存为预训练模型
    tf_model.save_pretrained(pytorch_dump_path)

# 主函数入口
if __name__ == "__main__":
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    # 必填的参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--convbert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained ConvBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，进行TF1检查点到pytorch模型的转换
    convert_orig_tf1_checkpoint_to_pytorch(args.tf_checkpoint_path, args.convbert_config_file, args.pytorch_dump_path)
```