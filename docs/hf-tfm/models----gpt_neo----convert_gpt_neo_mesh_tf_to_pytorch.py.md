# `.\models\gpt_neo\convert_gpt_neo_mesh_tf_to_pytorch.py`

```
# coding=utf-8
# 版权所有 2021 年 Eleuther AI 和 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下地址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何形式的担保或条件，无论是明示的还是隐含的。
# 请参阅许可证以获取具体语言规定的权限和限制
"""Convert GPT Neo checkpoint."""

# 导入所需的库和模块
import argparse
import json
# 从transformers模块中导入GPTNeoConfig、GPTNeoForCausalLM和load_tf_weights_in_gpt_neo
from transformers import GPTNeoConfig, GPTNeoForCausalLM, load_tf_weights_in_gpt_neo
from transformers.utils import logging

# 设置日志级别为info
logging.set_verbosity_info()

# 定义函数，将 TensorFlow 检查点转换为 PyTorch 检查点
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # 从配置文件中加载 JSON 配置
    config_json = json.load(open(config_file, "r"))
    # 根据 JSON 配置创建 GPTNeoConfig 对象
    config = GPTNeoConfig(
        hidden_size=config_json["n_embd"],
        num_layers=config_json["n_layer"],
        num_heads=config_json["n_head"],
        attention_types=config_json["attention_types"],
        max_position_embeddings=config_json["n_positions"],
        resid_dropout=config_json["res_dropout"],
        embed_dropout=config_json["embed_dropout"],
        attention_dropout=config_json["attn_dropout"],
    )
    # 打印创建的 PyTorch 模型配置信息
    print(f"Building PyTorch model from configuration: {config}")
    # 创建 GPTNeoForCausalLM 模型
    model = GPTNeoForCausalLM(config)

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_gpt_neo(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

# 当该脚本被直接运行时执行以下操作
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained mesh-tf model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    # 调用 convert_tf_checkpoint_to_pytorch 函数并传递参数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
```