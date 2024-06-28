# `.\models\gpt_neo\convert_gpt_neo_mesh_tf_to_pytorch.py`

```py
# 导入必要的模块和函数
import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 格式数据的模块

from transformers import GPTNeoConfig, GPTNeoForCausalLM, load_tf_weights_in_gpt_neo  # 导入 GPT-Neo 相关的类和函数
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志记录级别为信息

# 定义函数，用于将 TensorFlow 的检查点文件转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # 从配置文件中加载 JSON 数据
    config_json = json.load(open(config_file, "r"))
    
    # 根据加载的配置数据创建 GPTNeoConfig 对象
    config = GPTNeoConfig(
        hidden_size=config_json["n_embd"],  # 设置隐藏层的大小
        num_layers=config_json["n_layer"],  # 设置层数
        num_heads=config_json["n_head"],  # 设置注意力头数
        attention_types=config_json["attention_types"],  # 设置注意力类型
        max_position_embeddings=config_json["n_positions"],  # 设置最大位置编码长度
        resid_dropout=config_json["res_dropout"],  # 设置残差连接的 dropout 率
        embed_dropout=config_json["embed_dropout"],  # 设置嵌入层的 dropout 率
        attention_dropout=config_json["attn_dropout"],  # 设置注意力层的 dropout 率
    )
    
    # 打印配置信息
    print(f"Building PyTorch model from configuration: {config}")
    
    # 根据配置创建 GPTNeoForCausalLM 模型对象
    model = GPTNeoForCausalLM(config)

    # 加载 TensorFlow 检查点中的权重到 PyTorch 模型中
    load_tf_weights_in_gpt_neo(model, config, tf_checkpoint_path)

    # 将 PyTorch 模型保存到指定路径
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加必需的命令行参数
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

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数，将 TensorFlow 检查点转换为 PyTorch 模型
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
```